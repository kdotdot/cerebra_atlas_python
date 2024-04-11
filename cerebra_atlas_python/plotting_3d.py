import time
import open3d as o3d
# pylint: disable=no-name-in-module
from open3d.visualization import Visualizer # CUDA/CPU
import numpy as np

from cerebra_atlas_python.plotting import get_cmap_colors
from cerebra_atlas_python.cerebra_mne import CerebraMNE

def create_plot(background_color=[1, 1, 1], draw_bounding_box=True):
    vis = Visualizer()
    vis.create_window()
    vis.get_render_option().background_color = background_color

    # Get the view control and then the camera parameters
    # view_control = vis.get_view_control()
    # cam_params = view_control.convert_to_pinhole_camera_parameters()

    # Define the angle of rotation around the Y-axis (in radians)
    # angle = np.radians(45)

    # Create a rotation matrix around the Y-axis
    # R = np.array([[np.cos(angle), 0, np.sin(angle), 0],
    #             [0, 1, 0, 0],
    #             [-np.sin(angle), 0, np.cos(angle), 0],
    #             [0, 0, 0, 1]])

    # Apply the rotation to the camera's extrinsic matrix
    # cam_params.extrinsic = np.dot(R, cam_params.extrinsic)

    # Apply the new camera parameters
    # view_control.convert_from_pinhole_camera_parameters(cam_params)

    if draw_bounding_box:
        bb_points = np.array(   
        [
            [0, 0, 0],
            [0, 0, 255],
            [0, 255, 0],
            [0, 255, 255],
            [255, 0, 0],
            [255, 0, 255],
            [255, 255, 0],
            [255, 255, 255],
        ])
        bb_colors = np.array(
        [
            [100, 0, 0],
            [0, 0, 255],
            [0, 255, 0],
            [0, 255, 255],
            [255, 0, 0],
            [255, 0, 255],
            [255, 255, 0],
            [255, 255, 255],
        ])    # Get the view control and then the camera parameters
    # view_control = vis.get_view_control()
    # cam_params = view_control.convert_to_pinhole_camera_parameters()

    # Define the angle of rotation around the Y-axis (in radians)
    # angle = np.radians(45)

    # Create a rotation matrix around the Y-axis
    # R = np.array([[np.cos(angle), 0, np.sin(angle), 0],
    #             [0, 1, 0, 0],
    #             [-np.sin(angle), 0, np.cos(angle), 0],
    #             [0, 0, 0, 1]])

    # Apply the rotation to the camera's extrinsic matrix
    # cam_params.extrinsic = np.dot(R, cam_params.extrinsic)

    # Apply the new camera parameters
    # view_control.convert_from_pinhole_camera_parameters(cam_params)
        
    
    return vis

class Drawable:
    data = None
    def __init__(self, colors=None):
        if self.data is None:
            raise NotImplementedError("data should be defined")
        if colors is None:
            colors = np.zeros_like(self.data)
        elif len(colors) == 3 or len(colors) == 4:
            colors = np.repeat([colors], len(self.data), axis=0)
            print(colors.shape)
        else:
            assert len(colors) == len(self.data), "if colors are provided length should match points"
        self.colors = colors

class PointCloud(Drawable):
    def __init__(self, points: np.ndarray,*args,**kwargs):
        self.points = points
        self.data = points
        Drawable.__init__(self,*args,**kwargs)
    
    def get_o3d(self):
        points_o3d = o3d.utility.Vector3dVector(self.points)
        pc_o3d = o3d.geometry.PointCloud(points_o3d)
        pc_o3d.colors = o3d.utility.Vector3dVector(self.colors)
        
        # return pc_o3d.voxel_down_sample(voxel_size=100)
        return pc_o3d

class Mesh(Drawable):
    def __init__(self, mesh_vertices: np.ndarray, mesh_normals:np.ndarray, mesh_triangles:np.ndarray, *args,**kwargs):
        self.mesh_vertices = mesh_vertices
        self.data = mesh_vertices
        self.mesh_normals = mesh_normals
        self.mesh_triangles = mesh_triangles
        Drawable.__init__(self,*args,**kwargs)

    def get_o3d(self):
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(self.mesh_vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(self.mesh_triangles)
        mesh_o3d.vertex_normals = o3d.utility.Vector3dVector(self.mesh_normals)
        mesh_o3d.compute_vertex_normals()
        mesh_o3d.paint_uniform_color([0.1,0.1,0.9])
        return mesh_o3d



def add_drawable(vis, *drawables: Drawable):
    for obj in drawables:
        o3d_obj = obj.get_o3d()
        vis.add_geometry(o3d_obj)
        R = o3d_obj.get_rotation_matrix_from_xyz((np.pi/2+ np.pi/16, np.pi , np.pi/2- np.pi/4))
        o3d_obj.rotate(R, center=(128, 128, 128))
        o3d_obj.scale(0.5, center=(128, 128, 128))

    return vis

def run(vis, MAX_FRAME=int(1e6)):
    for _ in range(MAX_FRAME):
        vis.poll_events()
        vis.update_renderer()
        if not vis.poll_events():
            break
    vis.destroy_window()



if __name__ == "__main__":
    colors_hex = get_cmap_colors()

    cerebra_cortical = CerebraMNE(source_space_include_non_cortical=False)
    cerebra_non_cortical = CerebraMNE(source_space_include_non_cortical=True)

    print(cerebra_cortical.get_bem_triangles())
    #print(cerebra_non_cortical.bem["surfs"][0].keys())

    non_cortical_mask = np.logical_xor (cerebra_cortical.src_space_mask, cerebra_non_cortical.src_space_mask)
    non_cortical_points = np.indices([256, 256, 256])[:, non_cortical_mask].T
    colors_cortical = [colors_hex[label] for label in cerebra_cortical.src_space_labels]

    bem_colors = [[0,0.1,1],[0.1,0.2,0.9],[0.2,0.1,0.95]]
    

    # bem_meshes = [Mesh(cerebra_cortical.get_bem_vertices_ras()[i],cerebra_cortical.get_bem_normals_ras()[i],cerebra_cortical.get_bem_triangles()[i], colors=[0,0,1,0.5]) for i in range(3)]
    bem_pcs = [PointCloud(cerebra_cortical.get_bem_vertices_ras()[i], colors=bem_colors[i]) for i in range(3)]
    fiducials_pc = PointCloud(cerebra_cortical.get_fiducials_ras(), [0,1,0])

    vis = create_plot()
    # # Load the source space
    # src_space_pc = PointCloud(non_cortical_points, [0.2,0.2,0.2])
    src_space_pc_2 = PointCloud(cerebra_cortical.src_space_points, [0.9,0.0,0.9])
    # # 
    # src_space_pc_4 = PointCloud(cerebra_cortical.get_fiducials_ras(), [0,1,0])
    # vis = add_drawable(vis, *bem_meshes,fiducials_pc)
    vis = add_drawable(vis, src_space_pc_2, *bem_pcs, fiducials_pc)
    run(vis)

