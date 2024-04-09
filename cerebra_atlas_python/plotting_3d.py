import time
import open3d as o3d
import numpy as np

from cerebra_atlas_python.plotting import get_cmap_colors
from cerebra_atlas_python.cerebra_mne import CerebraMNE

def create_plot(background_color=[1, 1, 1], draw_bounding_box=True):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().background_color = background_color

    # Get the view control and then the camera parameters
    view_control = vis.get_view_control()
    cam_params = view_control.convert_to_pinhole_camera_parameters()

    # Define the angle of rotation around the Y-axis (in radians)
    angle = np.radians(45)

    # Create a rotation matrix around the Y-axis
    R = np.array([[np.cos(angle), 0, np.sin(angle), 0],
                [0, 1, 0, 0],
                [-np.sin(angle), 0, np.cos(angle), 0],
                [0, 0, 0, 1]])

    # Apply the rotation to the camera's extrinsic matrix
    cam_params.extrinsic = np.dot(R, cam_params.extrinsic)

    # Apply the new camera parameters
    view_control.convert_from_pinhole_camera_parameters(cam_params)

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
        ])
        points_o3d = o3d.utility.Vector3dVector(bb_points)
        pc = o3d.geometry.PointCloud(points_o3d)
        pc.colors = o3d.utility.Vector3dVector(bb_colors)
        vis.add_geometry(pc)
        
    
    return vis

class PointCloud:
    def __init__(self, points: np.ndarray, colors=None):
        self.points = points
        if colors is None:
            colors = np.zeros_like(points)
        elif len(colors) == 1:
            colors = np.repeat(colors, len(points), axis=0)
        else:
            assert len(colors) == len(points), "if colors are provided length should match points"
        self.colors = colors

def add_point_clouds(vis, *point_clouds: PointCloud):

    for pc in point_clouds:
        points_o3d = o3d.utility.Vector3dVector(pc.points)
        pc_o3d = o3d.geometry.PointCloud(points_o3d)
        pc_o3d.colors = o3d.utility.Vector3dVector(pc.colors)
        vis.add_geometry(pc_o3d)

        # trans_init = np.asarray(
        #     [[0, 0, -0,  0],
        #     [0, 0, 0,  0],
        #     [0, 0,  0, 0],
        #     [0.0, 0.0, 0.0, 1.0]])
        # pc_o3d.transform(trans_init)
        R = pc_o3d.get_rotation_matrix_from_xyz((np.pi/2+ np.pi/16, np.pi , np.pi/2- np.pi/4))
        pc_o3d.rotate(R, center=(128, 128, 128))
        pc_o3d.scale(0.5, center=(128, 128, 128))

    return vis

def add_meshes(*meshes):
    for mesh in meshes:
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)

    pass

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

    print(cerebra_non_cortical.bem_surfaces)

    # non_cortical_mask = np.logical_xor (cerebra_cortical.src_space_mask, cerebra_non_cortical.src_space_mask)
    # non_cortical_points = np.indices([256, 256, 256])[:, non_cortical_mask].T
    # colors_cortical = [colors_hex[label] for label in cerebra_cortical.src_space_labels]

    # vis = create_plot()
    # # Load the source space
    # src_space_pc = PointCloud(non_cortical_points)
    # src_space_pc_2 = PointCloud(cerebra_cortical.src_space_points, colors_cortical)
    # src_space_pc_3 = PointCloud(cerebra_cortical.bem_surfaces, [1,0,1])
    # vis = add_point_clouds(vis, src_space_pc, src_space_pc_2, src_space_pc_3)
    # run(vis)

