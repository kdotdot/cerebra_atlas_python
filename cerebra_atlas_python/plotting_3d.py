import time
import open3d as o3d
# pylint: disable=no-name-in-module
from open3d.visualization import Visualizer # CUDA/CPU
import numpy as np

from cerebra_atlas_python.plotting import get_cmap_colors
from cerebra_atlas_python import CerebrA
from cerebra_atlas_python.utils import rgb_to_hex_str


class Drawable:
    data = None
    def __init__(self, colors=None):
        if self.data is None:
            raise NotImplementedError("data should be defined")
        if colors is None:
            colors = np.zeros_like(self.data)
        elif len(colors) == 3 or len(colors) == 4:
            colors = np.repeat([colors], len(self.data), axis=0)
        else:
            assert len(colors) == len(self.data), "if colors are provided length should match points"
        self.colors = colors

        self.o3d = None
        self._set_o3d()

    def get_o3d(self):
        return self.o3d
    
    def _set_o3d(self):
        raise NotImplementedError("_set_o3d should be implemented")

class PointCloud(Drawable):
    def __init__(self, points: np.ndarray,*args,**kwargs):
        self.points = points
        self.data = points
        Drawable.__init__(self,*args,**kwargs)
        
    
    def _set_o3d(self):
        points_o3d = o3d.utility.Vector3dVector(self.points)
        pc_o3d = o3d.geometry.PointCloud(points_o3d)
        pc_o3d.colors = o3d.utility.Vector3dVector(self.colors)
        self.o3d = pc_o3d

    def update_colors(self, colors):
        self.o3d.colors = o3d.utility.Vector3dVector(colors)

class Mesh(Drawable):
    def __init__(self, mesh_vertices: np.ndarray, mesh_triangles:np.ndarray, mesh_normals:np.ndarray = None, *args,**kwargs):
        self.mesh_vertices = mesh_vertices
        self.data = mesh_vertices
        self.mesh_normals = mesh_normals
        self.mesh_triangles = mesh_triangles
        Drawable.__init__(self,*args,**kwargs)
        # if self.mesh_normals is None:
        #     mesh_o3d = self.get_o3d()
        #     mesh_o3d.compute_vertex_normals()
        #     self.mesh_normals = np.asarray(mesh_o3d.vertex_normals)

    def _set_o3d(self):
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.paint_uniform_color([0.1,0.1,0.9])
        self.o3d = mesh_o3d
        self.update_mesh(self.mesh_vertices, self.mesh_triangles, self.mesh_normals)

        
    def update_mesh(self, vertices, triangles, normals=None):
        self.o3d.vertices = o3d.utility.Vector3dVector(vertices)
        self.o3d.triangles = o3d.utility.Vector3iVector(triangles)
        if self.mesh_normals is not None:
            self.o3d.compute_vertex_normals()
            self.o3d.vertex_normals = o3d.utility.Vector3dVector(normals)
            
    
    def flip_normals(self):
        self.mesh_normals = self.mesh_normals * -1





def add_drawable(vis, *drawables: Drawable, translate=True, reset_bounding_box=True):
    for obj in drawables:
        o3d_obj = obj.get_o3d()
        vis.add_geometry(o3d_obj,reset_bounding_box=reset_bounding_box)
        if translate:
            cerebra = CerebrA()
            o3d_obj.translate(cerebra.affine[:3,3])
            #R = o3d_obj.get_rotation_matrix_from_xyz((np.pi/2- np.pi/32, np.pi , np.pi/2- np.pi/4- np.pi/16))
            # o3d_obj.rotate(R, center=(128, 128, 128))
            # o3d_obj.scale(0.7, center=(128, 128, 128))

    return vis

def create_plot(background_color=[1, 1, 1], draw_bounding_box=False, plot_coordinate_frame=False):
    vis = Visualizer()
    vis.create_window(width=1024, height=1024)# width=1080, height=1080
    vis.get_render_option().background_color = background_color
    vis.get_render_option().point_size = 15
    vis.get_render_option().show_coordinate_frame = True
    vis.get_render_option().mesh_show_wireframe = True
    vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Normal
    print(vis.get_render_option())
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
        add_drawable(vis, PointCloud(bb_points, colors=bb_colors))

    if plot_coordinate_frame:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=16)
        vis.add_geometry(axis)

    return vis

def rotate_camera(vis, rotate_mode = 1):
    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    
    if rotate_mode == 0 or rotate_mode == "0":
        pass
    elif rotate_mode == 1:
        ctr.rotate(0, -cam_params.intrinsic.height//2)#
        ctr.rotate(-3*cam_params.intrinsic.width//4 ,0)#
        ctr.rotate(0, cam_params.intrinsic.height//20)#
    elif rotate_mode == 2:
        ctr.rotate(0, -cam_params.intrinsic.height//2)#
        ctr.rotate(-1*cam_params.intrinsic.width//4 ,0)
        ctr.rotate(0, cam_params.intrinsic.height//20)#
    elif rotate_mode == 3:
        ctr.rotate(0, -cam_params.intrinsic.height)#
        # ctr.rotate(-1*cam_pararuninsic.width ,0)
    elif rotate_mode == "o1":
        ctr.rotate(0, -cam_params.intrinsic.height//2)
        ctr.rotate(cam_params.intrinsic.width ,0)
    elif rotate_mode == "o2":
        ctr.rotate(0, -cam_params.intrinsic.height//2)
        ctr.rotate(cam_params.intrinsic.width ,0)
    elif rotate_mode == "o3":
        ctr.rotate(0, -cam_params.intrinsic.height//2)
        ctr.rotate(cam_params.intrinsic.width ,0)
    elif rotate_mode == "iso":
        ctr.rotate(0, -cam_params.intrinsic.height//2)
        ctr.rotate(5*cam_params.intrinsic.width//4 ,0)
        ctr.rotate(0, cam_params.intrinsic.height//8)
        # ctr.rotate(0, cam_params.intrinsic.height/2)#
        pass
    elif rotate_mode == "planta":
        pass
    elif rotate_mode == "planta_inf":
        ctr.rotate(0, -cam_params.intrinsic.height)
    elif rotate_mode == "alzado":
        ctr.rotate(0,cam_params.intrinsic.height//2)
    elif rotate_mode == "alzado_post":
        ctr.rotate(0,-cam_params.intrinsic.height//2)
    elif rotate_mode == "perfil_derecho":
        ctr.rotate(-cam_params.intrinsic.width//2.0,0)
    elif rotate_mode == "perfil_izquierdo":
        ctr.rotate(cam_params.intrinsic.width//2.0,0)
    else:
        raise ValueError("Invalid rotate mode")
    ctr.change_field_of_view(step=-10)
    ctr.set_zoom(0.7)   
    

def run(vis, update_fn=None, *update_fn_args, MAX_FRAME=int(1e6), **update_fn_kwargs):
    for frame in range(MAX_FRAME):
        vis.poll_events()
        
        if not vis.poll_events():
            break
        if update_fn is not None:
            update_fn(vis,frame=frame, *update_fn_args,**update_fn_kwargs)

        vis.update_renderer()
    vis.destroy_window()

def create_text_mesh(text, translation=np.array([0, 0, 0]), scale=1, rotate = None):
    # Create text mesh
    mesh = o3d.t.geometry.TriangleMesh.create_text(text, depth=4)
    text_mesh_center = mesh.get_axis_aligned_bounding_box().get_center().numpy()
    translation = translation - text_mesh_center
    
    # mesh.flip_normals()

    # Color the mesh (optional)
    # mesh.vertex_colors = o3d.core.Tensor([[1, 0, 0]], dtype=o3d.core.Dtype.Float32, device=o3d.core.Device("CPU:0"))
    
    if rotate is None:
        rotate = np.eye(3)
    # Transform the mesh (optional)
    # transform = np.eye(4)
    # transform[:3, 3] = translation
    # transform[:3,:3] = rotate
    mesh = mesh.translate(translation)
    mesh = mesh.rotate(rotate, center=[0,0,0])
    # Flip normals
    mesh.compute_vertex_normals() 
    # mesh.mesh_normals = np.asarray(mesh_o3d.vertex_normals) * -1

    # if rotate is not None:
    #     mesh.o3d.rotate(rotate, center=(0, 0, 0) if translation is None else translation/2)

    # mesh.get_o3d().transform(transform)
    
    # Move mesh to GPU
    # mesh = mesh.to(device=o3d.core.Device("CUDA:0"))
    if scale != 1:
        mesh = mesh.scale(scale, np.array([0, 0, 0]))

    mesh = Mesh(mesh_vertices=mesh.vertex["positions"].numpy(), mesh_triangles=mesh.triangle["indices"].numpy(), mesh_normals=mesh.vertex["normals"].numpy())

    return mesh

def rotation_matrix(degrees):
    # Unpack the rotation degrees for x, y, z axes
    x, y, z = degrees
    
    # Convert degrees to radians
    x = np.deg2rad(x)
    y = np.deg2rad(y)
    z = np.deg2rad(z)
    
    # Rotation matrix for X-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]
    ])
    
    # Rotation matrix for Y-axis
    Ry = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])
    
    # Rotation matrix for Z-axis
    Rz = np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z), 0],
        [0, 0, 1]
    ])
    
    # Combine the rotations
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

# def tensor_mesh_to_legacy(tensor_mesh):
#     # Convert tensor vertices and triangles to numpy arrays
#     vertices = tensor_mesh.vertices.cpu().numpy()
#     triangles = tensor_mesh.triangles.cpu().numpy()
    
#     # Create a legacy mesh and set vertices and triangles
    
#     legacy_mesh.vertices = o3d.utility.Vector3dVector(vertices)
#     legacy_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
#     # If the tensor mesh has vertex colors, convert them too
#     if tensor_mesh.has_vertex_colors():
#         colors = tensor_mesh.vertex_colors.cpu().numpy()
#         legacy_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
#     return legacy_mesh

def text_3d(text, pos, direction=None, degree=0.0, font='/usr/share/fonts/truetype/lato/Lato-Regular.ttf', font_size=16):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    from PIL import Image, ImageFont, ImageDraw
    from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font, font_size)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 100.0)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd

if __name__ == "__main__":

    colors_hex = get_cmap_colors()
    cerebra = CerebrA(source_space_include_non_cortical=False)
    cerebra_non_cortical = CerebrA(source_space_include_non_cortical=True)
    colors_cortical = [colors_hex[label] for label in cerebra.src_space_labels]
    
    bem_vertices_vox_ras = cerebra.get_bem_vertices_vox_ras()
    non_cortical_mask = np.logical_xor (cerebra.src_space_mask, cerebra_non_cortical.src_space_mask)
    non_cortical_points = np.indices([256, 256, 256])[:, non_cortical_mask].T

    src_space_pc = PointCloud(cerebra.src_space_points, colors_cortical)
    non_cortical_pc = PointCloud(non_cortical_points, [0.8,0.8,0.8])
    bem_pcs = [PointCloud(bem_vertices_vox_ras[i],cerebra.bem_colors[i] ) for i in range(3)]
    bem_mesh = Mesh(bem_vertices_vox_ras[0],cerebra.get_bem_normals_vox_ras()[0],cerebra.get_bem_triangles()[0],cerebra.bem_colors[0] )
    vis = create_plot(draw_bounding_box=False)
    vis = add_drawable(vis,src_space_pc,non_cortical_pc,*bem_pcs)

    # Rotate camera 
    ctr = vis.get_view_control()
    # ctr.camera_local_rotate(256,32,19,0)
    # ctr.change_field_of_view(step=10)
    ctr.reset_camera_local_rotate()
    ctr.camera_local_translate(256,-100,100)
    ctr.camera_local_rotate(256,-100,100)
    ctr.rotate(1920, 500)
    ctr.set_zoom(200)
    run(vis)

