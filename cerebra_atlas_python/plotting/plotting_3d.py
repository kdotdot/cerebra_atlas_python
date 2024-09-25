#!/usr/bin/env python
"""
2D Plotting submodule for cerebra_atlas_python
"""


import numpy as np


from .colors import get_cmap_colors
from .cerebra_o3d import PointCloud, Mesh, add_drawable, create_plot, run, rotate_camera
from ..data._transforms import lia_points_to_ras_points


class Plots3D:
    def __init__(self, **kwargs):
        pass

    def plot_data_3d(
        self,
        plot_data,
        plot_src_space=True,
        plot_bem=False,
        plot_montage=False,
        **kwargs,
    ):
        """Plot 3D brain"""
        assert (
            "src_space_points" in plot_data.keys()
        ), "src_space_points should be provided in plot_data"

        cerebra_volume = plot_data["cerebra_volume"]
        src_space_points = plot_data["src_space_points"]
        src_space_labels = plot_data["src_space_labels"]
        cortical_color = plot_data["cortical_color"]
        bem_colors = plot_data["bem_colors"]
        bem_vertices_vox_ras = plot_data["bem_vertices_vox_ras"]
        bem_triangles = plot_data["bem_triangles"]
        bem_normals_vox_ras = plot_data["bem_normals_vox_ras"]
        info = plot_data["info"]
        fiducials = plot_data["fiducials"]
        colors = plot_data.get("colors", None)
        rotate_mode = plot_data.get("rotate_mode", 1)
        save_path = plot_data.get("save_path", None)

        print()

        vis = create_plot(draw_bounding_box=False)
        # SRC SPACE
        if plot_src_space:
            colors_hex = get_cmap_colors()
            colors_cortical = np.array(
                [colors_hex[label] for label in src_space_labels]
            )
            print(f"{colors=} {colors_cortical=}")
            src_space_pc = PointCloud(
                src_space_points, colors_cortical if colors is None else colors
            )
            vis = add_drawable(
                vis, src_space_pc, reset_bounding_box=True, translate=False
            )

        # BEM
        if plot_bem:
            bem_pcs = [
                PointCloud(bem_vertices_vox_ras[i], bem_colors[i]) for i in range(3)
            ]
            bem_mesh_0 = Mesh(
                bem_vertices_vox_ras[0],
                bem_triangles[0],
                bem_normals_vox_ras[0],
                bem_colors[0],
            )
            bem_mesh_1 = Mesh(
                bem_vertices_vox_ras[1],
                bem_triangles[1],
                bem_normals_vox_ras[1],
                bem_colors[1],
            )
            bem_mesh_2 = Mesh(
                bem_vertices_vox_ras[2],
                bem_triangles[2],
                bem_normals_vox_ras[2],
                bem_colors[2],
            )
            vis = add_drawable(
                vis, bem_mesh_0, reset_bounding_box=True, translate=False
            )

        # MONTAGE
        if plot_montage:
            fiducial_points = np.array([fiducial["r"] for fiducial in fiducials])
            montage = info.get_montage()
            montage_fiducials_mri = np.array([dig["r"] for dig in montage.dig[:3]])
            montage_pts_mri = np.array([dig["r"] for dig in montage.dig[3:]])

            montage_fiducials = self.apply_head_mri_trans(montage_fiducials_mri)
            montage_pts = self.apply_head_mri_trans(montage_pts_mri)

            montage_fiducials = self.apply_mri_vox_t(montage_fiducials)
            montage_fiducials_mri = self.apply_mri_vox_t(montage_fiducials_mri)
            montage_pts = self.apply_mri_vox_t(montage_pts)
            montage_pts_mri = self.apply_mri_vox_t(montage_pts_mri)
            fiducial_points = self.apply_mri_vox_t(fiducial_points)

            montage_fiducials = lia_points_to_ras_points(montage_fiducials)
            montage_fiducials_mri = lia_points_to_ras_points(montage_fiducials_mri)
            montage_pts = lia_points_to_ras_points(montage_pts)
            montage_pts_mri = lia_points_to_ras_points(montage_pts_mri)
            fiducial_points = lia_points_to_ras_points(fiducial_points)

            montage_fiducials_pc = PointCloud(montage_fiducials, [1, 0, 1])
            montage_pts_pc = PointCloud(montage_pts, [1, 0.4, 0])
            montage_fiducials_mri_pc = PointCloud(montage_fiducials_mri, [1, 0, 1])
            montage_pts_mri_pc = PointCloud(montage_pts_mri, [1, 0.4, 0])
            fiducial_points_pc = PointCloud(fiducial_points, [0, 1, 0])

            vis = add_drawable(
                vis,
                fiducial_points_pc,
                montage_pts_pc,
                montage_fiducials_pc,
                reset_bounding_box=True,
                translate=False,
            )

        rotate_camera(vis, rotate_mode=rotate_mode)
        # run(vis)
        if save_path is None:
            run(vis)

        else:

            def update_fn(vis, frame):
                vis.capture_screen_image(save_path)
                vis.close()

            run(vis, update_fn=update_fn)
