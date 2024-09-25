#!/usr/bin/env python
"""
2D Plotting submodule for cerebra_atlas_python
"""
import nibabel as nib
import numpy as np

from cerebra_atlas_python.plotting.colors import get_cmap_colors
from .cerebra_plt import *
from .volumes import *


def plot_brain_slice_2d(
    plot_data,
    axis=0,
    fixed_value=None,
    plot_regions=True,
    plot_whitematter=False,
    plot_empty=False,
    plot_affine=False,
    plot_planes=False,
    plot_region_names=False,
    region_names_text_size=14,
    src_space_points=None,
    bem_volume=None,
    # highlighted_region_ids=None,
    # highlighted_region_names=None,
    # highlighted_region_centroids=None,
    # highlighted_cortical_ids=None,
    region_centroid=None,
    pt_dist=None,
    cmap_name="default",
    s=2,
    fig=None,
    ax=None,
    pt=None,
    pt_text=None,
    plot_pt_lines=True,
    slice_figsize=(6, 6),
    n_layers: str | int = "max",
    n_layers_max=100,
    use_latex_figures=True,
    add_grid=False,
    add_top_left_info=True,
    top_left_info_text_size=14,
    add_coordinate_frame_info=True,
    coordinate_frame_info_text_size=14,
    add_ax_labels=True,
    add_ax_ticks=True,
    volume_colors=None,
    t1_volume=None,
    narrow_ax=True,
    adjust_ax=True,
    title=None,
    title_size=35,
    hide_ax=True,
):

    assert (
        "affine" in plot_data.keys() and "cerebra_volume" in plot_data.keys()
    ), "affine and cerebra_volume should be provided in plot_data"
    affine = plot_data["affine"]
    cerebra_volume = plot_data["cerebra_volume"]

    x_label, y_label = get_ax_labels(axis)

    if narrow_ax:
        min_x, max_x = 25, 226
        min_y, max_y = 25, 226
    else:
        min_x, max_x = 0, cerebra_volume.shape[x_label]
        min_y, max_y = 0, cerebra_volume.shape[y_label]

    # Obtain matplotlib ax handle
    if ax is None:
        fig, ax = get_2d_fig_ax(
            figsize=slice_figsize,
            use_latex_figures=use_latex_figures,
            add_grid=add_grid,
            x_lims=(min_x, max_x) if adjust_ax else None,
            y_lims=(min_y, max_y) if adjust_ax else None,
        )

    if adjust_ax:
        ax.set_xlim((min_x, max_x))
        ax.set_ylim((min_y, max_y))

    # Configure ax
    ax_labels = ["X", "Y", "Z"]

    if add_ax_labels:
        ax.set_xlabel(ax_labels[x_label])
        ax.set_ylabel(ax_labels[y_label])

    if not add_ax_ticks:
        ax.set_axis_off()

    if hide_ax:
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    if title is not None:
        ax.set_title(title, size=title_size, pad=20)

    # Setting fixed value (constant value for plotting plane)
    plot_plane_values = None
    if fixed_value is not None:
        pass
    elif pt is not None:
        fixed_value = pt[axis]
        plot_plane_values = pt
    # elif plot_highlighted_region is not None and region_centroid is not None:
    #     pt = region_centroid
    #     fixed_value = region_centroid[axis]
    #     plot_plane_values = region_centroid
    else:
        fixed_value = abs(int(affine[:, -1][axis]))
        plot_plane_values = (affine[:, -1][:3]).astype(int)

    codes = nib.orientations.aff2axcodes(affine)
    inverse_codes = {"R": "L", "A": "P", "S": "I", "L": "R", "P": "A", "I": "S"}

    if add_coordinate_frame_info:

        # if axis==1:
        #     xoffset= 40
        # elif axis ==2:
        #     xoffset = 30
        # else:
        #     xoffset = 0

        y_offset = coordinate_frame_info_text_size

        # BOTTOM
        ax.text(
            min_x + (max_x - min_x) // 2,
            min_y,
            "\\" + f"textbf{{{inverse_codes[codes[y_label]]}}}",
            size=coordinate_frame_info_text_size,
            c="white" if plot_empty else "black",
            horizontalalignment="center",
            verticalalignment="center",
        )
        # TOP
        ax.text(
            min_x + (max_x - min_x) // 2,
            max_y,
            "\\" + f"textbf{{{codes[y_label]}}}",
            size=coordinate_frame_info_text_size,
            c="white" if plot_empty else "black",
            horizontalalignment="center",
            verticalalignment="center",
        )

        # LEFT
        ax.text(
            min_x,
            max_y // 2,
            "\\" + f"textbf{{{inverse_codes[codes[x_label]]}}}",
            size=coordinate_frame_info_text_size,
            c="white" if plot_empty else "black",
            horizontalalignment="center",
            verticalalignment="center",
        )
        # RIGHT
        ax.text(
            max_x,
            max_y // 2,
            "\\" + f"textbf{{{codes[x_label]}}}",
            size=coordinate_frame_info_text_size,
            c="white" if plot_empty else "black",
            horizontalalignment="center",
            verticalalignment="center",
        )

    if add_top_left_info:
        ax.text(
            min_x + 10,
            max_y - top_left_info_text_size * 1.5,
            f"""{codes[axis]} ({ax_labels[axis]})= {fixed_value}
            """,
            c="white" if plot_empty else "black",
            horizontalalignment="left",
            verticalalignment="center",
        ).set_fontsize(top_left_info_text_size)
        ax.text(
            min_x + 10,
            min_y + top_left_info_text_size // 4,
            f"""{"".join(codes) + f"({cerebra_volume.shape[0]},{cerebra_volume.shape[1]},{cerebra_volume.shape[2]})"}
            """,
            c="white" if plot_empty else "black",
            horizontalalignment="left",
            verticalalignment="center",
        ).set_fontsize(top_left_info_text_size)

    # NOTE: Having repeated values for scatterplots
    # (i.e. [x=1,y=1,c='white',x=1,y=1,c='red'...]) increase processing time
    # Be careful when creating new scatterplots that overlap

    xs_ys, cs, alphas, sizes = None, None, None, None

    # PLOT VOLUMES
    # NOTE:FIRST PROCESSED ARE SHOWN ON UPPER LAYER
    # (FIRST SRC VOL THEN REGIONS THEN BEM...)

    # SRC SPACE
    # if src_space_points is not None:
    #     mask = src_space_points.T[axis] > fixed_value

    #     xs = src_space_points[mask].T[x_label]
    #     ys = src_space_points[mask].T[y_label]
    #     new_xs_ys = np.array([xs, ys]).T
    #     new_cs = None
    #     new_alphas = None
    #     new_sizes = np.full(len(new_xs_ys), 1)
    #     xs_ys, cs, alphas, sizes = merge_points_optimized(
    #         [xs_ys, new_xs_ys], [cs, new_cs], [alphas, new_alphas], [sizes, new_sizes]
    #     )

    # BEM SURFACES
    # if bem_volume is not None:
    #     bem_slice = slice_volume(
    #         bem_volume, fixed_value=fixed_value, axis=axis, n_layers=5
    #     )
    #     colors = get_cmap_colors("hsv", bem_volume.max())
    #     colors[-1] = [1, 0, 0]
    #     alpha_values = np.array([0, 0.10, 0.10, 1])
    #     new_xs_ys, new_cs, new_alphas, new_sizes = project_volume_2d(
    #         bem_slice,
    #         axis=axis,
    #         colors=colors,
    #         alpha_values=alpha_values,
    #         size_values=np.repeat(1, len(alpha_values)),
    #     )

    #     xs_ys, cs, alphas, sizes = merge_points_optimized(
    #         [xs_ys, new_xs_ys], [cs, new_cs], [alphas, new_alphas], [sizes, new_sizes]
    #     )

    if cmap_name != "default" or cerebra_volume.max() > 103:
        cmap_name = "gray" if cmap_name == "default" else cmap_name
        colors = get_cmap_colors(cmap_name, cerebra_volume.max())
    else:
        colors = get_cmap_colors()

    # REGIONS
    if plot_regions:
        # Set the number of layers used for plotting depth
        if n_layers == "max":
            n_layers = n_layers_max
        else:
            n_layers = int(n_layers)
        print(f"{fixed_value= } {cerebra_volume.shape= }")
        cerebra_slice = slice_volume(
            cerebra_volume, fixed_value=fixed_value, axis=axis, n_layers=n_layers
        )

        avoid_values = []
        if not plot_empty:
            avoid_values.append(0)
        if not plot_whitematter:
            avoid_values.append(103)

        alpha_values = None
        # if plot_highlighted_region:
        #     assert (
        #         region_centroid is not None
        #     ), "If plot_highlighted_region = (int) region_centroid should also be provided"
        #     alpha_values = np.ones(104) * 0.1
        #     alpha_values[plot_highlighted_region] = 1

        # if highlighted_region_ids is not None:
        #     alpha_values = np.ones(104) * 0.05
        #     alpha_values[highlighted_region_ids] = 1
        #     alpha_values[103] = 1

        # if (
        #     plot_region_names
        #     and highlighted_region_names is not None
        #     and highlighted_region_centroids is not None
        # ):

        #     npoints = len(highlighted_region_names)  # points to chose from

        #     if axis == 0:
        #         r = 100.5  # radius of the circle
        #     elif axis == 1:
        #         r = 98
        #     else:
        #         r = 100
        #     smaller_r = r - 5

        #     t = np.linspace(0, 2 * np.pi, npoints, endpoint=False)

        #     # if axis==1:
        #     #    aff_translate = affine[:-1, 3]
        #     #    x = r * np.cos(t) + aff_translate[x_label]
        #     #    y = r * np.sin(t) + aff_translate[y_label]
        #     # else:
        #     x = r * np.cos(t) + 128
        #     y = r * np.sin(t) + 128
        #     x_sm = smaller_r * np.cos(t) + 128
        #     y_sm = smaller_r * np.sin(t) + 128
        #     circle_points = np.array([x, y]).T
        #     circle_points_smaller = np.array([x_sm, y_sm]).T
        #     used_points = []
        #     used_ids = []
        #     for r_id, (region_name, region_centroid) in enumerate(
        #         zip(highlighted_region_names, highlighted_region_centroids)
        #     ):
        #         if highlighted_region_ids[r_id] not in cerebra_slice:
        #             continue
        #         x = region_centroid[x_label]
        #         y = region_centroid[y_label]
        #         region_id = (
        #             highlighted_cortical_ids[r_id]
        #             if highlighted_cortical_ids is not None
        #             else highlighted_region_ids[r_id]
        #         )
        #         # print(region_id)
        #         used_ids.append(region_id)
        #         # Get closest circle point
        #         min_dist = 100000
        #         min_i = 0
        #         for i, (cx, cy) in enumerate(circle_points):
        #             dist = (cx - x) ** 2 + (cy - y) ** 2
        #             if dist < min_dist and i not in used_points:
        #                 min_dist = dist
        #                 min_i = i
        #         used_points.append(min_i)
        #         x = circle_points[min_i][0]
        #         y = circle_points[min_i][1]

        #         x_text = x - 5 if x < 128 else x + 5
        #         y_text = y - 5 if (y < 120) else y + 5 if (y > 136) else y

        #         ax.text(
        #             x_text,
        #             y_text,
        #             f"{region_id}",
        #             c="white" if plot_empty else "black",
        #             size=region_names_text_size,
        #             verticalalignment="center",
        #             horizontalalignment="center",
        #         )
        #         x_sm = circle_points_smaller[min_i][0]
        #         y_sm = circle_points_smaller[min_i][1]
        #         # Plot straight line from point to centroid
        #         ax.plot(
        #             [x_sm, region_centroid[x_label]],
        #             [y_sm, region_centroid[y_label]],
        #             c="black",
        #             linewidth=0.5,
        #         )
        #         ax.scatter(
        #             x_sm, y_sm, color=colors[highlighted_region_ids[r_id]], s=108
        #         )

        new_xs_ys, new_cs, new_alphas, new_sizes = project_volume_2d(
            cerebra_slice,
            axis=axis,
            colors=colors if volume_colors is None else volume_colors,
            avoid_values=avoid_values,
            alpha_values=alpha_values,
            size_values=np.repeat(s, len(colors)),
        )
        xs_ys, cs, alphas, sizes = merge_points_optimized(
            (xs_ys, new_xs_ys), (cs, new_cs), (alphas, new_alphas), (sizes, new_sizes)
        )

    # Plot point
    if pt is not None:
        if plot_pt_lines:
            ax.vlines(pt[x_label], 0, 256, linestyles="dashed", alpha=0.4, colors="red")
            ax.hlines(pt[y_label], 0, 256, linestyles="dashed", alpha=0.4, colors="red")

        ax.scatter(pt[x_label], pt[y_label], s=100)

        if pt_text is not None:
            ax.text(
                pt[x_label] + 5,
                pt[y_label] + 5,
                pt_text,
                fontsize=8,
                c="white" if plot_empty else "black",
            )

    # if pt_dist is not None:
    #     inner_skull_pt, inner_skull_dist = pt_dist
    #     ax.plot(
    #         [inner_skull_pt[x_label], pt[x_label]],
    #         [inner_skull_pt[y_label], pt[y_label]],
    #         marker="o",
    #         c="red",
    #     )

    # T1 volume
    # if t1_volume is not None:
    #     t1_slice = slice_volume(
    #         t1_volume, fixed_value=fixed_value, axis=axis, n_layers=2
    #     )
    #     new_xs_ys, new_cs, new_alphas, new_sizes = project_volume_2d(
    #         t1_slice,
    #         axis=axis,
    #     )
    #     xs_ys, cs, alphas, sizes = merge_points_optimized(
    #         [xs_ys, new_xs_ys],
    #         [cs, new_cs],
    #         [alphas, new_alphas],
    #         [sizes, new_sizes],
    #         default_alpha=0.5,
    #     )

    if xs_ys is not None:
        xs, ys = xs_ys.T
        # TODO: check
        ax.scatter(xs, ys, c=cs, alpha=alphas, s=sizes)  # type:ignore

    # if plot_planes:
    #     ax.hlines(
    #         plot_plane_values[y_label],
    #         0,
    #         256,
    #         linestyles="solid",
    #         alpha=0.5,
    #         colors=colors[plot_highlighted_region]
    #         if plot_highlighted_region is not None
    #         else "gray",
    #     )
    #     ax.vlines(
    #         plot_plane_values[x_label],
    #         0,
    #         256,
    #         linestyles="solid",
    #         alpha=0.5,
    #         colors=colors[plot_highlighted_region]
    #         if plot_highlighted_region is not None
    #         else "gray",
    #     )

    if plot_affine:

        AFFINE_COLOR = "#FF00FF"

        aff_translate = affine[:-1, 3]
        ax.hlines(
            abs(aff_translate[y_label]),
            0,
            256,
            linestyles="solid",
            alpha=0.5,
            colors=AFFINE_COLOR,
        )
        ax.vlines(
            abs(aff_translate[x_label]),
            0,
            256,
            linestyles="solid",
            alpha=0.5,
            colors=AFFINE_COLOR,
        )

    return fig, ax


def orthoview(
    axs=None,
    fig=None,
    figsize=(15, 15),
    **kwargs,
):
    if axs is None:
        fig, axs = get_orthoview_axes(figsize=figsize)

    print(type(axs), axs, isinstance(axs, np.ndarray))
    if not isinstance(axs, np.ndarray):
        raise ValueError("axs should be a np array of Axes")

    plot_brain_slice_2d(
        axis=0,
        ax=axs[0, 0],
        **kwargs,
    )
    plot_brain_slice_2d(
        axis=1,
        ax=axs[0, 1],
        **kwargs,
    )
    plot_brain_slice_2d(
        axis=2,
        ax=axs[1, 0],
        **kwargs,
    )

    return fig, axs


class Plots2D:
    def __init__(self, tex=True, font="serif", dpi=180, **kwargs):
        figure_features(tex=tex, font=font, dpi=dpi)
        pass

    def plot_data_2d(self, **kwargs):
        """Plot 2D brain"""
        return plot_brain_slice_2d(
            **kwargs,
        )

    def plot_data_orthoview(self, **kwargs):
        """Plot 2D brain with orthoview"""
        return orthoview(
            **kwargs,
        )
