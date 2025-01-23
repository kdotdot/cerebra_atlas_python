#!/usr/bin/env python
"""
2D Plotting submodule for cerebra_atlas_python
"""
import logging
import nibabel as nib
import numpy as np
from cerebra_atlas_python.plotting.colors import get_cmap_colors
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator

logger = logging.getLogger(__name__)


def plot_brain_slice_2d(
    _plot_data,
    axis=0,
    fixed_value=None,
    plot_regions=True,
    plot_whitematter=False,
    plot_empty=False,
    plot_affine=False,
    affine_color="#ccc",
    plot_coordinate_frame_info=False,
    coordinate_frame_info_text_size=14,
    plot_relative_positions=False,
    relative_positions_text_size=18,
    plot_axes=True,
    ax_label_text_size=18,
    plot_axis_labels=True,
    plot_grid=True,
    plot_narrow_ax=True,
    plot_legend=True,
    cmap_name="default",
    volume_colors=None,
    pt=None,
    pt_text=None,
    pt_text_size=8,
    plot_pt_lines=True,
    # plot_region_names=False,
    # region_names_text_size=14,
    # src_space_points=None,
    # bem_volume=None,
    # highlighted_region_ids=None,
    # highlighted_region_names=None,
    # highlighted_region_centroids=None,
    # highlighted_cortical_ids=None,
    # region_centroid=None,
    # pt_dist=None,
    # t1_volume=None,
    s=2,
    n_layers: str | int = "max",
    n_layers_max=100,
    title=None,
    title_size=35,
    fig=None,
    ax=None,
    figsize=None,
):

    assert (
        "affine" in _plot_data.keys() and "cerebra_volume" in _plot_data.keys()
    ), "affine and cerebra_volume should be provided in _plot_data"
    affine = _plot_data["affine"]
    cerebra_volume = _plot_data["cerebra_volume"]

    x_label, y_label = _get_ax_labels(axis)

    if plot_narrow_ax:
        min_x, max_x = 25, 226
        min_y, max_y = 25, 226
    else:
        min_x, max_x = 0, cerebra_volume.shape[x_label]
        min_y, max_y = 0, cerebra_volume.shape[y_label]

    # Obtain matplotlib ax handle
    if ax is None:
        fig, ax = _get_2d_fig_ax(
            figsize=figsize,
            add_grid=plot_grid,
            x_lims=(min_x, max_x) if plot_narrow_ax else None,
            y_lims=(min_y, max_y) if plot_narrow_ax else None,
        )

    if plot_narrow_ax:
        ax.set_xlim((min_x, max_x))
        ax.set_ylim((min_y, max_y))

    # Configure ax
    ax_labels = ["X", "Y", "Z"]

    if plot_axis_labels:
        ax.set_xlabel(ax_labels[x_label], fontsize=ax_label_text_size)
        ax.set_ylabel(ax_labels[y_label], fontsize=ax_label_text_size)

    if plot_axes:
        ax.tick_params(axis="both", which="major", labelsize=ax_label_text_size)
        ax.tick_params(axis="both", which="minor", labelsize=ax_label_text_size)
    else:
        ax.set_axis_off()
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

    if plot_relative_positions:

        # BOTTOM
        ax.text(
            min_x + (max_x - min_x) // 2,
            min_y,
            "\\" + f"textbf{{{inverse_codes[codes[y_label]]}}}",
            size=relative_positions_text_size,
            c="white" if plot_empty else "black",
            horizontalalignment="center",
            verticalalignment="bottom",
        )
        # TOP
        ax.text(
            min_x + (max_x - min_x) // 2,
            max_y,
            "\\" + f"textbf{{{codes[y_label]}}}",
            size=relative_positions_text_size,
            c="white" if plot_empty else "black",
            horizontalalignment="center",
            verticalalignment="top",
        )

        # LEFT
        ax.text(
            min_x,
            max_y // 2,
            "\\" + f"textbf{{{inverse_codes[codes[x_label]]}}}",
            size=relative_positions_text_size,
            c="white" if plot_empty else "black",
            horizontalalignment="left",
            verticalalignment="center",
        )
        # RIGHT
        ax.text(
            max_x,
            max_y // 2,
            "\\" + f"textbf{{{codes[x_label]}}}",
            size=relative_positions_text_size,
            c="white" if plot_empty else "black",
            horizontalalignment="right",
            verticalalignment="center",
        )

    if plot_coordinate_frame_info:
        ax.text(
            min_x + 10,
            max_y - coordinate_frame_info_text_size * 1.5,
            f"""{codes[axis]} ({ax_labels[axis]})= {fixed_value}
            """,
            c="white" if plot_empty else "black",
            horizontalalignment="left",
            verticalalignment="center",
        ).set_fontsize(coordinate_frame_info_text_size)
        ax.text(
            min_x + 10,
            min_y + coordinate_frame_info_text_size // 4,
            f"""{"".join(codes) + f"({cerebra_volume.shape[0]},{cerebra_volume.shape[1]},{cerebra_volume.shape[2]})"}
            """,
            c="white" if plot_empty else "black",
            horizontalalignment="left",
            verticalalignment="center",
        ).set_fontsize(coordinate_frame_info_text_size)

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
    #     xs_ys, cs, alphas, sizes = _merge_points_optimized(
    #         [xs_ys, new_xs_ys], [cs, new_cs], [alphas, new_alphas], [sizes, new_sizes]
    #     )

    # BEM SURFACES
    # if bem_volume is not None:
    #     bem_slice = _slice_volume(
    #         bem_volume, fixed_value=fixed_value, axis=axis, n_layers=5
    #     )
    #     colors = get_cmap_colors("hsv", bem_volume.max())
    #     colors[-1] = [1, 0, 0]
    #     alpha_values = np.array([0, 0.10, 0.10, 1])
    #     new_xs_ys, new_cs, new_alphas, new_sizes = _project_volume_2d(
    #         bem_slice,
    #         axis=axis,
    #         colors=colors,
    #         alpha_values=alpha_values,
    #         size_values=np.repeat(1, len(alpha_values)),
    #     )

    #     xs_ys, cs, alphas, sizes = _merge_points_optimized(
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
        cerebra_slice = _slice_volume(
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

        new_xs_ys, new_cs, new_alphas, new_sizes = _project_volume_2d(
            cerebra_slice,
            axis=axis,
            colors=colors if volume_colors is None else volume_colors,
            avoid_values=avoid_values,
            alpha_values=alpha_values,
            size_values=np.repeat(s, len(colors)),
        )
        xs_ys, cs, alphas, sizes = _merge_points_optimized(
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
                fontsize=pt_text_size,
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
    #     t1_slice = _slice_volume(
    #         t1_volume, fixed_value=fixed_value, axis=axis, n_layers=2
    #     )
    #     new_xs_ys, new_cs, new_alphas, new_sizes = _project_volume_2d(
    #         t1_slice,
    #         axis=axis,
    #     )
    #     xs_ys, cs, alphas, sizes = _merge_points_optimized(
    #         [xs_ys, new_xs_ys],
    #         [cs, new_cs],
    #         [alphas, new_alphas],
    #         [sizes, new_sizes],
    #         default_alpha=0.5,
    #     )

    if xs_ys is not None and len(xs_ys) > 0:
        xs, ys = xs_ys.T
        # TODO: check
        ax.scatter(xs, ys, c=cs, alpha=alphas, s=sizes)  # type:ignore
    else:
        logger.warning(
            "Plot brain slice is empty, try chaning the fixed_value parameter"
        )

    if plot_affine:

        aff_translate = affine[:-1, 3]
        ax.hlines(
            abs(aff_translate[y_label]),
            0,
            256,
            linestyles="solid",
            alpha=0.5,
            colors=affine_color,
            label="Affine",
        )
        ax.vlines(
            abs(aff_translate[x_label]),
            0,
            256,
            linestyles="solid",
            alpha=0.5,
            colors=affine_color,
        )

    if plot_legend:
        ax.legend()

    return fig, ax


def orthoview(
    axs=None,
    fig=None,
    figsize=None,
    **kwargs,
):
    if axs is None:
        fig, axs = _get_orthoview_axes(figsize=figsize)

    print(type(axs), isinstance(axs, np.ndarray))

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


# https://github.com/RayleighLord/RayleighLordAnimations/blob/master/publication%20quality%20figures/fig_config.py
def figure_features(tex=True, font="serif", dpi=180):
    """Customize figure settings.
    Args:
        tex (bool, optional): use LaTeX. Defaults to True.
        font (str, optional): font type. Defaults to "serif".
        dpi (int, optional): dots per inch. Defaults to 180.
    """
    plt.rcParams.update(
        {
            "font.size": 20,
            "font.family": font,
            "text.usetex": tex,
            "figure.subplot.top": 0.9,
            "figure.subplot.right": 0.9,
            "figure.subplot.left": 0.15,
            "figure.subplot.bottom": 0.12,
            "figure.subplot.hspace": 0.4,
            "savefig.dpi": dpi,
            "savefig.format": "png",
            "axes.titlesize": 16,
            "axes.labelsize": 18,
            "axes.axisbelow": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 5,
            "xtick.minor.size": 2.25,
            "xtick.major.pad": 7.5,
            "xtick.minor.pad": 7.5,
            "ytick.major.pad": 7.5,
            "ytick.minor.pad": 7.5,
            "ytick.major.size": 5,
            "ytick.minor.size": 2.25,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "legend.framealpha": 1,
            "figure.titlesize": 16,
            "lines.linewidth": 2,
        }
    )


def _get_2d_fig_ax(
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] | None = None,
    add_grid: bool = False,
    x_lims: Optional[Tuple[int, int]] = None,
    y_lims: Optional[Tuple[int, int]] = None,
) -> Tuple[Optional[Figure], Axes]:
    """
        Creates a 2D figure and axes with optional LaTeX styling and grid.

        This function can take an existing matplotlib figure and axes objects or create new ones.
        It sets the limits of the axes and optionally applies LaTeX styling and adds a grid.

        Parameters:
        fig (Optional[plt.Figure]): An optional matplotlib figure object. Defaults to None.
        ax (Optional[plt.Axes]): An optional matplotlib axes object. Defaults to None.
        figsize (Tuple[int, int]): Size of the figure, defaults to (6, 6).
        add_grid (bool): If True, adds a grid to the axes. Defaults to False.
    self.montage_name is not None and self.head_size is not None
        Returns:
        Tuple[plt.Figure, plt.Axes]: A tuple containing the matplotlib figure and axes objects.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()

    if x_lims is not None:
        ax.set_xlim(x_lims)
    if y_lims is not None:
        ax.set_ylim(y_lims)

    if add_grid:
        _add_grid_to_ax(ax)

    return fig, ax


def _remove_ax(ax: Axes, keep_names: Optional[List[str]] = None) -> Axes:
    """
    Hides all the elements of a given matplotlib axis.

    This function takes a matplotlib axes object and hides its x-axis, y-axis,
    and all four spines (top, right, bottom, left).

    Parameters:
    ax (Axes): A matplotlib axes object on which the elements are to be hidden.
    keep_names (Optional[List[str]]): Specifies which spines to keep visible.
        options are "top", "right", "bottom", and "left". Defaults to None.
    """
    all_names = ["top", "right", "bottom", "left"]
    if keep_names is None:
        keep_names = []
    else:
        # Assert all keep names are valid
        if not all([name in all_names for name in keep_names]):
            raise ValueError(
                f"Invalid keep_names: {keep_names}. Must be a subset of {all_names}"
            )
    ax.set_yticks([])
    ax.set_xticks([])

    if "top" not in keep_names and "bottom" not in keep_names:
        ax.set_xticklabels([])

    if "right" not in keep_names and "left" not in keep_names:
        ax.set_yticklabels([])
    for name in all_names:
        if name in keep_names:
            continue
        ax.spines[name].set_visible(False)
    return ax


def _get_orthoview_axes(
    figsize: Tuple[int, int] | None = None,
    add_grid: bool = False,
) -> Tuple[Figure, List[Axes]]:
    """
    Creates a 2x2 grid of matplotlib subplots for displaying orthogonal views.

    This function sets up a 2x2 grid of subplots using matplotlib, with three of these subplots
    configured using the _get_2d_fig_ax function, and the fourth subplot hidden using _remove_ax.

    Parameters:
    figsize (Tuple[int, int]): Size of the figure, defaults to (7, 7).
    add_grid (bool): If True, adds a grid to the subplots. Defaults to False.

    Returns:
    Tuple[plt.Figure, np.ndarray]: A tuple containing the matplotlib figure and a 2x2 numpy array of axes objects.
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    # Configure the first three subplots
    _, axs[0, 0] = _get_2d_fig_ax(None, axs[0, 0], add_grid=add_grid)
    _, axs[0, 1] = _get_2d_fig_ax(None, axs[0, 1], add_grid=add_grid)
    _, axs[1, 0] = _get_2d_fig_ax(None, axs[1, 0], add_grid=add_grid)
    # Hide the fourth subplot
    _remove_ax(axs[1, 1])

    return fig, axs


def _add_grid_to_ax(ax, lines=True, locations=None):
    """Add a grid to the current plot.
    Args:
        ax (Axis): axis object in which to draw the grid.
        lines (bool, optional): add lines to the grid. Defaults to True.
        locations (tuple, optional):
            (xminor, xmajor, yminor, ymajor). Defaults to None.
    """

    if lines:
        ax.grid(lines, alpha=0.5, which="minor", ls=":")
        ax.grid(lines, alpha=0.7, which="major")

    if locations is not None:
        assert len(locations) == 4, "Invalid entry for the locations of the markers"

        xmin, xmaj, ymin, ymaj = locations

        ax.xaxis.set_minor_locator(MultipleLocator(xmin))
        ax.xaxis.set_major_locator(MultipleLocator(xmaj))
        ax.yaxis.set_minor_locator(MultipleLocator(ymin))
        ax.yaxis.set_major_locator(MultipleLocator(ymaj))


def _slice_volume(
    volume: np.ndarray, fixed_value: int, axis: int = 0, n_layers: int = 1
) -> np.ndarray:
    """
    Slices a given volume array along a specified axis.

    Args:
        volume (np.ndarray): The input volume array.
        fixed_value (int): The starting value for slicing.
        axis (int): The axis along which to slice the volume. Defaults to 0.
        n_layers (int): The number of layers to include in the slice. Defaults to 1.

    Returns:
        np.ndarray: The sliced volume array.
    """
    start_slice, end_slice = fixed_value, fixed_value + n_layers
    increment = 1
    logger.debug(
        "start_slice=%s  end_slice=%s  increment=%s ", start_slice, end_slice, increment
    )
    slice_idx = slice(start_slice, end_slice, increment)
    if axis == 0:
        return volume[slice_idx, :, :]
    elif axis == 1:
        return volume[:, slice_idx, :]
    elif axis == 2:
        return volume[:, :, slice_idx]
    else:
        raise ValueError(f"Invalid axis: {axis}")


def _get_ax_labels(axis: int) -> tuple[int, int]:
    """
    Determines the x and y axis labels based on the provided axis.

    This function takes an integer representing an axis (0, 1, or 2) and returns
    a tuple of integers representing the x and y labels. The labels are determined
    as follows:
    - If axis is 0, x_label is 1 and y_label is 2.
    - If axis is 1, x_label is 0 and y_label is 2.
    - If axis is 2, x_label is 0 and y_label is 1.

    Parameters:
    axis (int): An integer representing the axis (expected to be 0, 1, or 2).

    Returns:
    tuple[int, int]: A tuple containing two integers representing the x and y labels.
    """
    if axis == 0:
        x_label = 1
        y_label = 2
    elif axis == 1:
        x_label = 0
        y_label = 2
    elif axis == 2:
        x_label = 0
        y_label = 1
    else:
        raise ValueError("axis must be 0, 1, or 2")
    return x_label, y_label


def _project_volume_2d(
    volume_slice,
    axis=0,
    colors=None,
    alpha_values=None,
    size_values=None,
    avoid_values=None,
):
    avoid_values = avoid_values or [0]
    x_label, y_label = _get_ax_labels(axis)

    mask = ~np.isin(volume_slice, avoid_values)
    xyzs = np.where(mask)
    xs_ys = np.array([xyzs[x_label], xyzs[y_label]])

    # FILTER_DUPLICATES
    xs_ys, unique_indices = np.unique(xs_ys, axis=1, return_index=True)

    xyzs = np.take(xyzs, unique_indices, axis=1)

    new_values = np.array(volume_slice[tuple(xyzs)]).astype(int)
    cs = colors[new_values] if colors is not None else None
    alphas = alpha_values[new_values] if alpha_values is not None else None
    sizes = size_values[new_values] if size_values is not None else None

    return xs_ys.T, cs, alphas, sizes


def _merge_points_optimized(
    xs_ys_arr: Tuple[Optional[np.ndarray], np.ndarray],
    cs_arr: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
    alphas_arr: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
    sizes_arr: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
    default_color: Optional[list] = None,
    default_alpha: float = 1,
    default_size: float = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Merges two sets of points, colors, and alpha values while removing duplicates.

    """
    default_color = default_color or [
        1,
        0,
        1,
    ]
    xs_ys_keep, xs_ys_new = xs_ys_arr
    cs_keep, cs_new = cs_arr
    alphas_keep, alphas_new = alphas_arr
    sizes_keep, sizes_new = sizes_arr

    if alphas_new is None:
        alphas_new = np.full(len(xs_ys_new), default_alpha)
    if cs_new is None:
        cs_new = np.tile(default_color, (len(xs_ys_new), 1))
    if sizes_new is None:
        sizes_new = np.full(len(xs_ys_new), default_size)
    if xs_ys_keep is None:
        return xs_ys_new, cs_new, alphas_new, sizes_new

    # Step 1: Use a hash-based approach to identify non-duplicate points
    keep_set = set(map(tuple, xs_ys_keep))
    non_dup_indices = [
        i for i, point in enumerate(xs_ys_new) if tuple(point) not in keep_set
    ]
    non_dup_xs_ys_new = xs_ys_new[non_dup_indices]

    # Step 2: Efficiently handle color and alpha arrays
    if cs_keep is None:
        cs_keep = np.tile(default_color, (len(xs_ys_keep), 1))
    if cs_new is not None:
        cs_new = cs_new[non_dup_indices]  # Index the cs_new list

    if alphas_keep is None:
        alphas_keep = np.full(len(xs_ys_new), default_alpha)
    if alphas_new is not None:
        alphas_new = alphas_new[non_dup_indices]  # Index the alphas_new list

    if sizes_keep is None:
        sizes_keep = np.full(len(xs_ys_new), default_size)
    if sizes_new is not None:
        sizes_new = sizes_new[non_dup_indices]  # Index the alphas_new list

    # Step 3: Merge arrays
    xs_ys = np.vstack((xs_ys_keep, non_dup_xs_ys_new))
    cs = np.vstack((cs_keep, cs_new)) if cs_new is not None else cs_keep
    alphas = (
        np.concatenate((alphas_keep, alphas_new))
        if alphas_new is not None
        else alphas_keep
    )
    sizes = (
        np.concatenate((sizes_keep, sizes_new)) if sizes_new is not None else sizes_keep
    )

    return xs_ys, cs, alphas, sizes
