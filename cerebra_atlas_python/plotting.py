"""Plotting related utils"""
from typing import Optional, Tuple, List
import numpy as np
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import ListedColormap
from cerebra_atlas_python.utils import slice_volume, rgb_to_hex_str


ori_slice = dict(
    P="Coronal", A="Coronal", I="Axial", S="Axial", L="Sagittal", R="Saggital"
)
ori_names = dict(
    P="posterior", A="anterior", I="inferior", S="superior", L="left", R="right"
)

cortical_color = "#9EC8B9"
non_cortical_color = "#1B4242"


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


def add_grid_to_ax(ax, lines=True, locations=None):
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


def get_ax_labels(axis: int) -> tuple[int, int]:
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


def get_2d_fig_ax(
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (6, 6),
    use_latex_figures: bool = True,
    add_grid: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Creates a 2D figure and axes with optional LaTeX styling and grid.

    This function can take an existing matplotlib figure and axes objects or create new ones.
    It sets the limits of the axes and optionally applies LaTeX styling and adds a grid.

    Parameters:
    fig (Optional[plt.Figure]): An optional matplotlib figure object. Defaults to None.
    ax (Optional[plt.Axes]): An optional matplotlib axes object. Defaults to None.
    figsize (Tuple[int, int]): Size of the figure, defaults to (6, 6).
    use_latex_figures (bool): If True, applies LaTeX styling to the figure. Defaults to True.
    add_grid (bool): If True, adds a grid to the axes. Defaults to False.

    Returns:
    Tuple[plt.Figure, plt.Axes]: A tuple containing the matplotlib figure and axes objects.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()

    ax.set_xlim([0, 256])
    ax.set_ylim([0, 256])

    if use_latex_figures:
        figure_features()
    if add_grid:
        add_grid_to_ax(ax)

    return fig, ax


def remove_ax(ax: plt.Axes, keep_names: Optional[List[str]] = None) -> None:
    """
    Hides all the elements of a given matplotlib axis.

    This function takes a matplotlib axes object and hides its x-axis, y-axis,
    and all four spines (top, right, bottom, left).

    Parameters:
    ax (Axes): A matplotlib axes object on which the elements are to be hidden.
    keep_names (Optional[List[str]]): Specifies which spines to keep visible.
        options are "top", "right", "bottom", and "left". Defaults to None.
    """
    all_names: [str] = ["top", "right", "bottom", "left"]
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


def get_orthoview_axes(
    figsize: Tuple[int, int] = (7, 7),
    use_latex_figures: bool = True,
    add_grid: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Creates a 2x2 grid of matplotlib subplots for displaying orthogonal views.

    This function sets up a 2x2 grid of subplots using matplotlib, with three of these subplots
    configured using the get_2d_fig_ax function, and the fourth subplot hidden using remove_ax.

    Parameters:
    figsize (Tuple[int, int]): Size of the figure, defaults to (7, 7).
    use_latex_figures (bool): If True, applies LaTeX styling to the figure. Defaults to True.
    add_grid (bool): If True, adds a grid to the subplots. Defaults to False.

    Returns:
    Tuple[plt.Figure, np.ndarray]: A tuple containing the matplotlib figure and a 2x2 numpy array of axes objects.
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    # Configure the first three subplots
    _, axs[0, 0] = get_2d_fig_ax(
        None, axs[0, 0], use_latex_figures=use_latex_figures, add_grid=add_grid
    )
    _, axs[0, 1] = get_2d_fig_ax(
        None, axs[0, 1], use_latex_figures=use_latex_figures, add_grid=add_grid
    )
    _, axs[1, 0] = get_2d_fig_ax(
        None, axs[1, 0], use_latex_figures=use_latex_figures, add_grid=add_grid
    )
    # Hide the fourth subplot
    remove_ax(axs[1, 1])

    return fig, axs


def merge_points_optimized(
    xs_ys_arr: Tuple[np.ndarray, np.ndarray],
    cs_arr: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
    alphas_arr: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
    sizes_arr: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
    default_color: Optional[list] = None,
    default_alpha: float = 1,
    default_size: float = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def project_volume_2d(
    volume_slice,
    axis=0,
    colors=None,
    alpha_values=None,
    size_values=None,
    avoid_values=None,
):
    avoid_values = avoid_values or [0]
    x_label, y_label = get_ax_labels(axis)

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


def get_cmap_colors(cmap_name="gist_rainbow", n_classes=103):
    n_colors = int(n_classes) + 1
    cmap = matplotlib.colormaps[cmap_name]
    colors = cmap(np.linspace(0, 1, n_colors))
    white = np.array([1, 0.87, 0.87, 1])
    colors[-1] = white
    black = np.array([0, 0, 0, 1])
    colors[0] = black
    return colors[:, :3]


def get_cmap_colors_hex(**kwargs):
    colors = get_cmap_colors()
    return np.array([rgb_to_hex_str(c) for c in colors])


def get_cmap():
    newcmp = ListedColormap(get_cmap_colors())
    return newcmp


# takes in brain voxel volume (RAS)
# @time_func_decorator
def plot_brain_slice_2d(
    cerebra_volume,
    affine,
    axis=0,
    fixed_value=None,
    plot_regions=True,
    plot_whitematter=False,
    plot_empty=False,
    plot_affine=False,
    plot_planes=False,
    src_space_points=None,
    bem_volume=None,
    plot_highlighted_region=None,
    plot_highlighted_regions=None,
    highlighted_region_names=None,
    highlighted_region_centroids=None,
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
    n_layers: str or int = "max",
    n_layers_max=100,
    use_latex_figures=True,
    add_grid=False,
    add_top_left_info=True,
    add_coordinate_frame_info=True,
    add_ax_labels=True,
    add_ax_ticks=True,
    volume_colors=None,
    t1_volume=None,
):
    x_label, y_label = get_ax_labels(axis)

    # Obtain matplotlib ax handle
    if ax is None:
        fig, ax = get_2d_fig_ax(
            figsize=slice_figsize,
            use_latex_figures=use_latex_figures,
            add_grid=add_grid,
        )

    # Configure ax
    ax_labels = ["X", "Y", "Z"]

    if add_ax_labels:
        ax.set_xlabel(ax_labels[x_label])
        ax.set_ylabel(ax_labels[y_label])

    if not add_ax_ticks:
        ax.set_axis_off()

    # Setting fixed value (constant value for plotting plane)
    plot_plane_values = None
    if fixed_value is not None:
        pass
    elif pt is not None:
        fixed_value = pt[axis]
        plot_plane_values = pt
    elif plot_highlighted_region is not None and region_centroid is not None:
        pt = region_centroid
        fixed_value = region_centroid[axis]
        plot_plane_values = region_centroid
    else:
        fixed_value = int(affine[:, -1][axis])
        plot_plane_values = (affine[:, -1][:3]).astype(int)

    codes = nib.orientations.aff2axcodes(affine)
    inverse_codes = {"R": "L", "A": "P", "S": "I", "L": "R", "P": "A", "I": "S"}

    if add_coordinate_frame_info:
        ax.text(
            120, 10, inverse_codes[codes[y_label]], c="white" if plot_empty else "black"
        )
        ax.text(120, 240, codes[y_label], c="white" if plot_empty else "black")

        ax.text(
            10, 120, inverse_codes[codes[x_label]], c="white" if plot_empty else "black"
        )
        ax.text(240, 120, codes[x_label], c="white" if plot_empty else "black")

    if add_top_left_info:
        ax.text(
            10,
            220,
            f"""{codes[axis]} ({ax_labels[axis]})= {fixed_value}
            {"".join(codes)}
            {f"mm to surface={pt_dist[1]:.2f}" if pt_dist is not None else ""}
            """,
            c="white" if plot_empty else "black",
        ).set_fontsize(10)

    # NOTE: Having repeated values for scatterplots
    # (i.e. [x=1,y=1,c='white',x=1,y=1,c='red'...]) increase processing time
    # Be careful when creating new scatterplots that overlap

    xs_ys, cs, alphas, sizes = None, None, None, None

    # PLOT VOLUMES
    # NOTE:FIRST PROCESSED ARE SHOWN ON UPPER LAYER
    # (FIRST SRC VOL THEN REGIONS THEN BEM...)

    # SRC SPACE
    if src_space_points is not None:
        mask = src_space_points.T[axis] > fixed_value

        xs = src_space_points[mask].T[x_label]
        ys = src_space_points[mask].T[y_label]
        new_xs_ys = np.array([xs, ys]).T
        new_cs = None
        new_alphas = None
        new_sizes = np.full(len(new_xs_ys), 1)
        xs_ys, cs, alphas, sizes = merge_points_optimized(
            [xs_ys, new_xs_ys], [cs, new_cs], [alphas, new_alphas], [sizes, new_sizes]
        )

    # BEM SURFACES
    if bem_volume is not None:
        bem_slice = slice_volume(
            bem_volume, fixed_value=fixed_value, axis=axis, n_layers=5
        )
        colors = get_cmap_colors("hsv", bem_volume.max())
        colors[-1] = [1, 0, 0]
        alpha_values = np.array([0, 0.10, 0.10, 1])
        new_xs_ys, new_cs, new_alphas, new_sizes = project_volume_2d(
            bem_slice,
            axis=axis,
            colors=colors,
            alpha_values=alpha_values,
            size_values=np.repeat(1, len(alpha_values)),
        )

        xs_ys, cs, alphas, sizes = merge_points_optimized(
            [xs_ys, new_xs_ys], [cs, new_cs], [alphas, new_alphas], [sizes, new_sizes]
        )

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
        cerebra_slice = slice_volume(
            cerebra_volume, fixed_value=fixed_value, axis=axis, n_layers=n_layers
        )

        avoid_values = []
        if not plot_empty:
            avoid_values.append(0)
        if not plot_whitematter:
            avoid_values.append(103)

        alpha_values = None
        if plot_highlighted_region:
            assert (
                region_centroid is not None
            ), "If plot_highlighted_region = (int) region_centroid should also be provided"
            alpha_values = np.ones(104) * 0.1
            alpha_values[plot_highlighted_region] = 1

        if plot_highlighted_regions:
            alpha_values = np.ones(104) * 0.05
            alpha_values[plot_highlighted_regions] = 1
            if highlighted_region_names is not None and highlighted_region_centroids is not None:
                for i, name in enumerate(highlighted_region_names):
                    if plot_highlighted_regions[i] in cerebra_slice:
                        ax.text(
                            highlighted_region_centroids[i][x_label],
                            highlighted_region_centroids[i][y_label],
                            f"{name}",
                            c="white" if plot_empty else "black",
                            size=20 - len(plot_highlighted_regions)
                        )

        new_xs_ys, new_cs, new_alphas, new_sizes = project_volume_2d(
            cerebra_slice,
            axis=axis,
            colors=colors if volume_colors is None else volume_colors,
            avoid_values=avoid_values,
            alpha_values=alpha_values,
            size_values=np.repeat(1, len(colors)),
        )
        xs_ys, cs, alphas, sizes = merge_points_optimized(
            [xs_ys, new_xs_ys], [cs, new_cs], [alphas, new_alphas], [sizes, new_sizes]
        )

    # Plot point
    if pt is not None:
        if plot_pt_lines:
            ax.vlines(pt[x_label], 0, 256, linestyles="dashed", alpha=0.4, colors="red")
            ax.hlines(pt[y_label], 0, 256, linestyles="dashed", alpha=0.4, colors="red")

        ax.scatter(pt[x_label], pt[y_label])

        if pt_text is not None:
            ax.text(
                pt[x_label] + 5,
                pt[y_label] + 5,
                pt_text,
                fontsize=8,
                c="white" if plot_empty else "black",
            )

    if pt_dist is not None:
        inner_skull_pt, inner_skull_dist = pt_dist
        ax.plot(
            [inner_skull_pt[x_label], pt[x_label]],
            [inner_skull_pt[y_label], pt[y_label]],
            marker="o",
            c="red",
        )

    # T1 volume
    if t1_volume is not None:
        t1_slice = slice_volume(
            t1_volume, fixed_value=fixed_value, axis=axis, n_layers=2
        )
        new_xs_ys, new_cs, new_alphas, new_sizes = project_volume_2d(
            t1_slice,
            axis=axis,
        )
        xs_ys, cs, alphas, sizes = merge_points_optimized(
            [xs_ys, new_xs_ys],
            [cs, new_cs],
            [alphas, new_alphas],
            [sizes, new_sizes],
            default_alpha=0.5,
        )

    if xs_ys is not None:
        xs, ys = xs_ys.T
        ax.scatter(xs, ys, c=cs, alpha=alphas, s=sizes)

    if plot_planes:
        ax.hlines(
            plot_plane_values[y_label],
            0,
            256,
            linestyles="solid",
            alpha=0.5,
            colors=colors[plot_highlighted_region]
            if plot_highlighted_region is not None
            else "gray",
        )
        ax.vlines(
            plot_plane_values[x_label],
            0,
            256,
            linestyles="solid",
            alpha=0.5,
            colors=colors[plot_highlighted_region]
            if plot_highlighted_region is not None
            else "gray",
        )

    if plot_affine:
        aff_translate = affine[:-1, 3]
        ax.hlines(
            aff_translate[y_label],
            0,
            256,
            linestyles="dotted",
            alpha=0.5,
            colors="gray",
        )
        ax.vlines(
            aff_translate[x_label],
            0,
            256,
            linestyles="dotted",
            alpha=0.5,
            colors="gray",
        )

    return fig, ax


def orthoview(
    volume,
    affine,
    axs=None,
    fig=None,
    figsize=(15, 15),
    use_latex_figures=True,
    add_grid=False,
    **kwargs,
):
    if axs is None:
        fig, axs = get_orthoview_axes(
            figsize=figsize, use_latex_figures=use_latex_figures, add_grid=add_grid
        )

    plot_brain_slice_2d(
        volume,
        affine,
        axis=0,
        ax=axs[0, 0],
        **kwargs,
    )
    plot_brain_slice_2d(
        volume,
        affine,
        axis=1,
        ax=axs[0, 1],
        **kwargs,
    )
    plot_brain_slice_2d(
        volume,
        affine,
        axis=2,
        ax=axs[1, 0],
        **kwargs,
    )

    return fig, axs


def get_3d_fig_ax(min_ax_size=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.set_xlabel("X (R)")
    ax.set_ylabel("Y (A)")
    ax.set_zlabel("Z (S)")


    ax.set_xlim([0, 256])
    ax.set_ylim([0, 256])
    ax.set_zlim([0, 256])

    return fig, ax


def plot_volume_3d(
    volume,
    plot_whitematter=False,
    highlighted_regions_pts=None,
    density=8,
    alpha=0.1,
    ax=None,
    bem_surfaces=None,
    src_space_pc=None,
    plot_regions=True,
    volume_colors=None,
):
    fig = None
    if ax is None:
        fig, ax = get_3d_fig_ax()

    xs = []
    ys = []
    zs = []
    cs = []

    cmap_colors = get_cmap_colors()

    if highlighted_regions_pts is not None:
        for region_pts in highlighted_regions_pts:
            xs = []
            ys = []
            zs = []
            cs = []
            first_pt = region_pts[0]
            val = int(volume[first_pt[0], first_pt[1], first_pt[2]])
            # Not all points are plotted for performance reasons
            # The density parameter specifies the gap between points
            for i in range(0, len(region_pts), density):
                xs.append(region_pts.T[0][i])
                ys.append(region_pts.T[1][i])
                zs.append(region_pts.T[2][i])
            # xs, ys, zs = region_pts.T[0], region_pts.T[1], region_pts.T[2]
            ax.scatter(xs, ys, zs, color=cmap_colors[val], alpha=alpha)

    if plot_regions:
        xs = []
        ys = []
        zs = []
        cs = []
        for x in range(0, 256, density):
            for y in range(0, 256, density):
                for z in range(0, 256, density):
                    if volume[x, y, z] != 0:
                        if volume[x, y, z] == 103 and not plot_whitematter:
                            continue
                        val = int(volume[x, y, z])
                        if volume_colors is None:
                            cs.append(cmap_colors[val])
                        else:
                            cs.append(volume_colors[val])
                        xs.append(x)
                        ys.append(y)
                        zs.append(z)

        ax.scatter(xs, ys, zs, c=cs, alpha=0.1 if highlighted_regions is not None else alpha)

    if bem_surfaces is not None:
        xs = bem_surfaces[2].T[0]
        ys = bem_surfaces[2].T[1]
        zs = bem_surfaces[2].T[2]
        ax.scatter(xs, ys, zs, c="red", alpha=alpha, s=0.5)

        xs = bem_surfaces[1].T[0]
        ys = bem_surfaces[1].T[1]
        zs = bem_surfaces[1].T[2]
        ax.scatter(xs, ys, zs, c="gray", alpha=0.1, s=0.2)

        xs = bem_surfaces[0].T[0]
        ys = bem_surfaces[0].T[1]
        zs = bem_surfaces[0].T[2]
        ax.scatter(xs, ys, zs, c="gray", alpha=0.2, s=0.1)

    if src_space_pc is not None:
        xs, ys, zs = src_space_pc.T
        ax.scatter(xs, ys, zs, c="red", alpha=alpha, s=0.5)

    return fig, ax
