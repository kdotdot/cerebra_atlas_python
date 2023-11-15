import matplotlib
import matplotlib.pyplot as plt
import logging
from matplotlib.colors import ListedColormap
import numpy as np
import nibabel as nib
from cerebra_atlas_python.fig_config import figure_features
from cerebra_atlas_python.utils import slice_volume, time_func_decorator

figure_features()

ori_slice = dict(
    P="Coronal", A="Coronal", I="Axial", S="Axial", L="Sagittal", R="Saggital"
)
ori_names = dict(
    P="posterior", A="anterior", I="inferior", S="superior", L="left", R="right"
)


def get_ax_labels(axis):
    if axis == 0:
        x_label = 1
        y_label = 2
    if axis == 1:
        x_label = 0
        y_label = 2
    if axis == 2:
        x_label = 0
        y_label = 1
    return x_label, y_label


def get_2d_fig_ax(fig=None, ax=None, figsize=(6, 6)):
    if ax is None:
        fig = plt.figure(figsize)
        ax = fig.add_subplot()

    ax.set_xlim([0, 256])
    ax.set_ylim([0, 256])

    return fig, ax


def remove_ax(ax):
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def get_orthoview_axes(figsize=(7, 7)):
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    _, axs[0, 0] = get_2d_fig_ax(None, axs[0, 0])
    _, axs[0, 1] = get_2d_fig_ax(None, axs[0, 1])
    _, axs[1, 0] = get_2d_fig_ax(None, axs[1, 0])
    remove_ax(axs[1, 1])

    return fig, axs


# def merge_points_(xs_ys_arr, cs_arr, alphas_arr):
#     xs_ys_keep, xs_ys_new = xs_ys_arr
#     cs_keep, cs_new = cs_arr
#     alphas_keep, alphas_new = alphas_arr
#     print(xs_ys_new.shape, xs_ys_keep.shape)
#     non_dup_mask = ~np.isin(xs_ys_new, xs_ys_keep).all(axis=1)

#     print(xs_ys_new.shape, cs_new.shape, non_dup_mask.shape)
#     print(xs_ys_new, xs_ys_keep)
#     xs_ys = np.append(xs_ys_keep, xs_ys_new[non_dup_mask, :], 0)
#     cs = np.concatenate([cs_keep, cs_new[non_dup_mask, :]])
#     alphas = np.append(alphas_keep, alphas_new[non_dup_mask])

#     return xs_ys, cs, alphas


@time_func_decorator
def merge_points_(
    xs_ys_arr, cs_arr, alphas_arr, default_color=[0.2, 0.2, 0.2, 1], default_alpha=1
):
    xs_ys_keep, xs_ys_new = xs_ys_arr
    cs_keep, cs_new = cs_arr
    alphas_keep, alphas_new = alphas_arr

    # Identify non-duplicate points
    # We'll compare each point in xs_ys_new to all points in xs_ys_keep
    non_dup_mask = ~np.any(
        np.all(xs_ys_new[:, np.newaxis] == xs_ys_keep, axis=2), axis=1
    )

    # Merge arrays while keeping only non-duplicate points
    xs_ys = np.vstack((xs_ys_keep, xs_ys_new[non_dup_mask]))
    if cs_keep is None:
        cs_keep = np.repeat(default_color, xs_ys_keep.shape[0])
    elif cs_new is None:
        cs_new = np.repeat(default_color, xs_ys_new.shape[0])
    cs = np.vstack((cs_keep, cs_new[non_dup_mask]))
    if alphas_keep is None:
        alphas_keep = np.repeat(default_alpha, xs_ys_keep.shape[0])
    elif alphas_new is None:
        alphas_new = np.repeat(default_alpha, xs_ys_new.shape[0])

    alphas = np.concatenate((alphas_keep, alphas_new[non_dup_mask]))

    return xs_ys, cs, alphas


@time_func_decorator
def merge_points_optimized(
    xs_ys_arr, cs_arr, alphas_arr, default_color=[0.2, 0.2, 0.2, 1], default_alpha=1
):
    xs_ys_keep, xs_ys_new = xs_ys_arr
    cs_keep, cs_new = cs_arr
    alphas_keep, alphas_new = alphas_arr

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
        alphas_keep = np.full(len(xs_ys_keep), default_alpha)
    if alphas_new is not None:
        alphas_new = alphas_new[non_dup_indices]  # Index the alphas_new list

    # Step 3: Merge arrays
    xs_ys = np.vstack((xs_ys_keep, non_dup_xs_ys_new))
    cs = np.vstack((cs_keep, cs_new)) if cs_new is not None else cs_keep
    alphas = (
        np.concatenate((alphas_keep, alphas_new))
        if alphas_new is not None
        else alphas_keep
    )

    return xs_ys, cs, alphas


def project_volume_2d(
    volume_slice, axis=0, colors=None, alpha_values=None, avoid_values=[0]
):
    x_label, y_label = get_ax_labels(axis)

    mask = ~np.isin(volume_slice, avoid_values)
    xyzs = np.array(np.where(mask))
    xs_ys = np.array([xyzs[x_label], xyzs[y_label]])

    # FILTER_DUPLICATES
    xs_ys, unique_indices = np.unique(xs_ys, axis=1, return_index=True)

    xyzs = xyzs[:, unique_indices]

    new_values = volume_slice[xyzs[0], xyzs[1], xyzs[2]]
    cs = colors[new_values] if colors is not None else colors
    alphas = alpha_values[new_values] if alpha_values is not None else alpha_values

    return xs_ys.T, cs, alphas


def get_cmap_colors(cmap_name="gist_rainbow", n_classes=103):
    n_colors = int(n_classes) + 1
    cmap = matplotlib.colormaps[cmap_name]
    colors = cmap(np.linspace(0, 1, n_colors))
    white = np.array([1, 0.87, 0.87, 1])
    colors[-1] = white
    black = np.array([0, 0, 0, 1])
    colors[0] = black
    return colors


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
    src_volume=None,
    bem_volume=None,
    plot_highlighted_region=None,
    region_centroid=None,
    pt_dist=None,
    cmap_name="default",
    s=2,
    ax=None,
    pt=None,
    slice_figsize=(6, 6),
    n_layers: str or int = "max",
    n_layers_max=40,
):
    x_label, y_label = get_ax_labels(axis)

    # Obtain matplotlib ax handle
    if ax is None:
        _, ax = get_2d_fig_ax(figsize=slice_figsize)

    # Configure ax
    ax_labels = ["X", "Y", "Z"]

    ax.set_xlabel(ax_labels[x_label])
    ax.set_ylabel(ax_labels[y_label])

    # Setting fixed value (constant value for plotting plane)
    plot_plane_values = None
    if fixed_value is not None:
        pass
    elif pt is not None:
        fixed_value = pt[axis]
        plot_plane_values = pt
    elif plot_highlighted_region is not None and region_centroid is not None:
        fixed_value = region_centroid[axis]
        plot_plane_values = region_centroid
    else:
        fixed_value = int(affine[:, -1][axis])
        plot_plane_values = int(affine[:, -1])

    codes = nib.orientations.aff2axcodes(affine)

    inverse_codes = {"R": "L", "A": "P", "S": "I", "L": "R", "P": "A", "I": "S"}

    ax.text(
        120, 10, inverse_codes[codes[y_label]], c="white" if plot_empty else "black"
    )
    ax.text(120, 240, codes[y_label], c="white" if plot_empty else "black")

    ax.text(
        10, 120, inverse_codes[codes[x_label]], c="white" if plot_empty else "black"
    )
    ax.text(240, 120, codes[x_label], c="white" if plot_empty else "black")

    ax.text(
        10,
        220,
        f"""{codes[axis]} ({ax_labels[axis]})= {fixed_value}
        {"".join(codes)}
        {f"mm to surface={pt_dist[1]:.2f}" if pt_dist is not None else ""}
        """,
        c="white" if plot_empty else "black",
    ).set_fontsize(10)

    if cmap_name != "default" or cerebra_volume.max() > 103:
        cmap_name = "gray" if cmap_name == "default" else cmap_name
        colors = get_cmap_colors(cmap_name, cerebra_volume.max())
    else:
        colors = get_cmap_colors()

    # NOTE: Having repeated values for scatterplots
    # (i.e. [x=1,y=1,c='white',x=1,y=1,c='red'...]) increase processing time
    # Be careful when creating new scatterplots that overlap

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

        xs_ys, cs, alphas = project_volume_2d(
            cerebra_slice,
            axis=axis,
            colors=colors,
            avoid_values=avoid_values,
            alpha_values=alpha_values,
        )
        print(
            f"{xs_ys.shape= } {cs.shape if cs is not None else cs=  } {alpha_values.shape if alpha_values is not None else alpha_values=  }"
        )

    # BEM SURFACES
    if bem_volume is not None:
        bem_slice = slice_volume(
            bem_volume, fixed_value=fixed_value, axis=axis, n_layers=40
        )
        new_xys_ys, new_cs, new_alphas = project_volume_2d(
            bem_slice,
            axis=axis,
            colors=get_cmap_colors("hsv", bem_volume.max()),
            alpha_values=np.array([0, 0.10, 0.10, 0.05]),
        )
        xs_ys, cs, alphas = merge_points_optimized(
            [xs_ys, new_xys_ys], [cs, new_cs], [alphas, new_alphas]
        )
        print(
            f"{xs_ys.shape= } {cs.shape if cs is not None else cs=  } {alpha_values.shape if alpha_values is not None else alpha_values=  }"
        )

    # for i in range(3):  # NOTE: Layering (off)
    #     for x in range(0, 256, 1):
    #         for y in range(0, 256, 1):
    #             if axis == 0:
    #                 val = volume_data[fixed_value - i, x, y]
    #             elif axis == 1:
    #                 val = volume_data[x, fixed_value - i, y]
    #             elif axis == 2:
    #                 val = volume_data[x, y, fixed_value - i]

    #             if val == 0 and not plot_empty:
    #                 continue
    #             if val == 103 and not plot_whitematter:
    #                 continue
    #             if val != 0 and val != 103 and not plot_regions:
    #                 continue
    #             xs.append(x)
    #             ys.append(y)
    #             cs.append(colors[int(val)])

    #     ax.scatter(xs, ys, s=s, c=cs)  # , c=

    # # Plot point
    # if pt is not None:
    #     ax.vlines(pt[x_label], 0, 256, linestyles="dashed", alpha=0.4, colors="red")
    #     ax.hlines(pt[y_label], 0, 256, linestyles="dashed", alpha=0.4, colors="red")

    #     ax.scatter(pt[x_label], pt[y_label])

    # aff_translate = affine[:-1, 3]

    # # TODO: Comprobar que pt y src_volume funciona bien
    # if src_volume is not None:
    #     xs = []
    #     ys = []
    #     cs = []

    #     should_plot = False
    #     for i in range(10):  # Layering
    #         for x in range(0, 256, 1):
    #             for y in range(0, 256, 1):

    #                 if val > 0:
    #                     should_plot = True
    #                     xs.append(x)
    #                     ys.append(y)
    #                     cs.append("#333" if val == 1 else "#DDD")
    #         if should_plot:
    #             break
    #     ax.scatter(xs, ys, s=s, c=cs)  # , c=

    # if pt_dist is not None:
    #     inner_skull_pt, inner_skull_dist = pt_dist
    #     ax.plot(
    #         [inner_skull_pt[x_label], pt[x_label]],
    #         [inner_skull_pt[y_label], pt[y_label]],
    #         marker="o",
    #         c="red",
    #     )

    xs, ys = xs_ys.T

    # cs = np.concatenate(cs)
    # alphas = np.concatenate(alphas)

    ax.scatter(xs, ys, c=cs, alpha=alphas, s=s)
    # ax.scatter(xs, ys, c=cs, alpha=alphas, s=s)

    if plot_planes:
        ax.hlines(
            # -aff_translate[y_label] * affine[y_label, :-1].sum(),
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
            # 255 + aff_translate[x_label] * affine[x_label, :-1].sum(),
            plot_plane_values[x_label],
            0,
            256,
            linestyles="solid",
            alpha=0.5,
            colors=colors[plot_highlighted_region]
            if plot_highlighted_region is not None
            else "gray",
        )

    return ax


def orthoview(volume, affine, axs=None, fig=None, figsize=(15, 15), **kwargs):
    if axs is None:
        fig, axs = get_orthoview_axes(figsize=figsize)

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


def get_3d_fig_ax():
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
    volume, plot_whitematter=False, region_pts=None, density=8, alpha=0.1, ax=None
):
    fig = None
    if ax is None:
        fig, ax = get_3d_fig_ax()

    xs = []
    ys = []
    zs = []
    cs = []

    cmap_colors = get_cmap_colors()

    for x in range(0, 256, density):
        for y in range(0, 256, density):
            for z in range(0, 256, density):
                if volume[x, y, z] != 0:
                    if volume[x, y, z] == 103 and not plot_whitematter:
                        continue
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                    if region_pts is None:
                        cs.append(cmap_colors[int(volume[x, y, z])])
                    else:
                        cs.append((1, 1, 1))

    ax.scatter(xs, ys, zs, c=cs, alpha=0.01 if region_pts is not None else alpha)

    if region_pts is not None:
        xs = []
        ys = []
        zs = []
        cs = []
        for i in range(0, len(region_pts), density):
            xs.append(region_pts.T[0][i])
            ys.append(region_pts.T[1][i])
            zs.append(region_pts.T[2][i])
        # xs, ys, zs = region_pts.T[0], region_pts.T[1], region_pts.T[2]
        ax.scatter(xs, ys, zs, c="red", alpha=alpha)

    return fig, ax
