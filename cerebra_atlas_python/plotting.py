import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import nibabel as nib
import numpy as np

from cerebra_atlas_python.fig_config import figure_features, add_grid

figure_features()

ori_slice = dict(
    P="Coronal", A="Coronal", I="Axial", S="Axial", L="Sagittal", R="Saggital"
)
ori_names = dict(
    P="posterior", A="anterior", I="inferior", S="superior", L="left", R="right"
)


# takes in brain voxel volume (RAS)
def plot_brain_slice_2D(
    volume_data,
    affine,
    axis=0,
    fixed_value=None,
    plot_regions=True,
    plot_whitematter=False,
    plot_empty=False,
    src_space=None,
    bem_surfaces=None,
    pt_dist=None,
    cmap_name="default",
    s=0.3,
    ax=None,
    pt=None,
    plot_affine=False,
):
    if ax is None:
        ax = plt.figure(figsize=(6, 6)).add_subplot()

    if axis == 0:
        x_label = 1
        y_label = 2
    if axis == 1:
        x_label = 0
        y_label = 2
    if axis == 2:
        x_label = 0
        y_label = 1

    if pt is not None:
        fixed_value = pt[axis]
    elif fixed_value is None:
        fixed_value = int(affine[:, -1][axis])

    if bem_surfaces is not None:
        colors = ["#739072", "#4F6F52", "#3A4D39"]
        for i, surf in enumerate(bem_surfaces):
            mask = (surf.T[axis] > fixed_value - 1) & surf.T[axis] < fixed_value
            xs, ys = surf[mask, x_label], surf[mask, y_label]
            alpha = 0.10 + 0.1 * i
            ax.scatter(xs, ys, c=colors[i], alpha=alpha, s=alpha)  # , c=

    xs = []
    ys = []
    cs = []

    if cmap_name != "default" or volume_data.max() > 103:
        cmap_name = "gray" if cmap_name == "default" else cmap_name
        colors = get_cmap_colors(cmap_name, volume_data.max())
    else:
        colors = get_cmap_colors()

    for i in range(3):  # NOTE: Layering (off)
        for x in range(0, 256, 1):
            for y in range(0, 256, 1):
                if axis == 0:
                    val = volume_data[fixed_value - i, x, y]
                elif axis == 1:
                    val = volume_data[x, fixed_value - i, y]
                elif axis == 2:
                    val = volume_data[x, y, fixed_value - i]

                if val == 0 and not plot_empty:
                    continue
                if val == 103 and not plot_whitematter:
                    continue
                if val != 0 and val != 103 and not plot_regions:
                    continue
                xs.append(x)
                ys.append(y)
                cs.append(colors[int(val)])

        ax.scatter(xs, ys, s=s, c=cs)  # , c=

    codes = nib.orientations.aff2axcodes(affine)

    inverse_codes = {"R": "L", "A": "P", "S": "I", "L": "R", "P": "A", "I": "S"}

    ax_labels = ["X", "Y", "Z"]

    ax.set_xlabel(ax_labels[x_label])
    ax.set_ylabel(ax_labels[y_label])

    ax.set_xlim([0, 256])
    ax.set_ylim([0, 256])

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

    # Plot point
    if pt is not None:
        ax.vlines(pt[x_label], 0, 256, linestyles="dashed", alpha=0.4, colors="red")
        ax.hlines(pt[y_label], 0, 256, linestyles="dashed", alpha=0.4, colors="red")

        ax.scatter(pt[x_label], pt[y_label])

    aff_translate = affine[:-1, 3]

    if plot_affine:
        ax.hlines(
            # -aff_translate[y_label] * affine[y_label, :-1].sum(),
            aff_translate[y_label],
            0,
            256,
            linestyles="solid",
            alpha=0.5,
            colors="orange",
        )
        # print(256 + -aff_translate[x_label] * affine[:, x_label].sum())
        ax.vlines(
            # 255 + aff_translate[x_label] * affine[x_label, :-1].sum(),
            aff_translate[x_label],
            0,
            256,
            linestyles="solid",
            alpha=0.5,
            colors="orange",
        )

    # TODO: Comprobar que pt y src_space funciona bien
    if src_space is not None:
        xs = []
        ys = []
        cs = []

        should_plot = False
        for i in range(10):  # Layering
            for x in range(0, 256, 1):
                for y in range(0, 256, 1):
                    if axis == 0:
                        val = src_space[fixed_value - i, x, y]
                    elif axis == 1:
                        val = src_space[x, fixed_value - i, y]
                    elif axis == 2:
                        val = src_space[x, y, fixed_value - i]

                    if val > 0:
                        should_plot = True
                        xs.append(x)
                        ys.append(y)
                        cs.append("#333" if val == 1 else "#DDD")
            if should_plot:
                break
        ax.scatter(xs, ys, s=s, c=cs)  # , c=

    if pt_dist is not None:
        inner_skull_pt, inner_skull_dist = pt_dist
        ax.plot(
            [inner_skull_pt[x_label], pt[x_label]],
            [inner_skull_pt[y_label], pt[y_label]],
            marker="o",
            c="red",
        )

    return ax


def get_cmap_colors(cmap_name="gist_rainbow", n_classes=103):
    n_colors = int(n_classes) + 1
    cmap = matplotlib.colormaps[cmap_name]
    colors = cmap(np.linspace(0, 1, n_colors))
    white = np.array([1, 0.87, 0.87, 1])
    colors[-1] = white
    black = np.array([0, 0, 0, 1])
    colors[0] = black
    return colors


def add_region_plot_to_ax(
    ax, pts, centroid, region_id, axis=0, colors=get_cmap_colors()
):
    if axis == 0:
        x_label = 1
        y_label = 2
    if axis == 1:
        x_label = 0
        y_label = 2
    if axis == 2:
        x_label = 0
        y_label = 1

    fixed_value = centroid[axis]
    slice = pts[pts.T[axis] == fixed_value]
    xs = []
    ys = []
    for pt in slice:
        xs.append(pt[x_label])
        ys.append(pt[y_label])

    ax.scatter(xs, ys, s=0.7, color=colors[region_id])

    return ax


def get_cmap():
    newcmp = ListedColormap(get_cmap_colors())
    return newcmp


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


def remove_ax(ax):
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def orthoview(volume, affine, center_pt=None, axs=None, fig=None, **kwargs):
    if axs is None:
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    plot_brain_slice_2D(
        volume,
        affine,
        axis=0,
        ax=axs[0, 0],
        fixed_value=center_pt[0] if center_pt is not None else None,
        **kwargs,
    )
    plot_brain_slice_2D(
        volume,
        affine,
        axis=1,
        ax=axs[0, 1],
        fixed_value=center_pt[1] if center_pt is not None else None,
        **kwargs,
    )
    plot_brain_slice_2D(
        volume,
        affine,
        axis=2,
        ax=axs[1, 0],
        fixed_value=center_pt[2] if center_pt is not None else None,
        **kwargs,
    )
    remove_ax(axs[1, 1])

    return fig, axs


def orthoview_region(reg_points, reg_centroid, region_id, volume, affine, **kwargs):
    fig, axs = orthoview(volume, affine, center_pt=reg_centroid, **kwargs)
    colors = get_cmap_colors()
    add_region_plot_to_ax(
        axs[0, 0], reg_points, reg_centroid, region_id, axis=0, colors=colors
    )
    add_region_plot_to_ax(
        axs[0, 1], reg_points, reg_centroid, region_id, axis=1, colors=colors
    )
    add_region_plot_to_ax(
        axs[1, 0], reg_points, reg_centroid, region_id, axis=2, colors=colors
    )

    return fig, axs
