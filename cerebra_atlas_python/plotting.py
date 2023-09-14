import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
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
    affine=None,
    axis=0,
    fixed_value=None,
    n_classes=103,
    plot_whitematter=False,
    cmap_name="hsv",
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

    xs = []
    ys = []
    cs = []

    norm = plt.Normalize(0, n_classes)
    cmap = get_cmap(n_classes, name=cmap_name)

    for i in range(3):
        for x in range(0, 256, 1):
            for y in range(0, 256, 1):
                if axis == 0:
                    val = volume_data[fixed_value - i, x, y]
                elif axis == 1:
                    val = volume_data[x, fixed_value - i, y]
                elif axis == 2:
                    val = volume_data[x, y, fixed_value - i]

                if val != 0:
                    if val == 103 and not plot_whitematter:
                        continue

                    xs.append(x)
                    ys.append(y)
                    cs.append(val)

        ax.scatter(xs, ys, c=cs, cmap=cmap, s=0.3)

    codes = nib.orientations.aff2axcodes(affine)

    inverse_codes = {"R": "L", "A": "P", "S": "I", "L": "R", "P": "A", "I": "S"}

    ax_labels = ["X", "Y", "Z"]

    ax.set_xlabel(ax_labels[x_label])
    ax.set_ylabel(ax_labels[y_label])

    ax.set_xlim([0, 256])
    ax.set_ylim([0, 256])

    ax.text(120, 10, inverse_codes[codes[y_label]])
    ax.text(120, 240, codes[y_label])

    ax.text(10, 120, inverse_codes[codes[x_label]])
    ax.text(240, 120, codes[x_label])

    ax.text(
        10,
        220,
        f"""{codes[axis]} ({ax_labels[axis]})= {fixed_value}
        {"".join(codes)}     
        """,
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

    return ax


def add_region_plot_to_ax(ax, pts, centroid, axis=0):
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

    ax.scatter(xs, ys, s=0.7)

    return ax


def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def plot_volume_3d(volume, n_classes=103, plot_whitematter=False, region_pts=None):
    ax = plt.figure().add_subplot(projection="3d")

    xs = []
    ys = []
    zs = []
    cs = []

    norm = plt.Normalize(0, n_classes)
    cmap = get_cmap(n_classes)

    for x in range(0, 256, 4):
        for y in range(0, 256, 4):
            for z in range(0, 256, 4):
                if volume[x, y, z] != 0:
                    if volume[x, y, z] == 103 and not plot_whitematter:
                        continue
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                    if region_pts is None:
                        cs.append(volume[x, y, z])
                    else:
                        cs.append((0.8, 0.8, 0.8))

    ax.scatter(xs, ys, zs, c=cs, cmap=cmap, alpha=0.01 if region_pts is not None else 1)

    if region_pts is not None:
        xs, ys, zs = region_pts.T[0], region_pts.T[1], region_pts.T[2]
        ax.scatter(xs, ys, zs, c="red")

    ax.set_xlabel("X (R)")
    ax.set_ylabel("Y (A)")
    ax.set_zlabel("Z (S)")

    ax.set_xlim([0, 256])
    ax.set_ylim([0, 256])
    ax.set_zlim([0, 256])


def remove_ax(ax):
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def orthoview(volume, affine, center_pt=None, **kwargs):
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
        fixed_value=center_pt[0] if center_pt is not None else None,
        **kwargs,
    )
    plot_brain_slice_2D(
        volume,
        affine,
        axis=2,
        ax=axs[1, 0],
        fixed_value=center_pt[0] if center_pt is not None else None,
        **kwargs,
    )
    remove_ax(axs[1, 1])

    return fig, axs


def orthoview_region(reg_points, reg_centroid, volume, affine, **kwargs):
    fig, axs = orthoview(
        volume, affine, center_pt=reg_centroid, cmap_name="gray", **kwargs
    )
    add_region_plot_to_ax(axs[0, 0], reg_points, reg_centroid, axis=0)
    add_region_plot_to_ax(axs[0, 1], reg_points, reg_centroid, axis=1)
    add_region_plot_to_ax(axs[1, 0], reg_points, reg_centroid, axis=2)

    return fig, axs
