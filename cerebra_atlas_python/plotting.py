import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import nibabel as nib
import numpy as np

from cerebra_atlas_python.fig_config import figure_features, add_grid

figure_features()


# Plot voxel position in volume
# orientation: s: sagital, a:axial, c:coronal
def imshow_mri(data, img, vox, xyz, suptitle, orientation_axis=0, ax=None, slices=None):
    """Show an MRI slice with a voxel annotated."""
    i, j, k = vox

    print(vox)

    ax_passed = True
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(6, 6))
        ax_passed = False

    codes = nib.orientations.aff2axcodes(img.affine)
    # Figure out the title based on the code of this axis
    ori_slice = dict(
        P="Coronal", A="Coronal", I="Axial", S="Axial", L="Sagittal", R="Saggital"
    )
    ori_names = dict(
        P="posterior", A="anterior", I="inferior", S="superior", L="left", R="right"
    )
    title = ori_slice[codes[orientation_axis]]

    x_order = -1 if codes[2] in "LIP" else 1
    y_order = -1 if codes[1] in "LIP" else 1

    if orientation_axis == 0:
        ax.imshow(data[i, :, :], vmin=10, vmax=120, cmap="gray", origin="lower")
        ax.axvline(k, color="y")
        ax.axhline(j, color="y")

        ax.set(
            xlim=[0, data.shape[2] - 1][::x_order],
            ylim=[0, data.shape[1] - 1][::y_order],
            xlabel=f"k ({ori_names[codes[2]]}+)",
            ylabel=f"j ({ori_names[codes[1]]}+)",
            title=f"{title} view: i={i} ({ori_names[codes[0]]}+)",
        )
        if slices is not None:
            xs, ys, zs = slices[0].T[0], slices[0].T[1], slices[0].T[2]
            ax.scatter(zs, ys, c="red")

    elif orientation_axis == 1:
        ax.imshow(data[:, j, :], vmin=10, vmax=120, cmap="gray", origin="lower")
        ax.axvline(k, color="y")
        ax.axhline(i, color="y")

        ax.set(
            xlim=[0, data.shape[2] - 1][::x_order],
            ylim=[0, data.shape[0] - 1][::y_order],
            xlabel=f"k ({ori_names[codes[2]]}+)",
            ylabel=f"i ({ori_names[codes[0]]}+)",
            title=f"{title} view: j={j} ({ori_names[codes[orientation_axis]]}+)",
        )

        if slices is not None:
            xs, ys, zs = slices[1].T[0], slices[1].T[1], slices[1].T[2]
            ax.scatter(zs, xs, c="red")

    elif orientation_axis == 2:
        ax.imshow(data[:, :, k].T, vmin=10, vmax=120, cmap="gray", origin="lower")
        ax.axvline(i, color="y")
        ax.axhline(j, color="y")

        ax.set(
            xlim=[0, data.shape[0] - 1][::x_order],
            ylim=[0, data.shape[1] - 1][::y_order],
            xlabel=f"i ({ori_names[codes[1]]}+)",
            ylabel=f"j ({ori_names[codes[0]]}+)",
            title=f"{title} view: k={k} ({ori_names[codes[orientation_axis]]}+)",
        )

        if slices is not None:
            xs, ys, zs = slices[2].T[0], slices[2].T[1], slices[2].T[2]
            ax.scatter(xs, ys, c="red")

    for kind, coords in xyz.items():
        annotation = "{}: {}, {}, {} mm".format(kind, *np.round(coords).astype(int))
        if orientation_axis == 0:
            text = ax.text(
                k, j, annotation, va="baseline", ha="right", color=(1, 1, 0.7)
            )
        elif orientation_axis == 1:
            text = ax.text(
                k, i, annotation, va="baseline", ha="right", color=(1, 1, 0.7)
            )
        elif orientation_axis == 2:
            text = ax.text(
                i, j, annotation, va="baseline", ha="right", color=(1, 1, 0.7)
            )
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=2, foreground="black"),
                path_effects.Normal(),
            ]
        )
    # reorient view so that RAS is always rightward and upward

    if not ax_passed:
        fig.suptitle(suptitle)
        fig.subplots_adjust(0.1, 0.1, 0.95, 0.85)
        return fig


# takes in brain voxel volume (RAS)
def plot_brain_slice_2D(
    volume_data,
    affine=None,
    axis=0,
    fixed_value=128,
    n_classes=103,
    plot_whitematter=False,
    cmap_name="hsv",
    ax=None,
    pt=None,
    plot_midlines=False,
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

    xs = []
    ys = []
    cs = []

    norm = plt.Normalize(0, n_classes)
    cmap = get_cmap(n_classes, name=cmap_name)

    for x in range(0, 256, 1):
        for y in range(0, 256, 1):
            if axis == 0:
                val = volume_data[fixed_value, x, y]
            elif axis == 1:
                val = volume_data[x, fixed_value, y]
            elif axis == 2:
                val = volume_data[x, y, fixed_value]

            if val != 0:
                if val == 103 and not plot_whitematter:
                    continue

                xs.append(x)
                ys.append(y)
                cs.append(val)

    ax.scatter(xs, ys, c=cs, cmap=cmap)

    codes = nib.orientations.aff2axcodes(affine)

    inverse_codes = {"R": "L", "A": "P", "S": "I"}

    ax_labels = ["X", "Y", "Z"]

    ax.set_xlabel(ax_labels[x_label])
    ax.set_ylabel(ax_labels[y_label])

    ax.set_xlim([0, 256])
    ax.set_ylim([0, 256])

    ax.text(120, 10, inverse_codes[codes[y_label]])
    ax.text(120, 240, codes[y_label])

    ax.text(10, 120, inverse_codes[codes[x_label]])
    ax.text(240, 120, codes[x_label])

    # Plot point
    if pt is not None:
        ax.scatter(pt[x_label], pt[y_label])

    if plot_midlines:
        ax.hlines(124, 0, 256, linestyles="dotted", alpha=0.4, colors="red")
        ax.vlines(124, 0, 256, linestyles="dotted", alpha=0.4, colors="red")

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

    ax.scatter(xs, ys)

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
