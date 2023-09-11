import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import nibabel as nib
import numpy as np


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
