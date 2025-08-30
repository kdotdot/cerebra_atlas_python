import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
from cerebra_atlas_python import CerebrA

cerebra = CerebrA(source_space_include_non_cortical=False)
cmap_name = "hsv"


def get_scalar_colormap(vmin, vmax, cmap_name):
    cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    scalar_map = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap(cmap_name))
    return scalar_map


def apply_colormap(array, cmap_name, plot_cmap=True):
    vmin = np.min(array)
    vmax = np.max(array)
    scalar_map = get_scalar_colormap(vmin, vmax, cmap_name)

    if plot_cmap:
        cmap_colors = scalar_map.to_rgba(np.linspace(0, 1, 100))

        print(f"{cmap_colors.shape=}")
        image_data = [np.linspace(0, 1, 100)]
        image_data = np.repeat(image_data, 10, axis=0)
        print(image_data.shape)
        plt.imshow(image_data, cmap=cmap_name)

        # Remove y axis
        plt.gca().yaxis.set_visible(False)
        # Set X axis between 0 and 100 to be between vmin and vmax
        plt.xticks(
            ticks=[0, 25, 50, 75, 99],
            labels=[
                f"{vmin:.2f}",
                f"{vmin + (vmax - vmin) * 0.25:.2f}",
                f"{(vmin + vmax) / 2:.2f}",
                f"{vmin + (vmax - vmin) * 0.75:.2f}",
                f"{vmax:.2f}",
            ],
        )

        # Plt show non blocking
        plt.show(block=False)

    colors = scalar_map.to_rgba(array)
    return colors[..., :3]


# Plot using a list with shape [n_src_space_pts, t]


# print(cerebra.src_space_labels)
# plt.scatter(
#     np.arange(cerebra.src_space_n_total_points),
#     cerebra.src_space_labels,
#     c=cerebra.src_space_labels,
# )
# plt.show()


data = cerebra.src_space_points[:, 0]
colors = apply_colormap(data, cmap_name)


data = []
for t in range(1200):
    new_data = np.roll(colors.copy(), t)
    # for _ in range(20):
    data.append(new_data)
data = np.array(data)
data = data.swapaxes(0, 1)
print(data.shape)

print(data.shape)

# print(f"{data.shape=} {data.min()=} {data.max()=}")
colors = apply_colormap(data, cmap_name)


cerebra.plot_3d(colors=colors)
