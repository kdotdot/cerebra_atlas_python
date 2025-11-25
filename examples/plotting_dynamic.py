import numpy as np
from cerebra_atlas_python import CerebrA
from cerebra_atlas_python.plotting.colors import apply_colormap

cerebra = CerebrA(source_space_include_non_cortical=True)
cmap_name = "hsv"

data = cerebra.src_space_points[:, 0]
colors = apply_colormap(data, cmap_name)

data = []
for t in range(1200):
    new_data = np.roll(colors.copy(), t)
    # for _ in range(20):
    data.append(new_data)
data = np.array(data)
data = data.swapaxes(0, 1)

colors = apply_colormap(data, cmap_name)
cerebra.plot_3d(colors=colors)
