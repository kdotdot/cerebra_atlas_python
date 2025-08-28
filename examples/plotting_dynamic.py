import numpy as np
from cerebra_atlas_python import CerebrA

cerebra = CerebrA()


# Plot using a list with shape [n_src_space_pts, t]
colors = np.random.rand(len(cerebra.get_src_space_points()), 1200)
print(colors.shape)
cerebra.plot_3d(colors=colors)
# if __name__ == "__main__":
#     max_col = np.percentile(stc_eloreta_diff_s1_subject_0.data, 99)
#     cNorm = matplotlib.colors.Normalize(vmin=0, vmax=max_col)
#     scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap("YlOrRd"))
#     scalarMap.to_rgba(stc_eloreta_diff_s1_subject_0.data[:, 0] / max_col)
#     src_space_pts = np.random.rand()
