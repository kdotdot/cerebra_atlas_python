import numpy as np
from cerebra_atlas_python import CerebrA

import matplotlib.pyplot as plt

cerebra = CerebrA()

# Color based on region_id (default)
# cerebra.plot_2d()
# plt.show()

# cerebra.plot_2d(axis=0)
# plt.show()

# cerebra.plot_2d(axis=1)
# plt.show()

# cerebra.plot_2d(axis=2)
# plt.show()

cerebra.plot_2d(pt=[126, 125, 152], pt_text="[126, 125, 152]")
plt.show()

cerebra.plot_2d(kind="orthoview", fixed_value=150, plot_empty=False, plot_affine=True)
plt.show()

# cerebra.plot_2d(fixed_value=150, plot_regions=True, plot_whitematter=True)
# plt.show()

# Plot all pink
# cerebra.plot_3d(colors="#ff00ff", plot_src_space=True)

# cerebra.plot_3d(colors=(0, 1, 0))

# # Color based on position
# cerebra.plot_3d(colors=cerebra.src_space_points / 255)


# # Plot dynamic data
# MAX_FRAMES = 600


# def update(vis, source_space_pc, *, frame):
#     frame_looped = frame % MAX_FRAMES
#     elapsed_loop_frames = frame_looped / MAX_FRAMES
#     colors = np.repeat(
#         [[elapsed_loop_frames, elapsed_loop_frames, elapsed_loop_frames]],
#         len(source_space_pc.data),
#         axis=0,
#     )
#     source_space_pc.update_colors(colors)
#     vis.update_geometry(source_space_pc.get_o3d())


# cerebra.plot3d(colors="#ff00ff", update_fn=update)
