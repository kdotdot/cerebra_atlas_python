import numpy as np
from cerebra_atlas_python import CerebrA

cerebra = CerebrA()

# Color based on region_id (default)
cerebra.plot3d()

# Plot all pink
# cerebra.plot3d(colors="#ff00ff")

cerebra.plot3d(colors=(0, 1, 0))

# Color based on position
cerebra.plot3d(colors=cerebra.src_space_points / 255)


# Plot dynamic data
MAX_FRAMES = 600


def update(vis, source_space_pc, *, frame):
    frame_looped = frame % MAX_FRAMES
    elapsed_loop_frames = frame_looped / MAX_FRAMES
    colors = np.repeat(
        [[elapsed_loop_frames, elapsed_loop_frames, elapsed_loop_frames]],
        len(source_space_pc.data),
        axis=0,
    )
    source_space_pc.update_colors(colors)
    vis.update_geometry(source_space_pc.get_o3d())


cerebra.plot3d(colors="#ff00ff", update_fn=update)
