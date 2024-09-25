import time
import logging
import mne
import matplotlib
import os.path as op
import open3d as o3d
import numpy as np
import scipy

from cerebra_atlas_python import CerebrA
from cerebra_atlas_python.utils import setup_logging

setup_logging("DEBUG")

# CONSTANTS
MAX_FRAME = 75 * 60 * 60 * 60  # 75 frames one minute

# DRAW FUNCTIONS
def init():
    vis.create_window()

    # Bounding box (256,256,256)
    points = np.array(
        [
            [0, 0, 0],
            [0, 0, 255],
            [0, 255, 0],
            [0, 255, 255],
            [255, 0, 0],
            [255, 0, 255],
            [255, 255, 0],
            [255, 255, 255],
        ]
    )
    colors = np.array(
        [
            [0, 0, 0],
            [0, 0, 255],
            [0, 255, 0],
            [0, 255, 255],
            [255, 0, 0],
            [255, 0, 255],
            [255, 255, 0],
            [255, 255, 255],
        ]
    )
    points_o3d = o3d.utility.Vector3dVector(points)
    pc = o3d.geometry.PointCloud(points_o3d)
    pc.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pc)

    # CEREBRA
    # Points
    cerebra_points = cerebra.src_space_points
    cerebra_points_o3d = o3d.utility.Vector3dVector(cerebra_points)
    cerebra_pc = o3d.geometry.PointCloud(cerebra_points_o3d)
    colors = np.array(len(cerebra_points) * [[0, 0, 1]])
    cerebra_pc.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(cerebra_pc)



# Every frame
def update():
    global vis
    global total_updates
    vis.poll_events()
    total_updates += 1
    vis.update_renderer()


# Global variables
vis = o3d.visualization.Visualizer()
cerebra = CerebrA(source_space_include_non_cortical=True)

# Updates
update_every = 1  # seconds
total_updates = 0
last_update_time = None
should_update_data = True

# Logs
log_every = 1  # seconds
frames_since_last_log = 0
last_log_time = None

init()  # Run init
start_time = time.time()
# Main loop
for frame in range(MAX_FRAME):
    sim_time = (total_updates) * 0.008
    frames_since_last_log += 1
    if last_log_time is None or time.time() - last_log_time > log_every:
        fps = (
            frames_since_last_log / (time.time() - last_log_time)
            if last_log_time is not None
            else 0
        )
        real_time = time.time() - start_time
        # LOG
        print(
            f"\n\n\n\n\n\n\n\n\nFrame: {frame:.0f} fps({fps:.0f})\nReal time: {(real_time):.2f} \n\nSim time: {sim_time:.2f} \n\n\n\n\n\n\n\n\n"
        )
        last_log_time = time.time()
        frames_since_last_log = 0

    if last_update_time is None or time.time() - last_update_time > update_every:
        should_update_data = True
        last_update_time = time.time()


    update()  # Run update


# Cleanup
vis.destroy_window()