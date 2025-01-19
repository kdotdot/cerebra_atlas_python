from cerebra_atlas_python import CerebrA, setup_logging

setup_logging("DEBUG")
cerebra = CerebrA()
cerebra.montage_name = "standard_1020-downsample-64-10-10"
cerebra.head_size = 0.10

# cerebra.plot3d()
# cerebra.plot3d(plot_src_space=False, plot_montage=True)
# cerebra.plot3d(plot_src_space=False, plot_bem=True)

cerebra.plot3d(plot_src_space=True, plot_bem=True, plot_montage=True)
