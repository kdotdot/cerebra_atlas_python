import matplotlib.pyplot as plt
from cerebra_atlas_python import CerebrA, setup_logging

setup_logging("DEBUG")
cerebra = CerebrA()
cerebra.plot_3d()
