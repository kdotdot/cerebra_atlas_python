from cerebra_atlas_python import CerebrA

cerebra = CerebrA()

print(f"CerebrA: {cerebra.src_space_labels} {cerebra.src_space_labels.shape=}")
print(cerebra._label_details)


import numpy as np
import matplotlib.pyplot as plt


colors = [[0.0, 0.0, 1.0]] * len(cerebra.src_space_labels)
print(f"{colors=}")

cerebra.plot3d(colors=colors)
plt.show()
