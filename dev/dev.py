import mne
from cerebra_atlas_python import CerebrA, setup_logging

setup_logging()
cerebra = CerebrA()


cerebra.montage_name = "GSN-HydroCel-129-downsample-109"
cerebra.head_size = 0.1027

print(cerebra.get_bem_vertices_mri())

print(cerebra.src_space)
print(cerebra.bem)
print(cerebra.info)
print(cerebra.head_mri_trans)


# print(cerebra.trans_path)
print(cerebra.forward)
