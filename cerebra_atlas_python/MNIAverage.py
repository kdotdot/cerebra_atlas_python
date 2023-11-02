import os
import os.path as op
import numpy as np
import mne
import logging


######### Computed using FreeSurfer "MNIAverage" surfaces generated previosly
# NOTE: This class is common to all subjects and datasets, should be instantiated just once
class MNIAverage:
    def __init__(
        self,
        mniAverage_output_path="./generated",
        subjects_dir=os.getenv("SUBJECTS_DIR"),
        manual_fit_fiducials=True,
        bem_conductivity=(0.33, 0.0042, 0.33),
        bem_ico=4,
    ):
        self.vol_src = None
        self.bem = None
        self.fiducials = None
        self.output_path = mniAverage_output_path
        if not op.exists(mniAverage_output_path):
            os.mkdir(mniAverage_output_path)
        self.vol_src_path = op.join(mniAverage_output_path, "MNIAverage-v-src.fif")
        self.bem_conductivity_string = "".join(
            [str(x) + "_" for x in bem_conductivity]
        )[:-1]

        self.bem_ico = bem_ico
        self.subjects_dir = subjects_dir
        self.manual_fit_fiducials = manual_fit_fiducials
        self.bem_conductivity = bem_conductivity
        self.bem_path = op.join(
            mniAverage_output_path,
            f"MNIAverage-bem-{self.bem_conductivity_string}-{self.bem_ico}.fif",
        )
        self._set_bem()
        self._set_vol_src()
        self._set_fiducials()

    def get_src_volume(self, transform=None):
        src = self.vol_src[0]
        pts = mne.transforms.apply_trans(src["mri_ras_t"], src["rr"]) * 1000
        if transform is not None:
            pts = mne.transforms.apply_trans(transform, pts).astype(int)
        src_space = np.zeros((256, 256, 256)).astype(int)
        for i, pt in enumerate(pts):
            x, y, z = pt
            if i in src["vertno"]:
                src_space[x, y, z] = 1  # Usable source space
            else:
                src_space[x, y, z] = 2  # Box around source space
        return src_space

    def get_bem_surfaces(self, transform=None):
        src = self.vol_src[0]
        surfaces = []
        for surf in self.bem["surfs"]:
            pts = mne.transforms.apply_trans(src["mri_ras_t"], surf["rr"]) * 1000
            if transform is not None:
                pts = mne.transforms.apply_trans(transform, pts).astype(int)
            surfaces.append(pts)
        surfaces = np.array(surfaces)
        return surfaces

    # If vol src fif does not exist, create it, otherwise read it
    def _set_vol_src(self):
        if not op.exists(self.vol_src_path):
            logging.info(f"Generating volume source space...")
            surface = op.join(
                self.subjects_dir, "MNIAverage", "bem", "inner_skull.surf"
            )
            self.vol_src = mne.setup_volume_source_space(
                "MNIAverage",
                surface=surface,
                add_interpolator=False,  # Just for speed!
            )
            self.vol_src.save(self.vol_src_path, overwrite=True, verbose=True)
        else:
            logging.info(f"Reading volume source space from disk")
            self.vol_src = mne.read_source_spaces(self.vol_src_path)

    # Same for BEM
    def _set_bem(self):
        if not op.exists(self.bem_path):
            logging.info(
                f"Generating boundary element model... conductivity={self.bem_conductivity_string} ico= {self.bem_ico}"
            )
            # conductivity = (0.3,)  # for single layer

            model = mne.make_bem_model(
                subject="MNIAverage",
                ico=self.bem_ico,
                conductivity=self.bem_conductivity,
            )  # subjects_dir is env variable
            self.bem = mne.make_bem_solution(model)
            mne.write_bem_solution(
                self.bem_path, self.bem, overwrite=True, verbose=True
            )
        else:
            logging.info(
                f"Loading boundary element model from disk | conductivity={self.bem_conductivity_string} ico= {self.bem_ico}"
            )
            self.bem = mne.read_bem_solution(self.bem_path)

    # Same for BEM
    def _set_fiducials(self):
        self.fiducials = mne.coreg.get_mni_fiducials("MNIAverage")
        if self.manual_fit_fiducials:
            shift_x = 0
            shift_y = 0.005
            shift_z = -0.005
            # Manual fit (Works for EEG-MI & ERP-Core, might be different for other datasets)
            self.fiducials[0]["r"] = np.array(self.fiducials[0]["r"]) - np.array(
                [shift_x, shift_y, shift_z]
            )
            self.fiducials[1]["r"] = np.array(self.fiducials[1]["r"]) - np.array(
                [shift_x, shift_y, shift_z]
            )
            self.fiducials[2]["r"] = np.array(self.fiducials[2]["r"]) - np.array(
                [shift_x, shift_y, shift_z]
            )

    def index_to_ras(self, idx):
        src = self.vol_src[0]
        return mne.transforms.apply_trans(src["mri_ras_t"], src["rr"][idx, :]) * 1000


if __name__ == "__main__":
    mniAverage = MNIAverage()
    print(mniAverage)
