"""
Cerebra...
"""
import os
import os.path as op
import logging
import pickle
import nibabel as nib
import numpy as np
import pandas as pd
import mne

from cerebra_atlas_python.config import BaseConfig
from cerebra_atlas_python.mni_average import MNIAverage
from cerebra_atlas_python.utils import (
    setup_logging,
    move_volume_from_LIA_to_RAS,
    find_closest_point,
)
from cerebra_atlas_python.plotting import (
    plot_volume_3d,
    orthoview,
)


def preprocess_label_details(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the given dataframe by performing several operations such as removing rows and columns,
    converting data types, duplicating and modifying data, and appending new information.

    Args:
        df (pd.DataFrame): The dataframe to preprocess.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    # Remove first row
    df.drop(0, inplace=True)

    # Remove unused columns
    df.drop(columns=["Unnamed: 3", "Notes", "Dice Kappa"], inplace=True)

    # Change id column from string to int
    df["CerebrA ID"] = pd.to_numeric(df["CerebrA ID"])

    # Copy df and append
    df = pd.concat([df, df])
    df.reset_index(inplace=True, drop=True)

    # Modify left side labels
    df.loc["52":, "CerebrA ID"] = df.loc["52":, "CerebrA ID"] + 51

    # Modify names to include hemisphere
    df.loc["52":, "Label Name"] = "Left " + df.loc["52":, "Label Name"]
    df.loc[:"52", "Label Name"] = "Right " + df.loc[:"52", "Label Name"]

    # Add white matter to label details
    df.loc[len(df.index)] = [0, "White matter", 103]

    return df


def get_label_details(path):
    """
    Reads a CSV file from the given path and preprocesses its contents using the preprocess_label_details function.

    Args:
        path (str): The file path of the CSV file to be read and processed.

    Returns:
        pd.DataFrame: The preprocessed dataframe obtained from the CSV file.
    """
    return preprocess_label_details(pd.read_csv(path))


def get_volume_ras(path, dtype=np.uint8):
    """
    Loads a medical image volume from the given path and converts its coordinate frame from LIA to RAS.

    This function:
    1. Loads the volume using nibabel.
    2. Converts the volume's coordinate frame from LIA (Left, Inferior, Anterior) to RAS (Right, Anterior, Superior)
       using the move_volume_from_LIA_to_RAS function.

    Args:
        path (str): The file path of the medical image volume.
        dtype (type, optional): The data type to be used for the volume data. Defaults to np.uint8.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the transformed volume data and its affine matrix.
    """
    img = nib.load(path)  # All volumes are in LIA coordinate frame
    volume, affine = move_volume_from_LIA_to_RAS(
        np.array(img.dataobj, dtype=dtype), img.affine
    )
    return volume, affine


def point_cloud_to_voxel(
    point_cloud: np.ndarray, dtype=np.uint8, vox_value: int = 1
) -> np.ndarray:
    """
    Transforms a given point cloud into a voxel array.

    This function takes an array representing a point cloud where each point is a 3D coordinate,
    and converts it into a voxel representation with a specified size of [256, 256, 256].
    Each point in the point cloud is mapped to a voxel in this 3D grid. The values in the point
    cloud array should be in the range [0, 257) (RAS).

    Args:
        point_cloud (np.ndarray): A numpy array of shape [n_points, 3] representing the point cloud,
                                  where n_points is the number of points in the cloud and each point
                                  is a 3D coordinate.
        dtype (type, optional): The data type to be used for the voxel grid. Defaults to np.uint8.
        vox_value (int): set value for voxel grid

    Returns:
        np.ndarray: A voxel array of shape [256, 256, 256] representing the 3D grid, where each
                    element is set to 1 if it corresponds to a point in the input array, otherwise 0.
    """
    # Initialize a voxel grid of the specified size filled with zeros
    voxel_grid = np.zeros((256, 256, 256), dtype=dtype)

    # Iterate through each point in the point cloud
    for point in point_cloud:
        # Check if the point is within the valid range
        if all(0 <= coord < 256 for coord in point):
            # Convert the floating point coordinates to integers
            x, y, z = map(int, point)
            # Set the corresponding voxel to 1
            voxel_grid[x, y, z] = vox_value

    return voxel_grid


def get_cerebra_volume(cerebra_mgz, wm_mgz):
    """
    Processes cerebra and white matter medical image volumes to integrate
    white matter information into the cerebra volume.

    This function:
    1. Retrieves the volume data and affine matrices for both cerebra and white matter images in RAS coordinate frame.
    2. Modifies the cerebra volume by adding a specific label (103) to represent white matter regions
       that are not already labeled in the cerebra volume.

    Args:
        cerebra_mgz (str): The file path of the cerebra medical image volume.
        wm_mgz (str): The file path of the white matter medical image volume.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the modified cerebra volume data and its affine matrix.
    """
    cerebra_volume, cerebra_affine = get_volume_ras(cerebra_mgz)
    wm_volume, _ = get_volume_ras(wm_mgz)

    # Add whitematter to volume data
    cerebra_volume[(wm_volume != 0) & (cerebra_volume == 0)] = 103
    return cerebra_volume, cerebra_affine


def merge_voxel_grids(grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
    """
    Merges two voxel grids of the same size into one.

    The merge operation is a logical OR between the corresponding elements of the two voxel grids.
    If a voxel is set in either of the grids (value of 1), it will be set in the resulting merged grid.

    Args:
        grid1 (np.ndarray): The first voxel grid, expected to be of size [256, 256, 256].
        grid2 (np.ndarray): The second voxel grid, expected to be of the same size as grid1.

    Returns:
        np.ndarray: The merged voxel grid of size [256, 256, 256].

    Raises:
        ValueError: If the input grids are not of the same shape.
    """
    if grid1.shape != grid2.shape:
        raise ValueError("The two voxel grids must be of the same shape.")

    # Perform logical OR operation to merge the voxel grids
    merged_grid = grid1 + grid2

    # Make sure total number of voxels stayed the same
    # (No voxels overlap)
    assert (merged_grid != 0).sum() == (grid1 != 0).sum() + (grid2 != 0).sum()

    return merged_grid


class CerebrA(BaseConfig):
    """
    Initializes a CerebrA object with configurations for processing brain imaging data.

    This class is responsible for setting up paths and configurations for cerebra data processing,
    loading and processing brain volumes and label details, and providing functionalities for
    coordinate frame transformations and region-based analysis.

    Args:
        mni_average (Optional[MNIAverage]): An instance of MNIAverage. If None, a new instance is created.
        MNIAverageKwArgs (Optional[Dict[Any, Any]]): Keyword arguments for creating an MNIAverage instance.
        **kwargs: Additional keyword arguments for BaseConfig.

    Attributes:
        cerebra_output_path (str): Path for cerebra output data.
        default_data_path (str): Default path for cerebra data.
        bem_surfaces (Optional[np.ndarray]): Array of 3 BEM point clouds.
        bem_volume (Optional[np.ndarray]): Voxel grid of size [256, 256, 256] representing source space.
        src_space_pc (Optional[np.ndarray]): Source space point cloud.
        src_space_volume (Optional[np.ndarray]): Voxel grid of size [256, 256, 256] representing source space.
        mni_average (MNIAverage): MNIAverage instance.
        cerebra_volume (np.ndarray): Processed cerebra volume data.
        affine (np.ndarray): Affine matrix for coordinate transformations.
        label_details (pd.DataFrame): Dataframe containing label details for brain regions.
        region_ids (np.ndarray): Sorted array of unique region IDs in label_details.
        volume_data_sparse (Dict[int, np.ndarray]): Sparse representation of the volume data.
    """

    def __init__(
        self,
        mni_average=None,
        MNIAverageKwArgs=None,
        **kwargs,
    ):
        self.cerebra_output_path: str = None
        self.default_data_path: str = None
        default_config = {
            "cerebra_output_path": "./generated/cerebra",
            "default_data_path": "../cerebra_data/cerebra",
        }

        super().__init__(
            parent_name=self.__class__.__name__,
            default_config=default_config,
            **kwargs,
        )

        self.bem_surfaces = None  # Array of 3 bem point clouds
        self.bem_volume = (
            None  # Voxel grid of size [256, 256, 256] : 1 represent source space
        )
        self.src_space_pc = None  # Source space point cloud
        self.src_space_volume = (
            None  # Voxel grid of size [256, 256, 256] : 1 represent source space
        )

        # Instantiate/ assign MNIAverage object
        if mni_average is None:
            MNIAverageKwArgs = MNIAverageKwArgs or {}
            self.mni_average = MNIAverage(**MNIAverageKwArgs)
        else:
            assert isinstance(
                mni_average, MNIAverage
            ), f"Wrong class should be MNIAverage {type(mni_average)= }"

            self.mni_average = mni_average

        # If output folder does not exist, create it
        if not op.exists(self.cerebra_output_path):
            os.makedirs(self.cerebra_output_path, exist_ok=True)

        # Define volumes' path
        self.cerebra_path = cerebra_path = op.join(
            self.default_data_path, "CerebrA_in_head.mgz"
        )
        # t1_path = op.join(self.mni_average.fs_subjects_dir, "MNIAverage/mri/T1.mgz")
        wm_path = op.join(
            self.mni_average.fs_subjects_dir, "MNIAverage/mri/wm.asegedit.mgz"
        )

        # Set volume
        self.cerebra_volume, self.affine = get_cerebra_volume(cerebra_path, wm_path)

        # Read labels
        label_details_path = op.join(self.default_data_path, "CerebrA_LabelDetails.csv")
        self.label_details = get_label_details(label_details_path)

        # Metadata
        self.region_ids = np.sort(self.label_details["CerebrA ID"].unique())

        # Sparse representation
        cerebra_sparse_path = op.join(self.cerebra_output_path, "CerebrA_sparse.npy")
        if not op.exists(cerebra_sparse_path):
            logging.info("Generating sparse representation of Cerebra volume data...")
            self.volume_data_sparse = {
                region_id: self.get_points_from_region_id(region_id)
                for region_id in self.region_ids
            }
            with open(cerebra_sparse_path, "wb") as handle:
                pickle.dump(
                    self.volume_data_sparse, handle, protocol=pickle.HIGHEST_PROTOCOL
                )
        else:
            with open(cerebra_sparse_path, "rb") as handle:
                self.volume_data_sparse = pickle.load(handle)

    # COORDINATE FRAME TRANSFORMATIONS
    def center_ras(self, pts: np.ndarray) -> np.ndarray:
        #  RAS (non-zero origin) -> RAS
        return mne.transforms.apply_trans(self.affine, pts)

    # NOTE: Unused. Remove (?)
    def inverse_center_ras(self, pts: np.ndarray) -> np.ndarray:
        return np.round(mne.transforms.apply_trans(np.linalg.inv(self.affine), pts))

    # CENTERED MNI_AVERAGE POINTS
    # Returns array of length 3 of point clouds
    def get_bem_surfaces(self):
        if self.bem_surfaces is None:
            self.bem_surfaces = self.mni_average.get_bem_surfaces_ras_nzo(
                transform=self.affine
            )
        return self.bem_surfaces  # RAS coordinate frame

    def get_bem_volume(self):
        if self.bem_volume is None:
            bem_surfaces = self.get_bem_surfaces()
            bem_volume = None
            for surf, bem_id in zip(bem_surfaces, self.mni_average.bem_names.keys()):
                if bem_volume is None:
                    bem_volume = point_cloud_to_voxel(surf, vox_value=bem_id)
                else:
                    bem_volume = merge_voxel_grids(
                        bem_volume, point_cloud_to_voxel(surf, vox_value=bem_id)
                    )
            self.bem_volume = bem_volume
        return self.bem_volume

    # Returns point cloud
    def get_src_space_pc(self):
        if self.src_space_pc is None:
            self.src_space_pc = self.mni_average.get_src_space_ras_nzo(
                transform=self.affine
            )
        return self.src_space_pc

    def get_src_space_volume(self):
        if self.src_space_volume is None:
            src_space_pc = self.get_src_space_pc()
            self.src_space_volume = point_cloud_to_voxel(src_space_pc, vox_value=1)
        return self.src_space_volume

    # Functions
    def get_region_id_from_point(self, point):
        region_id = self.cerebra_volume[point[0], point[1], point[2]]
        return int(region_id)  # Can be 0 and 103

    def get_region_data_from_region_id(self, region_id):
        region_data = self.label_details[self.label_details["CerebrA ID"] == region_id]
        if len(region_data) < 1:
            return None
        return region_data

    def get_region_data_from_region_name(self, region_name):
        region_id = self.get_region_id_from_region_name(region_name)
        return self.get_region_data_from_region_id(region_id)

    def get_region_id_from_region_name(self, region_name):
        region_data = self.get_region_data_from_region_name(region_name)
        return region_data["CerebrA ID"].item()

    def get_region_name_from_region_id(self, region_id):
        if region_id == 0:
            return "Empty region"
        if region_id > 103:
            return "Non-valid id"

        return self.label_details[self.label_details["CerebrA ID"] == region_id][
            "Label Name"
        ].item()

    # Given a point in a (256,256,256) 3d space (RAS), determines
    # which brain region it belongs to
    def get_region_name_from_point(self, point):
        region_id = self.get_region_id_from_point(point)
        region_name = self.get_region_name_from_region_id(region_id)
        return region_name

    def get_points_from_region_id(self, region_id):
        return self.volume_data_sparse[region_id]

    # Given a brain region name, get an array of points that
    # make up said region
    def get_points_from_region_name(self, region_name):
        region_id = self.get_region_id_from_region_name(region_name)
        return self.get_points_from_region_id(region_id)

    def get_distance_to_inner_skull(self, pt):
        _, _, inner_skull_points = self.get_bem_surfaces()
        closest_point, distance = find_closest_point(inner_skull_points, pt)

        return closest_point, distance

    def get_closest_region_to_whitematter(self, pt, n_max=20):
        success = False

        region_id = self.get_region_id_from_point(pt)
        if region_id != 103 and region_id != 0:
            logging.warning(
                "Attempting to get closest region to whitematter from a non-whitematter region -> %s",
                region_id,
            )
            return success, None, None

        # Search around the point
        for i in range(n_max):
            for inc_id in range(1, 8):
                inc = (
                    np.array(list(bin(inc_id).split("b")[-1].rjust(3, "0"))).astype(int)
                    * i
                )  # 001, 010 ... 111

                pt1 = pt.copy() + inc
                pt2 = pt.copy() - inc

                region1 = self.get_region_id_from_point(pt1)
                if region1 != 0 and region1 != 103:
                    success = True
                    return success, pt1, region1

                region2 = self.get_region_id_from_point(pt2)
                if region2 != 0 and region2 != 103:
                    success = True
                    return success, pt2, region2

        logging.warning(
            "Get_closest_region_to_whitematter unable to find close region %s", n_max
        )
        return success, pt, region_id

    def find_region_centroid_from_name(self, region_name):
        region_id = self.get_region_id_from_region_name(region_name)
        return self.find_region_centroid_from_id(region_id)

    def find_region_centroid_from_id(self, region_id):
        return np.round(self.get_points_from_region_id(region_id).mean(axis=0))

    # @time_func_decorator
    def orthoview(
        self,
        plot_src_space: bool = False,
        plot_bem_surfaces: bool = False,
        plot_distance_to_inner_skull: bool = False,
        plot_highlighted_region: int = None,
        **kwargs,
    ):
        src_volume = None
        if plot_src_space:
            src_volume = self.get_src_space_volume()

        bem_volume = None
        if plot_bem_surfaces:
            bem_volume = self.get_bem_volume()

        region_volume = None
        region_centroid = None
        if plot_highlighted_region is not None:
            assert isinstance(plot_highlighted_region, int)
            # region_pts = self.get_points_from_region_id(plot_highlighted_region)
            # region_volume = point_cloud_to_voxel(region_pts)
            region_centroid = self.find_region_centroid_from_id(plot_highlighted_region)

        pt_dist = None
        if plot_distance_to_inner_skull:
            if "pt" not in kwargs:
                raise ValueError(
                    "pt=np.ndarray should be set if plot_distance_to_inner_skull=True"
                )
            pt = kwargs["pt"]
            pt_dist = self.get_distance_to_inner_skull(pt)

        fig, axs = orthoview(
            self.cerebra_volume,
            self.affine,
            src_volume=src_volume,
            bem_volume=bem_volume,
            pt_dist=pt_dist,
            plot_highlighted_region=plot_highlighted_region,
            region_centroid=region_centroid,
            **kwargs,
        )

        if "pt" in kwargs and kwargs["pt"] is not None:
            reg_name = self.get_region_name_from_point(kwargs["pt"])
            reg_id = self.get_region_id_from_point(kwargs["pt"])
            fig.suptitle(f"{reg_name} id ({reg_id})")

        return fig, axs

    def plot_3d(self, alpha=1):
        return plot_volume_3d(self.cerebra_volume, density=6, alpha=alpha)

    def plot_region_3d(
        self,
        region_id,
    ):
        pts = self.get_points_from_region_id(region_id)
        plot_volume_3d(self.cerebra_volume, region_pts=pts, density=6)

    def plot_whitematter_3d(self, **kwargs):
        pts = self.get_points_from_region_id(103)
        fig, ax = plot_volume_3d(self.cerebra_volume, density=6, alpha=0.5)
        _, ax = plot_volume_3d(
            self.cerebra_volume, region_pts=pts, density=64, ax=ax, **kwargs
        )


if __name__ == "__main__":
    setup_logging()
    cerebra = CerebrA()

    print(cerebra.volume_data.min(), cerebra.volume_data.max())
