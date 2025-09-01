import matplotlib
from matplotlib.colors import ListedColormap
import numpy as np
from typing import Union, Tuple, List
import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt

ColorsInputType = Union[
    str,
    List[str],
    Tuple[float, float, float],
    List[Tuple[float, float, float]],
    List[List[Tuple[float, float, float]]],
    List[List[str]],
    np.ndarray,
]


def rgb_to_hex_str(color_rgb: np.ndarray) -> str:
    """Transforms (r,g,b) (0,1) array into hex color string

    Args:
        color_rgb (np.ndarray): input array

    Returns:
        str: transformed hex string
    """
    color_rgb_list = [int(c * 255) for c in color_rgb]
    return f"#{color_rgb_list[0]:02x}{color_rgb_list[1]:02x}{color_rgb_list[2]:02x}"


def hex_str_to_rgb(color_hex: str) -> np.ndarray:
    """Transforms a hex color string (e.g. '#RRGGBB') into (r,g,b) (0,1) array.

    Args:
        color_hex (str): Hex string, e.g. '#3498db'

    Returns:
        np.ndarray: (r, g, b), each in the range [0, 1]
    """
    # Remove the '#' if present
    color_hex = color_hex.lstrip("#")

    # Convert each pair of hex digits to an integer and scale to [0, 1]
    r = int(color_hex[0:2], 16) / 255.0
    g = int(color_hex[2:4], 16) / 255.0
    b = int(color_hex[4:6], 16) / 255.0

    return np.array([r, g, b])


def get_cmap_colors(cmap_name="gist_rainbow", n_classes=103):
    n_colors = int(n_classes) + 1
    cmap = matplotlib.colormaps[cmap_name]
    colors = cmap(np.linspace(0, 1, n_colors))
    white = np.array([1, 0.87, 0.87, 1])
    colors[-1] = white
    black = np.array([0, 0, 0, 1])
    colors[0] = black
    return colors[:, :3]


def get_cmap_colors_hex(**kwargs):
    colors = get_cmap_colors()
    return np.array([rgb_to_hex_str(c) for c in colors])


def get_cmap():
    newcmp = ListedColormap(get_cmap_colors())
    return newcmp


def normalize_colors_input(colors: ColorsInputType, src_space_n_points: int):
    """This function validates the colors user input for plotting.
    The function rises error if the colors input is invalid.

    Valid colors input:
        - str | rgb: Solid color for whole brain
        - 1D array: Should be of shape (len(src_space_labels))
        - 2D array: Should be of shape (len(src_space_labels), time)

    Args:
        colors (ColorsInputType): colors
        src_space_n_points (int): Number of points in the source space.
          Used for validation of colors array length


    Returns:
        List[Tuple[int,int,int]] | List[List[Tuple[int,int,int]]]: normalized
        colors. List of rgb colors, one for each src space pt


    Raises:
        ValueError: if colors value is invalid
    """
    # print(f"{type(colors)= }")
    if type(colors) == np.ndarray:
        colors = list(colors)

    if type(colors) == str:
        _colors = hex_str_to_rgb(colors)
        _colors = [_colors] * src_space_n_points
    elif type(colors) == tuple:
        _colors = [colors] * src_space_n_points
    elif type(colors) == list:
        if len(colors) != src_space_n_points:
            raise ValueError(
                f"colors array ({len(colors)}) should match the total number of points in the src space"
            )
        _colors = colors
    else:
        raise ValueError(
            "colors input is not valid. Should be str, list[str], list[list[str]], rgb tuple, list[rgb tuple], list[list[rgb tuple]]"
        )

    return np.array(_colors)


def get_scalar_colormap(vmin, vmax, cmap_name):
    cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    scalar_map = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap(cmap_name))
    return scalar_map


def apply_colormap(array, cmap_name, vmin_vmax=None, plot_cmap=True):
    if vmin_vmax is None:
        vmin = np.min(array)
        vmax = np.max(array)
    else:
        vmin, vmax = vmin_vmax
    scalar_map = get_scalar_colormap(vmin, vmax, cmap_name)

    if plot_cmap:

        image_data = [np.linspace(0, 1, 100)]
        image_data = np.repeat(image_data, 10, axis=0)
        plt.imshow(image_data, cmap=cmap_name)

        # Remove y axis
        plt.gca().yaxis.set_visible(False)
        # Set X axis between 0 and 100 to be between vmin and vmax
        plt.xticks(
            ticks=[0, 25, 50, 75, 99],
            labels=[
                f"{vmin:.2e}",
                f"{vmin + (vmax - vmin) * 0.25:.2e}",
                f"{(vmin + vmax) / 2:.2e}",
                f"{vmin + (vmax - vmin) * 0.75:.2e}",
                f"{vmax:.2e}",
            ],
        )

        # Plt show non blocking
        plt.show(block=False)

    colors = scalar_map.to_rgba(array)
    return colors[..., :3]
