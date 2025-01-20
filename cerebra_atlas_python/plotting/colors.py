import matplotlib
from matplotlib.colors import ListedColormap
import numpy as np
from typing import cast, Dict, Tuple, TypedDict, List


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


def normalize_colors_input(
    src_space_n_points: int, _colors: List[str] | List[Tuple] | str | Tuple | None
):
    # """This function takes in the src_space_labels array of points
    # and the colors input for plotting. The function rises error if the
    # colors input (provided by user as a parameter) is invalid. If colors
    # is None then the default value is returned (cortical colors).

    # Valid colors input:
    #     - None: Defaults to cortical colors
    #     - str | rgb: Solid color for whole brain
    #     - 1D array: Should be of shape (len(src_space_labels))
    #     - 2D array: Should be of shape (len(src_space_labels), time)

    # Args:
    #     src_space_labels np.ndarray: Contains an array of src space labels [int]
    #     colors (None | str | list[str]): _description_

    # Returns:
    #     _type_: _description_

    # Raises:
    #     ValueError: if colors value is invalid
    # """
    # print(f"{type(colors)= }")
    if type(_colors) == str:
        colors = hex_str_to_rgb(_colors)
        colors = [colors] * src_space_n_points
    elif type(_colors) == tuple:
        colors = [_colors] * src_space_n_points
    else:
        colors = None  # Default color
    # print(f"{colors= }")

    # elif type(colors) ==

    return colors
