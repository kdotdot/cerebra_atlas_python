import matplotlib
from matplotlib.colors import ListedColormap
import numpy as np


def rgb_to_hex_str(color_rgb: np.ndarray) -> str:
    """Transforms (r,g,b) (0,1) array into hex color string

    Args:
        color_rgb (np.ndarray): input array

    Returns:
        str: transformed hex string
    """
    color_rgb_list = [int(c * 255) for c in color_rgb]
    return f"#{color_rgb_list[0]:02x}{color_rgb_list[1]:02x}{color_rgb_list[2]:02x}"


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
