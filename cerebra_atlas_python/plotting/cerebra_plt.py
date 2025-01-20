from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator


def get_2d_fig_ax(
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (6, 6),
    use_latex_figures: bool = True,
    add_grid: bool = False,
    x_lims: Optional[Tuple[int, int]] = None,
    y_lims: Optional[Tuple[int, int]] = None,
) -> Tuple[Optional[Figure], Axes]:
    """
        Creates a 2D figure and axes with optional LaTeX styling and grid.

        This function can take an existing matplotlib figure and axes objects or create new ones.
        It sets the limits of the axes and optionally applies LaTeX styling and adds a grid.

        Parameters:
        fig (Optional[plt.Figure]): An optional matplotlib figure object. Defaults to None.
        ax (Optional[plt.Axes]): An optional matplotlib axes object. Defaults to None.
        figsize (Tuple[int, int]): Size of the figure, defaults to (6, 6).
        use_latex_figures (bool): If True, applies LaTeX styling to the figure. Defaults to True.
        add_grid (bool): If True, adds a grid to the axes. Defaults to False.
    self.montage_name is not None and self.head_size is not None
        Returns:
        Tuple[plt.Figure, plt.Axes]: A tuple containing the matplotlib figure and axes objects.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    if x_lims is not None:
        ax.set_xlim(x_lims)
    if y_lims is not None:
        ax.set_ylim(y_lims)

    if add_grid:
        add_grid_to_ax(ax)

    return fig, ax


def remove_ax(ax: Axes, keep_names: Optional[List[str]] = None) -> Axes:
    """
    Hides all the elements of a given matplotlib axis.

    This function takes a matplotlib axes object and hides its x-axis, y-axis,
    and all four spines (top, right, bottom, left).

    Parameters:
    ax (Axes): A matplotlib axes object on which the elements are to be hidden.
    keep_names (Optional[List[str]]): Specifies which spines to keep visible.
        options are "top", "right", "bottom", and "left". Defaults to None.
    """
    all_names = ["top", "right", "bottom", "left"]
    if keep_names is None:
        keep_names = []
    else:
        # Assert all keep names are valid
        if not all([name in all_names for name in keep_names]):
            raise ValueError(
                f"Invalid keep_names: {keep_names}. Must be a subset of {all_names}"
            )
    ax.set_yticks([])
    ax.set_xticks([])

    if "top" not in keep_names and "bottom" not in keep_names:
        ax.set_xticklabels([])

    if "right" not in keep_names and "left" not in keep_names:
        ax.set_yticklabels([])
    for name in all_names:
        if name in keep_names:
            continue
        ax.spines[name].set_visible(False)
    return ax


def get_orthoview_axes(
    figsize: Tuple[int, int] = (7, 7),
    use_latex_figures: bool = True,
    add_grid: bool = False,
) -> Tuple[Figure, List[Axes]]:
    """
    Creates a 2x2 grid of matplotlib subplots for displaying orthogonal views.

    This function sets up a 2x2 grid of subplots using matplotlib, with three of these subplots
    configured using the get_2d_fig_ax function, and the fourth subplot hidden using remove_ax.

    Parameters:
    figsize (Tuple[int, int]): Size of the figure, defaults to (7, 7).
    use_latex_figures (bool): If True, applies LaTeX styling to the figure. Defaults to True.
    add_grid (bool): If True, adds a grid to the subplots. Defaults to False.

    Returns:
    Tuple[plt.Figure, np.ndarray]: A tuple containing the matplotlib figure and a 2x2 numpy array of axes objects.
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    # Configure the first three subplots
    _, axs[0, 0] = get_2d_fig_ax(
        None, axs[0, 0], use_latex_figures=use_latex_figures, add_grid=add_grid
    )
    _, axs[0, 1] = get_2d_fig_ax(
        None, axs[0, 1], use_latex_figures=use_latex_figures, add_grid=add_grid
    )
    _, axs[1, 0] = get_2d_fig_ax(
        None, axs[1, 0], use_latex_figures=use_latex_figures, add_grid=add_grid
    )
    # Hide the fourth subplot
    remove_ax(axs[1, 1])

    return fig, axs


def add_grid_to_ax(ax, lines=True, locations=None):
    """Add a grid to the current plot.
    Args:
        ax (Axis): axis object in which to draw the grid.
        lines (bool, optional): add lines to the grid. Defaults to True.
        locations (tuple, optional):
            (xminor, xmajor, yminor, ymajor). Defaults to None.
    """

    if lines:
        ax.grid(lines, alpha=0.5, which="minor", ls=":")
        ax.grid(lines, alpha=0.7, which="major")

    if locations is not None:
        assert len(locations) == 4, "Invalid entry for the locations of the markers"

        xmin, xmaj, ymin, ymaj = locations

        ax.xaxis.set_minor_locator(MultipleLocator(xmin))
        ax.xaxis.set_major_locator(MultipleLocator(xmaj))
        ax.yaxis.set_minor_locator(MultipleLocator(ymin))
        ax.yaxis.set_major_locator(MultipleLocator(ymaj))
