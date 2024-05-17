"""Main cerebra class
"""

from .data import SourceSpaceData
from .plotting import Plotting


class CerebrA(SourceSpaceData, Plotting):
    """Main cerebra class SA"""

    def __init__(self, **kwargs):
        SourceSpaceData.__init__(self, **kwargs)
        Plotting.__init__(self, **kwargs)
