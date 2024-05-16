"""Main cerebra class
"""

from cerebra_atlas_python.data import SourceSpace


class CerebrA(SourceSpace):
    """Main cerebra class SA"""

    def __init__(self, **kwargs):
        SourceSpace.__init__(self, **kwargs)
