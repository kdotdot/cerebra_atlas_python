"""Main cerebra class
"""

from .data.source_space import SourceSpace


class CerebrA(SourceSpace):
    """Main cerebra class SA"""

    def __init__(self, **kwargs):
        SourceSpace.__init__(self, **kwargs)
