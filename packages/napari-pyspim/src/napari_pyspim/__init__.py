"""
diSPIM Processing Pipeline - A napari plugin for processing dual-view SPIM data.
"""

# Check if pyspim is available for local computation
try:
    import pyspim
    HAS_PYSPIM = True
except ImportError:
    HAS_PYSPIM = False

# Import the main widget for npe2 discovery
from ._main_widget import DispimPipelineWidget

__all__ = ["DispimPipelineWidget", "HAS_PYSPIM"]
