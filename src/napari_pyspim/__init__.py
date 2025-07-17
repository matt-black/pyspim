"""
napari-pyspim: A napari plugin for diSPIM processing pipeline.

This plugin provides a comprehensive interface for processing dual-view SPIM
data, including data loading, ROI detection, deskewing, registration, and
deconvolution.
"""

from napari_plugin_engine import napari_hook_implementation

# Plugin manifest
manifest = {
    "name": "napari-pyspim",
    "display_name": "diSPIM Processing Pipeline",
    "version": "0.0.1",
    "description": "Complete pipeline for processing dual-view SPIM data",
    "author": "Matthew Black",
    "license": "GPL-3.0-only",
    "repository": "https://github.com/matt-black/pyspim",
    "python_version": ">=3.8",
    "contributions": {
        "widgets": [
            {
                "command": "napari_pyspim.main_widget",
                "display_name": "diSPIM Pipeline",
                "icon": "icon.png",
            }
        ]
    }
}


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    """Provide the main widget for napari."""
    # Import here to avoid CUDA compilation at module level
    from ._main_widget import DispimPipelineWidget
    return [DispimPipelineWidget] 