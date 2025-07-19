"""
Basic tests for the napari-pyspim plugin.
"""

import numpy as np
import pytest
from napari import Viewer

from napari_pyspim import DispimPipelineWidget


def test_plugin_import():
    """Test that the plugin can be imported."""
    from napari_pyspim import manifest

    assert manifest is not None
    assert manifest["name"] == "napari-pyspim"


def test_widget_creation():
    """Test that the main widget can be created."""
    viewer = Viewer()
    widget = DispimPipelineWidget(viewer)
    assert widget is not None
    assert hasattr(widget, "tab_widget")
    assert widget.tab_widget.count() == 5  # 5 processing steps


def test_data_loader_widget():
    """Test that the data loader widget can be created."""
    from napari_pyspim._data_loader import DataLoaderWidget

    viewer = Viewer()
    widget = DataLoaderWidget(viewer)
    assert widget is not None
    assert hasattr(widget, "load_button")


def test_roi_detection_widget():
    """Test that the ROI detection widget can be created."""
    from napari_pyspim._roi_detection import RoiDetectionWidget

    viewer = Viewer()
    widget = RoiDetectionWidget(viewer)
    assert widget is not None
    assert hasattr(widget, "detect_button")


def test_deskewing_widget():
    """Test that the deskewing widget can be created."""
    from napari_pyspim._deskewing import DeskewingWidget

    viewer = Viewer()
    widget = DeskewingWidget(viewer)
    assert widget is not None
    assert hasattr(widget, "deskew_button")


def test_registration_widget():
    """Test that the registration widget can be created."""
    from napari_pyspim._registration import RegistrationWidget

    viewer = Viewer()
    widget = RegistrationWidget(viewer)
    assert widget is not None
    assert hasattr(widget, "register_button")


def test_deconvolution_widget():
    """Test that the deconvolution widget can be created."""
    from napari_pyspim._deconvolution import DeconvolutionWidget

    viewer = Viewer()
    widget = DeconvolutionWidget(viewer)
    assert widget is not None
    assert hasattr(widget, "deconvolve_button")


def test_utils():
    """Test utility functions."""
    from napari_pyspim._utils import format_memory_usage, format_shape_string

    # Test memory formatting
    test_array = np.zeros((100, 100, 50))
    memory_str = format_memory_usage(test_array)
    assert "MB" in memory_str or "KB" in memory_str

    # Test shape formatting
    shape_str = format_shape_string((100, 100, 50))
    assert "100×100×50" in shape_str
    assert "(Z×Y×X)" in shape_str


if __name__ == "__main__":
    pytest.main([__file__])
