"""
Basic tests that don't require CUDA compilation.
"""

import pytest
import numpy as np


def test_manifest():
    """Test that the manifest can be accessed without importing the full plugin."""
    # Import just the manifest
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from napari_pyspim import manifest
    assert manifest is not None
    assert manifest["name"] == "napari-pyspim"
    assert manifest["display_name"] == "diSPIM Processing Pipeline"


def test_utils():
    """Test utility functions."""
    # Import utils directly
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
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