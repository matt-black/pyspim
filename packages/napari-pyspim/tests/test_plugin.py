#!/usr/bin/env python3
"""
Tests for the napari-pyspim plugin.
"""

import pytest
import numpy as np


def test_plugin_import():
    """Test that the plugin can be imported."""
    from napari_pyspim import DispimPipelineWidget
    assert DispimPipelineWidget is not None


def test_widget_class_exists():
    """Test that the widget class has the expected structure."""
    from napari_pyspim import DispimPipelineWidget
    
    # Check that the class exists and has expected methods
    assert hasattr(DispimPipelineWidget, '__init__')
    assert hasattr(DispimPipelineWidget, 'setup_ui')
    assert hasattr(DispimPipelineWidget, '_connect_signals')


def test_plugin_metadata():
    """Test that plugin metadata is correct."""
    # Import the widget class
    from napari_pyspim import DispimPipelineWidget
    
    # Check that it's a proper class
    assert isinstance(DispimPipelineWidget, type)
    
    # Check that it has the expected attributes
    assert hasattr(DispimPipelineWidget, '__doc__')
    if DispimPipelineWidget.__doc__:
        assert "diSPIM" in DispimPipelineWidget.__doc__


def test_plugin_structure():
    """Test that the plugin has the expected structure."""
    import napari_pyspim
    
    # Check that the module has the expected exports
    assert hasattr(napari_pyspim, 'DispimPipelineWidget')
    
    # Check that the widget is properly exported
    from napari_pyspim import DispimPipelineWidget
    assert DispimPipelineWidget is not None


def test_plugin_manifest():
    """Test that the plugin manifest exists and is valid."""
    import os
    import yaml
    
    # Check that napari.yaml exists
    manifest_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 'src', 'napari_pyspim', 'napari.yaml'
    )
    assert os.path.exists(manifest_path)
    
    # Check that it's valid YAML
    with open(manifest_path, 'r') as f:
        manifest = yaml.safe_load(f)
    
    # Check required fields
    assert 'name' in manifest
    assert 'display_name' in manifest
    assert 'contributions' in manifest
    assert 'commands' in manifest['contributions']
    assert 'widgets' in manifest['contributions']
    
    # Check that the command references the correct widget
    command = manifest['contributions']['commands'][0]
    assert command['id'] == 'napari-pyspim.main_widget'
    assert command['python_name'] == 'napari_pyspim._main_widget:DispimPipelineWidget' 