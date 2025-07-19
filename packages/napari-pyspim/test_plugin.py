#!/usr/bin/env python3
"""
Test script for the napari-pyspim plugin.
This script creates some test data and launches napari with the plugin.
"""

import numpy as np
import napari
from napari_pyspim import DispimPipelineWidget

def create_test_data():
    """Create some test data for the plugin."""
    # Create test 3D data
    shape = (50, 100, 100)
    
    # Channel A: some structured data
    a_data = np.random.randint(0, 1000, shape, dtype=np.uint16)
    # Add some structure
    a_data[20:30, 40:60, 40:60] += 500
    
    # Channel B: similar but different structure
    b_data = np.random.randint(0, 1000, shape, dtype=np.uint16)
    # Add some structure
    b_data[25:35, 45:65, 45:65] += 500
    
    return a_data, b_data

def main():
    """Main test function."""
    print("Creating test data...")
    a_data, b_data = create_test_data()
    
    print("Launching napari...")
    viewer = napari.Viewer()
    
    # Add test data as layers with metadata
    viewer.add_image(
        a_data,
        name="A_test",
        metadata={
            "step_size": 0.5,
            "pixel_size": 0.1625,
            "theta": 45 * np.pi / 180,
        }
    )
    
    viewer.add_image(
        b_data,
        name="B_test",
        metadata={
            "step_size": 0.5,
            "pixel_size": 0.1625,
            "theta": 45 * np.pi / 180,
        }
    )
    
    # Add the plugin widget
    widget = DispimPipelineWidget(viewer)
    viewer.window.add_dock_widget(widget, name="diSPIM Pipeline")
    
    print("Plugin loaded! You can now:")
    print("1. Select layers in the dropdowns")
    print("2. Adjust step size, pixel size, and theta values")
    print("3. Test the processing pipeline")
    
    napari.run()

if __name__ == "__main__":
    main() 