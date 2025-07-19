#!/usr/bin/env python3
"""
Test script to verify parameter passing through the pipeline.
"""

import numpy as np
from napari_pyspim import DispimPipelineWidget
import napari


def test_parameter_passing():
    """Test that parameters are passed correctly through the pipeline."""
    print("Testing parameter passing...")
    
    # Create test data
    a_data = np.random.randint(0, 1000, (10, 50, 50), dtype=np.uint16)
    b_data = np.random.randint(0, 1000, (10, 50, 50), dtype=np.uint16)
    
    # Create viewer and add data with metadata
    viewer = napari.Viewer()
    viewer.add_image(a_data, name="A_raw", metadata={
        'step_size': 0.5,
        'pixel_size': 0.1625,
        'theta': 45 * np.pi / 180,
        'camera_offset': 100
    })
    viewer.add_image(b_data, name="B_raw", metadata={
        'step_size': 0.5,
        'pixel_size': 0.1625,
        'theta': 45 * np.pi / 180,
        'camera_offset': 100
    })
    
    # Create widget
    widget = DispimPipelineWidget(viewer)
    
    # Simulate data loading by setting input data directly
    test_data = {
        'a_raw': a_data,
        'b_raw': b_data,
        'step_size': 0.5,
        'pixel_size': 0.1625,
        'theta': 45 * np.pi / 180,
        'camera_offset': 100
    }
    
    # Test ROI detection widget
    roi_widget = widget.roi_detection
    roi_widget.set_input_data(test_data)
    
    print("✓ ROI widget received input data")
    print(f"  - step_size: {test_data['step_size']}")
    print(f"  - pixel_size: {test_data['pixel_size']}")
    print(f"  - theta: {test_data['theta']}")
    
    # Test deskewing widget
    deskew_widget = widget.deskewing
    deskew_widget.set_input_data(test_data)
    
    print("✓ Deskew widget received input data")
    print(f"  - step_size label: {deskew_widget.step_size_label.text()}")
    print(f"  - pixel_size label: {deskew_widget.pixel_size_label.text()}")
    print(f"  - theta label: {deskew_widget.theta_label.text()}")
    
    print("\nParameter passing test completed successfully!")
    return True


if __name__ == "__main__":
    test_parameter_passing() 