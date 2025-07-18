#!/usr/bin/env python3
"""
Test script to demonstrate the improved ROI detection functionality.

This script shows how the ROI detection widget now supports:
1. Manual ROI selection using napari's rectangle tool
2. Single channel processing (A or B only)
3. Optional ROI detection (skip ROI and use full image)
"""

import numpy as np
import napari
from napari_pyspim import DispimPipelineWidget


def create_test_data():
    """Create synthetic test data for demonstration."""
    # Create synthetic 3D data
    z, y, x = 50, 100, 100
    
    # Create channel A with a bright region in the center
    a_data = np.random.poisson(10, (z, y, x)).astype(np.uint16)
    center_z, center_y, center_x = z//2, y//2, x//2
    a_data[center_z-10:center_z+10, center_y-20:center_y+20, 
           center_x-20:center_x+20] += 100
    
    # Create channel B with a different bright region
    b_data = np.random.poisson(10, (z, y, x)).astype(np.uint16)
    b_data[center_z-5:center_z+15, center_y-15:center_y+25, 
           center_x-15:center_x+25] += 80
    
    return a_data, b_data


def main():
    """Main function to demonstrate the improvements."""
    print("Creating napari viewer...")
    viewer = napari.Viewer()
    
    # Create test data
    print("Creating synthetic test data...")
    a_data, b_data = create_test_data()
    
    # Add data to viewer
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
    
    # Add the plugin widget
    print("Adding diSPIM pipeline widget...")
    widget = DispimPipelineWidget(viewer)
    viewer.window.add_dock_widget(widget, name="diSPIM Pipeline")
    
    print("\n" + "="*60)
    print("IMPROVED ROI DETECTION FEATURES:")
    print("="*60)
    print("1. CHANNEL SELECTION:")
    print("   - Process both channels (A & B)")
    print("   - Process only channel A")
    print("   - Process only channel B")
    print()
    print("2. ROI METHODS:")
    print("   - Otsu thresholding (automated)")
    print("   - Triangle thresholding (automated)")
    print("   - Manual ROI selection (use napari's rectangle tool)")
    print()
    print("3. OPTIONAL ROI:")
    print("   - Skip ROI detection and use full image")
    print()
    print("4. MANUAL ROI INSTRUCTIONS:")
    print("   - Select 'manual' method")
    print("   - Use napari's rectangle selection tool to draw ROI")
    print("   - Click 'Apply Manual ROI' button")
    print()
    print("5. SINGLE CHANNEL PROCESSING:")
    print("   - Select 'Process only channel A' or 'Process only channel B'")
    print("   - Only the selected channel will be processed in subsequent steps")
    print("="*60)
    
    # Start napari
    print("\nStarting napari viewer...")
    napari.run()


if __name__ == "__main__":
    main() 