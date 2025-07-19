#!/usr/bin/env python3
"""
Debug test to check parameter passing through the pipeline.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_parameter_passing_debug():
    """Test parameter passing with debug output."""
    print("Testing parameter passing with debug output...")

    # Simulate data loader output
    data_loader_output = {
        "a_raw": None,  # Placeholder
        "b_raw": None,  # Placeholder
        "data_path": "/test/path",
        "step_size": 0.5,
        "pixel_size": 0.1625,
        "theta": 45 * 3.14159 / 180,
        "camera_offset": 100,
    }

    print(f"Data loader output: {data_loader_output}")

    # Simulate ROI detection output (skip ROI)
    roi_output = {
        "a_cropped": data_loader_output["a_raw"],  # Full image
        "b_cropped": data_loader_output["b_raw"],  # Full image
        "roi_coords": None,
        "method": "none",
        "process_single_channel": None,
    }

    # Pass through parameters from input data
    for key in ["step_size", "pixel_size", "theta", "camera_offset"]:
        if key in data_loader_output:
            roi_output[key] = data_loader_output[key]

    print(f"ROI output: {roi_output}")

    # Check if deskewing would find parameters
    if "step_size" in roi_output:
        print("✓ step_size found in ROI output")
        step_size = roi_output["step_size"]
        pixel_size = roi_output.get("pixel_size", "Not set")
        theta_rad = roi_output.get("theta", "Not set")

        print(f"  - step_size: {step_size}")
        print(f"  - pixel_size: {pixel_size}")
        print(f"  - theta: {theta_rad}")

        if theta_rad != "Not set":
            theta_deg = theta_rad * 180 / 3.14159
            print(f"  - theta_deg: {theta_deg:.1f}°")
    else:
        print("✗ step_size NOT found in ROI output")

    # Test layer metadata simulation
    layer_metadata = {
        "roi_coords": None,
        "method": "none",
        "original_shape": None,
        "step_size": 0.5,
        "pixel_size": 0.1625,
        "theta": 45 * 3.14159 / 180,
        "camera_offset": 100,
    }

    print(f"\nLayer metadata: {layer_metadata}")

    if "step_size" in layer_metadata:
        print("✓ step_size found in layer metadata")
    else:
        print("✗ step_size NOT found in layer metadata")

    print("\nParameter passing debug test completed!")
    return True


if __name__ == "__main__":
    test_parameter_passing_debug()
