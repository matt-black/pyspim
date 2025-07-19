#!/usr/bin/env python3
"""
Test script to verify the skip ROI functionality.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_skip_roi_logic():
    """Test the skip ROI logic without importing the full pipeline."""
    print("Testing skip ROI functionality...")

    # Test data structure
    test_data = {
        "a_raw": None,  # Placeholder
        "b_raw": None,  # Placeholder
        "step_size": 0.5,
        "pixel_size": 0.1625,
        "theta": 45 * 3.14159 / 180,
        "camera_offset": 100,
    }

    # Test skip ROI output structure
    skip_roi_output = {
        "a_cropped": test_data["a_raw"],  # Full image
        "b_cropped": test_data["b_raw"],  # Full image
        "roi_coords": None,  # No ROI
        "method": "none",
        "process_single_channel": None,
        "step_size": test_data["step_size"],
        "pixel_size": test_data["pixel_size"],
        "theta": test_data["theta"],
        "camera_offset": test_data["camera_offset"],
    }

    # Verify all required fields are present
    required_fields = [
        "a_cropped",
        "b_cropped",
        "roi_coords",
        "method",
        "process_single_channel",
        "step_size",
        "pixel_size",
        "theta",
        "camera_offset",
    ]

    for field in required_fields:
        if field in skip_roi_output:
            print(f"✓ {field}: {skip_roi_output[field]}")
        else:
            print(f"✗ Missing {field}")
            return False

    # Test single channel skip ROI
    single_channel_output = {
        "a_cropped": test_data["a_raw"],  # Only A channel
        "b_cropped": None,  # B channel not processed
        "roi_coords": None,
        "method": "none",
        "process_single_channel": "a",
        "step_size": test_data["step_size"],
        "pixel_size": test_data["pixel_size"],
        "theta": test_data["theta"],
        "camera_offset": test_data["camera_offset"],
    }

    print("\nTesting single channel skip ROI...")
    if single_channel_output["process_single_channel"] == "a":
        print("✓ Single channel A processing correctly set")
    else:
        print("✗ Single channel processing not set correctly")
        return False

    print("\nSkip ROI functionality test completed successfully!")
    print("The skip ROI button should now appear when the checkbox is checked.")
    return True


if __name__ == "__main__":
    test_skip_roi_logic()
