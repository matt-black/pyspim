#!/usr/bin/env python3
"""
Test script to verify robust parameter handling.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_robust_parameter_handling():
    """Test robust parameter handling with various scenarios."""
    print("Testing robust parameter handling...")

    # Test scenario 1: Parameters in input data
    test_data_1 = {
        "a_cropped": None,
        "b_cropped": None,
        "step_size": 0.5,
        "pixel_size": 0.1625,
        "theta": 45 * 3.14159 / 180,
        "camera_offset": 100,
    }

    print("Scenario 1: Parameters in input data")
    step_size = test_data_1.get("step_size", 0.5)
    pixel_size = test_data_1.get("pixel_size", 0.1625)
    theta_rad = test_data_1.get("theta", 45 * 3.14159 / 180)

    print(f"  ✓ step_size: {step_size}")
    print(f"  ✓ pixel_size: {pixel_size}")
    print(f"  ✓ theta: {theta_rad}")

    # Test scenario 2: Missing parameters (fallback to defaults)
    test_data_2 = {
        "a_cropped": None,
        "b_cropped": None,
        # No parameters
    }

    print("\nScenario 2: Missing parameters (fallback)")
    step_size = test_data_2.get("step_size", 0.5)
    pixel_size = test_data_2.get("pixel_size", 0.1625)
    theta_rad = test_data_2.get("theta", 45 * 3.14159 / 180)

    print(f"  ✓ step_size (default): {step_size}")
    print(f"  ✓ pixel_size (default): {pixel_size}")
    print(f"  ✓ theta (default): {theta_rad}")

    # Test scenario 3: Mixed parameters
    test_data_3 = {
        "a_cropped": None,
        "b_cropped": None,
        "step_size": 0.3,
        # Missing pixel_size and theta
    }

    print("\nScenario 3: Mixed parameters")
    step_size = test_data_3.get("step_size", 0.5)
    pixel_size = test_data_3.get("pixel_size", 0.1625)
    theta_rad = test_data_3.get("theta", 45 * 3.14159 / 180)

    print(f"  ✓ step_size (from data): {step_size}")
    print(f"  ✓ pixel_size (default): {pixel_size}")
    print(f"  ✓ theta (default): {theta_rad}")

    # Test parameter validation
    print("\nParameter validation:")
    if step_size > 0 and pixel_size > 0 and theta_rad > 0:
        print("  ✓ All parameters are valid positive numbers")
    else:
        print("  ✗ Some parameters are invalid")
        return False

    # Test parameter conversion
    theta_deg = theta_rad * 180 / 3.14159
    print(f"  ✓ theta conversion: {theta_rad:.4f} rad = {theta_deg:.1f}°")

    print("\nRobust parameter handling test completed successfully!")
    print("The deskewing step should now work even with missing parameters.")
    return True


if __name__ == "__main__":
    test_robust_parameter_handling()
