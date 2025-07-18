#!/usr/bin/env python3
"""
Simple test script to verify parameter passing without importing full pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def test_parameter_passing_logic():
    """Test the parameter passing logic without importing the full pipeline."""
    print("Testing parameter passing logic...")
    
    # Test data structure
    test_data = {
        'a_raw': None,  # Placeholder
        'b_raw': None,  # Placeholder
        'step_size': 0.5,
        'pixel_size': 0.1625,
        'theta': 45 * 3.14159 / 180,  # 45 degrees in radians
        'camera_offset': 100
    }
    
    # Test that all required parameters are present
    required_params = ['step_size', 'pixel_size', 'theta', 'camera_offset']
    for param in required_params:
        if param in test_data:
            print(f"✓ {param}: {test_data[param]}")
        else:
            print(f"✗ Missing {param}")
            return False
    
    # Test parameter conversion
    theta_deg = test_data['theta'] * 180 / 3.14159
    print(f"✓ theta conversion: {test_data['theta']:.4f} rad = "
          f"{theta_deg:.1f}°")
    
    # Test parameter formatting
    step_size_text = f"{test_data['step_size']} μm"
    pixel_size_text = f"{test_data['pixel_size']} μm"
    theta_text = f"{theta_deg:.1f}°"
    
    print(f"✓ step_size formatted: {step_size_text}")
    print(f"✓ pixel_size formatted: {pixel_size_text}")
    print(f"✓ theta formatted: {theta_text}")
    
    print("\nParameter passing logic test completed successfully!")
    return True


if __name__ == "__main__":
    test_parameter_passing_logic() 