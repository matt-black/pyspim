#!/usr/bin/env python3
"""Simple test script to verify deskewing methods work."""

import os
import math
import time
import numpy as np
import tifffile

from pyspim.data.dispim import uManagerAcquisitionOnePos
from pyspim.deskew import deskew_stage_scan


def test_methods():
    """Test both raw and ortho deskewing methods."""
    # Use a small test dataset
    acq_path = ("/scratch/gpfs/swwolf/git/pyspim/data/"
                "example_dispim_data/fruiting_body001")
    
    if not os.path.exists(acq_path):
        print(f"Test data not found at {acq_path}")
        print("Please update the path or create test data")
        return 1
    
    # Load a small volume
    with uManagerAcquisitionOnePos(acq_path, np) as acq:
        vol = acq.get('a', 0, 0)
        # Take a small subset for quick testing
        # vol = vol[:50, :100, :100]  # Small subset
    
    print(f"Testing with volume shape: {vol.shape}")
    
    # Test parameters
    pixel_size = 0.1625
    step_size = 0.5
    direction = 1
    theta = math.pi / 4
    preserve_dtype = True
    
    methods = ['raw', 'orthogonal']
    results = {}
    
    for method in methods:
        print(f"\nTesting {method} method...")
        try:
            time_start = time.perf_counter()
            result = deskew_stage_scan(
                vol, pixel_size, step_size, direction, theta,
                method=method, preserve_dtype=preserve_dtype
            )
            time_end = time.perf_counter()
            print(f"  Time taken: {time_end - time_start:.3f} seconds")
            results[method] = result
            print(f"  ✓ {method} method successful")
            print(f"  Output shape: {result.shape}")
            print(f"  Output dtype: {result.dtype}")
            print(f"  Output range: [{result.min()}, {result.max()}]")
        except Exception as e:
            print(f"  ✗ {method} method failed: {e}")
            results[method] = None
    
    # Save results to files
    print("\nSAVING RESULTS:")
    print("-" * 30)
    
    # Create output directory
    output_dir = "test_deskew_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original volume
    original_path = os.path.join(output_dir, "original_volume.tif")
    tifffile.imwrite(original_path, vol,
                     bigtiff=True,
                     imagej=True,
                     compression='zlib',
                     compressionargs={'level': 6},
                     metadata={'axes': 'ZYX'})
    print(f"✓ Original volume saved: {original_path}")
    print(f"  Shape: {vol.shape}, dtype: {vol.dtype}")
    print(f"  Range: [{vol.min()}, {vol.max()}]")
    
    # Save deskewed results
    for method, result in results.items():
        if result is not None:
            # Handle both NumPy and CuPy arrays
            if hasattr(result, 'get'):
                result_data = result.get()  # CuPy array
            else:
                result_data = result  # NumPy array
            
            # Ensure data is in uint16 format for Fiji compatibility
            if result_data.dtype != np.uint16:
                # Clip to valid range and convert
                result_data = np.clip(result_data, 0, 65535).astype(np.uint16)
            
            output_path = os.path.join(output_dir, f"deskewed_{method}.tif")
            tifffile.imwrite(output_path, result_data,
                           bigtiff=True,
                           imagej=True,
                           compression='zlib',
                           compressionargs={'level': 6},
                           metadata={'axes': 'ZYX'})
            print(f"✓ {method} result saved: {output_path}")
            print(f"  Shape: {result_data.shape}, dtype: {result_data.dtype}")
            print(f"  Range: [{result_data.min()}, {result_data.max()}]")
        else:
            print(f"✗ {method} result not saved (failed)")
    
    # Compare results if both succeeded
    if all(v is not None for v in results.values()):
        print("\nCOMPARISON:")
        print("-" * 30)
        
        # Check if outputs have same shape
        shapes = {k: v.shape for k, v in results.items()}
        if len(set(shapes.values())) == 1:
            print("✓ All methods produce same output shape")
        else:
            print("✗ Methods produce different output shapes:")
            for method, shape in shapes.items():
                print(f"  {method}: {shape}")
        
        # Check if outputs are similar (within tolerance)
        raw_result = results['raw']
        ortho_result = results['orthogonal']
        
        if raw_result.shape == ortho_result.shape:
            # Handle both NumPy and CuPy arrays
            if hasattr(raw_result, 'get'):
                raw_data = raw_result.get()  # CuPy array
            else:
                raw_data = raw_result  # NumPy array
                
            if hasattr(ortho_result, 'get'):
                ortho_data = ortho_result.get()  # CuPy array
            else:
                ortho_data = ortho_result  # NumPy array
            
            # Flatten arrays for correlation analysis
            raw_flat = raw_data.flatten()
            ortho_flat = ortho_data.flatten()
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(raw_flat, ortho_flat)[0, 1]
            
            # Calculate differences
            diff = np.abs(raw_data - ortho_data)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"✓ Correlation coefficient: {correlation:.6f}")
            print(f"✓ Max difference: {max_diff:.2f}")
            print(f"✓ Mean difference: {mean_diff:.4f}")
            
            if correlation > 0.99:
                print("✓ Results are highly correlated (>0.99)")
            elif correlation > 0.95:
                print("✓ Results are well correlated (>0.95)")
            elif correlation > 0.9:
                print("⚠ Results are moderately correlated (>0.9)")
            else:
                print("⚠ Results have low correlation (<0.9)")
                
            if max_diff < 1.0:  # Allow for small numerical differences
                print("✓ Results are very similar")
            else:
                print("⚠ Results differ significantly")
        else:
            print("✗ Cannot compare results due to different shapes")
    
    print(f"\nAll results saved to: {output_dir}/")
    return 0


if __name__ == "__main__":
    exit(test_methods()) 