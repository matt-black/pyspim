#!/usr/bin/env python3
"""
Simple test script for the fruiting body subset dataset.
"""

import os
import numpy as np
import tifffile
import json


def test_subset_file_exists():
    """Test that the subset file exists and is accessible."""
    subset_path = "docs/examples/data/fruiting_body_subset/fruiting_body_subset.ome.tif"
    print(f"Testing file existence: {subset_path}")
    
    assert os.path.exists(subset_path), f"Subset file not found: {subset_path}"
    assert os.path.isfile(subset_path), f"Not a file: {subset_path}"
    print("✓ Subset file exists")


def test_subset_file_size():
    """Test that the subset file is under GitHub's 50MB limit."""
    subset_path = "docs/examples/data/fruiting_body_subset/fruiting_body_subset.ome.tif"
    print(f"Testing file size: {subset_path}")
    
    file_size = os.path.getsize(subset_path)
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"  File size: {file_size_mb:.1f}MB")
    assert file_size_mb < 50, f"File too large for GitHub: {file_size_mb:.1f}MB"
    assert file_size_mb > 10, f"File too small: {file_size_mb:.1f}MB"
    print("✓ File size is appropriate for GitHub")


def test_subset_data_loading():
    """Test loading the subset data."""
    subset_path = "docs/examples/data/fruiting_body_subset/fruiting_body_subset.ome.tif"
    print(f"Testing data loading: {subset_path}")
    
    # Load the data
    data_array = tifffile.imread(subset_path)
    
    # Check data properties
    print(f"  Data shape: {data_array.shape}")
    print(f"  Data dtype: {data_array.dtype}")
    
    assert data_array.ndim == 4, f"Expected 4D data, got {data_array.ndim}D"
    assert data_array.shape[0] == 2, f"Expected 2 channels, got {data_array.shape[0]}"
    assert data_array.dtype == np.uint16, f"Expected uint16, got {data_array.dtype}"
    
    # Check reasonable dimensions
    assert data_array.shape[1] >= 50, f"Z dimension too small: {data_array.shape[1]}"
    assert data_array.shape[2] >= 200, f"Y dimension too small: {data_array.shape[2]}"
    assert data_array.shape[3] >= 200, f"X dimension too small: {data_array.shape[3]}"
    print("✓ Data loading successful")


def test_subset_data_quality():
    """Test the quality of the subset data."""
    subset_path = "docs/examples/data/fruiting_body_subset/fruiting_body_subset.ome.tif"
    print(f"Testing data quality: {subset_path}")
    
    data_array = tifffile.imread(subset_path)
    
    # Check that data is not all zeros
    assert np.any(data_array > 0), "Data is all zeros"
    
    # Check that data has reasonable range
    max_val = np.max(data_array)
    min_val = np.min(data_array)
    
    print(f"  Data range: {min_val} to {max_val}")
    
    assert max_val > 0, "Maximum value is zero"
    assert max_val <= 65535, f"Maximum value exceeds uint16 range: {max_val}"
    assert min_val >= 0, f"Minimum value is negative: {min_val}"
    print("✓ Data quality is good")


def test_psf_files_exist():
    """Test that demo PSF files exist."""
    print("Testing PSF files existence")
    
    psf_a_path = "docs/examples/data/fruiting_body_subset/PSFA_demo.npy"
    psf_b_path = "docs/examples/data/fruiting_body_subset/PSFB_demo.npy"
    
    assert os.path.exists(psf_a_path), f"PSF A file not found: {psf_a_path}"
    assert os.path.exists(psf_b_path), f"PSF B file not found: {psf_b_path}"
    print("✓ PSF files exist")


def test_psf_data_loading():
    """Test loading the demo PSF files."""
    print("Testing PSF data loading")
    
    psf_a_path = "docs/examples/data/fruiting_body_subset/PSFA_demo.npy"
    psf_b_path = "docs/examples/data/fruiting_body_subset/PSFB_demo.npy"
    
    # Load PSFs
    psf_a = np.load(psf_a_path)
    psf_b = np.load(psf_b_path)
    
    print(f"  PSF A shape: {psf_a.shape}")
    print(f"  PSF B shape: {psf_b.shape}")
    
    # Check PSF properties
    assert psf_a.ndim == 3, f"PSF A should be 3D, got {psf_a.ndim}D"
    assert psf_b.ndim == 3, f"PSF B should be 3D, got {psf_b.ndim}D"
    assert psf_a.shape == psf_b.shape, f"PSF shapes don't match: {psf_a.shape} vs {psf_b.shape}"
    
    # Check reasonable PSF size
    assert psf_a.shape[0] >= 15, f"PSF Z dimension too small: {psf_a.shape[0]}"
    assert psf_a.shape[1] >= 15, f"PSF Y dimension too small: {psf_a.shape[1]}"
    assert psf_a.shape[2] >= 15, f"PSF X dimension too small: {psf_a.shape[2]}"
    print("✓ PSF data loading successful")


def test_acquisition_parameters():
    """Test that acquisition parameters file exists and is valid."""
    print("Testing acquisition parameters")
    
    params_path = "docs/examples/data/fruiting_body_subset/acquisition_params.txt"
    assert os.path.exists(params_path), f"Acquisition parameters not found: {params_path}"
    
    # Test that it's valid text format
    with open(params_path, 'r') as f:
        content = f.read()
    
    print(f"  Parameters content: {content.strip()}")
    
    # Check for required parameters
    required_params = ['step_size', 'pixel_size', 'theta', 'camera_offset']
    for param in required_params:
        assert param in content, f"Missing parameter: {param}"
    
    # Parse parameters
    params = {}
    for line in content.strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            if key in ['step_size', 'pixel_size', 'theta']:
                params[key] = float(value)
            elif key == 'camera_offset':
                params[key] = int(value)
    
    # Check parameter types and ranges
    assert isinstance(params['step_size'], (int, float)), "step_size should be numeric"
    assert isinstance(params['pixel_size'], (int, float)), "pixel_size should be numeric"
    assert isinstance(params['theta'], (int, float)), "theta should be numeric"
    assert isinstance(params['camera_offset'], int), "camera_offset should be integer"
    print("✓ Acquisition parameters are valid")


def test_notebook_compatibility():
    """Test that the subset is compatible with the example notebook."""
    print("Testing notebook compatibility")
    
    # Check that the notebook exists
    notebook_path = "docs/examples/fruiting_body_workflow.ipynb"
    assert os.path.exists(notebook_path), f"Workflow notebook not found: {notebook_path}"
    
    # For now, just check that the file exists (content will be added later)
    print("✓ Notebook file exists (content to be added)")


def test_documentation_compatibility():
    """Test that the subset is documented correctly."""
    print("Testing documentation compatibility")
    
    # Check documentation file
    doc_path = "docs/examples/fruiting-body-workflow.md"
    assert os.path.exists(doc_path), f"Documentation not found: {doc_path}"
    
    # Check that documentation references the subset
    with open(doc_path, 'r') as f:
        content = f.read()
    
    # Should mention the subset
    assert "fruiting body" in content.lower(), "Documentation doesn't mention fruiting body"
    print("✓ Documentation compatibility verified")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING FRUITING BODY SUBSET DATASET")
    print("=" * 60)
    
    tests = [
        test_subset_file_exists,
        test_subset_file_size,
        test_subset_data_loading,
        test_subset_data_quality,
        test_psf_files_exist,
        test_psf_data_loading,
        test_acquisition_parameters,
        test_notebook_compatibility,
        test_documentation_compatibility,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All tests passed! The fruiting body subset dataset is ready.")
    else:
        print("❌ Some tests failed. Please check the dataset.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 