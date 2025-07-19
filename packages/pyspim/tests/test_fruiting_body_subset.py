"""Tests for the fruiting body subset dataset."""

import os
import numpy as np
import pytest
import tifffile
import json


class TestFruitingBodySubset:
    """Test the fruiting body subset dataset."""

    @pytest.fixture
    def subset_path(self):
        """Path to the fruiting body subset dataset."""
        return "docs/examples/data/fruiting_body_subset/fruiting_body_subset.ome.tif"

    def test_subset_file_exists(self, subset_path):
        """Test that the subset file exists and is accessible."""
        assert os.path.exists(subset_path), f"Subset file not found: {subset_path}"
        assert os.path.isfile(subset_path), f"Not a file: {subset_path}"

    def test_subset_file_size(self, subset_path):
        """Test that the subset file is under GitHub's 50MB limit."""
        file_size = os.path.getsize(subset_path)
        file_size_mb = file_size / (1024 * 1024)
        
        assert file_size_mb < 50, f"File too large for GitHub: {file_size_mb:.1f}MB"
        assert file_size_mb > 10, f"File too small: {file_size_mb:.1f}MB"

    def test_subset_data_loading(self, subset_path):
        """Test loading the subset data."""
        # Load the data
        data_array = tifffile.imread(subset_path)
        
        # Check data properties
        assert data_array.ndim == 4, f"Expected 4D data, got {data_array.ndim}D"
        assert data_array.shape[0] == 2, f"Expected 2 channels, got {data_array.shape[0]}"
        assert data_array.dtype == np.uint16, f"Expected uint16, got {data_array.dtype}"
        
        # Check reasonable dimensions
        assert data_array.shape[1] >= 50, f"Z dimension too small: {data_array.shape[1]}"
        assert data_array.shape[2] >= 200, f"Y dimension too small: {data_array.shape[2]}"
        assert data_array.shape[3] >= 200, f"X dimension too small: {data_array.shape[3]}"

    def test_subset_data_quality(self, subset_path):
        """Test the quality of the subset data."""
        data_array = tifffile.imread(subset_path)
        
        # Check that data is not all zeros
        assert np.any(data_array > 0), "Data is all zeros"
        
        # Check that data has reasonable range
        max_val = np.max(data_array)
        min_val = np.min(data_array)
        
        assert max_val > 0, "Maximum value is zero"
        assert max_val <= 65535, f"Maximum value exceeds uint16 range: {max_val}"
        assert min_val >= 0, f"Minimum value is negative: {min_val}"

    def test_subset_metadata(self, subset_path):
        """Test that the subset has proper metadata."""
        # Load metadata
        with tifffile.TiffFile(subset_path) as tif:
            metadata = tif.imagej_metadata
            
        # Check for required metadata
        assert metadata is not None, "No ImageJ metadata found"
        assert "channels" in metadata, "No channels metadata"
        assert "slices" in metadata, "No slices metadata"
        assert metadata["channels"] == 2, f"Expected 2 channels, got {metadata['channels']}"

    def test_psf_files_exist(self):
        """Test that demo PSF files exist."""
        psf_a_path = "docs/examples/data/fruiting_body_subset/demo_psf_a.npy"
        psf_b_path = "docs/examples/data/fruiting_body_subset/demo_psf_b.npy"
        
        assert os.path.exists(psf_a_path), f"PSF A file not found: {psf_a_path}"
        assert os.path.exists(psf_b_path), f"PSF B file not found: {psf_b_path}"

    def test_psf_data_loading(self):
        """Test loading the demo PSF files."""
        psf_a_path = "docs/examples/data/fruiting_body_subset/demo_psf_a.npy"
        psf_b_path = "docs/examples/data/fruiting_body_subset/demo_psf_b.npy"
        
        # Load PSFs
        psf_a = np.load(psf_a_path)
        psf_b = np.load(psf_b_path)
        
        # Check PSF properties
        assert psf_a.ndim == 3, f"PSF A should be 3D, got {psf_a.ndim}D"
        assert psf_b.ndim == 3, f"PSF B should be 3D, got {psf_b.ndim}D"
        assert psf_a.shape == psf_b.shape, f"PSF shapes don't match: {psf_a.shape} vs {psf_b.shape}"
        
        # Check reasonable PSF size
        assert psf_a.shape[0] >= 15, f"PSF Z dimension too small: {psf_a.shape[0]}"
        assert psf_a.shape[1] >= 15, f"PSF Y dimension too small: {psf_a.shape[1]}"
        assert psf_a.shape[2] >= 15, f"PSF X dimension too small: {psf_a.shape[2]}"

    def test_acquisition_parameters(self):
        """Test that acquisition parameters file exists and is valid JSON."""
        params_path = "docs/examples/data/fruiting_body_subset/acquisition_params.json"
        assert os.path.exists(params_path), f"Acquisition parameters not found: {params_path}"
        
        # Test that it's valid JSON
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        # Check for required parameters
        required_params = ['step_size', 'pixel_size', 'theta', 'camera_offset']
        for param in required_params:
            assert param in params, f"Missing parameter: {param}"
        
        # Check parameter types and ranges
        assert isinstance(params['step_size'], (int, float)), "step_size should be numeric"
        assert isinstance(params['pixel_size'], (int, float)), "pixel_size should be numeric"
        assert isinstance(params['theta'], (int, float)), "theta should be numeric"
        assert isinstance(params['camera_offset'], int), "camera_offset should be integer"

    def test_data_processing_workflow_simulation(self, subset_path):
        """Test basic data processing workflow simulation with the subset."""
        # Load data
        data_array = tifffile.imread(subset_path)
        
        # Extract channels
        a_raw = data_array[0]  # Channel A
        b_raw = data_array[1]  # Channel B
        
        # Simulate camera offset correction
        camera_offset = 100
        a_corrected = np.clip(a_raw.astype(np.int32) - camera_offset, 0, 65535).astype(np.uint16)
        b_corrected = np.clip(b_raw.astype(np.int32) - camera_offset, 0, 65535).astype(np.uint16)
        
        # Check correction worked
        assert np.max(a_corrected) <= np.max(a_raw)
        assert np.max(b_corrected) <= np.max(b_raw)
        assert np.min(a_corrected) >= 0
        assert np.min(b_corrected) >= 0

    def test_deskewing_parameters(self):
        """Test deskewing parameter calculations for the subset."""
        # Standard diSPIM parameters
        step_size = 0.5  # microns
        pixel_size = 0.1625  # microns
        theta = np.pi / 4  # radians
        
        # Calculate expected values
        step_pix = step_size / pixel_size
        step_size_lat = step_size / np.cos(theta)
        step_pix_lat = step_pix / np.cos(theta)
        
        # Verify calculations
        assert np.isclose(step_pix, 3.0769, rtol=1e-3)
        assert np.isclose(step_size_lat, 0.7071, rtol=1e-3)
        assert np.isclose(step_pix_lat, 4.3514, rtol=1e-3)

    def test_notebook_compatibility(self):
        """Test that the subset is compatible with the example notebook."""
        # Check that the notebook exists
        notebook_path = "docs/examples/fruiting_body_workflow.ipynb"
        assert os.path.exists(notebook_path), f"Workflow notebook not found: {notebook_path}"
        
        # Check that the notebook references the subset
        with open(notebook_path, 'r') as f:
            content = f.read()
        
        # Should reference the subset data
        assert "fruiting_body_subset" in content, "Notebook doesn't reference subset data"
        assert "demo_psf" in content, "Notebook doesn't reference demo PSFs"

    def test_documentation_compatibility(self):
        """Test that the subset is documented correctly."""
        # Check documentation file
        doc_path = "docs/examples/fruiting-body-workflow.md"
        assert os.path.exists(doc_path), f"Documentation not found: {doc_path}"
        
        # Check that documentation references the subset
        with open(doc_path, 'r') as f:
            content = f.read()
        
        # Should mention the subset
        assert "fruiting body subset" in content.lower(), "Documentation doesn't mention subset"
        assert "49mb" in content.lower() or "48.8mb" in content.lower(), "Documentation doesn't mention file size"

    def test_dataset_completeness(self):
        """Test that all required files for the dataset are present."""
        required_files = [
            "docs/examples/data/fruiting_body_subset/fruiting_body_subset.ome.tif",
            "docs/examples/data/fruiting_body_subset/demo_psf_a.npy",
            "docs/examples/data/fruiting_body_subset/demo_psf_b.npy",
            "docs/examples/data/fruiting_body_subset/acquisition_params.json",
            "docs/examples/fruiting_body_workflow.ipynb",
            "docs/examples/fruiting-body-workflow.md",
        ]
        
        for file_path in required_files:
            assert os.path.exists(file_path), f"Required file missing: {file_path}"

    def test_dataset_consistency(self, subset_path):
        """Test that the dataset is internally consistent."""
        # Load data and parameters
        data_array = tifffile.imread(subset_path)
        params_path = "docs/examples/data/fruiting_body_subset/acquisition_params.json"
        
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        # Check that data dimensions are reasonable for the parameters
        z_dim, y_dim, x_dim = data_array.shape[1:]
        
        # Data should be large enough to be useful
        assert z_dim >= 50, f"Z dimension too small for useful processing: {z_dim}"
        assert y_dim >= 200, f"Y dimension too small for useful processing: {y_dim}"
        assert x_dim >= 200, f"X dimension too small for useful processing: {x_dim}"
        
        # Data should not be too large for GitHub
        total_voxels = z_dim * y_dim * x_dim * 2  # 2 channels
        total_mb = total_voxels * 2 / (1024 * 1024)  # uint16 = 2 bytes
        assert total_mb < 50, f"Dataset too large for GitHub: {total_mb:.1f}MB"


if __name__ == "__main__":
    pytest.main([__file__]) 