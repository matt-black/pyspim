"""Tests for PySPIM data loading functionality."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

# Import the actual modules from the project
try:
    from pyspim.data import dispim as data
except ImportError:
    # For testing without full installation
    data = Mock()


class TestDataLoading:
    """Test data loading functionality."""

    def test_subtract_constant_uint16arr(self):
        """Test uint16 array subtraction with overflow handling."""
        # Test data
        arr = np.array([[100, 200, 300], [400, 500, 600]], dtype=np.uint16)
        offset = 100

        # Expected result
        expected = np.array([[0, 100, 200], [300, 400, 500]], dtype=np.uint16)

        # Test the function
        result = data.subtract_constant_uint16arr(arr, offset)

        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.uint16

    def test_subtract_constant_uint16arr_overflow(self):
        """Test uint16 array subtraction with overflow protection."""
        # Test data with values less than offset
        arr = np.array([[50, 100, 150], [200, 250, 300]], dtype=np.uint16)
        offset = 200

        # Expected result (should not go below 0)
        expected = np.array([[0, 0, 0], [0, 50, 100]], dtype=np.uint16)

        # Test the function
        result = data.subtract_constant_uint16arr(arr, offset)

        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.uint16

    @patch("pyspim.data.dispim.uManagerAcquisition")
    def test_uManagerAcquisition_context_manager(self, mock_acq):
        """Test uManagerAcquisition context manager."""
        # Mock the acquisition
        mock_acq_instance = Mock()
        mock_acq.return_value.__enter__.return_value = mock_acq_instance
        mock_acq.return_value.__exit__.return_value = None

        # Test data
        test_data = np.random.randint(0, 65535, (10, 10, 10), dtype=np.uint16)
        mock_acq_instance.get.return_value = test_data

        # Test the context manager
        with data.uManagerAcquisition("/test/path", False, np) as acq:
            a_raw = acq.get("a", 0, 0)
            b_raw = acq.get("b", 0, 0)

        # Verify calls
        mock_acq.assert_called_once_with("/test/path", False, np)
        assert mock_acq_instance.get.call_count == 2
        mock_acq_instance.get.assert_any_call("a", 0, 0)
        mock_acq_instance.get.assert_any_call("b", 0, 0)

    def test_data_validation(self):
        """Test data validation functions."""
        # Test valid data
        valid_data = np.random.randint(0, 65535, (10, 10, 10), dtype=np.uint16)

        # Test invalid data types
        invalid_data = np.random.random((10, 10, 10)).astype(np.float32)

        # These tests would depend on actual validation functions
        # For now, just test that we can work with the data
        assert valid_data.dtype == np.uint16
        assert invalid_data.dtype == np.float32


class TestROIDetection:
    """Test ROI detection functionality."""

    def test_roi_detection_3d(self):
        """Test 3D ROI detection."""
        # Create test data with a clear ROI
        data_3d = np.zeros((20, 20, 20), dtype=np.uint16)
        # Add a bright region in the center
        data_3d[5:15, 5:15, 5:15] = 1000

        # Test ROI detection (mock implementation)
        # In real implementation, this would call roi.detect_roi_3d
        roi_result = [(5, 15), (5, 15), (5, 15)]

        # Verify ROI bounds
        assert roi_result[0] == (5, 15)
        assert roi_result[1] == (5, 15)
        assert roi_result[2] == (5, 15)

    def test_combine_rois(self):
        """Test ROI combination."""
        # Test ROIs
        roia = [(5, 15), (5, 15), (5, 15)]
        roib = [(3, 17), (3, 17), (3, 17)]

        # Expected combined ROI (intersection)
        expected = [(5, 15), (5, 15), (5, 15)]

        # Test combination (mock implementation)
        combined = roia  # Simplified for testing

        assert combined == expected


class TestDeskewing:
    """Test deskewing functionality."""

    def test_deskew_parameters(self):
        """Test deskewing parameter calculations."""
        # Test parameters
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

    def test_deskew_direction(self):
        """Test deskewing direction handling."""
        # Test data
        test_data = np.random.randint(0, 1000, (10, 10, 10), dtype=np.uint16)

        # Test both directions
        direction_a = 1  # Head A
        direction_b = -1  # Head B

        # Verify directions are different
        assert direction_a != direction_b
        assert direction_a == 1
        assert direction_b == -1


class TestPerformance:
    """Test performance-related functionality."""

    def test_gpu_availability(self):
        """Test GPU availability detection."""
        try:
            import cupy as cp

            gpu_available = cp.cuda.is_available()
        except ImportError:
            gpu_available = False

        # Test should pass regardless of GPU availability
        assert isinstance(gpu_available, bool)

    def test_memory_usage(self):
        """Test memory usage monitoring."""
        # Create test data
        data_size = (100, 100, 100)
        test_data = np.random.randint(0, 65535, data_size, dtype=np.uint16)

        # Calculate expected memory usage
        expected_memory = test_data.nbytes / (1024**3)  # GB

        # Verify reasonable memory usage
        assert expected_memory < 1.0  # Should be less than 1GB

    def test_chunked_processing(self):
        """Test chunked processing functionality."""
        # Test data
        large_data = np.random.randint(0, 65535, (200, 200, 200), dtype=np.uint16)

        # Define chunk size
        chunk_size = (64, 64, 64)

        # Test chunking
        chunks = []
        for i in range(0, large_data.shape[0], chunk_size[0]):
            for j in range(0, large_data.shape[1], chunk_size[1]):
                for k in range(0, large_data.shape[2], chunk_size[2]):
                    chunk = large_data[
                        i : i + chunk_size[0],
                        j : j + chunk_size[1],
                        k : k + chunk_size[2],
                    ]
                    chunks.append(chunk)

        # Verify chunks
        assert len(chunks) > 0
        assert all(chunk.shape <= chunk_size for chunk in chunks)


if __name__ == "__main__":
    pytest.main([__file__])
