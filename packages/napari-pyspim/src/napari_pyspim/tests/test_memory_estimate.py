"""Tests for the GPU memory estimation function in _deconvolution."""

import pytest

from napari_pyspim._deconvolution import estimate_chunk_gpu_memory_mb


class TestEstimateChunkGpuMemory:
    """Tests for estimate_chunk_gpu_memory_mb."""

    def test_basic_calculation_boundary_correction(self):
        """Test memory estimate with boundary correction enabled."""
        chunk_size = (64, 128, 128)
        overlap = (40, 40, 40)
        # PSF shape: ceil(max(2.277, 7.385) * 3) = ceil(22.155) = 23
        psf_shape = (23, 23, 23)
        bp_shape = (23, 23, 23)

        memory_mb = estimate_chunk_gpu_memory_mb(
            chunk_size=chunk_size,
            overlap=overlap,
            psf_shape_a=psf_shape,
            psf_shape_b=psf_shape,
            bp_shape_a=bp_shape,
            bp_shape_b=bp_shape,
            boundary_correction=True,
            safety_margin=1.0,  # No margin for predictable test
        )

        # Effective chunk: (64+80) x (128+80) x (128+80) = 144 x 208 x 208 = 6,193,152
        c_eff = 144 * 208 * 208
        p = 2 * (23 * 23 * 23)  # Two PSFs
        q = 2 * (23 * 23 * 23)  # Two backprojectors
        expected_bytes = 480 * c_eff + 8 * (p + q)
        expected_mb = expected_bytes / (1024 * 1024)

        assert abs(memory_mb - expected_mb) < 0.01

    def test_basic_calculation_no_boundary_correction(self):
        """Test memory estimate with boundary correction disabled."""
        chunk_size = (64, 128, 128)
        overlap = (40, 40, 40)
        psf_shape = (23, 23, 23)
        bp_shape = (23, 23, 23)

        memory_mb = estimate_chunk_gpu_memory_mb(
            chunk_size=chunk_size,
            overlap=overlap,
            psf_shape_a=psf_shape,
            psf_shape_b=psf_shape,
            bp_shape_a=bp_shape,
            bp_shape_b=bp_shape,
            boundary_correction=False,
            safety_margin=1.0,
        )

        c_eff = 144 * 208 * 208
        p = 2 * (23 * 23 * 23)
        q = 2 * (23 * 23 * 23)
        expected_bytes = 44 * c_eff + 8 * (p + q)
        expected_mb = expected_bytes / (1024 * 1024)

        assert abs(memory_mb - expected_mb) < 0.01

    def test_safety_margin(self):
        """Test that safety margin multiplies the peak memory."""
        chunk_size = (64, 128, 128)
        overlap = (0, 0, 0)
        psf_shape = (15, 15, 15)
        bp_shape = (15, 15, 15)

        memory_no_margin = estimate_chunk_gpu_memory_mb(
            chunk_size=chunk_size,
            overlap=overlap,
            psf_shape_a=psf_shape,
            psf_shape_b=psf_shape,
            bp_shape_a=bp_shape,
            bp_shape_b=bp_shape,
            boundary_correction=True,
            safety_margin=1.0,
        )

        memory_with_margin = estimate_chunk_gpu_memory_mb(
            chunk_size=chunk_size,
            overlap=overlap,
            psf_shape_a=psf_shape,
            psf_shape_b=psf_shape,
            bp_shape_a=bp_shape,
            bp_shape_b=bp_shape,
            boundary_correction=True,
            safety_margin=1.3,
        )

        assert abs(memory_with_margin / memory_no_margin - 1.3) < 0.001

    def test_zero_overlap(self):
        """Test with zero overlap (effective chunk = raw chunk)."""
        chunk_size = (128, 256, 256)
        overlap = (0, 0, 0)
        psf_shape = (15, 15, 15)
        bp_shape = (15, 15, 15)

        memory_mb = estimate_chunk_gpu_memory_mb(
            chunk_size=chunk_size,
            overlap=overlap,
            psf_shape_a=psf_shape,
            psf_shape_b=psf_shape,
            bp_shape_a=bp_shape,
            bp_shape_b=bp_shape,
            boundary_correction=False,
            safety_margin=1.0,
        )

        c_eff = 128 * 256 * 256
        p = 2 * (15 * 15 * 15)
        q = 2 * (15 * 15 * 15)
        expected_bytes = 44 * c_eff + 8 * (p + q)
        expected_mb = expected_bytes / (1024 * 1024)

        assert abs(memory_mb - expected_mb) < 0.01

    def test_large_chunk(self):
        """Test with a large chunk that should produce GB-scale memory."""
        chunk_size = (256, 512, 512)
        overlap = (80, 80, 80)
        psf_shape = (23, 23, 23)
        bp_shape = (23, 23, 23)

        memory_mb = estimate_chunk_gpu_memory_mb(
            chunk_size=chunk_size,
            overlap=overlap,
            psf_shape_a=psf_shape,
            psf_shape_b=psf_shape,
            bp_shape_a=bp_shape,
            bp_shape_b=bp_shape,
            boundary_correction=True,
            safety_margin=1.3,
        )

        # Should be well over 1 GB
        assert memory_mb > 1024

    def test_small_chunk(self):
        """Test with a small chunk."""
        chunk_size = (32, 64, 64)
        overlap = (10, 10, 10)
        psf_shape = (9, 9, 9)
        bp_shape = (9, 9, 9)

        memory_mb = estimate_chunk_gpu_memory_mb(
            chunk_size=chunk_size,
            overlap=overlap,
            psf_shape_a=psf_shape,
            psf_shape_b=psf_shape,
            bp_shape_a=bp_shape,
            bp_shape_b=bp_shape,
            boundary_correction=True,
            safety_margin=1.3,
        )

        # Should be in the hundreds of MB range
        assert 0 < memory_mb < 1024

    def test_different_psf_shapes(self):
        """Test with different PSF shapes for view A and B."""
        chunk_size = (64, 128, 128)
        overlap = (20, 20, 20)
        psf_a = (23, 23, 23)
        psf_b = (31, 31, 31)
        bp_a = (23, 23, 23)
        bp_b = (31, 31, 31)

        memory_mb = estimate_chunk_gpu_memory_mb(
            chunk_size=chunk_size,
            overlap=overlap,
            psf_shape_a=psf_a,
            psf_shape_b=psf_b,
            bp_shape_a=bp_a,
            bp_shape_b=bp_b,
            boundary_correction=True,
            safety_margin=1.0,
        )

        c_eff = 104 * 168 * 168
        p = 23**3 + 31**3
        q = 23**3 + 31**3
        expected_bytes = 480 * c_eff + 8 * (p + q)
        expected_mb = expected_bytes / (1024 * 1024)

        assert abs(memory_mb - expected_mb) < 0.01

    def test_positive_memory(self):
        """Memory estimate should always be positive."""
        memory_mb = estimate_chunk_gpu_memory_mb(
            chunk_size=(16, 32, 32),
            overlap=(0, 0, 0),
            psf_shape_a=(7, 7, 7),
            psf_shape_b=(7, 7, 7),
            bp_shape_a=(7, 7, 7),
            bp_shape_b=(7, 7, 7),
            boundary_correction=False,
            safety_margin=1.0,
        )
        assert memory_mb > 0

    def test_boundary_correction_uses_correct_coefficient(self):
        """Verify boundary correction uses 480x and no-correction uses 44x."""
        chunk_size = (64, 128, 128)
        overlap = (0, 0, 0)
        psf_shape = (9, 9, 9)
        bp_shape = (9, 9, 9)

        memory_bc = estimate_chunk_gpu_memory_mb(
            chunk_size=chunk_size,
            overlap=overlap,
            psf_shape_a=psf_shape,
            psf_shape_b=psf_shape,
            bp_shape_a=bp_shape,
            bp_shape_b=bp_shape,
            boundary_correction=True,
            safety_margin=1.0,
        )

        memory_no_bc = estimate_chunk_gpu_memory_mb(
            chunk_size=chunk_size,
            overlap=overlap,
            psf_shape_a=psf_shape,
            psf_shape_b=psf_shape,
            bp_shape_a=bp_shape,
            bp_shape_b=bp_shape,
            boundary_correction=False,
            safety_margin=1.0,
        )

        # The ratio should be approximately 480/44 ≈ 10.91
        # (dominated by the c_eff term when PSF is small)
        ratio = memory_bc / memory_no_bc
        assert 10 < ratio < 12
