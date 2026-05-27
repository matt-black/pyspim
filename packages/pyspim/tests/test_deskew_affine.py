"""Tests for affine deskewing functionality."""

from math import sqrt
import numpy as np
import pytest

try:
    import cupy as cp
    HAS_GPU = cp.cuda.is_available()
except ImportError:
    cp = None
    HAS_GPU = False

from pyspim.deskew import affine


def test_fwd_deskew_matrix():
    """Test that fwd_deskew_matrix is constructed correctly for both directions."""
    pixel_size = 0.1625
    step_size = 0.5
    
    # direction = 1
    T1 = affine.fwd_deskew_matrix(pixel_size, step_size, 1, 0, 0, 0)
    assert T1.shape == (4, 4)
    
    sq2 = sqrt(2)
    a = 1 / sqrt(2)
    r = step_size / pixel_size
    
    # Expected T1 when zed=0, row=0, col=0
    # u_c = -0.5, v_c = -0.5, w_c = -0.5
    expected_T1 = np.asarray([
        [ a, 0, 0, 0.5 * a],
        [ 0, 1, 0, 0.5],
        [ a, 0, r * sq2, 0.5 * a + 0.5 * r * sq2],
        [ 0, 0, 0, 1],
    ])
    np.testing.assert_allclose(T1, expected_T1, rtol=1e-6)
    
    # direction = -1
    T2 = affine.fwd_deskew_matrix(pixel_size, step_size, -1, 0, 0, 0)
    assert T2.shape == (4, 4)
    
    # Expected T2 when zed=0, row=0, col=0
    # u_c = -0.5, v_c = -0.5, w_c = -0.5
    expected_T2 = np.asarray([
        [-a, 0, 0, -0.5 * a],
        [ 0, 1, 0, 0.5],
        [ a, 0, -r * sq2, 0.5 * a - 0.5 * r * sq2],
        [ 0, 0, 0, 1],
    ])
    np.testing.assert_allclose(T2, expected_T2, rtol=1e-6)


def test_deskewing_transform_shapes():
    """Test that deskewing_transform returns the correct transform matrix and shapes."""
    z, r, c = 100, 100, 100
    pixel_size = 0.1625
    step_size = 0.5
    
    # Test direction = 1
    D1, shp1 = affine.deskewing_transform(z, r, c, pixel_size, step_size, 1)
    assert D1.shape == (4, 4)
    assert len(shp1) == 3
    
    # Test direction = -1
    D2, shp2 = affine.deskewing_transform(z, r, c, pixel_size, step_size, -1)
    assert D2.shape == (4, 4)
    assert len(shp2) == 3
    
    # Make sure output shapes are reasonable (positive values, close to each other)
    assert all(s > 0 for s in shp1)
    assert all(s > 0 for s in shp2)
    # The output bounding boxes should have the exact same dimensions
    assert shp1 == shp2


@pytest.mark.skipif(not HAS_GPU, reason="CuPy/CUDA GPU not available")
def test_deskew_alignment_synthetic():
    """Verify that a synthetic point in View A and View B deskews to the exact same location in real-space."""
    pixel_size = 0.1625
    step_size = 0.5
    
    # Define input volume sizes
    z_size, r_size, c_size = 128, 128, 128
    
    # Compute transforms and output shapes
    D1, out_shape_1 = affine.deskewing_transform(z_size, r_size, c_size, pixel_size, step_size, 1)
    D2, out_shape_2 = affine.deskewing_transform(z_size, r_size, c_size, pixel_size, step_size, -1)
    
    # Choose a common physical target voxel coordinate in the deskewed space
    # Target in (x_out, y_out, z_out) - should fit in the output bounds of both views
    x_tgt = int(out_shape_1[2] / 2)
    y_tgt = int(out_shape_1[1] / 2)
    z_tgt = int(out_shape_1[0] / 2)
    
    # Map target output coordinates back to input coordinates using D
    voxel_tgt = np.array([x_tgt, y_tgt, z_tgt, 1.0])
    
    # Map back for View A (direction = 1)
    in_coord_1 = D1 @ voxel_tgt
    x_in_1 = int(np.round(in_coord_1[0]))
    y_in_1 = int(np.round(in_coord_1[1]))
    z_in_1 = int(np.round(in_coord_1[2]))
    
    # Map back for View B (direction = -1)
    in_coord_2 = D2 @ voxel_tgt
    x_in_2 = int(np.round(in_coord_2[0]))
    y_in_2 = int(np.round(in_coord_2[1]))
    z_in_2 = int(np.round(in_coord_2[2]))
    
    # Check that mapped coordinates fall within input bounds
    assert 0 <= x_in_1 < c_size and 0 <= y_in_1 < r_size and 0 <= z_in_1 < z_size
    assert 0 <= x_in_2 < c_size and 0 <= y_in_2 < r_size and 0 <= z_in_2 < z_size
    
    # Create input images on GPU
    im_1 = cp.zeros((z_size, r_size, c_size), dtype=cp.uint16)
    im_1[z_in_1, y_in_1, x_in_1] = 1000
    
    im_2 = cp.zeros((z_size, r_size, c_size), dtype=cp.uint16)
    im_2[z_in_2, y_in_2, x_in_2] = 1000
    
    # Run deskewing
    block_size = (8, 8, 8)
    dsk_1 = affine.deskew_stage_scan(im_1, pixel_size, step_size, 1, "nearest", True, block_size)
    dsk_2 = affine.deskew_stage_scan(im_2, pixel_size, step_size, -1, "nearest", True, block_size)
    
    # Locate peaks in deskewed output stacks
    # deskew_stage_scan returns volumes with axis order swapped: (x_out, y_out, z_out)
    peak_1 = cp.unravel_index(cp.argmax(dsk_1), dsk_1.shape)
    peak_2 = cp.unravel_index(cp.argmax(dsk_2), dsk_2.shape)
    
    # Check that they both deskew to the exact target coordinate!
    # Allowed tolerance of 1 voxel due to nearest-neighbor interpolation discretization
    assert abs(int(peak_1[0]) - x_tgt) <= 1
    assert abs(int(peak_1[1]) - y_tgt) <= 1
    assert abs(int(peak_1[2]) - z_tgt) <= 1
    
    assert abs(int(peak_2[0]) - x_tgt) <= 1
    assert abs(int(peak_2[1]) - y_tgt) <= 1
    assert abs(int(peak_2[2]) - z_tgt) <= 1
    
    # Verify both views map to the exact same position relative to each other (i.e. overlap perfectly)
    np.testing.assert_allclose(np.array(peak_1), np.array(peak_2), atol=1)


if __name__ == "__main__":
    pytest.main([__file__])
