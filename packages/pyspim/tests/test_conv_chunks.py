"""Tests for calculate_conv_chunks in pyspim.decon._util.

This module tests the chunkwise convolution scheduling logic used for
deconvolving large 3D volumes.  The function is pure-Python (no GPU
required) and returns slice windows / padding metadata that describe
how each chunk of a volume should be read, processed, and assembled
back into the full output.

Key invariants verified:
  1. Every voxel in the original volume belongs to exactly one
     chunk's ``data_window`` (full coverage, no overlap).
  2. ``read_window`` always contains ``data_window`` and is clamped
     to the data bounds.
  3. ``out_window`` correctly indexes into the processed-chunk
     output (same shape as ``read_window``) to extract the
     ``data_window`` region.
  4. Reconstructing the original volume from chunk-wise reads/
     writes produces a perfect copy.

NOTE: CuPy is mocked at import time so these tests run without CUDA/GPU.
"""

# Mock cupy before importing the module under test (no GPU required).
import sys
from unittest.mock import MagicMock

sys.modules["cupy"] = MagicMock()

import pytest
import numpy as np

from pyspim.decon._util import (
    _pad_amount,
    _pad_splits,
    calculate_conv_chunks,
    ChunkProps,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pattern_volume(shape, seed=42):
    """Create a deterministic volume where voxel (z, r, c) has a unique value.

    Using a linear mapping makes any misalignment immediately detectable
    because every voxel value encodes its own coordinates.
    """
    z, r, c = shape
    z_idx, r_idx, c_idx = np.ogrid[:z, :r, :c]
    # Use distinct multipliers so that each coordinate contributes
    # uniquely to the value.
    vol = (z_idx * r * c * 100
           + r_idx * c * 100
           + c_idx).astype(np.int64)
    return vol


def _reconstruct_volume(shape, chunks, ndim_zyc=3):
    """Simulate chunk-wise processing to reconstruct the original volume.

    For each chunk:
      1. Read ``read_window`` from the input volume.
      2. Pad to ``chunk_shape`` (symmetric-style; content doesn't matter
         because we only keep ``out_window``).
      3. "Process" (identity).
      4. Extract ``out_window`` from the processed chunk.
      5. Write to ``data_window`` in the output volume.

    Returns the reconstructed volume.
    """
    # Build input volume with unique values per voxel.
    if ndim_zyc == 4:
        ch, z, r, c = shape
        vol = np.zeros(shape, dtype=np.int64)
        z_idx, r_idx, c_idx = np.ogrid[:z, :r, :c]
        for ci in range(ch):
            vol[ci] = (z_idx * r * c + r_idx * c + c_idx + ci * z * r * c)
    else:
        vol = _make_pattern_volume(shape)

    out = np.zeros_like(vol)

    for chunk_idx, props in chunks.items():
        data_win = props.data_window
        read_win = props.read_window
        out_win = props.out_window

        # Step 1: Read from input volume using read_window.
        chunk_data = vol[read_win].copy()

        # Step 4: Extract out_window from processed chunk (same shape
        # as read_window result).
        result = chunk_data[out_win]

        # Step 5: Write to data_window in output.
        out[data_win] = result

    return out, vol


# ===========================================================================
#  Helper function tests: _pad_amount, _pad_splits
# ===========================================================================


class TestPadAmount:
    """Tests for _pad_amount helper."""

    def test_exact_division(self):
        """No padding when dim divides chunk_dim evenly."""
        assert _pad_amount(64, 32) == 0

    def test_small_padding(self):
        assert _pad_amount(50, 32) == 14  # 32*2 - 50 = 14

    def test_equal_dim(self):
        assert _pad_amount(32, 32) == 0

    def test_chunk_larger_than_dim_error(self):
        """Should raise when dim < chunk_dim."""
        with pytest.raises(AssertionError):
            _pad_amount(10, 32)

    def test_large_padding(self):
        assert _pad_amount(100, 32) == 28  # 32*4 - 100 = 28


class TestPadSplits:
    """Tests for _pad_splits helper."""

    def test_even_split(self):
        left, right = _pad_splits(4)
        assert left == 2
        assert right == 2

    def test_odd_split(self):
        left, right = _pad_splits(5)
        assert left == 2
        assert right == 3

    def test_zero(self):
        left, right = _pad_splits(0)
        assert left == 0
        assert right == 0

    def test_sum_matches(self):
        for n in range(10):
            left, right = _pad_splits(n)
            assert left + right == n


# ===========================================================================
#  Basic function tests
# ===========================================================================


class TestCalculateConvChunksBasic:
    """Basic tests for calculate_conv_chunks."""

    def test_single_chunk(self):
        """Volume exactly equals chunk size produces one chunk."""
        chunks = calculate_conv_chunks(32, 32, 32, (32, 32, 32), (0, 0, 0), None)
        assert len(chunks) == 1
        assert 0 in chunks

    def test_single_chunk_windows(self):
        """Single chunk should cover the full volume with no padding."""
        chunks = calculate_conv_chunks(32, 32, 32, (32, 32, 32), (0, 0, 0), None)
        props = chunks[0]
        # data_window should cover [0:32] in all dims
        for s in props.data_window:
            assert s.start == 0
            assert s.stop == 32
        # read_window should be same as data_window (no overlap)
        assert props.read_window == props.data_window
        # padding should be all zeros
        for p in props.paddings:
            assert p == (0, 0)

    def test_multiple_chunks(self):
        """Volume larger than chunk in each dim produces multiple chunks."""
        chunks = calculate_conv_chunks(64, 64, 64, (32, 32, 32), (0, 0, 0), None)
        assert len(chunks) == 8  # 2*2*2

    def test_returns_dict_of_ChunkProps(self):
        chunks = calculate_conv_chunks(64, 64, 64, (32, 32, 32), (0, 0, 0), None)
        assert isinstance(chunks, dict)
        for k, v in chunks.items():
            assert isinstance(k, int)
            assert isinstance(v, ChunkProps)

    def test_chunk_count_formula(self):
        """Number of chunks = product of ceil(dim/chunk_dim) for each dim."""
        z, r, c = 50, 60, 70
        cs = (32, 32, 32)
        expected = 1
        for d, cd in zip((z, r, c), cs):
            n = -(-d // cd)  # ceil division
            expected *= n
        chunks = calculate_conv_chunks(z, r, c, cs, (0, 0, 0), None)
        assert len(chunks) == expected

    def test_ChunkProps_attributes(self):
        """ChunkProps should have all four attributes."""
        chunks = calculate_conv_chunks(64, 64, 64, (32, 32, 32), (5, 5, 5), None)
        props = chunks[0]
        assert hasattr(props, "data_window")
        assert hasattr(props, "read_window")
        assert hasattr(props, "paddings")
        assert hasattr(props, "out_window")


# ===========================================================================
#  Coverage tests: every voxel belongs to exactly one chunk
# ===========================================================================


class TestDataWindowCoverage:
    """Verify that data_windows cover the volume exactly once."""

    @pytest.mark.parametrize(
        "shape,chunk_shape,overlap",
        [
            ((32, 32, 32), (32, 32, 32), (0, 0, 0)),
            ((64, 64, 64), (32, 32, 32), (0, 0, 0)),
            ((50, 60, 70), (32, 32, 32), (0, 0, 0)),
            ((100, 80, 90), (32, 32, 32), (10, 10, 10)),
            ((48, 48, 48), (16, 24, 32), (5, 5, 5)),   # anisotropic chunks
        ],
    )
    def test_data_windows_cover_volume_3d(self, shape, chunk_shape, overlap):
        """Union of all data_window regions covers every voxel exactly once."""
        z, r, c = shape
        chunks = calculate_conv_chunks(z, r, c, chunk_shape, overlap, None)

        # Build a coverage grid: each voxel should be assigned exactly once.
        coverage = np.zeros(shape, dtype=np.int32)

        for chunk_idx, props in chunks.items():
            dw = props.data_window
            # Slice should have 3 elements for 3D
            assert len(dw) == 3
            # Check data_window is non-empty
            sizes = [dw[d].stop - dw[d].start for d in range(3)]
            assert all(s > 0 for s in sizes), (
                f"Chunk {chunk_idx} has empty data_window: {sizes}"
            )
            coverage[dw] += 1

        np.testing.assert_array_equal(coverage, 1,
            err_msg="Some voxels are covered 0 times or more than once")

    @pytest.mark.parametrize(
        "shape,chunk_shape,overlap",
        [
            ((3, 32, 32, 32), (32, 32, 32), (0, 0, 0)),
            ((2, 50, 60, 70), (32, 32, 32), (5, 5, 5)),
        ],
    )
    def test_data_windows_cover_volume_4d(self, shape, chunk_shape, overlap):
        """Coverage for 4D (CZYX) volumes with channel_slice."""
        ch, z, r, c = shape
        chunks = calculate_conv_chunks(
            z, r, c, chunk_shape, overlap, slice(None)
        )

        coverage = np.zeros(shape, dtype=np.int32)

        for chunk_idx, props in chunks.items():
            dw = props.data_window
            assert len(dw) == 4  # CZYX
            # Channel dimension (dw[0]) is slice(None) so its start/stop
            # are None — check only ZYX dims for non-emptiness.
            zyx_sizes = [dw[d].stop - dw[d].start for d in range(1, 4)]
            assert all(s > 0 for s in zyx_sizes), (
                f"Chunk {chunk_idx} has empty ZYX data_window: {zyx_sizes}"
            )
            coverage[dw] += 1

        np.testing.assert_array_equal(coverage, 1,
            err_msg="Some voxels are covered 0 times or more than once")

    @pytest.mark.parametrize(
        "shape,chunk_shape,overlap",
        [
            ((50, 60, 70), (32, 32, 32), (0, 0, 0)),
            ((50, 60, 70), (32, 32, 32), (10, 10, 10)),
        ],
    )
    def test_no_overlapping_data_windows(self, shape, chunk_shape, overlap):
        """No two chunks share a data_window voxel."""
        z, r, c = shape
        chunks = calculate_conv_chunks(z, r, c, chunk_shape, overlap, None)

        # Track which chunk owns each voxel.
        owner = np.full(shape, -1, dtype=np.int32)

        for chunk_idx, props in chunks.items():
            dw = props.data_window
            mask = np.zeros(shape, dtype=bool)
            mask[dw] = True
            # Check no voxel already owned.
            assert not np.any((owner[dw] != -1) & mask[dw]), (
                f"Chunk {chunk_idx} overlaps with another chunk"
            )
            owner[dw] = chunk_idx


# ===========================================================================
#  Read window & overlap tests
# ===========================================================================


class TestReadWindow:
    """Tests for read_window correctness."""

    def test_read_window_contains_data_window(self):
        """For every chunk, data_window is a subset of read_window."""
        chunks = calculate_conv_chunks(50, 60, 70, (32, 32, 32), (10, 10, 10), None)
        for idx, props in chunks.items():
            for dim in range(3):
                assert props.read_window[dim].start <= props.data_window[dim].start
                assert props.read_window[dim].stop >= props.data_window[dim].stop

    def test_read_window_respects_data_bounds(self):
        """read_window should never exceed [0, shape)."""
        z, r, c = 50, 60, 70
        chunks = calculate_conv_chunks(z, r, c, (32, 32, 32), (10, 10, 10), None)
        for idx, props in chunks.items():
            for dim, sz in enumerate((z, r, c)):
                assert 0 <= props.read_window[dim].start <= sz
                assert 0 <= props.read_window[dim].stop <= sz

    def test_overlap_expands_read_window_internal(self):
        """Internal chunks should expand data_window by overlap amount."""
        # Use a large volume so that internal chunks exist.
        z, r, c = 128, 128, 128
        cs = (32, 32, 32)
        ov = (8, 8, 8)
        chunks = calculate_conv_chunks(z, r, c, cs, ov, None)

        # Find an internal chunk (not touching any boundary).
        for idx, props in chunks.items():
            is_internal = True
            for dim, sz in enumerate((z, r, c)):
                if props.data_window[dim].start == 0:
                    is_internal = False
                if props.data_window[dim].stop == sz:
                    is_internal = False
            if is_internal:
                for dim in range(3):
                    dw = props.data_window[dim]
                    rw = props.read_window[dim]
                    assert rw.start == dw.start - ov[dim], (
                        f"Chunk {idx} dim {dim}: left read={rw.start} "
                        f"!= data_left-{ov[dim]}={dw.start - ov[dim]}"
                    )
                    assert rw.stop == dw.stop + ov[dim], (
                        f"Chunk {idx} dim {dim}: right read={rw.stop} "
                        f"!= data_right+{ov[dim]}={dw.stop + ov[dim]}"
                    )
                return  # found one, good
        pytest.fail("No internal chunk found in test volume")

    def test_boundary_chunks_have_clamped_read_window(self):
        """Chunks at volume boundaries should have read_window clamped."""
        z, r, c = 50, 50, 50
        cs = (32, 32, 32)
        ov = (20, 20, 20)  # large overlap
        chunks = calculate_conv_chunks(z, r, c, cs, ov, None)

        for idx, props in chunks.items():
            for dim, sz in enumerate((z, r, c)):
                # Clamp left to 0
                if props.data_window[dim].start == 0:
                    assert props.read_window[dim].start == 0
                # Clamp right to sz
                if props.data_window[dim].stop == sz:
                    assert props.read_window[dim].stop == sz

    def test_zero_overlap(self):
        """With zero overlap, read_window == data_window."""
        chunks = calculate_conv_chunks(50, 60, 70, (32, 32, 32), (0, 0, 0), None)
        for idx, props in chunks.items():
            assert props.read_window == props.data_window


# ===========================================================================
#  Padding tests
# ===========================================================================


class TestPadding:
    """Tests for paddings correctness."""

    def test_padding_non_negative(self):
        """All padding values must be >= 0."""
        chunks = calculate_conv_chunks(50, 60, 70, (32, 32, 32), (10, 10, 10), None)
        for idx, props in chunks.items():
            for dim, (l, r) in enumerate(props.paddings):
                assert l >= 0, f"Chunk {idx} dim {dim}: left_pad={l} < 0"
                assert r >= 0, f"Chunk {idx} dim {dim}: right_pad={r} < 0"

    def test_padding_matches_boundary_clamping(self):
        """padding = how much read_window is smaller than data_window + overlap."""
        z, r, c = 50, 60, 70
        ov = (10, 10, 10)
        chunks = calculate_conv_chunks(z, r, c, (32, 32, 32), ov, None)

        for idx, props in chunks.items():
            for dim in range(3):
                dw = props.data_window[dim]
                rw = props.read_window[dim]
                pad = props.paddings[dim]
                # left_pad = data_left - read_left
                expected_left = dw.start - rw.start
                # right_pad = read_right - data_right
                expected_right = rw.stop - dw.stop
                assert pad[0] == expected_left, (
                    f"Chunk {idx} dim {dim}: left_pad={pad[0]} "
                    f"!= {expected_left}"
                )
                assert pad[1] == expected_right, (
                    f"Chunk {idx} dim {dim}: right_pad={pad[1]} "
                    f"!= {expected_right}"
                )

    def test_internal_chunks_padding_equals_overlap(self):
        """Internal chunks have padding equal to overlap (read extends past data).

        The padding fields represent the gap between read_window edge and
        data_window edge.  For an internal chunk this gap is exactly the
        overlap in each direction — i.e. padding = (overlap, overlap).
        Boundary chunks have larger padding because the read window is
        clamped to the data bounds.
        """
        z, r, c = 128, 128, 128
        cs = (32, 32, 32)
        ov = (8, 8, 8)
        chunks = calculate_conv_chunks(z, r, c, cs, ov, None)

        for idx, props in chunks.items():
            is_internal = True
            for dim, sz in enumerate((z, r, c)):
                if props.data_window[dim].start == 0:
                    is_internal = False
                if props.data_window[dim].stop == sz:
                    is_internal = False
            if is_internal:
                for dim in range(3):
                    expected = (ov[dim], ov[dim])
                    assert props.paddings[dim] == expected, (
                        f"Internal chunk {idx} dim {dim}: padding "
                        f"{props.paddings[dim]} != expected {expected}"
                    )
                return  # found one
        pytest.fail("No internal chunk found")

    def test_read_window_size_equals_data_plus_padding(self):
        """read_window size = data_window size + left_pad + right_pad."""
        chunks = calculate_conv_chunks(50, 60, 70, (32, 32, 32), (10, 10, 10), None)

        for idx, props in chunks.items():
            for dim in range(3):
                dw_size = props.data_window[dim].stop - props.data_window[dim].start
                rw_size = props.read_window[dim].stop - props.read_window[dim].start
                pad = props.paddings[dim]
                expected_rw = dw_size + pad[0] + pad[1]
                assert rw_size == expected_rw, (
                    f"Chunk {idx} dim {dim}: read_size={rw_size} "
                    f"!= data_size({dw_size}) + padding({pad[0]}+{pad[1]})"
                )


# ===========================================================================
#  Output window tests
# ===========================================================================


class TestOutWindow:
    """Tests for out_window correctness."""

    def test_out_window_size_matches_data_window(self):
        """out_window should extract the same number of elements as data_window."""
        chunks = calculate_conv_chunks(50, 60, 70, (32, 32, 32), (10, 10, 10), None)

        for idx, props in chunks.items():
            for dim in range(3):
                dw_size = props.data_window[dim].stop - props.data_window[dim].start
                ow_size = props.out_window[dim].stop - props.out_window[dim].start
                assert dw_size == ow_size, (
                    f"Chunk {idx} dim {dim}: data_size={dw_size} "
                    f"!= out_size={ow_size}"
                )

    def test_out_window_corresponds_to_padding(self):
        """left_out should equal left_pad; right_out = left_pad + data_size."""
        chunks = calculate_conv_chunks(50, 60, 70, (32, 32, 32), (10, 10, 10), None)

        for idx, props in chunks.items():
            for dim in range(3):
                pad = props.paddings[dim]
                ow = props.out_window[dim]
                dw_size = props.data_window[dim].stop - props.data_window[dim].start

                assert ow.start == pad[0], (
                    f"Chunk {idx} dim {dim}: left_out={ow.start} "
                    f"!= left_pad={pad[0]}"
                )
                assert ow.stop == pad[0] + dw_size, (
                    f"Chunk {idx} dim {dim}: right_out={ow.stop} "
                    f"!= left_pad({pad[0]}) + data_size({dw_size})"
                )

    def test_out_window_within_read_bounds(self):
        """out_window indices must be within [0, read_window_size)."""
        chunks = calculate_conv_chunks(50, 60, 70, (32, 32, 32), (10, 10, 10), None)

        for idx, props in chunks.items():
            for dim in range(3):
                rw_size = props.read_window[dim].stop - props.read_window[dim].start
                ow = props.out_window[dim]
                assert 0 <= ow.start <= rw_size, (
                    f"Chunk {idx} dim {dim}: out_start={ow.start} "
                    f"outside read_range [0, {rw_size})"
                )
                assert 0 <= ow.stop <= rw_size, (
                    f"Chunk {idx} dim {dim}: out_stop={ow.stop} "
                    f"outside read_range [0, {rw_size})"
                )


# ===========================================================================
#  Channel dimension (CZYX) tests
# ===========================================================================


class TestChannelDimension:
    """Tests for channel_slice handling in 4D volumes."""

    def test_channel_slice_inserted(self):
        """When channel_slice is not None, all windows have 4 elements."""
        chunks = calculate_conv_chunks(
            32, 32, 32, (32, 32, 32), (0, 0, 0), slice(None)
        )
        for idx, props in chunks.items():
            assert len(props.data_window) == 4
            assert len(props.read_window) == 4
            assert len(props.out_window) == 4
            # First element should be the channel_slice
            assert props.data_window[0] == slice(None)
            assert props.read_window[0] == slice(None)
            assert props.out_window[0] == slice(None)

    def test_channel_padding_is_zero(self):
        """Padding for channel dim should be (0, 0)."""
        chunks = calculate_conv_chunks(
            32, 32, 32, (32, 32, 32), (5, 5, 5), slice(None)
        )
        for idx, props in chunks.items():
            # First padding tuple is for channel dim
            assert props.paddings[0] == (0, 0)

    def test_3d_no_channel_dim(self):
        """When channel_slice is None, windows have 3 elements."""
        chunks = calculate_conv_chunks(
            32, 32, 32, (32, 32, 32), (0, 0, 0), None
        )
        for idx, props in chunks.items():
            assert len(props.data_window) == 3
            assert len(props.read_window) == 3
            assert len(props.out_window) == 3
            # 3 paddings (ZYX), not 4
            assert len(props.paddings) == 3


# ===========================================================================
#  Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge-case and boundary tests."""

    def test_volume_exactly_divisible(self):
        """No padding needed when volume dims divide chunk dims evenly."""
        chunks = calculate_conv_chunks(64, 64, 64, (32, 32, 32), (0, 0, 0), None)
        # All chunks should be exactly 32x32x32 in data_window
        for idx, props in chunks.items():
            for dim in range(3):
                sz = props.data_window[dim].stop - props.data_window[dim].start
                assert sz == 32

    def test_anisotropic_chunks(self):
        """Non-cubic chunk shapes."""
        chunks = calculate_conv_chunks(64, 48, 96, (32, 24, 48), (5, 5, 5), None)
        assert len(chunks) > 0
        # Verify coverage.
        coverage = np.zeros((64, 48, 96), dtype=np.int32)
        for props in chunks.values():
            coverage[props.data_window] += 1
        np.testing.assert_array_equal(coverage, 1)

    def test_anisotropic_overlap(self):
        """Different overlap per dimension."""
        chunks = calculate_conv_chunks(
            64, 64, 64, (32, 32, 32), (10, 5, 0), None
        )
        assert len(chunks) > 0
        # Read windows should still contain data windows.
        for props in chunks.values():
            for dim in range(3):
                assert props.read_window[dim].start <= props.data_window[dim].start
                assert props.read_window[dim].stop >= props.data_window[dim].stop

    def test_volume_smaller_than_chunk_raises(self):
        """Volume smaller than chunk_dim raises AssertionError (documented constraint)."""
        with pytest.raises(AssertionError):
            calculate_conv_chunks(16, 16, 16, (32, 32, 32), (10, 10, 10), None)

    def test_small_volume_equal_to_chunk(self):
        """Volume equal to chunk size works with any overlap."""
        chunks = calculate_conv_chunks(32, 32, 32, (32, 32, 32), (10, 10, 10), None)
        assert len(chunks) == 1
        props = chunks[0]
        # data_window covers entire volume
        for dim in range(3):
            assert props.data_window[dim].start == 0
            assert props.data_window[dim].stop == 32

    def test_no_empty_chunks(self):
        """All chunks must have non-empty data_window."""
        for shape, cs, ov in [
            ((50, 60, 70), (32, 32, 32), (10, 10, 10)),
            ((33, 49, 61), (32, 32, 32), (0, 0, 0)),
            ((100, 80, 90), (32, 32, 32), (15, 10, 5)),
        ]:
            chunks = calculate_conv_chunks(*shape, cs, ov, None)
            for idx, props in chunks.items():
                sizes = [
                    props.data_window[d].stop - props.data_window[d].start
                    for d in range(3)
                ]
                assert all(s > 0 for s in sizes), (
                    f"Chunk {idx} has empty data_window: sizes={sizes}"
                )


# ===========================================================================
#  Alignment reconstruction tests
# ===========================================================================


class TestReconstruction:
    """Critical tests: reconstructing the volume from chunks must be perfect.

    These tests simulate chunk-wise processing with identity operation and
    verify that the assembled output matches the original input exactly.
    """

    @pytest.mark.parametrize(
        "shape,chunk_shape,overlap",
        [
            # --- Evenly divisible by chunk_shape ---
            ((32, 32, 32), (32, 32, 32), (0, 0, 0)),
            ((64, 64, 64), (32, 32, 32), (0, 0, 0)),
            ((64, 64, 64), (32, 32, 32), (10, 10, 10)),
            ((96, 96, 96), (32, 32, 32), (0, 0, 0)),
            ((96, 96, 96), (32, 32, 32), (16, 16, 16)),
            # --- Not evenly divisible ---
            ((50, 50, 50), (32, 32, 32), (0, 0, 0)),
            ((50, 50, 50), (32, 32, 32), (10, 10, 10)),
            ((33, 49, 61), (32, 32, 32), (0, 0, 0)),
            ((33, 49, 61), (32, 32, 32), (5, 5, 5)),
            ((100, 80, 90), (32, 32, 32), (0, 0, 0)),
            ((100, 80, 90), (32, 32, 32), (15, 10, 5)),
            # --- Anisotropic chunks ---
            ((64, 48, 96), (32, 24, 48), (0, 0, 0)),
            ((64, 48, 96), (32, 24, 48), (8, 4, 16)),
            ((50, 60, 70), (32, 32, 32), (10, 5, 0)),
            # --- Large overlap (relative to chunk) ---
            ((64, 64, 64), (32, 32, 32), (16, 16, 16)),
            ((50, 50, 50), (32, 32, 32), (20, 20, 20)),
        ],
    )
    def test_reconstruct_volume_3d(
        self, shape, chunk_shape, overlap
    ):
        """Reconstruct 3D volume from chunks must match original."""
        z, r, c = shape
        chunks = calculate_conv_chunks(z, r, c, chunk_shape, overlap, None)
        reconstructed, original = _reconstruct_volume(shape, chunks, ndim_zyc=3)
        np.testing.assert_array_equal(
            reconstructed, original,
            err_msg=(
                f"Reconstruction failed for shape={shape}, "
                f"chunk_shape={chunk_shape}, overlap={overlap}"
            ),
        )

    @pytest.mark.parametrize(
        "shape,chunk_shape,overlap",
        [
            # Evenly divisible
            ((2, 32, 32, 32), (32, 32, 32), (0, 0, 0)),
            ((2, 64, 64, 64), (32, 32, 32), (10, 10, 10)),
            # Not evenly divisible
            ((3, 50, 50, 50), (32, 32, 32), (0, 0, 0)),
            ((3, 50, 50, 50), (32, 32, 32), (5, 5, 5)),
            ((1, 33, 49, 61), (32, 32, 32), (0, 0, 0)),
        ],
    )
    def test_reconstruct_volume_4d(
        self, shape, chunk_shape, overlap
    ):
        """Reconstruct 4D (CZYX) volume from chunks must match original."""
        ch, z, r, c = shape
        chunks = calculate_conv_chunks(
            z, r, c, chunk_shape, overlap, slice(None)
        )
        reconstructed, original = _reconstruct_volume(shape, chunks, ndim_zyc=4)
        np.testing.assert_array_equal(
            reconstructed, original,
            err_msg=(
                f"Reconstruction failed for shape={shape}, "
                f"chunk_shape={chunk_shape}, overlap={overlap}"
            ),
        )

    def test_reconstruct_with_ramp_pattern(self):
        """Reconstruct with a ramp pattern to catch subtle misalignments."""
        shape = (50, 60, 70)
        chunk_shape = (32, 32, 32)
        overlap = (10, 8, 5)
        z, r, c = shape
        chunks = calculate_conv_chunks(z, r, c, chunk_shape, overlap, None)

        # Create a ramp: value = z * R * C + r * C + c
        z_idx, r_idx, c_idx = np.ogrid[:z, :r, :c]
        vol = (z_idx * r * c + r_idx * c + c_idx).astype(np.int64)
        out = np.zeros_like(vol)

        for props in chunks.values():
            chunk_data = vol[props.read_window].copy()
            result = chunk_data[props.out_window]
            out[props.data_window] = result

        np.testing.assert_array_equal(out, vol)

    def test_reconstruct_large_overlap(self):
        """Reconstruct with overlap close to chunk size."""
        shape = (64, 64, 64)
        chunk_shape = (32, 32, 32)
        overlap = (24, 24, 24)  # large overlap
        z, r, c = shape
        chunks = calculate_conv_chunks(z, r, c, chunk_shape, overlap, None)
        reconstructed, original = _reconstruct_volume(shape, chunks, ndim_zyc=3)
        np.testing.assert_array_equal(reconstructed, original)

    def test_reconstruct_volume_equal_to_chunk_large_overlap(self):
        """Volume equal to chunk size with large overlap reconstructs correctly."""
        shape = (32, 32, 32)
        chunk_shape = (32, 32, 32)
        overlap = (10, 10, 10)
        z, r, c = shape
        chunks = calculate_conv_chunks(z, r, c, chunk_shape, overlap, None)
        reconstructed, original = _reconstruct_volume(shape, chunks, ndim_zyc=3)
        np.testing.assert_array_equal(reconstructed, original)

    @pytest.mark.parametrize(
        "shape,chunk_shape,overlap",
        [
            ((50, 60, 70), (32, 32, 32), (0, 0, 0)),
            ((50, 60, 70), (32, 32, 32), (10, 10, 10)),
            ((100, 80, 90), (32, 32, 32), (15, 10, 5)),
            ((33, 49, 61), (32, 32, 32), (5, 5, 5)),
            ((64, 48, 96), (32, 24, 48), (8, 4, 16)),
        ],
    )
    def test_reconstruct_various_shapes(
        self, shape, chunk_shape, overlap
    ):
        """Grid of shapes, chunk shapes, and overlaps."""
        z, r, c = shape
        chunks = calculate_conv_chunks(z, r, c, chunk_shape, overlap, None)
        reconstructed, original = _reconstruct_volume(shape, chunks, ndim_zyc=3)
        np.testing.assert_array_equal(
            reconstructed, original,
            err_msg=(
                f"Reconstruction failed for shape={shape}, "
                f"chunk_shape={chunk_shape}, overlap={overlap}"
            ),
        )

    @pytest.mark.parametrize(
        "overlap",
        [
            (0, 0, 0),
            (1, 1, 1),
            (5, 5, 5),
            (10, 10, 10),
            (15, 15, 15),
            (20, 20, 20),
            (5, 10, 15),  # anisotropic
        ],
    )
    def test_reconstruct_overlap_grid(self, overlap):
        """Test reconstruction across a grid of overlap values.

        Uses a non-evenly-divisible volume shape to exercise padding logic.
        """
        shape = (50, 60, 70)
        chunk_shape = (32, 32, 32)
        z, r, c = shape
        chunks = calculate_conv_chunks(z, r, c, chunk_shape, overlap, None)
        reconstructed, original = _reconstruct_volume(shape, chunks, ndim_zyc=3)
        np.testing.assert_array_equal(
            reconstructed, original,
            err_msg=f"Reconstruction failed for overlap={overlap}",
        )

    @pytest.mark.parametrize(
        "shape,chunk_shape",
        [
            # Evenly divisible
            ((64, 64, 64), (32, 32, 32)),
            ((96, 96, 96), (32, 32, 32)),
            ((64, 48, 96), (32, 24, 48)),
            # Not evenly divisible (+1)
            ((33, 33, 33), (32, 32, 32)),
            ((50, 50, 50), (32, 32, 32)),
            # Not evenly divisible (arbitrary)
            ((100, 80, 90), (32, 32, 32)),
            ((73, 85, 97), (32, 32, 32)),
        ],
    )
    def test_reconstruct_divisibility_grid(self, shape, chunk_shape):
        """Test reconstruction for various divisibility combinations.

        Tests both zero and non-zero overlap for each shape/chunk pair.
        """
        for overlap in [(0, 0, 0), (10, 10, 10)]:
            z, r, c = shape
            chunks = calculate_conv_chunks(z, r, c, chunk_shape, overlap, None)
            reconstructed, original = _reconstruct_volume(
                shape, chunks, ndim_zyc=3
            )
            np.testing.assert_array_equal(
                reconstructed, original,
                err_msg=(
                    f"Reconstruction failed for shape={shape}, "
                    f"chunk_shape={chunk_shape}, overlap={overlap}"
                ),
            )
