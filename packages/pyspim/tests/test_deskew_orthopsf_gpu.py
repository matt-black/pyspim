"""Tests for PSF-weighted orthogonal deskewing (GPU).

Requires: cupy, NVIDIA GPU.
Run with: pytest test_deskew_orthopsf_gpu.py -v
"""

import math

import numpy as np
import pytest

HAS_CUPY = False
HAS_ORTHOSPF = False

try:
    import cupy
    HAS_CUPY = True
except ImportError:
    pass

try:
    # Import directly from orthopsf module to avoid deskew/__init__.py chain
    import importlib.util
    _spec = importlib.util.find_spec("pyspim.deskew.orthopsf")
    if _spec is not None:
        _orthopsf = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_orthopsf)
        deskew_stage_scan = _orthopsf.deskew_stage_scan
        output_shape = _orthopsf.output_shape
        PSF_GAUSSIAN = _orthopsf.PSF_GAUSSIAN
        PSF_AIRY = _orthopsf.PSF_AIRY
        PSF_LORENTZIAN = _orthopsf.PSF_LORENTZIAN
        _fwhm_to_param = _orthopsf._fwhm_to_param
        _compute_N = _orthopsf._compute_N
        HAS_ORTHOSPF = True
    else:
        pass
except Exception:
    pass

_SKIP_REASON = "cupy and/or orthopsf module not available"
_GPU_SKIP = not (HAS_CUPY and HAS_ORTHOSPF)

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

PIXEL_SIZE = 0.1625
STEP_SIZE = 0.5
THETA = math.pi / 4
DIRECTION = 1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_volume():
    """Small test volume."""
    rng = np.random.default_rng(42)
    vol = rng.random((20, 32, 64), dtype=np.float32) * 255.0
    return vol


@pytest.fixture
def sample_volume_gpu(sample_volume):
    return cupy.asarray(sample_volume)


@pytest.fixture
def uniform_volume():
    """Uniform volume for energy conservation tests."""
    return np.ones((20, 32, 64), dtype=np.float32) * 1000.0


@pytest.fixture
def uniform_volume_gpu(uniform_volume):
    return cupy.asarray(uniform_volume)


@pytest.fixture
def ramp_volume():
    """Ramp volume for banding tests."""
    n_planes, ny, h = 20, 32, 64
    vol = np.zeros((n_planes, ny, h), dtype=np.float32)
    for i in range(n_planes):
        vol[i, :, :] = i * 50.0
    return vol


@pytest.fixture
def ramp_volume_gpu(ramp_volume):
    return cupy.asarray(ramp_volume)


# ---------------------------------------------------------------------------
# Test GPU output shape
# ---------------------------------------------------------------------------


class TestGPUOutputShape:

    def test_gpu_output_shape(self, sample_volume_gpu):
        """GPU output shape matches CPU output shape."""
        result = deskew_stage_scan(
            sample_volume_gpu, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            psf_model="gaussian", preserve_dtype=False,
        )
        expected_shape = output_shape(
            *sample_volume_gpu.shape, PIXEL_SIZE, STEP_SIZE, THETA
        )
        assert result.shape == expected_shape

    def test_gpu_input_is_cupy(self, sample_volume_gpu):
        """Result should be a CuPy array when input is CuPy."""
        result = deskew_stage_scan(
            sample_volume_gpu, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
        )
        assert isinstance(result, cupy.ndarray)


# ---------------------------------------------------------------------------
# Test GPU basic functionality
# ---------------------------------------------------------------------------


class TestGPUBasicFunctionality:

    def test_gpu_preserve_dtype_u16(self, sample_volume_gpu):
        """Test preserve_dtype=True with uint16 input."""
        vol_u16 = sample_volume_gpu.astype(cupy.uint16)
        result = deskew_stage_scan(
            vol_u16, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            preserve_dtype=True,
        )
        assert result.dtype == cupy.uint16

    def test_gpu_preserve_dtype_float32(self, sample_volume_gpu):
        """Test preserve_dtype=True with float32 input."""
        result = deskew_stage_scan(
            sample_volume_gpu, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            preserve_dtype=True,
        )
        assert result.dtype == cupy.float32

    def test_gpu_float_output_default(self, sample_volume_gpu):
        """Default (preserve_dtype=False) should give float32."""
        result = deskew_stage_scan(
            sample_volume_gpu, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
        )
        assert result.dtype == cupy.float32

    def test_gpu_direction_negative(self, sample_volume_gpu):
        """Test with direction=-1 produces valid output."""
        result = deskew_stage_scan(
            sample_volume_gpu, PIXEL_SIZE, STEP_SIZE, -1, THETA,
        )
        assert cupy.get_array_module(result) == cupy
        expected_shape = output_shape(
            *sample_volume_gpu.shape, PIXEL_SIZE, STEP_SIZE, THETA
        )
        assert result.shape == expected_shape

    def test_gpu_psf_in_plane_flag(self, sample_volume_gpu):
        """Test use_psf_in_plane=True produces valid output."""
        result = deskew_stage_scan(
            sample_volume_gpu, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            use_psf_in_plane=True,
        )
        assert isinstance(result, cupy.ndarray)

    def test_gpu_invalid_psf_model(self, sample_volume_gpu):
        """Invalid psf_model should raise ValueError."""
        with pytest.raises(ValueError, match="psf_model"):
            deskew_stage_scan(
                sample_volume_gpu, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
                psf_model="invalid",
            )


# ---------------------------------------------------------------------------
# Test GPU PSF models
# ---------------------------------------------------------------------------


class TestGPUPSModels:

    @pytest.mark.parametrize("psf_model", ["gaussian", "airy", "lorentzian"])
    def test_all_psf_models(self, sample_volume_gpu, psf_model):
        """All PSF models should produce valid output."""
        result = deskew_stage_scan(
            sample_volume_gpu, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            psf_model=psf_model,
        )
        assert isinstance(result, cupy.ndarray)
        assert not cupy.any(cupy.isnan(result))
        assert not cupy.any(cupy.isinf(result))

    @pytest.mark.parametrize("psf_model", ["gaussian", "airy", "lorentzian"])
    def test_psf_in_plane_all_models(self, sample_volume_gpu, psf_model):
        """PSF in-plane should work for all models."""
        result = deskew_stage_scan(
            sample_volume_gpu, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            psf_model=psf_model, use_psf_in_plane=True,
        )
        assert isinstance(result, cupy.ndarray)
        assert not cupy.any(cupy.isnan(result))


# ---------------------------------------------------------------------------
# Test GPU energy conservation
# ---------------------------------------------------------------------------


class TestGPUEnergyConservation:

    def test_uniform_gaussian(self, uniform_volume_gpu):
        """Uniform input should produce approximately uniform output."""
        result = deskew_stage_scan(
            uniform_volume_gpu, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            psf_model="gaussian",
        )
        expected = cupy.float32(1000.0)
        mask = (result > 0)
        mean = cupy.mean(result[mask])
        rel_error = abs(float(mean) - float(expected)) / float(expected)
        assert rel_error < 0.05, f"Mean {mean} differs from {expected} by {rel_error:.1%}"

    def test_uniform_lorentzian(self, uniform_volume_gpu):
        """Uniform input with Lorentzian PSF."""
        result = deskew_stage_scan(
            uniform_volume_gpu, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            psf_model="lorentzian",
        )
        expected = cupy.float32(1000.0)
        mask = (result > 0)
        mean = cupy.mean(result[mask])
        rel_error = abs(float(mean) - float(expected)) / float(expected)
        assert rel_error < 0.05, f"Mean {mean} differs from {expected} by {rel_error:.1%}"


# ---------------------------------------------------------------------------
# Test GPU banding
# ---------------------------------------------------------------------------


class TestGPUBanding:

    def test_ramp_smoothness(self, ramp_volume_gpu):
        """Ramp input should produce smooth output (no large jumps)."""
        result = deskew_stage_scan(
            ramp_volume_gpu, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            psf_model="gaussian",
        )
        diff_z = cupy.diff(result, axis=0)
        mask = (result[1:, :, :] > 0) & (result[:-1, :, :] > 0)
        masked_diff = diff_z[mask]
        if masked_diff.size > 0:
            max_jump = float(cupy.max(cupy.abs(masked_diff)))
            assert max_jump < 200.0, f"Max jump in z is {max_jump}, indicating banding"


# ---------------------------------------------------------------------------
# Test GPU/CPU consistency
# ---------------------------------------------------------------------------


class TestGPUConsistency:

    def test_cpu_gpu_consistency_multi_plane(self, sample_volume):
        """CPU and GPU outputs should match for multi-plane."""
        vol_gpu = cupy.asarray(sample_volume)

        result_cpu = deskew_stage_scan(
            sample_volume, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            psf_model="gaussian", preserve_dtype=False,
        )
        result_gpu = deskew_stage_scan(
            vol_gpu, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            psf_model="gaussian", preserve_dtype=False,
        )

        result_gpu_np = cupy.asnumpy(result_gpu)

        # Diagnostic: find worst voxels and compare shapes
        nz_cpu, ny_cpu, nx_cpu = result_cpu.shape
        nz_gpu, ny_gpu, nx_gpu = result_gpu_np.shape
        assert (nz_cpu, ny_cpu, nx_cpu) == (nz_gpu, ny_gpu, nx_gpu), (
            f"Shape mismatch: CPU {result_cpu.shape} vs GPU {result_gpu_np.shape}"
        )

        # Compute absolute and relative differences
        abs_diff = np.abs(result_cpu - result_gpu_np)
        max_abs_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        max_abs_diff = abs_diff[max_abs_idx]

        # Count matching vs non-matching voxels
        mask = (np.abs(result_cpu) > 1e-6) & (np.abs(result_gpu_np) > 1e-6)
        total_nonzero = np.sum(mask)
        tol = 1e-4
        matching = np.sum(np.abs(result_cpu[mask] - result_gpu_np[mask]) < tol)
        pct_match = 100.0 * matching / total_nonzero if total_nonzero > 0 else 100.0

        if np.any(mask):
            rel_diff = np.abs(result_cpu[mask] - result_gpu_np[mask]) / np.abs(result_cpu[mask])
            max_rel_diff = np.max(rel_diff)
            diag_msg = (
                f"Max relative diff={max_rel_diff:.6f}, "
                f"max abs diff={max_abs_diff:.4f} at {max_abs_idx}, "
                f"match={pct_match:.1f}%, "
                f"CPU val={result_cpu[max_abs_idx]:.6f}, "
                f"GPU val={result_gpu_np[max_abs_idx]:.6f}"
            )
            assert max_rel_diff < 1e-2, (
                f"{diag_msg}"
            )

    def test_cpu_gpu_consistency_psf_in_plane(self, sample_volume):
        """CPU and GPU outputs should match for PSF in-plane."""
        vol_gpu = cupy.asarray(sample_volume)

        result_cpu = deskew_stage_scan(
            sample_volume, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            psf_model="gaussian", preserve_dtype=False, use_psf_in_plane=True,
        )
        result_gpu = deskew_stage_scan(
            vol_gpu, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            psf_model="gaussian", preserve_dtype=False, use_psf_in_plane=True,
        )

        result_gpu_np = cupy.asnumpy(result_gpu)

        # Compute relative difference where both are nonzero
        mask = (np.abs(result_cpu) > 1e-6) & (np.abs(result_gpu_np) > 1e-6)
        if np.any(mask):
            rel_diff = np.abs(result_cpu[mask] - result_gpu_np[mask]) / np.abs(result_cpu[mask])
            max_rel_diff = np.max(rel_diff)
            assert max_rel_diff < 1e-2, (
                f"Max relative difference CPU/GPU = {max_rel_diff:.4f}"
            )

    def test_cpu_gpu_consistency_u16(self, sample_volume):
        """CPU and GPU outputs should match for uint16 output."""
        vol_u16 = (sample_volume).astype(np.uint16)
        vol_gpu_u16 = cupy.asarray(vol_u16)

        result_cpu = deskew_stage_scan(
            vol_u16, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            psf_model="gaussian", preserve_dtype=True,
        )
        result_gpu = deskew_stage_scan(
            vol_gpu_u16, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            psf_model="gaussian", preserve_dtype=True,
        )

        result_gpu_np = cupy.asnumpy(result_gpu)
        assert result_cpu.dtype == np.uint16
        assert result_gpu_np.dtype == np.uint16

        max_abs_diff = np.max(np.abs(result_cpu.astype(np.int32) - result_gpu_np.astype(np.int32)))
        assert max_abs_diff <= 2, f"Max absolute difference = {max_abs_diff}"


# ---------------------------------------------------------------------------
# Test GPU with stream
# ---------------------------------------------------------------------------


class TestGPUStream:

    def test_stream_execution(self, sample_volume_gpu):
        """Test that stream parameter is accepted."""
        stream = cupy.cuda.Stream()
        result = deskew_stage_scan(
            sample_volume_gpu, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            stream=stream,
        )
        assert isinstance(result, cupy.ndarray)


# ---------------------------------------------------------------------------
# Test GPU triangle zeroing
# ---------------------------------------------------------------------------


class TestGPUDriangleZeroing:

    def test_triangle_zeros_direction_positive(self, sample_volume_gpu):
        """Check that triangle corners are zeroed for direction=+1.

        After _zero_triangle_gpu, the upper-left triangle is zeroed:
        z=0 has :stop cols zeroed, z=1 has :stop-1 cols zeroed, etc.
        So at z, cols :stop-z should be zero.
        """
        result = deskew_stage_scan(
            sample_volume_gpu, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
        )
        nz, _, nx = result.shape
        stop = min(nz, nx)
        for z in range(stop):
            zero_width = stop - z
            col_sum = cupy.sum(result[z, :, :zero_width])
            assert col_sum == 0, (
                f"Triangle not zeroed at z={z}, zero_width={zero_width}"
            )

    def test_triangle_zeros_direction_negative(self, sample_volume_gpu):
        """Check that triangle corners are zeroed for direction=-1.

        For direction=-1, the x-axis is reversed via dsk[..., ::-1].
        Before reversal, upper-left is zeroed (:stop-z cols).
        After reversal, upper-RIGHT is zeroed (cols nx-(stop-z) : nx).
        """
        result = deskew_stage_scan(
            sample_volume_gpu, PIXEL_SIZE, STEP_SIZE, -1, THETA,
        )
        nz, _, nx = result.shape
        stop = min(nz, nx)
        for z in range(stop):
            zero_width = stop - z
            start = nx - zero_width
            col_sum = cupy.sum(result[z, :, start:])
            assert col_sum == 0, (
                f"Triangle not zeroed at z={z}, zero_width={zero_width}, start={start}"
            )


# ---------------------------------------------------------------------------
# Test GPU PSF model consistency
# ---------------------------------------------------------------------------


class TestGPUPSModelConsistency:

    def test_large_fwhm_consistency(self, uniform_volume_gpu):
        """All models should give similar results for large FWHM."""
        large_fwhm = 10.0
        results = {}
        for model in ["gaussian", "airy", "lorentzian"]:
            results[model] = deskew_stage_scan(
                uniform_volume_gpu, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
                psf_fwhm_axial=large_fwhm, psf_model=model, preserve_dtype=False,
            )

        for model_a in results:
            for model_b in results:
                if model_a >= model_b:
                    continue
                diff = cupy.abs(results[model_a] - results[model_b])
                max_diff = float(cupy.max(diff))
                assert max_diff < 10.0, (
                    f"Models {model_a} and {model_b} differ by {max_diff} "
                    f"for uniform input with large FWHM"
                )


# ---------------------------------------------------------------------------
# Test GPU default parameters
# ---------------------------------------------------------------------------


class TestGPUDefaultParameters:

    def test_default_fwhm_axial(self, sample_volume_gpu):
        result = deskew_stage_scan(
            sample_volume_gpu, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
        )
        expected_shape = output_shape(
            *sample_volume_gpu.shape, PIXEL_SIZE, STEP_SIZE, THETA
        )
        assert result.shape == expected_shape

    def test_default_fwhm_lateral(self, sample_volume_gpu):
        result = deskew_stage_scan(
            sample_volume_gpu, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            use_psf_in_plane=True,
        )
        expected_shape = output_shape(
            *sample_volume_gpu.shape, PIXEL_SIZE, STEP_SIZE, THETA
        )
        assert result.shape == expected_shape
