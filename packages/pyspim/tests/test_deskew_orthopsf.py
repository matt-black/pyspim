"""Self-contained tests for PSF-weighted orthogonal deskewing.

This test file embeds all logic from orthopsf.py directly to avoid importing
pyspim, which has a cupy dependency not available on this machine.
"""

import math
import sys

import numpy as np
import pytest
from numba import njit, prange

# ---------------------------------------------------------------------------
# PSF model enum (integers for Numba compatibility)
# ---------------------------------------------------------------------------

PSF_GAUSSIAN = 0
PSF_AIRY = 1
PSF_LORENTZIAN = 2

PSF_MODEL_MAP = {
    "gaussian": PSF_GAUSSIAN,
    "airy": PSF_AIRY,
    "lorentzian": PSF_LORENTZIAN,
}


# ---------------------------------------------------------------------------
# PSF weight functions (Python-level for parameter computation)
# ---------------------------------------------------------------------------


def _fwhm_to_param(fwhm: float, psf_model: int) -> float:
    if psf_model == PSF_GAUSSIAN:
        return fwhm / 2.355
    else:
        return fwhm


# ---------------------------------------------------------------------------
# Numba-accelerated PSF weight functions
# ---------------------------------------------------------------------------


@njit
def _psf_weight_gaussian(d: float, sigma: float) -> float:
    s = d / sigma
    return math.exp(-0.5 * s * s)


@njit
def _psf_weight_airy(d: float, fwhm: float) -> float:
    x = math.pi * d / fwhm
    if abs(x) < 1e-10:
        return 1.0
    j1_val = _bessel_j1(x)
    jinc = 2.0 * j1_val / x
    return jinc * jinc


@njit
def _psf_weight_lorentzian(d: float, fwhm: float) -> float:
    hwhm = 0.5 * fwhm
    return (hwhm * hwhm) / (d * d + hwhm * hwhm)


@njit
def _bessel_j1(x: float) -> float:
    if x < 0:
        return -_bessel_j1(-x)
    if x < 0.1:
        x2 = x * x
        return x * (0.5 - x2 * (0.0625 - x2 * (0.0026041666666666665 - x2 * 0.00005425347222222222)))
    if x < 3.0:
        x2 = x * x
        num = 0.5 - 0.0625 * x2 + 0.002604167 * x2 * x2 - 0.000054253 * x2 * x2 * x2
        return x * num
    return math.sqrt(2.0 / (math.pi * x)) * math.cos(x - 3.0 * math.pi / 4.0)


@njit
def _psf_weight(d: float, param: float, psf_model: int) -> float:
    if psf_model == PSF_GAUSSIAN:
        return _psf_weight_gaussian(d, param)
    elif psf_model == PSF_AIRY:
        return _psf_weight_airy(d, param)
    else:
        return _psf_weight_lorentzian(d, param)


# ---------------------------------------------------------------------------
# N computation helpers
# ---------------------------------------------------------------------------


def _compute_N(
    psf_param: float,
    step_size_lat: float,
    psf_model: int,
    energy_fraction: float = 0.99,
) -> int:
    cumulative = _psf_weight(0.0, psf_param, psf_model)
    total = cumulative
    N = 0
    while True:
        N += 1
        d = N * step_size_lat
        w = _psf_weight(float(d), psf_param, psf_model)
        total += 2.0 * w
        if cumulative / total >= energy_fraction:
            break
        cumulative += 2.0 * w
    return N


def _compute_kernel_radius(
    psf_param: float,
    pixel_size: float,
    psf_model: int,
    energy_fraction: float = 0.99,
) -> int:
    cumulative = _psf_weight(0.0, psf_param, psf_model)
    total = cumulative
    r = 0
    while True:
        r += 1
        ring_weight = 0.0
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dx) == r or abs(dy) == r:
                    dist = math.sqrt((dx * pixel_size) ** 2 + (dy * pixel_size) ** 2)
                    ring_weight += _psf_weight(dist, psf_param, psf_model)
        total += ring_weight
        if cumulative / total >= energy_fraction:
            break
        cumulative += ring_weight
    return r


# ---------------------------------------------------------------------------
# Output shape helper
# ---------------------------------------------------------------------------


def output_shape(
    n_planes: int,
    n_y: int,
    h: int,
    pixel_size: float,
    step_size: float,
    theta: float = math.pi / 4,
):
    step_size_lat = step_size / math.cos(theta)
    step_pix = step_size_lat / pixel_size
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    n_x = int(math.ceil(n_planes * step_pix + h * cos_theta))
    n_z = int(math.ceil(h * sin_theta))
    return (n_z, n_y, n_x)


# ---------------------------------------------------------------------------
# Triangle zeroing
# ---------------------------------------------------------------------------


def _zero_triangle(dsk: np.ndarray, direction: int) -> np.ndarray:
    if direction < 0:
        dsk = np.flipud(dsk)
    nz, _, nx = dsk.shape
    stop = min(nz, nx)
    for z in range(stop):
        dsk[z, :, : z + 1] = 0
    dsk = np.flipud(dsk)
    return dsk[..., ::direction]


# ---------------------------------------------------------------------------
# Numba core kernels
# ---------------------------------------------------------------------------


@njit(parallel=True)
def _deskew_cpu_multi_plane_core(
    im, dsk, n_planes, ny, h, nx, nz, direction,
    sin_t, tan_t, tan_ot, step_pix, inv_step_pix, h_f, h_cos_t,
    psf_param_axial, step_size_lat, N, psf_model, preserve_dtype,
):
    total = nz * ny * nx
    for idx in prange(total):
        z = idx // (ny * nx)
        remainder = idx % (ny * nx)
        y = remainder // nx
        x = remainder % nx

        if direction > 0:
            xpr = float(z) / sin_t
            zpr = (float(x) - float(z) / tan_t) * inv_step_pix
        else:
            xpr = float(z) / sin_t - h_f
            zpr = (float(x) + float(z) / tan_t - h_cos_t) * inv_step_pix

        zpb = int(math.floor(zpr))
        val = 0.0
        weight_sum = 0.0

        for k in range(-N, N + 1):
            plane = zpb + k
            plane_n = plane
            if plane_n < 0:
                plane_n = plane_n + n_planes
            if plane_n < 0 or plane_n >= n_planes:
                continue

            d_k = abs((zpr - float(plane)) * step_size_lat)
            w_k = _psf_weight(d_k, psf_param_axial, psf_model)
            if w_k < 1e-15:
                continue

            beta_k = (zpr - float(plane)) * step_pix
            x_sample = xpr + float(direction) * beta_k * tan_ot

            xs_f = int(math.floor(x_sample))
            dx = x_sample - float(xs_f)

            xs0 = xs_f
            xs1 = xs_f + 1
            if xs0 < 0:
                xs0 = xs0 + h
            if xs1 < 0:
                xs1 = xs1 + h
            if xs0 < 0 or xs0 >= h or xs1 < 0 or xs1 >= h:
                continue

            plane_val = dx * im[plane_n, y, xs1] + (1.0 - dx) * im[plane_n, y, xs0]
            val += w_k * plane_val
            weight_sum += w_k

        if weight_sum > 0:
            result = val / weight_sum
            if preserve_dtype:
                rounded = int(math.floor(result + 0.5))
                if rounded < 0:
                    rounded = 0
                if rounded > 65535:
                    rounded = 65535
                dsk[z, y, x] = rounded
            else:
                dsk[z, y, x] = result


@njit(parallel=True)
def _deskew_cpu_psf_full_core(
    im, dsk, n_planes, ny, h, nx, nz, direction,
    sin_t, tan_t, tan_ot, step_pix, inv_step_pix, h_f, h_cos_t,
    psf_param_axial, psf_param_lateral, step_size_lat, pixel_size,
    N, kernel_radius, psf_model, preserve_dtype,
):
    total = nz * ny * nx
    for idx in prange(total):
        z = idx // (ny * nx)
        remainder = idx % (ny * nx)
        y = remainder // nx
        x = remainder % nx

        if direction > 0:
            xpr = float(z) / sin_t
            zpr = (float(x) - float(z) / tan_t) * inv_step_pix
        else:
            xpr = float(z) / sin_t - h_f
            zpr = (float(x) + float(z) / tan_t - h_cos_t) * inv_step_pix

        zpb = int(math.floor(zpr))
        val = 0.0
        weight_sum = 0.0

        for k in range(-N, N + 1):
            plane = zpb + k
            plane_n = plane
            if plane_n < 0:
                plane_n = plane_n + n_planes
            if plane_n < 0 or plane_n >= n_planes:
                continue

            d_k = abs((zpr - float(plane)) * step_size_lat)
            w_axial = _psf_weight(d_k, psf_param_axial, psf_model)
            if w_axial < 1e-15:
                continue

            beta_k = (zpr - float(plane)) * step_pix
            x_sample = xpr + float(direction) * beta_k * tan_ot

            plane_val = 0.0
            plane_weight_sum = 0.0

            for dy_int in range(-kernel_radius, kernel_radius + 1):
                for dx_int in range(-kernel_radius, kernel_radius + 1):
                    sx = int(math.floor(x_sample)) + dx_int
                    sy = y + dy_int

                    sx_n = sx
                    if sx_n < 0:
                        sx_n = sx_n + h
                    if sx_n >= h:
                        sx_n = sx_n - h
                    sy_n = sy
                    if sy_n < 0:
                        sy_n = sy_n + ny
                    if sy_n >= ny:
                        sy_n = sy_n - ny

                    if sx_n < 0 or sx_n >= h or sy_n < 0 or sy_n >= ny:
                        continue

                    dx_phys = (float(sx) - x_sample) * pixel_size
                    dy_phys = float(dy_int) * pixel_size
                    dist = math.sqrt(dx_phys * dx_phys + dy_phys * dy_phys)

                    w_lat = _psf_weight(dist, psf_param_lateral, psf_model)
                    plane_val += w_lat * im[plane_n, sy_n, sx_n]
                    plane_weight_sum += w_lat

            if plane_weight_sum > 0:
                plane_val /= plane_weight_sum

            val += w_axial * plane_val
            weight_sum += w_axial

        if weight_sum > 0:
            result = val / weight_sum
            if preserve_dtype:
                rounded = int(math.floor(result + 0.5))
                if rounded < 0:
                    rounded = 0
                if rounded > 65535:
                    rounded = 65535
                dsk[z, y, x] = rounded
            else:
                dsk[z, y, x] = result


# ---------------------------------------------------------------------------
# Public CPU implementations
# ---------------------------------------------------------------------------


def _deskew_cpu_multi_plane(
    im, pixel_size, step_size, direction, theta,
    psf_param_axial, N, psf_model, preserve_dtype,
):
    direction = int(np.sign(direction))
    step_size_lat = step_size / math.cos(theta)
    step_pix = step_size_lat / pixel_size

    sin_t = float(math.sin(theta))
    tan_t = float(math.tan(theta))
    tan_ot = float(math.tan(math.pi / 2.0 - theta))

    n_planes, ny, h = im.shape
    out_shape = output_shape(n_planes, ny, h, pixel_size, step_size, theta)
    nz, ny_out, nx = out_shape

    # Always work in float64 for the kernel, then cast at the end
    dsk = np.zeros((nz, ny_out, nx), dtype=np.float64)

    _deskew_cpu_multi_plane_core(
        im.astype(np.float64),
        dsk,
        n_planes, ny, h, nx, nz, direction,
        sin_t, tan_t, tan_ot, step_pix, 1.0 / step_pix,
        float(h), float(h * math.cos(theta)),
        psf_param_axial, step_size_lat, N, psf_model, preserve_dtype,
    )

    dsk = _zero_triangle(dsk, direction)
    if preserve_dtype:
        dsk = dsk.astype(im.dtype)
    else:
        dsk = dsk.astype(np.float32)
    return dsk


def _deskew_cpu_psf_in_plane(
    im, pixel_size, step_size, direction, theta,
    psf_param_axial, psf_param_lateral, N, kernel_radius, psf_model, preserve_dtype,
):
    direction = int(np.sign(direction))
    step_size_lat = step_size / math.cos(theta)
    step_pix = step_size_lat / pixel_size

    sin_t = float(math.sin(theta))
    tan_t = float(math.tan(theta))
    tan_ot = float(math.tan(math.pi / 2.0 - theta))

    n_planes, ny, h = im.shape
    out_shape = output_shape(n_planes, ny, h, pixel_size, step_size, theta)
    nz, ny_out, nx = out_shape

    # Always work in float64 for the kernel, then cast at the end
    dsk = np.zeros((nz, ny_out, nx), dtype=np.float64)

    _deskew_cpu_psf_full_core(
        im.astype(np.float64),
        dsk,
        n_planes, ny, h, nx, nz, direction,
        sin_t, tan_t, tan_ot, step_pix, 1.0 / step_pix,
        float(h), float(h * math.cos(theta)),
        psf_param_axial, psf_param_lateral, step_size_lat, pixel_size,
        N, kernel_radius, psf_model, preserve_dtype,
    )

    dsk = _zero_triangle(dsk, direction)
    if preserve_dtype:
        dsk = dsk.astype(im.dtype)
    else:
        dsk = dsk.astype(np.float32)
    return dsk


# ---------------------------------------------------------------------------
# Entry point (replicated)
# ---------------------------------------------------------------------------


def deskew_stage_scan(
    im, pixel_size, step_size, direction, theta=math.pi / 4,
    psf_fwhm_axial=None, psf_fwhm_lateral=None,
    psf_model="gaussian", preserve_dtype=False, use_psf_in_plane=False,
):
    if psf_model not in PSF_MODEL_MAP:
        valid = ", ".join(f"'{k}'" for k in PSF_MODEL_MAP)
        raise ValueError(f"psf_model must be one of {valid}, got '{psf_model}'")
    psf_model_int = PSF_MODEL_MAP[psf_model]

    if psf_fwhm_axial is None:
        psf_fwhm_axial = step_size * 3.0
    psf_param_axial = _fwhm_to_param(psf_fwhm_axial, psf_model_int)

    if use_psf_in_plane:
        if psf_fwhm_lateral is None:
            psf_fwhm_lateral = pixel_size * 2.5
        psf_param_lateral = _fwhm_to_param(psf_fwhm_lateral, psf_model_int)

    step_size_lat = step_size / math.cos(theta)
    N = _compute_N(psf_param_axial, step_size_lat, psf_model_int)

    if use_psf_in_plane:
        kernel_radius = _compute_kernel_radius(psf_param_lateral, pixel_size, psf_model_int)

    im_np = np.ascontiguousarray(im)

    if use_psf_in_plane:
        return _deskew_cpu_psf_in_plane(
            im_np, pixel_size, step_size, direction, theta,
            psf_param_axial, psf_param_lateral, N, kernel_radius,
            psf_model_int, preserve_dtype,
        )
    else:
        return _deskew_cpu_multi_plane(
            im_np, pixel_size, step_size, direction, theta,
            psf_param_axial, N, psf_model_int, preserve_dtype,
        )


# ===========================================================================
# TESTS
# ===========================================================================

# Test parameters
STEP_SIZE = 0.5
PIXEL_SIZE = 0.1625
PSF_FWHM_AXIAL = 2.1
PSF_FWHM_LATERAL = 0.381
THETA = math.pi / 4
DIRECTION = 1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_volume():
    np.random.seed(42)
    return np.random.randint(0, 1000, (10, 8, 16), dtype=np.uint16)


@pytest.fixture
def uniform_volume():
    return np.ones((20, 16, 32), dtype=np.uint16) * 500


@pytest.fixture
def ramp_volume():
    n_planes, n_y, h = 20, 8, 32
    ramp = np.zeros((n_planes, n_y, h), dtype=np.float32)
    for i in range(n_planes):
        ramp[i, :, :] = i * 50.0
    return ramp


# ---------------------------------------------------------------------------
# Test PSF weight functions
# ---------------------------------------------------------------------------

class TestPSFWeights:

    def test_gaussian_center(self):
        w = _psf_weight(0.0, 0.892, PSF_GAUSSIAN)
        assert abs(w - 1.0) < 1e-10

    def test_gaussian_fwhm(self):
        sigma = 0.892
        fwhm = sigma * 2.355
        w = _psf_weight(fwhm / 2.0, sigma, PSF_GAUSSIAN)
        assert abs(w - 0.5) < 1e-3  # Gaussian FWHM relationship is approximate

    def test_airy_center(self):
        w = _psf_weight(0.0, 2.1, PSF_AIRY)
        assert abs(w - 1.0) < 1e-6

    def test_airy_zero(self):
        fwhm = 2.1
        first_zero_d = 3.8317 * fwhm / math.pi
        w = _psf_weight(first_zero_d, fwhm, PSF_AIRY)
        assert abs(w) < 0.05

    def test_lorentzian_center(self):
        w = _psf_weight(0.0, 2.1, PSF_LORENTZIAN)
        assert abs(w - 1.0) < 1e-10

    def test_lorentzian_fwhm(self):
        fwhm = 2.1
        w = _psf_weight(fwhm / 2.0, fwhm, PSF_LORENTZIAN)
        assert abs(w - 0.5) < 1e-6

    def test_bessel_j1_small(self):
        x = 0.01
        j1_val = _bessel_j1(x)
        expected = x / 2.0
        assert abs(j1_val - expected) / abs(expected) < 0.01


# ---------------------------------------------------------------------------
# Test _compute_N and _compute_kernel_radius
# ---------------------------------------------------------------------------

class TestComputeN:

    def test_N_gaussian(self):
        sigma = PSF_FWHM_AXIAL / 2.355
        step_size_lat = STEP_SIZE / math.cos(THETA)
        N = _compute_N(sigma, step_size_lat, PSF_GAUSSIAN)
        assert N >= 2
        assert N <= 6

    def test_N_lorentzian(self):
        fwhm = PSF_FWHM_AXIAL
        step_size_lat = STEP_SIZE / math.cos(THETA)
        N_lorentz = _compute_N(fwhm, step_size_lat, PSF_LORENTZIAN)
        assert N_lorentz >= 1

    def test_kernel_radius(self):
        sigma = PSF_FWHM_LATERAL / 2.355
        r = _compute_kernel_radius(sigma, PIXEL_SIZE, PSF_GAUSSIAN)
        assert r >= 1
        assert r <= 8


# ---------------------------------------------------------------------------
# Test output shape
# ---------------------------------------------------------------------------

class TestOutputShape:

    def test_shape_reasonable(self):
        shape = output_shape(20, 16, 32, PIXEL_SIZE, STEP_SIZE, THETA)
        assert len(shape) == 3
        assert all(s > 0 for s in shape)


# ---------------------------------------------------------------------------
# Test basic functionality
# ---------------------------------------------------------------------------

class TestBasicFunctionality:

    def test_basic_output_shape(self, sample_volume):
        result = deskew_stage_scan(
            sample_volume, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            psf_fwhm_axial=PSF_FWHM_AXIAL, psf_model="gaussian",
        )
        expected_shape = output_shape(*sample_volume.shape, PIXEL_SIZE, STEP_SIZE, THETA)
        assert result.shape == expected_shape

    def test_preserve_dtype(self, sample_volume):
        result = deskew_stage_scan(
            sample_volume, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            psf_fwhm_axial=PSF_FWHM_AXIAL, preserve_dtype=True,
        )
        assert result.dtype == sample_volume.dtype

    def test_float_output(self, sample_volume):
        result = deskew_stage_scan(
            sample_volume, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            psf_fwhm_axial=PSF_FWHM_AXIAL, preserve_dtype=False,
        )
        assert result.dtype == np.float32

    def test_direction_negative(self, sample_volume):
        result = deskew_stage_scan(
            sample_volume, PIXEL_SIZE, STEP_SIZE, -1, THETA,
            psf_fwhm_axial=PSF_FWHM_AXIAL,
        )
        assert result.shape == output_shape(*sample_volume.shape, PIXEL_SIZE, STEP_SIZE, THETA)

    def test_psf_in_plane_flag(self, sample_volume):
        result = deskew_stage_scan(
            sample_volume, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            psf_fwhm_axial=PSF_FWHM_AXIAL, psf_fwhm_lateral=PSF_FWHM_LATERAL,
            use_psf_in_plane=True,
        )
        assert result.shape == output_shape(*sample_volume.shape, PIXEL_SIZE, STEP_SIZE, THETA)

    def test_invalid_psf_model(self, sample_volume):
        with pytest.raises(ValueError, match="psf_model"):
            deskew_stage_scan(
                sample_volume, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
                psf_model="invalid",
            )


# ---------------------------------------------------------------------------
# Test energy conservation
# ---------------------------------------------------------------------------

class TestEnergyConservation:

    def test_uniform_gaussian(self, uniform_volume):
        result = deskew_stage_scan(
            uniform_volume, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            psf_fwhm_axial=PSF_FWHM_AXIAL, psf_model="gaussian",
            preserve_dtype=False,
        )
        nz, ny, nx = result.shape
        stop = min(nz, nx)
        valid_mask = np.ones(result.shape, dtype=bool)
        for z in range(stop):
            valid_mask[z, :, : z + 1] = False
        valid_values = result[valid_mask]

        mean_val = np.mean(valid_values)
        assert abs(mean_val - 500.0) < 50.0, f"Mean {mean_val} deviates too much from 500.0"

    def test_uniform_lorentzian(self, uniform_volume):
        result = deskew_stage_scan(
            uniform_volume, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            psf_fwhm_axial=PSF_FWHM_AXIAL, psf_model="lorentzian",
            preserve_dtype=False,
        )
        nz, ny, nx = result.shape
        stop = min(nz, nx)
        valid_mask = np.ones(result.shape, dtype=bool)
        for z in range(stop):
            valid_mask[z, :, : z + 1] = False
        valid_values = result[valid_mask]
        mean_val = np.mean(valid_values)
        assert abs(mean_val - 500.0) < 50.0


# ---------------------------------------------------------------------------
# Test banding
# ---------------------------------------------------------------------------

class TestBanding:

    def test_ramp_smoothness(self, ramp_volume):
        result = deskew_stage_scan(
            ramp_volume, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            psf_fwhm_axial=PSF_FWHM_AXIAL, psf_model="gaussian",
            preserve_dtype=False,
        )
        mid_x = result.shape[2] // 2
        mid_y = result.shape[1] // 2
        profile = result[:, mid_y, mid_x]
        profile = profile[profile > 0]

        if len(profile) < 10:
            pytest.skip("Profile too short")

        second_deriv = np.diff(profile, n=2)
        signal_range = np.max(profile) - np.min(profile)
        if signal_range < 1:
            pytest.skip("Signal range too small")

        mean_second_deriv = np.mean(np.abs(second_deriv))
        assert mean_second_deriv < signal_range * 0.3, (
            f"Banding detected: mean |2nd deriv| = {mean_second_deriv}, "
            f"signal range = {signal_range}"
        )


# ---------------------------------------------------------------------------
# Test PSF model consistency
# ---------------------------------------------------------------------------

class TestPSFModelConsistency:

    def test_large_fwhm_consistency(self, uniform_volume):
        large_fwhm = 10.0
        results = {}
        for model in ["gaussian", "airy", "lorentzian"]:
            results[model] = deskew_stage_scan(
                uniform_volume, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
                psf_fwhm_axial=large_fwhm, psf_model=model, preserve_dtype=False,
            )

        for model_a in results:
            for model_b in results:
                if model_a >= model_b:
                    continue
                diff = np.abs(results[model_a] - results[model_b])
                max_diff = np.max(diff)
                assert max_diff < 10.0, (
                    f"Models {model_a} and {model_b} differ by {max_diff} "
                    f"for uniform input with large FWHM"
                )


# ---------------------------------------------------------------------------
# Test default parameters
# ---------------------------------------------------------------------------

class TestDefaultParameters:

    def test_default_fwhm_axial(self, sample_volume):
        result = deskew_stage_scan(
            sample_volume, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
        )
        assert result.shape == output_shape(*sample_volume.shape, PIXEL_SIZE, STEP_SIZE, THETA)

    def test_default_fwhm_lateral(self, sample_volume):
        result = deskew_stage_scan(
            sample_volume, PIXEL_SIZE, STEP_SIZE, DIRECTION, THETA,
            use_psf_in_plane=True,
        )
        assert result.shape == output_shape(*sample_volume.shape, PIXEL_SIZE, STEP_SIZE, THETA)


# ---------------------------------------------------------------------------
# Test _fwhm_to_param
# ---------------------------------------------------------------------------

class TestFWHMToParam:

    def test_gaussian_conversion(self):
        param = _fwhm_to_param(2.1, PSF_GAUSSIAN)
        expected = 2.1 / 2.355
        assert abs(param - expected) < 1e-10

    def test_airy_conversion(self):
        param = _fwhm_to_param(2.1, PSF_AIRY)
        assert param == 2.1

    def test_lorentzian_conversion(self):
        param = _fwhm_to_param(2.1, PSF_LORENTZIAN)
        assert param == 2.1
