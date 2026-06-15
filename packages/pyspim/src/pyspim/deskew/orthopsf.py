"""PSF-Weighted Orthogonal Deskewing.

Deskews stage-scanned tilted light-sheet data using PSF-weighted orthogonal
interpolation. Unlike the standard "orthogonal" method which blends only two
adjacent planes with linear distance weights, this method integrates
contributions from N adjacent planes weighted by a configurable PSF model
(Gaussian, Airy, or Lorentzian).

This addresses the fundamental limitation that the axial PSF typically spans
multiple stage steps, so a two-plane linear blend truncates a significant
portion of the PSF energy.

References
----------
Based on the orthogonal interpolation scheme from V. Maioli's Ph.D. thesis [1],
extended with PSF-aware multi-plane blending.

[1] Vincent Maioli's PhD Thesis doi: 10.25560/68022
"""

import math
from typing import Tuple

import numpy
try:
    import cupy
    from cupy import get_array_module
    _HAS_GPU = True
except ImportError:
    def get_array_module(x):
        return numpy
    _HAS_GPU = False
    
from numba import njit, prange
from scipy.special import j1

# Use local type alias to avoid importing from pyspim.typing
NDArray = numpy.ndarray

# PSF model enum (integers for Numba/CUDA compatibility)
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
    """Convert FWHM to model-specific parameter.

    For Gaussian: returns sigma = FWHM / 2.355
    For Airy:     returns FWHM directly (used as scaling parameter)
    For Lorentzian: returns FWHM directly (HWHM computed inside weight function)
    """
    if psf_model == PSF_GAUSSIAN:
        return fwhm / 2.355
    else:
        # Airy and Lorentzian use FWHM directly
        return fwhm


# ---------------------------------------------------------------------------
# Numba-accelerated PSF weight functions (CPU path)
# ---------------------------------------------------------------------------


@njit
def _psf_weight_gaussian(d: float, sigma: float) -> float:
    """Gaussian PSF weight: exp(-d^2 / (2*sigma^2))."""
    s = d / sigma
    return math.exp(-0.5 * s * s)


@njit
def _psf_weight_airy(d: float, fwhm: float) -> float:
    """Airy PSF weight: [jinc(pi * d / FWHM)]^2."""
    x = math.pi * d / fwhm
    if abs(x) < 1e-10:
        return 1.0
    j1_val = _bessel_j1(x)
    jinc = 2.0 * j1_val / x
    return jinc * jinc


@njit
def _psf_weight_lorentzian(d: float, fwhm: float) -> float:
    """Lorentzian PSF weight: HWHM^2 / (d^2 + HWHM^2)."""
    hwhm = 0.5 * fwhm
    return (hwhm * hwhm) / (d * d + hwhm * hwhm)


@njit
def _bessel_j1(x: float) -> float:
    """Approximation of Bessel function J1(x) for Numba.

    Uses Taylor series for small x and asymptotic approximation for large x.
    """
    if x < 0:
        return -_bessel_j1(-x)
    if x < 0.1:
        x2 = x * x
        return x * (0.5 - x2 * (0.0625 - x2 * (0.0026041666666666665 - x2 * 0.00005425347222222222)))
    if x < 3.0:
        x2 = x * x
        num = 0.5 - 0.0625 * x2 + 0.002604167 * x2 * x2 - 0.000054253 * x2 * x2 * x2
        return x * num
    # Asymptotic for x >= 3: J1(x) ≈ sqrt(2/(pi*x)) * cos(x - 3*pi/4)
    return math.sqrt(2.0 / (math.pi * x)) * math.cos(x - 3.0 * math.pi / 4.0)


@njit
def _psf_weight(d: float, param: float, psf_model: int) -> float:
    """Dispatch to the appropriate PSF weight function."""
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
    """Compute the number of adjacent planes needed to capture energy_fraction of PSF energy.

    Iteratively sums PSF weights from plane k=0 outward until the cumulative
    fraction exceeds energy_fraction.
    """
    cumulative = _psf_weight(0.0, psf_param, psf_model)
    total = cumulative
    N = 0
    while True:
        N += 1
        d = N * step_size_lat
        w = _psf_weight(float(d), psf_param, psf_model)
        # Both +N and -N planes contribute the same weight
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
    """Compute the in-plane kernel radius to capture energy_fraction of PSF energy."""
    cumulative = _psf_weight(0.0, psf_param, psf_model)
    total = cumulative
    r = 0
    while True:
        r += 1
        # For a 2D kernel, sum the ring at radius r
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
# Output shape helper (same as ortho.py)
# ---------------------------------------------------------------------------


def output_shape(
    n_planes: int,
    n_y: int,
    h: int,
    pixel_size: float,
    step_size: float,
    theta: float = math.pi / 4,
) -> Tuple[int, int, int]:
    """Compute output shape of deskewed volume.

    Parameters
    ----------
    n_planes : int
        Number of input planes (z-dimension of raw data).
    n_y : int
        Number of rows (y-dimension).
    h : int
        Width of each plane (x-dimension within plane).
    pixel_size : float
        Lateral pixel size in physical units.
    step_size : float
        Stage step size in physical units.
    theta : float
        Angle of objective w.r.t coverslip in radians.

    Returns
    -------
    Tuple[int, int, int]
        (n_z, n_y, n_x) shape of deskewed volume.
    """
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


def _zero_triangle(dsk: numpy.ndarray, direction: int) -> numpy.ndarray:
    """Zero the unfilled triangular corner of the deskewed volume.

    Mirrors the logic from ortho.py lines 128-135.
    """
    if direction < 0:
        dsk = numpy.flipud(dsk)
    nz, _, nx = dsk.shape
    stop = min(nz, nx)
    for z in range(stop):
        dsk[z, :, : z + 1] = 0
    dsk = numpy.flipud(dsk)
    return dsk[..., ::direction]


# ---------------------------------------------------------------------------
# Numba core kernels (CPU path)
# ---------------------------------------------------------------------------

# NOTE: Both kernels use flat indexing (idx -> z,y,x) to avoid Numba's
# inability to broadcast scalar += scalar * array. This is necessary because
# each output voxel must be processed independently with scalar arithmetic.


@njit(parallel=True)
def _deskew_cpu_multi_plane_core(
    im, dsk, n_planes, ny, h, nx, nz, direction,
    sin_t, tan_t, tan_ot, step_pix, inv_step_pix, h_f, h_cos_t,
    psf_param_axial, step_size_lat, N, psf_model, preserve_dtype,
):
    """Core Numba kernel: PSF-weighted multi-plane blending + bilinear in-plane.

    Each output voxel (z,y,x) is processed independently to avoid array operations
    inside the Numba parallel loop.
    """
    total = nz * ny * nx
    for idx in prange(total):
        z = idx // (ny * nx)
        remainder = idx % (ny * nx)
        y = remainder // nx
        x = remainder % nx

        # Map output (z, x) to raw coordinates (xpr, zpr)
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

            # Per-voxel bilinear interpolation (scalar, not array)
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
    """Full PSF kernel: PSF-weighted multi-plane + PSF-weighted in-plane.

    Loops over (z, y, x) in parallel for full 3D processing with per-voxel
    2D in-plane PSF kernel.
    """
    total = nz * ny * nx
    for idx in prange(total):
        z = idx // (ny * nx)
        remainder = idx % (ny * nx)
        y = remainder // nx
        x = remainder % nx

        # Map output (z, x) to raw coordinates
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

            # PSF-weighted in-plane interpolation at (plane_n, y, x_sample)
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
    im: numpy.ndarray,
    pixel_size: float,
    step_size: float,
    direction: int,
    theta: float,
    psf_param_axial: float,
    N: int,
    psf_model: int,
    preserve_dtype: bool,
) -> numpy.ndarray:
    """CPU implementation: PSF-weighted multi-plane blending + bilinear in-plane."""
    direction = int(numpy.sign(direction))
    step_size_lat = step_size / math.cos(theta)
    step_pix = step_size_lat / pixel_size

    sin_t = float(math.sin(theta))
    tan_t = float(math.tan(theta))
    tan_ot = float(math.tan(math.pi / 2.0 - theta))

    n_planes, ny, h = im.shape
    out_shape = output_shape(n_planes, ny, h, pixel_size, step_size, theta)
    nz, ny_out, nx = out_shape

    # Always work in float64 for the kernel, then cast at the end
    dsk = numpy.zeros((nz, ny_out, nx), dtype=numpy.float64)

    _deskew_cpu_multi_plane_core(
        im.astype(numpy.float64),
        dsk,
        n_planes, ny, h, nx, nz, direction,
        sin_t, tan_t, tan_ot, step_pix, 1.0 / step_pix,
        float(h), float(h * math.cos(theta)),
        psf_param_axial, step_size_lat, N, psf_model, preserve_dtype,
    )

    if preserve_dtype:
        dsk = dsk.astype(im.dtype)
    else:
        dsk = dsk.astype(numpy.float32)
    return dsk


def _deskew_cpu_psf_in_plane(
    im: numpy.ndarray,
    pixel_size: float,
    step_size: float,
    direction: int,
    theta: float,
    psf_param_axial: float,
    psf_param_lateral: float,
    N: int,
    kernel_radius: int,
    psf_model: int,
    preserve_dtype: bool,
) -> numpy.ndarray:
    """CPU implementation: PSF-weighted multi-plane + PSF-weighted in-plane."""
    direction = int(numpy.sign(direction))
    step_size_lat = step_size / math.cos(theta)
    step_pix = step_size_lat / pixel_size

    sin_t = float(math.sin(theta))
    tan_t = float(math.tan(theta))
    tan_ot = float(math.tan(math.pi / 2.0 - theta))

    n_planes, ny, h = im.shape
    out_shape = output_shape(n_planes, ny, h, pixel_size, step_size, theta)
    nz, ny_out, nx = out_shape

    # Always work in float64 for the kernel, then cast at the end
    dsk = numpy.zeros((nz, ny_out, nx), dtype=numpy.float64)

    _deskew_cpu_psf_full_core(
        im.astype(numpy.float64),
        dsk,
        n_planes, ny, h, nx, nz, direction,
        sin_t, tan_t, tan_ot, step_pix, 1.0 / step_pix,
        float(h), float(h * math.cos(theta)),
        psf_param_axial, psf_param_lateral, step_size_lat, pixel_size,
        N, kernel_radius, psf_model, preserve_dtype,
    )

    if preserve_dtype:
        dsk = dsk.astype(im.dtype)
    else:
        dsk = dsk.astype(numpy.float32)
    return dsk


# ---------------------------------------------------------------------------
# GPU Implementation (CUDA kernels via CuPy RawKernel)
# ---------------------------------------------------------------------------

if _HAS_GPU:

    # Base CUDA kernel source for multi-plane PSF blending
    # {PSF_MODEL} will be replaced with GAUSSIAN, AIRY, or LORENTZIAN
    # {OUTPUT_TYPE} will be replaced with FLOAT32 or UINT16
    _ORTHOPSF_MULTI_PLANE_SRC = r'''
    #define CUDA_PI 3.14159265358979323846f
    // PSF weight functions
    __device__ inline float psfWeight(float d, float param) {
    #ifdef GAUSSIAN
        float s = d / param;
        return expf(-0.5f * s * s);
    #elif defined(AIRY)
        float x = CUDA_PI * d / param;
        if (fabsf(x) < 1e-6f) return 1.0f;
        // Approximation of J1(x) using series for small x, asymptotic for large
        float j1v;
        if (x < 0.1f) {
            float x2 = x * x;
            j1v = x * (0.5f - x2 * (0.0625f - x2 * 0.002604167f));
        } else if (x < 3.0f) {
            float x2 = x * x;
            j1v = x * (0.5f - 0.0625f*x2 + 0.002604167f*x2*x2 - 0.000054253f*x2*x2*x2);
        } else {
            j1v = sqrtf(2.0f / (CUDA_PI * x)) * cosf(x - 3.0f * CUDA_PI / 4.0f);
        }
        float jinc = 2.0f * j1v / x;
        return jinc * jinc;
    #elif defined(LORENTZIAN)
        float hwhm = 0.5f * param;
        return (hwhm * hwhm) / (d * d + hwhm * hwhm);
    #endif
    }

    extern "C" __global__
    void deskewPSFMultiPlane(
        const float* __restrict__ im,
        float* __restrict__ out,
        const int n_planes, const int ny, const int h,
        const int nz, const int nx,
        const int direction,
        const double sin_t, const double tan_t, const double tan_ot,
        const double step_pix, const double inv_step_pix,
        const double h_f, const double h_cos_t,
        const double psf_param_axial, const double step_size_lat,
        const int N)
    {
        long long total = (long long)nz * ny * nx;
        for (long long tid = (blockDim.x * blockIdx.x + threadIdx.x);
             tid < total;
             tid += (long long)blockDim.x * gridDim.x)
        {
            long long tmp = tid / nx;
            int x = (int)(tid % nx);
            int y = (int)(tmp % ny);
            int z = (int)(tmp / ny);

            double xpr, zpr;
            if (direction > 0) {
                xpr = (double)z / sin_t;
                zpr = ((double)x - (double)z / tan_t) * inv_step_pix;
            } else {
                xpr = (double)z / sin_t - h_f;
                zpr = ((double)x + (double)z / tan_t - h_cos_t) * inv_step_pix;
            }

            int zpb = (int)floor(zpr);
            double val = 0.0;
            double weight_sum = 0.0;

            for (int k = -N; k <= N; k++) {
                int plane = zpb + k;
                int plane_n = plane;
                if (plane_n < 0) plane_n = plane_n + n_planes;
                if (plane_n < 0 || plane_n >= n_planes) continue;

                double d_k = fabs((zpr - (double)plane) * step_size_lat);
                double w_k = psfWeight((float)d_k, (float)psf_param_axial);
                if (w_k < 1e-15) continue;

                double beta_k = (zpr - (double)plane) * step_pix;
                double x_sample = xpr + (direction > 0 ? 1.0 : -1.0) * beta_k * tan_ot;

                int xs_f = (int)floor(x_sample);
                double dx = x_sample - (double)xs_f;

                int xs0 = xs_f;
                int xs1 = xs_f + 1;
                if (xs0 < 0) xs0 = xs0 + h;
                if (xs1 < 0) xs1 = xs1 + h;
                if (xs0 < 0 || xs0 >= h || xs1 < 0 || xs1 >= h) continue;

                long long base = ((long long)plane_n * ny + y) * h;
                double plane_val = dx * im[base + xs1] + (1.0 - dx) * im[base + xs0];
                val += w_k * plane_val;
                weight_sum += w_k;
            }

            if (weight_sum > 0.0) {
                double result = val / weight_sum;
                if (result < 0.0) result = 0.0;
                if (result > 65535.0) result = 65535.0;
                out[tid] = (float)result;
            }
        }
    }
    '''.strip()

    # Base CUDA kernel source for full PSF (multi-plane + in-plane)
    _ORTHOPSF_FULL_SRC = r'''
    #define CUDA_PI 3.14159265358979323846f
    // PSF weight functions
    __device__ inline float psfWeight(float d, float param) {
    #ifdef GAUSSIAN
        float s = d / param;
        return expf(-0.5f * s * s);
    #elif defined(AIRY)
        float x = CUDA_PI * d / param;
        if (fabsf(x) < 1e-6f) return 1.0f;
        float j1v;
        if (x < 0.1f) {
            float x2 = x * x;
            j1v = x * (0.5f - x2 * (0.0625f - x2 * 0.002604167f));
        } else if (x < 3.0f) {
            float x2 = x * x;
            j1v = x * (0.5f - 0.0625f*x2 + 0.002604167f*x2*x2 - 0.000054253f*x2*x2*x2);
        } else {
            j1v = sqrtf(2.0f / (CUDA_PI * x)) * cosf(x - 3.0f * CUDA_PI / 4.0f);
        }
        float jinc = 2.0f * j1v / x;
        return jinc * jinc;
    #elif defined(LORENTZIAN)
        float hwhm = 0.5f * param;
        return (hwhm * hwhm) / (d * d + hwhm * hwhm);
    #endif
    }

    extern "C" __global__
    void deskewPSFFull(
        const float* __restrict__ im,
        float* __restrict__ out,
        const int n_planes, const int ny, const int h,
        const int nz, const int nx,
        const int direction,
        const double sin_t, const double tan_t, const double tan_ot,
        const double step_pix, const double inv_step_pix,
        const double h_f, const double h_cos_t,
        const double psf_param_axial, const double psf_param_lateral,
        const double step_size_lat, const double pixel_size,
        const int N, const int kernel_radius)
    {
        long long total = (long long)nz * ny * nx;
        for (long long tid = (blockDim.x * blockIdx.x + threadIdx.x);
             tid < total;
             tid += (long long)blockDim.x * gridDim.x)
        {
            long long tmp = tid / nx;
            int x = (int)(tid % nx);
            int y = (int)(tmp % ny);
            int z = (int)(tmp / ny);

            double xpr, zpr;
            if (direction > 0) {
                xpr = (double)z / sin_t;
                zpr = ((double)x - (double)z / tan_t) * inv_step_pix;
            } else {
                xpr = (double)z / sin_t - h_f;
                zpr = ((double)x + (double)z / tan_t - h_cos_t) * inv_step_pix;
            }

            int zpb = (int)floor(zpr);
            double val = 0.0;
            double weight_sum = 0.0;

            for (int k = -N; k <= N; k++) {
                int plane = zpb + k;
                int plane_n = plane;
                if (plane_n < 0) plane_n = plane_n + n_planes;
                if (plane_n < 0 || plane_n >= n_planes) continue;

                double d_k = fabs((zpr - (double)plane) * step_size_lat);
                double w_axial = psfWeight((float)d_k, (float)psf_param_axial);
                if (w_axial < 1e-15) continue;

                double beta_k = (zpr - (double)plane) * step_pix;
                double x_sample = xpr + (direction > 0 ? 1.0 : -1.0) * beta_k * tan_ot;

                // PSF-weighted in-plane interpolation
                double plane_val = 0.0;
                double plane_weight_sum = 0.0;

                for (int dy_int = -kernel_radius; dy_int <= kernel_radius; dy_int++) {
                    for (int dx_int = -kernel_radius; dx_int <= kernel_radius; dx_int++) {
                        int sx = (int)floor(x_sample) + dx_int;
                        int sy = y + dy_int;

                        int sx_n = sx;
                        if (sx_n < 0) sx_n = sx_n + h;
                        if (sx_n >= h) sx_n = sx_n - h;
                        int sy_n = sy;
                        if (sy_n < 0) sy_n = sy_n + ny;
                        if (sy_n >= ny) sy_n = sy_n - ny;

                        if (sx_n < 0 || sx_n >= h || sy_n < 0 || sy_n >= ny) continue;

                        double dx_phys = ((double)sx - x_sample) * pixel_size;
                        double dy_phys = (double)dy_int * pixel_size;
                        double dist = sqrt(dx_phys * dx_phys + dy_phys * dy_phys);

                        double w_lat = psfWeight((float)dist, (float)psf_param_lateral);
                        long long idx = ((long long)plane_n * ny + sy_n) * h + sx_n;
                        plane_val += w_lat * im[idx];
                        plane_weight_sum += w_lat;
                    }
                }

                if (plane_weight_sum > 0.0) {
                    plane_val /= plane_weight_sum;
                }

                val += w_axial * plane_val;
                weight_sum += w_axial;
            }

            if (weight_sum > 0.0) {
                double result = val / weight_sum;
                if (result < 0.0) result = 0.0;
                if (result > 65535.0) result = 65535.0;
                out[tid] = (float)result;
            }
        }
    }
    '''.strip()

    # Lazy-compiled kernel caches
    _gpu_kernel_multi_plane = {}  # (psf_model, preserve_dtype, want_u16) -> RawKernel
    _gpu_kernel_full = {}

    def _get_gpu_kernel(use_psf_in_plane: bool, psf_model: int, want_u16: bool):
        """Get or compile the appropriate GPU kernel.

        Note: want_u16 parameter is kept for API compatibility but ignored.
        All kernels output float32; dtype conversion is done in Python.
        """
        cache = _gpu_kernel_full if use_psf_in_plane else _gpu_kernel_multi_plane
        key = (psf_model,)
        if key in cache:
            return cache[key]

        psf_define = {PSF_GAUSSIAN: "GAUSSIAN", PSF_AIRY: "AIRY", PSF_LORENTZIAN: "LORENTZIAN"}[psf_model]
        kernel_name = "deskewPSFFull" if use_psf_in_plane else "deskewPSFMultiPlane"

        src = (_ORTHOPSF_FULL_SRC if use_psf_in_plane else _ORTHOPSF_MULTI_PLANE_SRC)

        module_txt = src
        module = cupy.RawModule(
            code=f"#define {psf_define}\n{module_txt}",
            name_expressions=(kernel_name,),
        )
        module.compile()
        kernel = module.get_function(kernel_name)
        cache[key] = kernel
        return kernel

    def _deskew_gpu_multi_plane(
        im: cupy.ndarray,
        pixel_size: float,
        step_size: float,
        direction: int,
        theta: float,
        psf_param_axial: float,
        N: int,
        psf_model: int,
        preserve_dtype: bool,
        in_dtype,
        stream: cupy.cuda.Stream | None = None,
    ) -> cupy.ndarray:
        """GPU implementation: PSF-weighted multi-plane blending + bilinear in-plane."""
        xim = cupy.asarray(im)
        if xim.dtype != cupy.float32:
            xim = xim.astype(cupy.float32, copy=False)

        n_planes, ny, h = xim.shape
        direction = 1 if direction >= 0 else -1

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        tan_t = math.tan(theta)
        tan_ot = math.tan(math.pi / 2.0 - theta)

        step_pix = (step_size / cos_t) / pixel_size
        inv_step_pix = 1.0 / step_pix

        nx = int(math.ceil(n_planes * step_pix + h * cos_t))
        nz = int(math.ceil(h * sin_t))

        out = cupy.zeros((nz, ny, nx), dtype=cupy.float32)

        total = nz * ny * nx
        threads = 256
        blocks = min((total + threads - 1) // threads, 65535)

        kernel = _get_gpu_kernel(False, psf_model, False)  # Always float32 output from kernel
        args = (
            xim, out,
            numpy.int32(n_planes), numpy.int32(ny), numpy.int32(h),
            numpy.int32(nz), numpy.int32(nx),
            numpy.int32(direction),
            numpy.float64(sin_t), numpy.float64(tan_t), numpy.float64(tan_ot),
            numpy.float64(step_pix), numpy.float64(inv_step_pix),
            numpy.float64(h), numpy.float64(h * cos_t),
            numpy.float64(psf_param_axial), numpy.float64(step_size / cos_t),
            numpy.int32(N),
        )

        if stream is None:
            kernel((blocks,), (threads,), args)
        else:
            with stream:
                kernel((blocks,), (threads,), args)

        if preserve_dtype:
            if numpy.issubdtype(in_dtype, numpy.integer):
                info = numpy.iinfo(in_dtype)
                out = cupy.clip(cupy.rint(out), info.min, info.max).astype(in_dtype)
            else:
                out = out.astype(in_dtype)

        return out

    def _deskew_gpu_psf_in_plane(
        im: cupy.ndarray,
        pixel_size: float,
        step_size: float,
        direction: int,
        theta: float,
        psf_param_axial: float,
        psf_param_lateral: float,
        N: int,
        kernel_radius: int,
        psf_model: int,
        preserve_dtype: bool,
        in_dtype,
        stream: cupy.cuda.Stream | None = None,
    ) -> cupy.ndarray:
        """GPU implementation: PSF-weighted multi-plane + PSF-weighted in-plane."""
        xim = cupy.asarray(im)
        if xim.dtype != cupy.float32:
            xim = xim.astype(cupy.float32, copy=False)

        n_planes, ny, h = xim.shape
        direction = 1 if direction >= 0 else -1

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        tan_t = math.tan(theta)
        tan_ot = math.tan(math.pi / 2.0 - theta)

        step_pix = (step_size / cos_t) / pixel_size
        inv_step_pix = 1.0 / step_pix

        nx = int(math.ceil(n_planes * step_pix + h * cos_t))
        nz = int(math.ceil(h * sin_t))

        out = cupy.zeros((nz, ny, nx), dtype=cupy.float32)

        total = nz * ny * nx
        threads = 256
        blocks = min((total + threads - 1) // threads, 65535)

        kernel = _get_gpu_kernel(True, psf_model, False)  # Always float32 output from kernel
        args = (
            xim, out,
            numpy.int32(n_planes), numpy.int32(ny), numpy.int32(h),
            numpy.int32(nz), numpy.int32(nx),
            numpy.int32(direction),
            numpy.float64(sin_t), numpy.float64(tan_t), numpy.float64(tan_ot),
            numpy.float64(step_pix), numpy.float64(inv_step_pix),
            numpy.float64(h), numpy.float64(h * cos_t),
            numpy.float64(psf_param_axial), numpy.float64(psf_param_lateral),
            numpy.float64(step_size / cos_t), numpy.float64(pixel_size),
            numpy.int32(N), numpy.int32(kernel_radius),
        )

        if stream is None:
            kernel((blocks,), (threads,), args)
        else:
            with stream:
                kernel((blocks,), (threads,), args)

        if preserve_dtype:
            if numpy.issubdtype(in_dtype, numpy.integer):
                info = numpy.iinfo(in_dtype)
                out = cupy.clip(cupy.rint(out), info.min, info.max).astype(in_dtype)
            else:
                out = out.astype(in_dtype)

        return out

    def _zero_triangle_gpu(dsk: cupy.ndarray, direction: int) -> cupy.ndarray:
        """Zero lower-left triangular wrap artifact on GPU."""
        if direction < 0:
            dsk = cupy.flipud(dsk)
        nz, _, nx = dsk.shape
        stop = min(nz, nx)
        for z in range(stop):
            dsk[z, :, : z + 1] = 0
        dsk = cupy.flipud(dsk)
        return dsk[..., ::direction]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def deskew_stage_scan(
    im: NDArray,
    pixel_size: float,
    step_size: float,
    direction: int,
    theta: float = math.pi / 4,
    psf_fwhm_axial: float = None,
    psf_fwhm_lateral: float = None,
    psf_model: str = "gaussian",
    preserve_dtype: bool = False,
    use_psf_in_plane: bool = False,
    stream = None,
) -> NDArray:
    """Deskew stage-scan data using PSF-weighted orthogonal interpolation.

    Parameters
    ----------
    im : NDArray
        Input volume (n_planes, n_y, h). NumPy or CuPy array.
    pixel_size : float
        Lateral pixel size in physical units (e.g., um).
    step_size : float
        Stage step size in physical units.
    direction : int
        Scan direction (+1 or -1).
    theta : float
        Angle of objective w.r.t coverslip in radians. Default: pi/4.
    psf_fwhm_axial : float, optional
        Axial PSF FWHM in physical units. If None, estimated as step_size * 3.0.
    psf_fwhm_lateral : float, optional
        Lateral PSF FWHM in physical units. If None, estimated as pixel_size * 2.5.
    psf_model : str
        PSF model for weighting. One of "gaussian", "airy", "lorentzian".
        Default: "gaussian".
    preserve_dtype : bool
        If True, output matches input dtype. Default: False.
    use_psf_in_plane : bool
        If True, use PSF-weighted interpolation within each plane.
        If False, use standard bilinear interpolation. Default: False.
    stream : cupy.cuda.Stream, optional
        CuPy stream for async execution (GPU only).

    Returns
    -------
    NDArray
        Deskewed volume (n_z, n_y, n_x) in XYZ lab frame (z normal to coverslip).
    """
    # Validate input
    xp = get_array_module(im)

    # Validate psf_model
    if psf_model not in PSF_MODEL_MAP:
        valid = ", ".join(f"'{k}'" for k in PSF_MODEL_MAP)
        raise ValueError(
            f"psf_model must be one of {valid}, got '{psf_model}'"
        )
    psf_model_int = PSF_MODEL_MAP[psf_model]

    # Compute PSF parameters
    if psf_fwhm_axial is None:
        psf_fwhm_axial = step_size * 3.0
    psf_param_axial = _fwhm_to_param(psf_fwhm_axial, psf_model_int)

    if use_psf_in_plane:
        if psf_fwhm_lateral is None:
            psf_fwhm_lateral = pixel_size * 2.5
        psf_param_lateral = _fwhm_to_param(psf_fwhm_lateral, psf_model_int)

    # Compute N and kernel radius
    step_size_lat = step_size / math.cos(theta)
    N = _compute_N(psf_param_axial, step_size_lat, psf_model_int)

    if use_psf_in_plane:
        kernel_radius = _compute_kernel_radius(
            psf_param_lateral, pixel_size, psf_model_int
        )

    # GPU path
    if _HAS_GPU and xp == cupy:
        in_dtype = im.dtype
        if use_psf_in_plane:
            out = _deskew_gpu_psf_in_plane(
                im, pixel_size, step_size, direction, theta,
                psf_param_axial, psf_param_lateral, N, kernel_radius,
                psf_model_int, preserve_dtype, in_dtype, stream,
            )
        else:
            out = _deskew_gpu_multi_plane(
                im, pixel_size, step_size, direction, theta,
                psf_param_axial, N, psf_model_int, preserve_dtype, in_dtype, stream,
            )
        out = _zero_triangle_gpu(out, direction)
        return out

    # CPU path
    if xp != numpy:
        raise NotImplementedError(
            "GPU (CuPy) support requires cupy to be installed."
        )

    im_np = numpy.ascontiguousarray(im)
    if use_psf_in_plane:
        out = _deskew_cpu_psf_in_plane(
            im_np, pixel_size, step_size, direction, theta,
            psf_param_axial, psf_param_lateral, N, kernel_radius,
            psf_model_int, preserve_dtype,
        )
    else:
        out = _deskew_cpu_multi_plane(
            im_np, pixel_size, step_size, direction, theta,
            psf_param_axial, N, psf_model_int, preserve_dtype,
        )
    out = _zero_triangle(out, direction)
    return out
