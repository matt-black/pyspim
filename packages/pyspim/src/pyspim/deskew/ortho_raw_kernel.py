"""Deskewing by orthogonal interpolation (GPU-only).

Orthogonal interpolation deskew for stage-scanned tilted light-sheet data.
Original approach: V. Maioli Ph.D. Thesis (doi:10.25560/68022).
Implementation inspired by QI2lab/OPM (D. Shepherd et al.).

All computation performed on GPU using CuPy RawKernels.
"""

from __future__ import annotations

import math
from typing import Tuple, Union

import numpy as np
import cupy as cp

try:
    # project-typed alias
    from ..typing import NDArray  # noqa: F401
except Exception:  # pragma: no cover - fallback
    NDArray = np.ndarray  # type: ignore


# -----------------------------------------------------------------------------
# CUDA kernels
# -----------------------------------------------------------------------------

# Float32-output kernel (analysis mode / non-preserve)
_deskew_kernel_src = r'''
extern "C" __global__
void deskew_kernel(const unsigned short* __restrict__ im, // raw uint16 input
                   float* __restrict__ out,               // float32 working out
                   const int n_planes,
                   const int ny,
                   const int h,
                   const int nz,
                   const int nx,
                   const int direction,   // +1 or -1
                   const float sin_t,
                   const float cos_t,
                   const float tan_t,
                   const float tan_ot,
                   const float cos_ot,
                   const float step_pix,
                   const float inv_step_pix,
                   const float h_f,       // h as float
                   const float h_cos_t)   // h * cos_t precomputed
{
    // Comments in the _deskew_kernel_u16_src kernel apply here too.
    // only difference is the output type
    long long total = (long long)nz * ny * nx;

    for (long long tid = (blockDim.x * blockIdx.x + threadIdx.x);
         tid < total;
         tid += (long long)blockDim.x * gridDim.x)
    {
        int x = tid % nx;
        long long tmp = tid / nx;
        int y = tmp % ny;
        int z = tmp / ny;

        float xpr, zpr;
        if (direction > 0) {
            xpr = z / sin_t;
            zpr = (x - (z / tan_t)) * inv_step_pix;
        } else {
            xpr = z / sin_t - h_f;
            zpr = (x + (z / tan_t) - h_cos_t) * inv_step_pix;
        }

        int z0 = (int)floorf(zpr);
        if (z0 < 0 || z0 >= n_planes - 1) {
            out[tid] = 0.0f;
            continue;
        }

        float beta = step_pix * (zpr - (float)z0);

        float xib = xpr + (direction > 0 ? +1.0f : -1.0f) * beta * tan_ot;
        int xb0 = (int)floorf(xib);
        if (xb0 < 0 || xb0 >= h - 1) {
            out[tid] = 0.0f;
            continue;
        }
        float dxib = xib - (float)xb0;

        float xia = xpr - (direction > 0 ? +1.0f : -1.0f) * (step_pix - beta) * tan_ot;
        int xa0 = (int)floorf(xia);
        if (xa0 < 0 || xa0 >= h - 1) {
            out[tid] = 0.0f;
            continue;
        }
        float dxia = xia - (float)xa0;

        float wb = beta / cos_ot;              // weight for z0+1 plane
        float wa = (step_pix - beta) / cos_ot; // weight for z0 plane

        long long baseA = ((long long)z0     * ny + y) * h; // lower plane
        long long baseB = ((long long)(z0+1) * ny + y) * h; // upper plane

        float A0 = (float)im[baseA + xb0    ];
        float A1 = (float)im[baseA + xb0 + 1];
        float B0 = (float)im[baseB + xa0    ];
        float B1 = (float)im[baseB + xa0 + 1];

        float v_a = dxib * A1 + (1.0f - dxib) * A0;
        float v_b = dxia * B1 + (1.0f - dxia) * B0;

        float val = (wa * v_a + wb * v_b) * inv_step_pix;
        out[tid] = val;
    }
}
'''.strip()


# Direct uint16-output kernel (preserve_dtype fast path)
_deskew_kernel_u16_src = r'''
extern "C" __global__
void deskew_kernel_u16(const unsigned short* __restrict__ im,
                       unsigned short* __restrict__ out,
                       const int n_planes,
                       const int ny,
                       const int h,
                       const int nz,
                       const int nx,
                       const int direction,
                       const float sin_t,
                       const float cos_t,
                       const float tan_t,
                       const float tan_ot,
                       const float cos_ot,
                       const float step_pix,
                       const float inv_step_pix,
                       const float h_f,
                       const float h_cos_t)
{
    // Total number of points to process
    long long total = (long long)nz * ny * nx;

    // 1-D grid-stride loop (https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)
    for (long long tid = (blockDim.x * blockIdx.x + threadIdx.x);
         tid < total;
         tid += (long long)blockDim.x * gridDim.x)
    {

        // Convert 1D thread ID to 3D coordinates (x,y,z)
        int x = tid % nx;
        long long tmp = tid / nx;
        int y = tmp % ny;
        int z = tmp / ny;

        // NOTE: possible to avoid if-statement, but make code less readable?
        //       for (x + direction * (z / tan_t)) * inv_step_pix
        //       and 2*(direction-1)* h_cos_t
        float xpr, zpr;
        if (direction > 0) {
            xpr = z / sin_t;
            zpr = (x - (z / tan_t)) * inv_step_pix;
        } else {
            xpr = z / sin_t - h_f;
            zpr = (x + (z / tan_t) - h_cos_t) * inv_step_pix;
        }

        // Determine the z0 plane index and check bounds, if so re-use thread with next point
        int z0 = (int)floorf(zpr);
        if (z0 < 0 || z0 >= n_planes - 1) {
            out[tid] = 0u;
            continue;
        }

        float beta = step_pix * (zpr - (float)z0);

        // NOTE, any speedup from storing ternary operator result since reused?
        // or have direction as a float?
        float xib = xpr + (direction > 0 ? +1.0f : -1.0f) * beta * tan_ot;
        int xb0 = (int)floorf(xib);
        if (xb0 < 0 || xb0 >= h - 1) {
            out[tid] = 0u;
            continue;
        }
        float dxib = xib - (float)xb0;

        float xia = xpr - (direction > 0 ? +1.0f : -1.0f) * (step_pix - beta) * tan_ot;
        int xa0 = (int)floorf(xia);
        if (xa0 < 0 || xa0 >= h - 1) {
            out[tid] = 0u;
            continue;
        }
        float dxia = xia - (float)xa0;

        float wb = beta / cos_ot;
        float wa = (step_pix - beta) / cos_ot;

        long long baseA = ((long long)z0     * ny + y) * h;
        long long baseB = ((long long)(z0+1) * ny + y) * h;

        float A0 = (float)im[baseA + xb0    ];
        float A1 = (float)im[baseA + xb0 + 1];
        float B0 = (float)im[baseB + xa0    ];
        float B1 = (float)im[baseB + xa0 + 1];

        float v_a = dxib * A1 + (1.0f - dxib) * A0;
        float v_b = dxia * B1 + (1.0f - dxia) * B0;
        float val = (wa * v_a + wb * v_b) * inv_step_pix;

        val = nearbyintf(val);
        if (val < 0.f)      val = 0.f;
        if (val > 65535.f)  val = 65535.f;
        out[tid] = (unsigned short)(val);
    }
}
'''.strip()


# Lazy-compiled kernel handles
_deskew_kernel = None       # float32-out
_deskew_kernel_u16 = None   # uint16-out


# -----------------------------------------------------------------------------
# Internal GPU helper
# -----------------------------------------------------------------------------

def _deskew_gpu(
    im: Union[np.ndarray, cp.ndarray],
    pixel_size: float,
    step_size: float,
    direction: int,
    theta: float,
    preserve_dtype: bool,
    stream: cp.cuda.Stream | None,
) -> cp.ndarray:
    """Run CUDA deskew kernel; return CuPy array (uint16 or float32)."""
    global _deskew_kernel, _deskew_kernel_u16

    xim = cp.asarray(im)  # host->device if needed
    assert xim.ndim == 3, f"Input must be a 3D array (z,y,x), received shape: {xim.shape}"
    n_planes, ny, h = xim.shape

    direction = 1 if direction >= 0 else -1

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    tan_t = math.tan(theta)
    otheta = (math.pi / 2.0) - theta
    tan_ot = math.tan(otheta)
    cos_ot = math.cos(otheta)

    step_pix = (step_size / cos_t) / pixel_size
    inv_step_pix = 1.0 / step_pix

    nx = int(math.ceil(n_planes * step_pix + h * cos_t))
    nz = int(math.ceil(h * sin_t))

    in_dtype = im.dtype if isinstance(im, np.ndarray) else xim.dtype
    want_u16 = preserve_dtype and (in_dtype == np.uint16 or in_dtype == cp.uint16)


    # SUGGESTION: Used 3D structure for kernel(?)
    total = nz * ny * nx
    threads = 256 #consider [8, 8 8] to get x,y,z values in the kernel instead of having to do % and / to get 3D from 1D
    blocks = min((total + threads - 1) // threads, 65535) # same [8, 8, 8] consideration idea for these blocks

    if want_u16:
        if _deskew_kernel_u16 is None:
            _deskew_kernel_u16 = cp.RawKernel(_deskew_kernel_u16_src, 'deskew_kernel_u16', options = ("-lineinfo",)) #compile for debugging with ncu for "source" tab
        out = cp.empty((nz, ny, nx), dtype=cp.uint16)
        args = (
            xim, out,
            np.int32(n_planes), np.int32(ny), np.int32(h),
            np.int32(nz), np.int32(nx),
            np.int32(direction),
            np.float32(sin_t), np.float32(cos_t), np.float32(tan_t),
            np.float32(tan_ot), np.float32(cos_ot),
            np.float32(step_pix), np.float32(inv_step_pix),
            np.float32(h), np.float32(h * cos_t),
        )
        if stream is None:
            _deskew_kernel_u16((blocks,), (threads,), args)
        else:
            with stream:
                _deskew_kernel_u16((blocks,), (threads,), args, stream=stream)
        return out  # already clipped & cast

    # float32 path
    if _deskew_kernel is None:
        _deskew_kernel = cp.RawKernel(_deskew_kernel_src, 'deskew_kernel')
    out = cp.empty((nz, ny, nx), dtype=cp.float32)
    args = (
        xim, out,
        np.int32(n_planes), np.int32(ny), np.int32(h),
        np.int32(nz), np.int32(nx),
        np.int32(direction),
        np.float32(sin_t), np.float32(cos_t), np.float32(tan_t),
        np.float32(tan_ot), np.float32(cos_ot),
        np.float32(step_pix), np.float32(inv_step_pix),
        np.float32(h), np.float32(h * cos_t),
    )
    if stream is None:
        _deskew_kernel((blocks,), (threads,), args)
    else:
        with stream:
            _deskew_kernel((blocks,), (threads,), args, stream=stream)

    if preserve_dtype:
        # Non-uint16 preserve path; generic cast (allocates extra temp if large).
        if np.issubdtype(in_dtype, np.integer):
            info = np.iinfo(in_dtype)
            out = cp.clip(cp.rint(out), info.min, info.max).astype(in_dtype)
        else:
            out = out.astype(in_dtype)
    return out


# -----------------------------------------------------------------------------
# Triangle zeroing (GPU)
# -----------------------------------------------------------------------------

def _zero_triangle_gpu(dsk: cp.ndarray, direction: int) -> cp.ndarray:
    """Zero lower-left triangular wrap artifact on GPU. dsk shape (z,y,x)."""
    if direction < 0:
        dsk = cp.flipud(dsk)
    nz, _, nx = dsk.shape
    stop = min(nz, nx)
    # Slice loop: cheap; avoids allocating large boolean masks.
    for z in range(stop):
        dsk[z, :, : z + 1] = 0
    dsk = cp.flipud(dsk)
    return dsk[..., ::direction]


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def deskew_stage_scan(
    im: NDArray,
    pixel_size: float,
    step_size: float,
    direction: int,
    theta: float = math.pi / 4,
    preserve_dtype: bool = False,
    stream: cp.cuda.Stream | None = None,
) -> cp.ndarray:
    """Deskew stage-scan data (GPU only).

    Parameters
    ----------
    im : array-like (z,y,x) raw stack (uint16 typical). NumPy or CuPy accepted.
    pixel_size : lateral pixel size (physical units).
    step_size  : stage step (same units).
    direction  : scan direction sign (+/-1).
    theta      : sheet tilt angle (rad).
    preserve_dtype : bool
        If True and input is uint16, result written directly as uint16 (memory-light).
        Otherwise float32 (analysis mode).
    stream : optional CuPy stream.

    Returns
    -------
    CuPy array (z,y,x) deskewed (dtype float32 or uint16).
    """
    dsk = _deskew_gpu(im, pixel_size, step_size, direction, theta, preserve_dtype, stream)
    dsk = _zero_triangle_gpu(dsk, direction)
    return dsk


# -----------------------------------------------------------------------------
# Shape utility
# -----------------------------------------------------------------------------


# Same as ortho.py:output_shape
def output_shape(
    z: int,
    y: int,
    x: int,
    pixel_size: float,
    step_size: float,
    theta: float = math.pi / 4,
) -> Tuple[int, int, int]:
    """Compute deskewed output shape for given input dims & geometry."""
    step_pix = (step_size / math.cos(theta)) / pixel_size
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    n_x = int(np.ceil(z * step_pix + x * cos_t))
    n_y = int(y)
    n_z = int(np.ceil(x * sin_t))
    return (n_z, n_y, n_x)