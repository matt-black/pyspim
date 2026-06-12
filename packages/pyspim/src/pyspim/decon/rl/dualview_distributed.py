"""Distributed deconvolution algorithms for dual-view microscopy data.

This module provides distributed implementations of Richardson-Lucy joint
deconvolution algorithms using nvmath-python's distributed FFT (cuFFTMp)
with NVSHMEM backend for multi-GPU multi-node computation.

References
---
[1] Wu, et. al., "Spatially isotropic four-dimensional...", doi:10.1038/nbt.2713
[2] Preibsch, et al. "Efficient Bayesian-based...", doi:10.1038/nmeth.2929
[3] Guo, et al. "Rapid image deconvolution..." doi:10.1038/s41587-020-0560-x
[4] Anconelli, B., et al. "Reduction of boundary effects...", doi:10.1051/0004-6361:20053848
"""

from __future__ import annotations

import atexit
import threading
from typing import Optional, Tuple

import cupy
import numpy
from cupyx.scipy.signal import fftconvolve as fftconv_gpu
from scipy.signal import fftconvolve as fftconv_cpu

from ..._util import supported_float_type
from ...typing import NDArray, PadType

# Lazy imports for nvmath.distributed - only imported when actually used
# to avoid import errors when nvmath-python is not installed
_nvmath_distributed = None
_nvmath_slab = None
_nvmath_fft = None


def _import_nvmath():
    """Lazily import nvmath.distributed and related modules."""
    global _nvmath_distributed, _nvmath_slab, _nvmath_fft
    if _nvmath_distributed is None:
        import nvmath.distributed
        import nvmath.distributed.fft
        from nvmath.distributed.distribution import Slab

        _nvmath_distributed = nvmath.distributed
        _nvmath_fft = nvmath.distributed.fft
        _nvmath_slab = Slab


# ---------------------------------------------------------------------------
# Initialization / Finalization
# ---------------------------------------------------------------------------

_initialize_mutex = threading.Lock()
_initialized = False


def initialize_distributed(
    device_id: int, comm, backends: list[str] | None = None
) -> None:
    """Initialize nvmath.distributed runtime.

    Must be called by ALL processes before using any distributed deconvolution
    functions. This is a collective operation.

    Args:
        device_id: CUDA device ID for this process.
        comm: MPI communicator (mpi4py.MPI.Comm).
        backends: Communication backends to use. Defaults to ["nvshmem"].

    Raises:
        ImportError: If nvmath-python is not installed.
        RuntimeError: If already initialized.
    """
    global _initialized

    _import_nvmath()

    if backends is None:
        backends = ["nvshmem"]

    with _initialize_mutex:
        if _initialized:
            raise RuntimeError(
                "nvmath.distributed has already been initialized. "
                "Call finalize_distributed() first if you need to reinitialize."
            )

        _nvmath_distributed.initialize(device_id, comm, backends=backends)
        atexit.register(finalize_distributed)
        _initialized = True


def finalize_distributed() -> None:
    """Finalize nvmath.distributed runtime.

    This is a collective operation called by all processes. Also registered
    with atexit for safety.
    """
    global _initialized

    with _initialize_mutex:
        if not _initialized:
            return
        _initialized = False

    if _nvmath_distributed is not None:
        _nvmath_distributed.finalize()


def is_distributed_initialized() -> bool:
    """Check if nvmath.distributed has been initialized."""
    return _initialized


def get_slab() -> type:
    """Get the Slab distribution class from nvmath.distributed."""
    _import_nvmath()
    return _nvmath_slab


# ---------------------------------------------------------------------------
# Distributed FFT Convolution Helper
# ---------------------------------------------------------------------------

def _distributed_fft_convolve(
    kernel_fft: cupy.ndarray,
    operand: cupy.ndarray,
    distribution: Slab,
    stream=None,
) -> cupy.ndarray:
    """Compute conv(kernel, operand, mode='same') using distributed FFT.

    Uses the convolution theorem: IFFT(FFT(kernel) * FFT(operand))

    The kernel's FFT is pre-computed on the full global volume shape and
    replicated on all ranks. The operand is distributed across processes.

    Args:
        kernel_fft: Pre-computed FFT of the kernel, shape = global volume shape.
            Replicated on all ranks, lives on local GPU memory.
        operand: Distributed operand on NVSHMEM symmetric heap, distributed
            according to `distribution`.
        distribution: Slab distribution of the input operand (e.g., Slab.X).
        stream: Optional CUDA stream for async execution.

    Returns:
        Distributed CuPy ndarray with the same distribution as the input operand.

    Note:
        For Slab.X input: FFT changes to Slab.Y, element-wise multiply preserves
        Slab.Y, IFFT changes back to Slab.X.
    """
    _import_nvmath()

    # Determine complementary distribution for the inverse FFT.
    # Slab.X -> Slab.Y, Slab.Y -> Slab.X
    partition_dim = distribution.partition_dim
    if partition_dim == 0:
        output_dist_after_fft = _nvmath_slab.Y
        inverse_dist = _nvmath_slab.Y
    elif partition_dim == 1:
        output_dist_after_fft = _nvmath_slab.X
        inverse_dist = _nvmath_slab.X
    else:
        # For Slab.Z or other dimensions, we need to handle differently
        # For now, default to reshaping back
        output_dist_after_fft = distribution
        inverse_dist = distribution

    fft_options = {"reshape": False}
    if stream is not None:
        fft_options["stream"] = stream

    # Step 1: Forward FFT of distributed operand
    # Slab.X -> Slab.Y (or complementary)
    operand_fft = _nvmath_fft.fft(
        operand, distribution=distribution, options=fft_options
    )

    # Step 2: Element-wise multiply with replicated kernel FFT
    # kernel_fft is replicated (broadcast automatically)
    product = operand_fft * kernel_fft

    # Step 3: Inverse FFT to get convolution result
    # Slab.Y -> Slab.X (back to original distribution)
    result = _nvmath_fft.ifft(
        product, distribution=inverse_dist, options=fft_options
    )

    return result


# ---------------------------------------------------------------------------
# Pre-compute PSF FFTs
# ---------------------------------------------------------------------------

def _precompute_psf_ffts(
    psf_a: NDArray,
    psf_b: NDArray,
    backproj_a: Optional[NDArray],
    backproj_b: Optional[NDArray],
    global_shape: Tuple[int, ...],
    float_type: numpy.dtype,
) -> Tuple[cupy.ndarray, cupy.ndarray, cupy.ndarray, cupy.ndarray]:
    """Pre-compute FFTs of PSFs and backprojectors.

    Each kernel is zero-padded to the global volume shape and its 3D FFT
    is computed locally on each rank. Results are replicated.

    Args:
        psf_a: Point spread function for view A.
        psf_b: Point spread function for view B.
        backproj_a: Backprojector for view A. If None, uses mirrored psf_a.
        backproj_b: Backprojector for view B. If None, uses mirrored psf_b.
        global_shape: Global (non-distributed) volume shape.
        float_type: Floating point dtype for computation.

    Returns:
        Tuple of (psf_a_fft, psf_b_fft, backproj_a_fft, backproj_b_fft).
    """
    # Ensure float type
    psf_a = numpy.asarray(psf_a, dtype=float_type)
    psf_b = numpy.asarray(psf_b, dtype=float_type)

    # Default back-projectors are mirrored PSFs
    if backproj_a is None:
        backproj_a = numpy.ascontiguousarray(psf_a[::-1, ::-1, ::-1])
    else:
        backproj_a = numpy.asarray(backproj_a, dtype=float_type)

    if backproj_b is None:
        backproj_b = numpy.ascontiguousarray(psf_b[::-1, ::-1, ::-1])
    else:
        backproj_b = numpy.asarray(backproj_b, dtype=float_type)

    def _pad_and_fft(kernel: numpy.ndarray) -> cupy.ndarray:
        """Zero-pad kernel to global shape and compute FFT."""
        padded = numpy.zeros(global_shape, dtype=float_type)
        # Place kernel at origin (matching fftconvolve convention)
        slices = tuple(slice(0, k) for k in kernel.shape)
        padded[slices] = kernel
        return cupy.fft.fftn(cupy.asarray(padded))

    psf_a_fft = _pad_and_fft(psf_a)
    psf_b_fft = _pad_and_fft(psf_b)
    backproj_a_fft = _pad_and_fft(backproj_a)
    backproj_b_fft = _pad_and_fft(backproj_b)

    return psf_a_fft, psf_b_fft, backproj_a_fft, backproj_b_fft


def _distributed_efficient_bayesian_backprojectors(
    psf_a: NDArray,
    psf_b: NDArray,
    float_type: numpy.dtype,
) -> Tuple[cupy.ndarray, cupy.ndarray]:
    """Calculate efficient Bayesian backprojectors.

    As described in [3], the backprojectors for "quadruple-view deconvolution"
    are computed as compound convolutions of the PSFs. Since PSFs are small,
    these convolutions are done locally using CuPy FFT.

    Args:
        psf_a: Point spread function for view A.
        psf_b: Point spread function for view B.
        float_type: Floating point dtype.

    Returns:
        Tuple of (bp_a, bp_b) backprojector arrays.
    """
    psf_a = cupy.asarray(psf_a, dtype=float_type)
    psf_b = cupy.asarray(psf_b, dtype=float_type)

    # Flipped PSFs
    flp_a = cupy.ascontiguousarray(psf_a[::-1, ::-1, ::-1])
    flp_b = cupy.ascontiguousarray(psf_b[::-1, ::-1, ::-1])

    # Compound backprojectors: bp_a = flp_a * conv(conv(flp_a, psf_b), flp_b)
    conv_ab = fftconv_gpu(fftconv_gpu(flp_a, psf_b, mode="same"), flp_b, mode="same")
    bp_a = flp_a * conv_ab

    conv_ba = fftconv_gpu(fftconv_gpu(flp_b, psf_a, mode="same"), flp_a, mode="same")
    bp_b = flp_b * conv_ba

    return bp_a, bp_b


# ---------------------------------------------------------------------------
# Global Reduction Helpers
# ---------------------------------------------------------------------------

def _distributed_sum(arr: cupy.ndarray) -> float:
    """Compute global sum of a distributed array.

    Uses MPI allreduce to sum the local CuPy sum across all ranks.

    Args:
        arr: Distributed CuPy array on symmetric heap.

    Returns:
        Global sum as a Python float.
    """
    import mpi4py.MPI

    comm = mpi4py.MPI.COMM_WORLD
    local_sum = float(cupy.sum(arr).get())
    global_sum = comm.allreduce(local_sum, op=mpi4py.MPI.SUM)
    return global_sum


def _div_stable_distributed(
    a: cupy.ndarray, b: cupy.ndarray, eps: float = 1e-5
) -> cupy.ndarray:
    """Element-wise stable division for distributed arrays.

    Computes a / b where b > eps, else 0.

    Args:
        a: Numerator (distributed).
        b: Denominator (distributed).
        eps: Stability threshold.

    Returns:
        Distributed result array.
    """
    return cupy.where(b > eps, a / b, 0.0)


# ---------------------------------------------------------------------------
# Helper: Get local shape for a given global shape and Slab distribution
# ---------------------------------------------------------------------------

def _get_local_shape(
    global_shape: Tuple[int, ...], rank: int, nranks: int, partition_dim: int = 0
) -> Tuple[int, ...]:
    """Calculate local shape for Slab distribution.

    Args:
        global_shape: Global tensor shape.
        rank: Process rank.
        nranks: Total number of processes.
        partition_dim: Axis along which data is partitioned.

    Returns:
        Local shape on this rank.
    """
    local_shape = list(global_shape)
    total = global_shape[partition_dim]
    base = total // nranks
    remainder = total % nranks
    local_shape[partition_dim] = base + (1 if rank < remainder else 0)
    return tuple(local_shape)


def _get_global_offset(
    global_shape: Tuple[int, ...], rank: int, nranks: int, partition_dim: int = 0
) -> int:
    """Calculate global offset for this rank's slab.

    Args:
        global_shape: Global tensor shape.
        rank: Process rank.
        nranks: Total number of processes.
        partition_dim: Axis along which data is partitioned.

    Returns:
        Starting index of this rank's slab along the partition dimension.
    """
    total = global_shape[partition_dim]
    base = total // nranks
    remainder = total % nranks
    if rank < remainder:
        return rank * (base + 1)
    else:
        return remainder * (base + 1) + (rank - remainder) * base
