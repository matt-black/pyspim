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
from cupyx.scipy.signal import fftconvolve

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
    global_shape: Tuple[int, ...] | None = None,
    stream=None,
) -> cupy.ndarray:
    """Compute conv(kernel, operand, mode='same') using distributed FFT.

    Uses the convolution theorem: IFFT(FFT(kernel) * FFT(operand)) / N

    The kernel's FFT is pre-computed on the full global volume shape and
    replicated on all ranks. The operand is distributed across processes
    on NVSHMEM symmetric heap.

    Note: nvmath.distributed.fft() only supports C2C (complex-to-complex)
    transforms and requires GPU operand on symmetric memory. For real-valued
    input, a temporary complex symmetric buffer is allocated.

    Note: cuFFT (which nvmath wraps) follows the unnormalized convention
    where ifft() does NOT apply 1/N. We apply this normalization manually
    to match scipy.signal.fftconvolve convention.

    Args:
        kernel_fft: Pre-computed FFT of the kernel, shape = global volume shape.
            Replicated on all ranks, lives on local GPU memory. Must be complex.
        operand: Distributed operand on NVSHMEM symmetric heap, distributed
            according to `distribution`. Can be real or complex.
        distribution: Slab distribution of the input operand (e.g., Slab.X).
        global_shape: Global shape of the distributed operand. Required for
            normalization. If None, will be inferred from kernel_fft.shape.
        stream: Optional CUDA stream for async execution.

    Returns:
        Distributed CuPy ndarray with the same distribution as the input operand.
        If input was real, output is real; if input was complex, output is complex.

    Note:
        Uses reshape=True to keep Slab distribution consistent between FFT and IFFT,
        avoiding stride mismatches that occur with reshape=False.
    """
    _import_nvmath()

    # Determine global shape for normalization
    if global_shape is None:
        global_shape = kernel_fft.shape
    n_elements = cupy.prod(cupy.asarray(global_shape, dtype=cupy.float64))

    # Use reshape=True so both FFT and IFFT keep the same distribution.
    # With reshape=False, FFT changes Slab.X -> Slab.Y (different strides), and
    # the element-wise product array would have C-order strides that don't match
    # the Slab.Y layout expected by IFFT, causing numerical corruption.
    fft_options = {"reshape": True}
    if stream is not None:
        fft_options["stream"] = stream

    # nvmath.distributed.fft() only supports C2C (complex-to-complex) and requires
    # operand on symmetric memory. cupy.astype() does NOT preserve symmetric memory,
    # so we need to explicitly allocate complex symmetric memory and copy.
    was_real = not cupy.issubdtype(operand.dtype, cupy.complexfloating)
    fft_operand = operand
    temp_buf = None

    if was_real:
        # Allocate complex symmetric memory and copy real data into it
        if operand.dtype == cupy.float32:
            complex_dtype = cupy.complex64
        else:
            complex_dtype = cupy.complex128

        temp_buf = _nvmath_distributed.allocate_symmetric_memory(
            operand.shape, cupy, dtype=complex_dtype
        )
        temp_buf[:] = operand
        fft_operand = temp_buf

    # Step 1: Forward FFT of distributed operand (in-place on symmetric memory)
    # With reshape=True, output keeps the same Slab distribution
    operand_fft = _nvmath_fft.fft(
        fft_operand, distribution=distribution, options=fft_options
    )

    # Step 2: Element-wise multiply with replicated kernel FFT, writing back
    # into the same symmetric buffer to preserve correct strides for IFFT.
    fft_operand[:] = operand_fft * kernel_fft

    # Step 3: Inverse FFT to get convolution result (in-place on same symmetric memory)
    # With reshape=True, output keeps the same Slab distribution
    result = _nvmath_fft.ifft(
        fft_operand, distribution=distribution, options=fft_options
    )

    # Step 4: Apply 1/N normalization to match scipy.signal.fftconvolve convention
    # cuFFT's IFFT does NOT apply 1/N, so we do it here
    result = result / n_elements

    # Free temporary symmetric memory
    if temp_buf is not None:
        _nvmath_distributed.free_symmetric_memory(temp_buf)

    # Extract real part if input was real
    if was_real:
        result = cupy.real(result)

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
    # Note: fftconvolve expects the larger array as the first argument for efficiency.
    # Here the inner result (convolved PSF-sized volume) is the larger operand,
    # while the outer kernel (flp_b) is the smaller PSF.
    conv_ab = fftconvolve(flp_b, fftconvolve(psf_b, flp_a, mode="same"), mode="same")
    bp_a = flp_a * conv_ab

    conv_ba = fftconvolve(flp_a, fftconvolve(psf_a, flp_b, mode="same"), mode="same")
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


# ---------------------------------------------------------------------------
# Distributed diSPIM Joint RL
# ---------------------------------------------------------------------------

def distributed_joint_rl_dispim(
    view_a: NDArray,
    view_b: NDArray,
    est_i: Optional[NDArray],
    psf_a: NDArray,
    psf_b: NDArray,
    backproj_a: Optional[NDArray] = None,
    backproj_b: Optional[NDArray] = None,
    num_iter: int = 10,
    epsilon: float = 1e-5,
    boundary_correction: bool = False,
    req_both: bool = False,
    zero_padding: Optional[PadType] = None,
    boundary_sigma_a: float = 1e-2,
    boundary_sigma_b: float = 1e-2,
    verbose: bool = False,
) -> NDArray:
    """Distributed diSPIM joint Richardson-Lucy deconvolution.

    Matches the API of :func:`pyspim.decon.rl.dualview_fft.joint_rl_dispim`.

    Args:
        view_a: Data array for view A (distributed on Slab.X symmetric memory).
        view_b: Data array for view B (distributed on Slab.X symmetric memory).
        est_i: Initial estimate. If None, uses (view_a + view_b) / 2.
        psf_a: Point spread function for view A (replicated on all ranks).
        psf_b: Point spread function for view B (replicated on all ranks).
        backproj_a: Backprojector for view A. If None, uses mirrored psf_a.
        backproj_b: Backprojector for view B. If None, uses mirrored psf_b.
        num_iter: Number of iterations.
        epsilon: Small parameter to prevent division by zero.
        boundary_correction: If True, use OSEM boundary correction.
        req_both: Zero out areas where either view has no data.
        zero_padding: Padding for boundary correction.
        boundary_sigma_a: Threshold for view A window function.
        boundary_sigma_b: Threshold for view B window function.
        verbose: Display progress bar.

    Returns:
        Deconvolved estimate (distributed on Slab.X symmetric memory).
    """
    _import_nvmath()
    Slab = _nvmath_slab

    import mpi4py.MPI

    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    # Ensure float type
    float_type = supported_float_type(view_a.dtype)
    view_a = cupy.asarray(view_a, dtype=float_type)
    view_b = cupy.asarray(view_b, dtype=float_type)

    # Get global shape from the distributed view arrays
    # Reconstruct global shape from local shape
    local_shape = view_a.shape
    global_shape = list(local_shape)
    global_shape[0] = 0
    for r in range(nranks):
        r_local = _get_local_shape(tuple(global_shape) if global_shape[0] else local_shape, r, nranks, 0)
        # Sum up partition dimension across all ranks
        pass
    # Simpler: all-gather the local partition sizes
    local_partition_size = local_shape[0]
    total_partition_size = comm.allreduce(local_partition_size, op=mpi4py.MPI.SUM)
    global_shape = list(local_shape)
    global_shape[0] = total_partition_size
    global_shape = tuple(global_shape)

    # Initialize estimate
    if est_i is None:
        est = (view_a + view_b) / 2.0
    else:
        est = cupy.asarray(est_i, dtype=float_type)

    # Pre-compute PSF FFTs
    psf_a_fft, psf_b_fft, backproj_a_fft, backproj_b_fft = _precompute_psf_ffts(
        psf_a, psf_b, backproj_a, backproj_b, global_shape, float_type
    )

    if boundary_correction:
        return _distributed_joint_rl_osem(
            view_a, view_b, est,
            psf_a_fft, psf_b_fft, backproj_a_fft, backproj_b_fft,
            global_shape, float_type,
            num_iter, epsilon, zero_padding,
            boundary_sigma_a, boundary_sigma_b,
            req_both, verbose,
        )
    else:
        return _distributed_joint_rl_dispim_uncorr(
            view_a, view_b, est,
            psf_a_fft, psf_b_fft, backproj_a_fft, backproj_b_fft,
            global_shape, float_type,
            num_iter, epsilon, req_both, verbose,
        )


def _distributed_joint_rl_dispim_uncorr(
    view_a: cupy.ndarray,
    view_b: cupy.ndarray,
    est: cupy.ndarray,
    psf_a_fft: cupy.ndarray,
    psf_b_fft: cupy.ndarray,
    backproj_a_fft: cupy.ndarray,
    backproj_b_fft: cupy.ndarray,
    global_shape: Tuple[int, ...],
    float_type: numpy.dtype,
    num_iter: int,
    epsilon: float,
    req_both: bool,
    verbose: bool,
) -> cupy.ndarray:
    """Distributed diSPIM joint RL without boundary correction.

    Sequential update: view A updates estimate, then view B updates from result.
    """
    _import_nvmath()
    Slab = _nvmath_slab

    # Import tqdm for progress bar (optional)
    try:
        from tqdm.auto import trange
        use_tqdm = True
    except ImportError:
        trange = range
        use_tqdm = False

    import mpi4py.MPI
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()

    iter_range = trange(num_iter) if (verbose and use_tqdm and rank == 0) else range(num_iter)

    for _ in iter_range:
        # cona = conv(psf_a, est)
        cona = _distributed_fft_convolve(psf_a_fft, est, Slab.X, global_shape)
        # ratio_a = view_a / cona (stable division)
        ratio_a = _div_stable_distributed(view_a, cona, epsilon)
        # est_a = est * conv(backproj_a, ratio_a)
        est_a = est * _distributed_fft_convolve(backproj_a_fft, ratio_a, Slab.X, global_shape)

        # conb = conv(psf_b, est_a)
        conb = _distributed_fft_convolve(psf_b_fft, est_a, Slab.X, global_shape)
        # ratio_b = view_b / conb (stable division)
        ratio_b = _div_stable_distributed(view_b, conb, epsilon)
        # est = est_a * conv(backproj_b, ratio_b)
        est = est_a * _distributed_fft_convolve(backproj_b_fft, ratio_b, Slab.X, global_shape)

    if req_both:
        est[cupy.logical_or(view_a == 0, view_b == 0)] = 0

    return est


def _distributed_joint_rl_osem(
    view_a: cupy.ndarray,
    view_b: cupy.ndarray,
    est: cupy.ndarray,
    psf_a_fft: cupy.ndarray,
    psf_b_fft: cupy.ndarray,
    backproj_a_fft: cupy.ndarray,
    backproj_b_fft: cupy.ndarray,
    global_shape: Tuple[int, ...],
    float_type: numpy.dtype,
    num_iter: int,
    epsilon: float,
    zero_padding: Optional[PadType],
    boundary_sigma_a: float,
    boundary_sigma_b: float,
    req_both: bool,
    verbose: bool,
) -> cupy.ndarray:
    """Distributed diSPIM joint RL with OSEM boundary correction.

    Uses ordered subset EM where each sub-iteration uses one view,
    followed by flux rescaling via global reduction.
    """
    _import_nvmath()
    Slab = _nvmath_slab

    import mpi4py.MPI
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    # Import tqdm for progress bar (optional)
    try:
        from tqdm.auto import trange
        use_tqdm = True
    except ImportError:
        trange = range
        use_tqdm = False

    # Determine padding
    if zero_padding is None:
        pad = tuple([(d // 2, d // 2) for d in global_shape])
    else:
        if isinstance(zero_padding, int):
            pad = tuple([(zero_padding, zero_padding) for _ in global_shape])
        else:
            assert len(zero_padding) == len(global_shape), (
                "zero padding must be specified for all dimensions of input"
            )
            pad = tuple([(p, p) for p in zero_padding])

    # Compute padded global shape
    padded_global_shape = tuple(d + pad[i][0] + pad[i][1] for i, d in enumerate(global_shape))

    # Allocate padded distributed arrays on symmetric memory
    padded_local_shape = _get_local_shape(padded_global_shape, rank, nranks, 0)

    # Pad the views, estimate, and create ones masks
    # Each rank pads its local slab
    view_a_padded = _distributed_pad_array(view_a, pad, padded_global_shape, global_shape, float_type)
    view_b_padded = _distributed_pad_array(view_b, pad, padded_global_shape, global_shape, float_type)
    est_padded = _distributed_pad_array(est, pad, padded_global_shape, global_shape, float_type)

    # Create ones masks (distributed) for window computation
    M_a = _distributed_ones(padded_global_shape, float_type)
    M_b = _distributed_ones(padded_global_shape, float_type)

    # Compute flux constant (global reduction)
    c = _distributed_sum((view_a_padded + view_b_padded) / 2.0)

    # Calculate window functions: alpha = conv(backproj, ones)
    alpha_a = _distributed_fft_convolve(backproj_a_fft, M_a, Slab.X, padded_global_shape)
    alpha_b = _distributed_fft_convolve(backproj_b_fft, M_b, Slab.X, padded_global_shape)
    window_a = cupy.where(alpha_a > boundary_sigma_a, 1.0 / alpha_a, 0.0)
    window_b = cupy.where(alpha_b > boundary_sigma_b, 1.0 / alpha_b, 0.0)

    # Free masks
    _nvmath_distributed.free_symmetric_memory(M_a)
    _nvmath_distributed.free_symmetric_memory(M_b)

    # Compute combined alpha for flux rescaling
    alpha = alpha_a + alpha_b
    _nvmath_distributed.free_symmetric_memory(alpha_a)
    _nvmath_distributed.free_symmetric_memory(alpha_b)

    # Pre-compute PSF FFTs for padded shape
    psf_a_fft_pad, psf_b_fft_pad, bp_a_fft_pad, bp_b_fft_pad = _precompute_psf_ffts_for_padded(
        psf_a_fft, psf_b_fft, backproj_a_fft, backproj_b_fft,
        padded_global_shape, global_shape,
    )

    iter_range = trange(num_iter) if (verbose and use_tqdm and rank == 0) else range(num_iter)

    for _ in iter_range:
        # Sub-iteration A
        con = _distributed_fft_convolve(psf_a_fft_pad, est_padded, Slab.X, padded_global_shape)
        ratio = _div_stable_distributed(view_a_padded, con, epsilon)
        est_padded = window_a * est_padded * _distributed_fft_convolve(
            bp_a_fft_pad, ratio, Slab.X, padded_global_shape
        )
        # Flux rescale
        c_tilde = _distributed_sum(alpha * est_padded) / 2.0
        est_padded = (c / c_tilde) * est_padded

        # Sub-iteration B
        con = _distributed_fft_convolve(psf_b_fft_pad, est_padded, Slab.X, padded_global_shape)
        ratio = _div_stable_distributed(view_b_padded, con, epsilon)
        est_padded = window_b * est_padded * _distributed_fft_convolve(
            bp_b_fft_pad, ratio, Slab.X, padded_global_shape
        )
        # Flux rescale
        c_tilde = _distributed_sum(alpha * est_padded) / 2.0
        est_padded = (c / c_tilde) * est_padded

    if req_both:
        est_padded[cupy.logical_or(view_a_padded == 0, view_b_padded == 0)] = 0

    # Trim padding from each rank's local slab
    result = _distributed_trim_array(est_padded, pad, global_shape)

    # Cleanup padded arrays
    _nvmath_distributed.free_symmetric_memory(view_a_padded)
    _nvmath_distributed.free_symmetric_memory(view_b_padded)
    _nvmath_distributed.free_symmetric_memory(est_padded)
    _nvmath_distributed.free_symmetric_memory(window_a)
    _nvmath_distributed.free_symmetric_memory(window_b)
    _nvmath_distributed.free_symmetric_memory(alpha)

    return result


def _distributed_pad_array(
    arr: cupy.ndarray,
    pad: Tuple[Tuple[int, int], ...],
    padded_global_shape: Tuple[int, ...],
    original_global_shape: Tuple[int, ...],
    dtype: numpy.dtype,
) -> cupy.ndarray:
    """Pad a distributed array by allocating a new symmetric buffer with zero padding.

    Each rank copies its local data to the center region of the padded buffer.
    """
    _import_nvmath()

    import mpi4py.MPI
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    padded_local_shape = _get_local_shape(padded_global_shape, rank, nranks, 0)

    padded_arr = _nvmath_distributed.allocate_symmetric_memory(
        padded_local_shape, cupy, dtype=dtype
    )
    padded_arr[:] = 0.0

    # Compute the slice within the padded array where this rank's data belongs
    # The global offset in the padded array for this rank's original data
    global_offset_orig = _get_global_offset(original_global_shape, rank, nranks, 0)
    # In the padded array, this data starts at pad[0][0] + global_offset_orig
    padded_offset = pad[0][0] + global_offset_orig
    local_shape = arr.shape

    # Compute the slice within the padded local array
    padded_global_offset = _get_global_offset(padded_global_shape, rank, nranks, 0)

    # The local data needs to go from padded_offset to padded_offset + local_shape[0]
    # But we need to map this to the local padded array's coordinates
    local_start = padded_offset - padded_global_offset
    local_end = local_start + local_shape[0]

    # Clamp to local bounds
    if local_start < 0 or local_end > padded_local_shape[0]:
        # This rank's padded slab spans the boundary where data lives on another rank
        # For simplicity, handle the case where the original data falls entirely within
        # this rank's padded slab
        if local_start < 0:
            # Part of the data is on the previous rank - should not happen for reasonable padding
            pass
        actual_start = max(0, local_start)
        actual_end = min(padded_local_shape[0], local_end)
        data_start = max(0, -local_start)
        data_end = data_start + (actual_end - actual_start)
        padded_arr[actual_start:actual_end, :, :] = arr[data_start:data_end, :, :]
    else:
        padded_arr[local_start:local_end, :, :] = arr

    return padded_arr


def _distributed_trim_array(
    padded_arr: cupy.ndarray,
    pad: Tuple[Tuple[int, int], ...],
    original_global_shape: Tuple[int, ...],
) -> cupy.ndarray:
    """Trim padding from a distributed array.

    Each rank extracts the center region corresponding to the original volume.
    """
    _import_nvmath()

    import mpi4py.MPI
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    padded_global_shape = tuple(
        d + pad[i][0] + pad[i][1] for i, d in enumerate(original_global_shape)
    )
    padded_global_offset = _get_global_offset(padded_global_shape, rank, nranks, 0)

    # The trimmed data for the original global shape starts at pad[0][0] in the padded array
    # Find which rank(s) own this data and extract the relevant portion
    local_shape = _get_local_shape(original_global_shape, rank, nranks, 0)

    trimmed = _nvmath_distributed.allocate_symmetric_memory(
        local_shape, cupy, dtype=padded_arr.dtype
    )

    # The local start in the padded array for the trimmed data
    trimmed_global_offset = _get_global_offset(original_global_shape, rank, nranks, 0)
    padded_start = pad[0][0] + trimmed_global_offset
    local_padded_start = padded_start - padded_global_offset
    local_padded_end = local_padded_start + local_shape[0]

    trimmed[:, :, :] = padded_arr[local_padded_start:local_padded_end, :, :]

    return trimmed


def _distributed_ones(
    global_shape: Tuple[int, ...], dtype: numpy.dtype
) -> cupy.ndarray:
    """Create a distributed array filled with ones on symmetric memory."""
    _import_nvmath()

    import mpi4py.MPI
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    local_shape = _get_local_shape(global_shape, rank, nranks, 0)
    arr = _nvmath_distributed.allocate_symmetric_memory(
        local_shape, cupy, dtype=dtype
    )
    arr[:] = 1.0
    return arr


def _precompute_psf_ffts_for_padded(
    psf_a_fft: cupy.ndarray,
    psf_b_fft: cupy.ndarray,
    backproj_a_fft: cupy.ndarray,
    backproj_b_fft: cupy.ndarray,
    padded_global_shape: Tuple[int, ...],
    original_global_shape: Tuple[int, ...],
) -> Tuple[cupy.ndarray, cupy.ndarray, cupy.ndarray, cupy.ndarray]:
    """Re-compute PSF FFTs for the padded global shape.

    The PSFs themselves don't change, but their FFTs need to be computed
    at the padded volume shape for the OSEM convolution.
    """
    def _pad_and_fft_at_shape(kernel_fft: cupy.ndarray, orig_shape: Tuple[int, ...], new_shape: Tuple[int, ...]) -> cupy.ndarray:
        """IFFT the original FFT to get the kernel, then re-FFT at new shape."""
        # Get the original padded kernel (which is at orig_shape)
        n_orig = cupy.prod(cupy.asarray(orig_shape, dtype=cupy.float64))
        kernel = cupy.real(cupy.fft.ifftn(kernel_fft) * n_orig)

        # Zero-pad to new shape and recompute FFT
        padded = cupy.zeros(new_shape, dtype=kernel.dtype)
        slices = tuple(slice(0, k) for k in kernel.shape)
        padded[slices] = kernel
        return cupy.fft.fftn(padded)

    psf_a_fft_pad = _pad_and_fft_at_shape(psf_a_fft, original_global_shape, padded_global_shape)
    psf_b_fft_pad = _pad_and_fft_at_shape(psf_b_fft, original_global_shape, padded_global_shape)
    bp_a_fft_pad = _pad_and_fft_at_shape(backproj_a_fft, original_global_shape, padded_global_shape)
    bp_b_fft_pad = _pad_and_fft_at_shape(backproj_b_fft, original_global_shape, padded_global_shape)

    return psf_a_fft_pad, psf_b_fft_pad, bp_a_fft_pad, bp_b_fft_pad
