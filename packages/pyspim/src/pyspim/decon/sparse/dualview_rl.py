"""Sparse, Structurally-Regularized, Dual-View Richardson-Lucy deconvolution.

Provides both whole-volume deconvolution (`deconvolve`) and chunk-by-chunk
deconvolution (`deconvolve_chunkwise`) with support for CPU (numpy/scipy)
and GPU (cupy/cupyx).
"""

import concurrent.futures
import gc
import multiprocessing
import os
from contextlib import nullcontext
from functools import partial
from itertools import repeat
from typing import Iterable, Optional, Tuple

import cupy
import numpy
import zarr
from cupyx.scipy.signal import fftconvolve as fftconv_gpu
from scipy.signal import fftconvolve as fftconv_cpu
from tqdm.auto import tqdm, trange

from ..._util import supported_float_type, get_ndimage_module
from ...typing import NDArray
from .._util import ChunkProps, calculate_conv_chunks, div_stable


def safe_div_stable(a: NDArray, b: NDArray, eps: float) -> NDArray:
    """Perform stable element-wise division, avoiding division by zero/small values."""
    xp = cupy.get_array_module(a)
    if xp == cupy:
        return div_stable(a, b, eps)
    else:
        return xp.where(b > eps, a / b, 0.0)


def _convolve_derivative(v: NDArray, axis_a: int, axis_b: int) -> NDArray:
    """Compute second-order partial derivatives along axis_a and axis_b using 1D convolutions."""
    xp = cupy.get_array_module(v)
    ndi = get_ndimage_module(v)

    if 0 in v.shape:
        return xp.zeros_like(v)

    if axis_a == axis_b:
        # Second derivative along one axis: kernel [1.0, -2.0, 1.0]
        weights = numpy.array([1.0, -2.0, 1.0], dtype=numpy.float32)
        if xp == cupy:
            weights = cupy.asarray(weights)
        return ndi.convolve1d(v, weights, axis=axis_a, mode="reflect")
    else:
        # Mixed partial derivative: [0.5, 0.0, -0.5] along both axes
        weights = numpy.array([0.5, 0.0, -0.5], dtype=numpy.float32)
        if xp == cupy:
            weights = cupy.asarray(weights)
        res = ndi.convolve1d(v, weights, axis=axis_a, mode="reflect")
        return ndi.convolve1d(res, weights, axis=axis_b, mode="reflect")


def _deconvolve_single_channel(
    view_a: NDArray,
    view_b: NDArray,
    est_i: NDArray | None,
    psf_a: NDArray,
    psf_b: NDArray,
    backproj_a: NDArray | None,
    backproj_b: NDArray | None,
    num_iter: int,
    epsilon: float,
    lambda1: float,
    lambda2: float,
    epsilon_hess: float,
    req_both: bool,
    verbose: bool,
) -> NDArray:
    xp = cupy.get_array_module(view_a)
    float_type = supported_float_type(view_a.dtype)

    view_a = view_a.astype(float_type, copy=False)
    view_b = view_b.astype(float_type, copy=False)
    psf_a = psf_a.astype(float_type, copy=False)
    psf_b = psf_b.astype(float_type, copy=False)

    if backproj_a is None:
        backproj_a = xp.ascontiguousarray(psf_a[::-1, ::-1, ::-1])
    else:
        backproj_a = backproj_a.astype(float_type, copy=False)

    if backproj_b is None:
        backproj_b = xp.ascontiguousarray(psf_b[::-1, ::-1, ::-1])
    else:
        backproj_b = backproj_b.astype(float_type, copy=False)

    if est_i is None:
        est = (view_a + view_b) / 2.0
    else:
        est = est_i.astype(float_type, copy=True)

    if xp == cupy:
        conv = lambda k, i: fftconv_gpu(i, k, mode="same")
    else:
        conv = lambda k, i: fftconv_cpu(i, k, mode="same")

    # Precompute normalisation factors: H_1^T 1 and H_2^T 1
    norm_a = conv(backproj_a, xp.ones_like(view_a))
    norm_b = conv(backproj_b, xp.ones_like(view_b))
    norm_total = norm_a + norm_b

    for _ in trange(num_iter) if verbose else range(num_iter):
        con_a = conv(psf_a, est)
        con_b = conv(psf_b, est)

        ratio_a = safe_div_stable(view_a, con_a, epsilon)
        ratio_b = safe_div_stable(view_b, con_b, epsilon)

        bp_ratio_a = conv(backproj_a, ratio_a)
        bp_ratio_b = conv(backproj_b, ratio_b)

        G_hess = xp.zeros_like(est)
        if lambda2 > 0:
            # We calculate 6 unique combinations of Hessian coordinates
            # and weight cross-terms by 2.
            u_list = []
            axes = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]

            # 1. Compute u = D_ab * est and sum of squares
            ss = xp.zeros_like(est)
            for axis_a, axis_b in axes:
                u = _convolve_derivative(est, axis_a, axis_b)
                u_list.append(u)
                if axis_a == axis_b:
                    ss += u**2
                else:
                    ss += 2.0 * (u**2)

            # 2. Shared denominator (Frobenius norm of Hessian)
            denom = xp.sqrt(ss + epsilon_hess)

            # 3. Compute D_ab^T * v where v = D_ab * est / denom
            for i, (axis_a, axis_b) in enumerate(axes):
                u = u_list[i]
                v = u / denom
                w = _convolve_derivative(v, axis_a, axis_b)
                if axis_a == axis_b:
                    G_hess += w
                else:
                    G_hess += 2.0 * w

            G_hess *= lambda2

        G_hess_pos = xp.maximum(G_hess, 0)
        G_hess_neg = xp.maximum(-G_hess, 0)

        numerator = bp_ratio_a + bp_ratio_b + G_hess_neg
        denominator = norm_total + lambda1 + G_hess_pos

        est = est * safe_div_stable(numerator, denominator, epsilon)

    if req_both:
        est[xp.logical_or(view_a == 0, view_b == 0)] = 0

    return est


def _deconvolve_multichannel(
    view_a: NDArray,
    view_b: NDArray,
    est_i: NDArray | None,
    psf_a: NDArray | Iterable[NDArray],
    psf_b: NDArray | Iterable[NDArray],
    backproj_a: NDArray | Iterable[NDArray] | None,
    backproj_b: NDArray | Iterable[NDArray] | None,
    num_iter: int,
    epsilon: float,
    lambda1: float,
    lambda2: float,
    epsilon_hess: float,
    req_both: bool,
    verbose: bool,
) -> NDArray:
    xp = cupy.get_array_module(view_a)
    n_chan = view_a.shape[0]

    psf_a = (
        repeat(psf_a) if isinstance(psf_a, (numpy.ndarray, cupy.ndarray)) 
        else psf_a
    )
    psf_b = (
        repeat(psf_b) if isinstance(psf_b, (numpy.ndarray, cupy.ndarray)) 
        else psf_b
    )

    if backproj_a is None or isinstance(backproj_a, (numpy.ndarray, cupy.ndarray)):
        backproj_a = repeat(backproj_a)
    if backproj_b is None or isinstance(backproj_b, (numpy.ndarray, cupy.ndarray)):
        backproj_b = repeat(backproj_b)

    pfun = partial(
        _deconvolve_single_channel,
        num_iter=num_iter,
        epsilon=epsilon,
        lambda1=lambda1,
        lambda2=lambda2,
        epsilon_hess=epsilon_hess,
        req_both=req_both,
        verbose=verbose,
    )

    return xp.stack(
        [
            pfun(
                view_a[i, ...],
                view_b[i, ...],
                (est_i[i, ...] if est_i is not None else None),
                pa,
                pb,
                bpa,
                bpb,
            )
            for i, pa, pb, bpa, bpb in zip(
                range(n_chan), psf_a, psf_b, backproj_a, backproj_b
            )
        ],
        axis=0,
    )


def deconvolve(
    view_a: NDArray,
    view_b: NDArray,
    est_i: NDArray | None,
    psf_a: NDArray | Iterable[NDArray],
    psf_b: NDArray | Iterable[NDArray],
    backproj_a: Optional[NDArray | Iterable[NDArray]] = None,
    backproj_b: Optional[NDArray | Iterable[NDArray]] = None,
    num_iter: int = 10,
    epsilon: float = 1e-5,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    epsilon_hess: float = 1e-5,
    req_both: bool = False,
    verbose: bool = False,
) -> NDArray:
    """Joint deconvolution of 2 views into a single one using Sparse, Hessian-regularized RL.

    Args:
        view_a (NDArray): array from 1st view
        view_b (NDArray): array from 2nd view
        est_i (NDArray | None): initial estimate
        psf_a (NDArray | Iterable[NDArray]): psf arrays for view A (single one, or 1 per channel)
        psf_b (NDArray | Iterable[NDArray]): psf arrays for view B (single, or 1 per channel)
        backproj_a (NDArray | Iterable[NDArray], optional): backprojector array(s) for view A
        backproj_b (NDArray | Iterable[NDArray], optional): backprojector array(s) for view B
        num_iter (int): number of iterations
        epsilon (float): small parameter to prevent division by zero
        lambda1 (float): L1 sparsity prior coefficient
        lambda2 (float): Structural Hessian penalty coefficient
        epsilon_hess (float): Small smoothing parameter for Hessian differentiation
        req_both (bool): set all areas where either view doesn't have data to 0
        verbose (bool): show progressbar for iterations

    Raises:
        ValueError: input array is not 3 or 4D

    Returns:
        NDArray
    """
    if len(view_a.shape) == 4:
        return _deconvolve_multichannel(
            view_a,
            view_b,
            est_i,
            psf_a,
            psf_b,
            backproj_a,
            backproj_b,
            num_iter,
            epsilon,
            lambda1,
            lambda2,
            epsilon_hess,
            req_both,
            verbose,
        )
    elif len(view_a.shape) == 3:
        return _deconvolve_single_channel(
            view_a,
            view_b,
            est_i,
            psf_a,
            psf_b,
            backproj_a,
            backproj_b,
            num_iter,
            epsilon,
            lambda1,
            lambda2,
            epsilon_hess,
            req_both,
            verbose,
        )
    else:
        raise ValueError("invalid shape, must be 3- or 4D")


def deconvolve_chunkwise(
    view_a: zarr.Array,
    view_b: zarr.Array,
    out: zarr.Array,
    chunk_size: int | Tuple[int, int, int],
    overlap: int | Tuple[int, int, int],
    psf_a: numpy.ndarray,
    psf_b: numpy.ndarray,
    bp_a: numpy.ndarray,
    bp_b: numpy.ndarray,
    num_iter: int,
    epsilon: float,
    lambda1: float,
    lambda2: float,
    epsilon_hess: float,
    verbose: bool,
    decon_function: str = "sparse",  # for signature compatibility
):
    """Joint deconvolution of the input views A & B, done chunk-by-chunk on available GPUs."""
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(
        "[deconvolve_chunkwise] volume=%s, chunk_size=%s, overlap=%s, "
        "num_iter=%d, lambda1=%f, lambda2=%f",
        view_a.shape, chunk_size, overlap, num_iter, lambda1, lambda2,
    )

    if len(view_a.shape) == 4:
        channel_slice = slice(None)
        ch_a, z_a, r_a, c_a = view_a.shape
        ch_b, z_b, r_b, c_b = view_b.shape
        assert ch_a == ch_b, "input volumes must have same # channels"
    else:
        channel_slice = None
        z_a, r_a, c_a = view_a.shape
        z_b, r_b, c_b = view_b.shape
    assert all([a == b for a, b in zip([z_a, r_a, c_a], [z_b, r_b, c_b])]), (
        "input volumes must be same shape"
    )
    chunks = calculate_conv_chunks(z_a, r_a, c_a, chunk_size, overlap, channel_slice)
    n_gpu = cupy.cuda.runtime.getDeviceCount()
    logger.warning(
        "[deconvolve_chunkwise] n_gpu=%d, n_chunks=%d, "
        "view_a.store=%r, view_b.store=%r, out.store=%r",
        n_gpu, len(chunks), view_a.store, view_b.store, out.store,
    )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_gpu, mp_context=multiprocessing.get_context("spawn")
    ) as executor, multiprocessing.Manager() as manager:
        gpu_queue = manager.Queue()
        for gpu_id in range(n_gpu):
            gpu_queue.put(gpu_id)
        log_path = os.environ.get("PYSPIM_LOG_PATH", default=None)
        fun = partial(
            _decon_chunk,
            out,
            view_a,
            view_b,
            psf_a,
            psf_b,
            bp_a,
            bp_b,
            num_iter,
            epsilon,
            lambda1,
            lambda2,
            epsilon_hess,
            log_path,
        )
        futures = []
        for chunk_id, chunk in chunks.items():
            future = executor.submit(fun, chunk, gpu_queue)
            futures.append((chunk_id, future))
        if verbose:
            pbar = tqdm(total=len(futures), desc="Deconvolving Chunks")
        else:
            pbar = nullcontext()
        with pbar:
            for chunk_id, future in futures:
                try:
                    val = future.result()
                except Exception as e:
                    logger.error(
                        "[deconvolve_chunkwise] Chunk %s failed: %s",
                        chunk_id, e, exc_info=True,
                    )
                    raise
                if verbose:
                    pbar.update(1)
                    pbar.set_postfix({"data": val > 0})


def _decon_chunk(
    # shared arguments across chunks
    out: zarr.Array,
    view_a: zarr.Array,
    view_b: zarr.Array,
    psf_a: numpy.ndarray,
    psf_b: numpy.ndarray,
    bp_a: numpy.ndarray,
    bp_b: numpy.ndarray,
    num_iter: int,
    epsilon: float,
    lambda1: float,
    lambda2: float,
    epsilon_hess: float,
    log_path: str | None,
    # iterate over these
    chunk_props: ChunkProps,
    gpu_queue,
):
    """Deconvolve a single chunk on a assigned GPU."""
    import logging
    import sys
    import traceback

    logger = logging.getLogger(__name__)

    if log_path is not None:
        try:
            log_dir = os.path.dirname(log_path)
            os.makedirs(log_dir, exist_ok=True)
            sys.stderr = open(log_path, "a", buffering=1)
            handler = logging.StreamHandler(sys.stderr)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s [pid=%(process)d] %(name)s: %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
        except Exception:
            pass

    gpu_id = None
    try:
        logger.warning(
            "[_decon_chunk] pid=%d gpu_queue type=%r, "
            "view_a.store=%r, out.store=%r",
            multiprocessing.current_process().pid,
            type(gpu_queue).__name__,
            getattr(view_a, "store", None),
            getattr(out, "store", None),
        )
        gpu_id = gpu_queue.get()
        logger.warning("[_decon_chunk] acquired gpu_id=%d", gpu_id)

        # Log GPU memory before deconvolution
        with cupy.cuda.Device(gpu_id):
            free_mem, total_mem = cupy.cuda.runtime.memGetInfo()
            logger.warning(
                "[_decon_chunk] gpu=%d memory before: free=%.1fMB, total=%.1fMB",
                gpu_id, free_mem / 1024**2, total_mem / 1024**2,
            )

        # load data
        a = view_a.oindex[chunk_props.read_window]
        b = view_b.oindex[chunk_props.read_window]

        if numpy.all(a == 0) and numpy.all(b == 0):
            gpu_queue.put(gpu_id)
            return 0

        # pad the inputs according to the chunking properties
        a = numpy.pad(a, chunk_props.paddings)
        b = numpy.pad(b, chunk_props.paddings)
        logger.warning(
            "[_decon_chunk] gpu=%d padded_a=%s padded_b=%s, launching deconvolve...",
            gpu_id, a.shape, b.shape,
        )

        with cupy.cuda.Device(gpu_id):
            psf_a_cp = cupy.asarray(psf_a, dtype=cupy.float32)
            psf_b_cp = cupy.asarray(psf_b, dtype=cupy.float32)
            bp_a_cp = cupy.asarray(bp_a, dtype=cupy.float32)
            bp_b_cp = cupy.asarray(bp_b, dtype=cupy.float32)
            
            dec = deconvolve(
                cupy.asarray(a, dtype=cupy.float32),
                cupy.asarray(b, dtype=cupy.float32),
                None,
                psf_a_cp,
                psf_b_cp,
                bp_a_cp,
                bp_b_cp,
                num_iter=num_iter,
                epsilon=epsilon,
                lambda1=lambda1,
                lambda2=lambda2,
                epsilon_hess=epsilon_hess,
                req_both=True,
                verbose=False,
            ).get()[chunk_props.out_window]

            # Explicit GPU memory cleanup
            del psf_a_cp, psf_b_cp, bp_a_cp, bp_b_cp
            cupy.get_default_memory_pool().free_all_blocks()
            gc.collect()

            # Log GPU memory after cleanup
            free_mem, total_mem = cupy.cuda.runtime.memGetInfo()
            logger.warning(
                "[_decon_chunk] gpu=%d memory after cleanup: free=%.1fMB, total=%.1fMB",
                gpu_id, free_mem / 1024**2, total_mem / 1024**2,
            )

        logger.warning("[_decon_chunk] gpu=%d deconvolve done, writing result...", gpu_id)
        gpu_queue.put(gpu_id)
        out.set_orthogonal_selection(chunk_props.data_window, dec)
        
        # Clear numpy arrays after writing
        del a, b, dec
        gc.collect()
        logger.warning("[_decon_chunk] gpu=%d done", gpu_id)
        return 1
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        try:
            if gpu_id is not None:
                gpu_queue.put(gpu_id)
        except Exception:
            pass
        raise
