"""deconvolution algorithms for jointly deconvolving dual-view microscopy data
"""
from contextlib import nullcontext
from typing import Callable, Optional

import cupy
import numpy
import cupyx.scipy
import dask.array
from scipy import signal as cpusig
from cupyx.scipy import signal as cusig
from dask.array import map_overlap
from tqdm.auto import trange
from tqdm.dask import TqdmCallback

from ..typing import NDArray
from .._util import supported_float_type


def joint_rl_dispim(view_a : NDArray, view_b : NDArray,
                    psf_a  : NDArray, psf_b  : NDArray,
                    backproj_a : Optional[NDArray]=None,
                    backproj_b : Optional[NDArray]=None,
                    num_iter   : int=10,
                    epsilon    : float=1e-5,
                    req_both   : bool=False,
                    verbose    : bool=False) -> NDArray:
    """modified Richardson-Lucy joint deconvolution, as described in [1]
        originally developed by the Shroff group at the NIH
        for use in diSPIM imaging.

    References
    ---
    [1] Wu, et. al., "Spatially isotropic four-dimensional...",
        doi:10.1038/nbt.2713

    :param view_a: array of data corresponding to first view (A)
    :type view_a: NDArray
    :param view_b: array of data corresponding to second view (B)
    :type view_b: NDArray
    :param psf_a: array with real-space point spread function for view A
    :type psf_a: NDArray
    :param psf_b: array with real-space point spread function for view B
    :type psf_b: NDArray
    :param backproj_a: array with real-space backprojector for view A.
        if `None`, the mirrored point spread function is used.
    :type backproj_a: Optional[NDArray]
    :param backproj_b: array with real-space backprojector for view B.
        if `None`, the mirrored point spread function is used.
    :type backproj_b: Optional[NDArray]
    :param num_iter: number of iterations to deconvolve for
    :type num_iter: int
    :param epsilon: small parameter to prevent division by zero errors
    :type epsilon: float
    :param req_both: only deconvolve areas where views a & b have data
    :type req_both: bool
    :param verbose: display progress bar
    :type verbose: bool
    :returns: deconvolved volume
    :rtype: NDArray
    """
    xp = cupy.get_array_module(view_a)
    # make sure all inputs are floats
    float_type = supported_float_type(view_a.dtype)
    view_a = view_a.astype(float_type, copy=False)
    view_b = view_b.astype(float_type, copy=False)
    psf_a = psf_a.astype(float_type, copy=False)
    psf_b = psf_b.astype(float_type, copy=False)
    # default back-projectors are mirrored PSFs
    if backproj_a is None:
        backproj_a = xp.ascontiguousarray(psf_a[::-1,::-1,::-1])
    if backproj_b is None:
        backproj_b = xp.ascontiguousarray(psf_b[::-1,::-1,::-1])
    # initial guess for decon
    est_i = (view_a + view_b) / 2
    # utility function for doing convolution
    # NOTE: by default `fftconvolve` wants the larger array as first arg
    # but for clarity, I like specifying the PSF first
    if xp == cupy:
        conv = lambda k, i: cusig.fftconvolve(i, k, mode='same')
    else:
        conv = lambda k, i: cpusig.fftconvolve(i, k, mode='same')
    for _ in (trange(num_iter) if verbose else range(num_iter)):
        cona = conv(psf_a, est_i)
        est_a = xp.multiply(
            est_i, conv(backproj_a, xp.where(cona < epsilon, 0, view_a / cona))
        )
        conb = conv(psf_b, est_a)
        est_i = xp.multiply(
            est_a, conv(backproj_b, xp.where(conb < epsilon, 0, view_b / conb))
        )
    if req_both:
        est_i[xp.logical_or(view_a==0, view_b==0)] = 0
    return est_i


def efficient_bayesian(view_a : NDArray, view_b : NDArray,
                       psf_a  : NDArray, psf_b  : NDArray,
                       num_iter : int=10,
                       epsilon  : float=1e-5,
                       req_both : bool=False,
                       verbose  : bool=False) -> NDArray:
    """efficient bayesian multiview deconvolution
        originally described in [1], this is just additive joint deconvolution
        with backprojectors calculated as described in the
        "quadruple-view deconvolution" section of [2]
    
    References
    ---
    [1] Preibsch, et al. "Efficient Bayesian-based...", doi:10.1038/nmeth.2929
    [2] Guo, et al. "Rapid image deconvolution..." doi:10.1038/s41587-020-0560-x

    :param view_a: array of data corresponding to first view (A)
    :type view_a: NDArray
    :param view_b: array of data corresponding to second view (B)
    :type view_b: NDArray
    :param psf_a: array with real-space point spread function for view A
    :type psf_a: NDArray
    :param psf_b: array with real-space point spread function for view B
    :type psf_b: NDArray
    :param num_iter: number of iterations to deconvolve for
    :type num_iter: int
    :param epsilon: small parameter to prevent division by zero errors
    :type epsilon: float
    :param req_both: only deconvolve areas where views a & b have data
    :type req_both: bool
    :param verbose: display progress bar
    :type verbose: bool
    :returns: deconvolved volume
    :rtype: NDArray
    """
    xp = cupy.get_array_module(view_a)
    # make sure all inputs are floats
    float_type = supported_float_type(view_a.dtype)
    view_a = view_a.astype(float_type, copy=False)
    view_b = view_b.astype(float_type, copy=False)
    psf_a = psf_a.astype(float_type, copy=False)
    psf_b = psf_b.astype(float_type, copy=False)
    # formulate backprojectors for "virtual view" updates
    flp_a = xp.ascontiguousarray(psf_a[::-1,::-1,::-1])
    flp_b = xp.ascontiguousarray(psf_b[::-1,::-1,::-1])
    # utility function for convolution
    if xp == cupy:
        conv = lambda k, i: cusig.fftconvolve(i, k, mode='same')
    else:
        conv = lambda k, i: cpusig.fftconvolve(i, k, mode='same')
    backproj_a = flp_a * conv(conv(flp_a, psf_b), flp_b)
    backproj_b = flp_b * conv(conv(flp_b, psf_a), flp_a)
    est_i = (view_a + view_b) / 2
    for _ in (trange(num_iter) if verbose else range(num_iter)):
        cona = conv(psf_a, est_i)
        est_a = xp.multiply(
            est_i, conv(backproj_a, xp.where(cona < epsilon, 0, view_a / cona))
        )
        conb = conv(psf_b, est_i)
        est_b = xp.multiply(
            est_i, conv(backproj_b, xp.where(conb < epsilon, 0, view_b / conb))
        )
        est_i = (est_a + est_b) / 2
    if req_both:
        est_i[xp.logical_or(view_a==0, view_b==0)] = 0
    return est_i


def additive_joint_rl(view_a : NDArray, view_b : NDArray,
                      psf_a  : NDArray, psf_b  : NDArray,
                      backproj_a : Optional[NDArray]=None,
                      backproj_b : Optional[NDArray]=None,
                      num_iter   : int=10,
                      epsilon    : float=1e-5,
                      req_both   : bool=False,
                      verbose    : bool=False) -> NDArray:
    """additive joint deconvolution.
        each view is deconvolved separately and averaged at each iteration

    :param view_a: array of data corresponding to first view (A)
    :type view_a: NDArray
    :param view_b: array of data corresponding to second view (B)
    :type view_b: NDArray
    :param psf_a: array with real-space point spread function for view A
    :type psf_a: NDArray
    :param psf_b: array with real-space point spread function for view B
    :type psf_b: NDArray
    :param backproj_a: array with real-space backprojector for view A.
        if `None`, the mirrored point spread function is used.
    :type backproj_a: Optional[NDArray]
    :param backproj_b: array with real-space backprojector for view B.
        if `None`, the mirrored point spread function is used.
    :type backproj_b: Optional[NDArray]
    :param num_iter: number of iterations to deconvolve for
    :type num_iter: int
    :param epsilon: small parameter to prevent division by zero errors
    :type epsilon: float
    :param req_both: only deconvolve areas where views a & b have data
    :type req_both: bool
    :param verbose: display progress bar
    :type verbose: bool
    :returns: deconvolved volume
    :rtype: NDArray
    """
    xp = cupy.get_array_module(view_a)
    sp = cupyx.scipy.get_array_module(view_a)
    # make sure all inputs are floats
    float_type = supported_float_type(view_a.dtype)
    view_a = view_a.astype(float_type, copy=False)
    view_b = view_b.astype(float_type, copy=False)
    psf_a = psf_a.astype(float_type, copy=False)
    psf_b = psf_b.astype(float_type, copy=False)
    # utility function for convolution
    if xp == cupy:
        conv = lambda k, i: cusig.fftconvolve(i, k, mode='same')
    else:
        conv = lambda k, i: cpusig.fftconvolve(i, k, mode='same')
    # default back-projectors are mirrored PSFs
    if backproj_a is None:
        backproj_a = xp.ascontiguousarray(psf_a[::-1,::-1,::-1])
    if backproj_b is None:
        backproj_b = xp.ascontiguousarray(psf_b[::-1,::-1,::-1])
    # initial guess for decon
    est_i = (view_a + view_b) / 2
    # deconvolution loop
    for _ in (trange(num_iter) if verbose else range(num_iter)):
        cona = conv(psf_a, est_i)
        est_a = xp.multiply(
            est_i, conv(backproj_a, xp.where(cona < epsilon, 0, view_a / cona))
        )
        conb = conv(psf_b, est_i)
        est_b = xp.multiply(
            est_i, conv(backproj_b, xp.where(conb < epsilon, 0, view_b / conb))
        )
        est_i = (est_a + est_b) / 2
    if req_both:
        est_i[xp.logical_or(view_a==0, view_b==0)] = 0
    return est_i


def deconvolve_dask(decon_fun : str, device : str,
                    view_a : dask.array.Array, view_b : dask.array.Array,
                    psf_a : NDArray, psf_b : NDArray,
                    backproj_a : Optional[NDArray]=None,
                    backproj_b : Optional[NDArray]=None,
                    num_iter   : int=10,
                    epsilon    : float=1e-5,
                    req_both   : bool=False,
                    verbose    : bool=False,
                    overlap    : Optional[int]=None) -> numpy.ndarray:
    """jointly deconvolve (large) input volumes block-wise using `dask`

    :param decon_fun: string indicating which deconvolution algorithm to use.
        one of 'joint', 'effbayes', or 'additive'
    :type decon_fun: str
    :param device: string indicating which device to do the computation on.
        one of 'cpu' or 'gpu'
    :type device: str
    :param view_a: array of data corresponding to first view (A)
    :type view_a: dask.array.Array
    :param view_b: array of data corresponding to second view (B)
    :type view_b: dask.array.Array
    :param psf_a: array with real-space point spread function for view A
    :type psf_a: NDArray
    :param psf_b: array with real-space point spread function for view B
    :type psf_b: NDArray
    :param backproj_a: array with real-space backprojector for view A.
        if `None`, the mirrored point spread function is used.
    :type backproj_a: Optional[NDArray]
    :param backproj_b: array with real-space backprojector for view B.
        if `None`, the mirrored point spread function is used.
    :type backproj_b: Optional[NDArray]
    :param num_iter: number of iterations to deconvolve for
    :type num_iter: int
    :param epsilon: small parameter to prevent division by zero errors
    :type epsilon: float
    :param req_both: only deconvolve areas where views a & b have data
    :type req_both: bool
    :param verbose: display progress bar
    :type verbose: bool
    :param overlap: number of pixels each block shares with its neighbor
        (same as `depth` parameter in `dask.array.map_overlap`)
    :type overlap: Optional[int]
    :returns: deconvolved volume
    :rtype: NDArray
    """
    # if `overlap` not specified, determine from size of psf
    if overlap is None:
        overlap = max([max(psf_a.shape), max(psf_b.shape)])
    # figure out which function
    if decon_fun == 'joint':
        fun = joint_rl_dispim
    elif decon_fun == 'effbayes':
        fun = efficient_bayesian
    elif decon_fun == 'additive':
        fun = additive_joint_rl
    else:
        raise ValueError('invalid decon function')
    if device == 'gpu':
        psf_a = cupy.asarray(psf_a)
        psf_b = cupy.asarray(psf_b)
        mempool = cupy.get_default_memory_pool()
    else:
        mempool = None
    if backproj_a is None:
        backproj_a = psf_a[::-1,::-1,::-1]
    if backproj_b is None:
        backproj_b = psf_b[::-1,::-1,::-1]
    # if verbose, make a tqdm progress bar
    # otherwise, this should be a no-op
    ctx_mgr = TqdmCallback(desc="dask-decon") if verbose else nullcontext
    with ctx_mgr:
        return dask.array.map_overlap(
            _decon_dask, view_a, view_b,
            # keyword-arguments for _decon_dask
            psf_a=psf_a, psf_b=psf_b,
            decon_fun=fun, device=device,
            backproj_a=backproj_a, backproj_b=backproj_b,
            num_iter=num_iter, epsilon=1e-5,
            req_both=False, verbose=False,
            mempool=mempool, crop=overlap,
            # keyword-arguments for map_overlap
            trim=False, depth=overlap,
            meta=view_a._meta,
            boundary='reflect'
        ).compute()
    

def _decon_dask(view_a : NDArray, view_b : NDArray,
                # everything past here should be specified as keyword-arg
                # when called by `map_overlap`
                decon_fun : Callable, device : str,
                psf_a : NDArray, psf_b : NDArray,
                backproj_a : Optional[NDArray]=None,
                backproj_b : Optional[NDArray]=None,
                num_iter : int=10,
                epsilon : float=1e-5,
                req_both : bool=False,
                verbose : bool=False,
                mempool=None, crop : Optional[int]=None) -> numpy.ndarray:
    if device == 'gpu':
        if cupy.get_array_module(psf_a) != cupy:
            psf_a = cupy.asarray(psf_a)
        if cupy.get_array_module(psf_b) != cupy:
            psf_b = cupy.asarray(psf_b)
        out = decon_fun(
            cupy.asarray(view_a), cupy.asarray(view_b), psf_a, psf_b,
            (psf_a[::-1,::-1,::-1] if backproj_a is None else backproj_a),
            (psf_b[::-1,::-1,::-1] if backproj_b is None else backproj_b),
            num_iter=num_iter, epsilon=epsilon,
            req_both=req_both, verbose=verbose
        ).get()
        if mempool is not None:
            mempool.free_all_blocks()
    else:  # cpu
        out = decon_fun(
            view_a, view_b, psf_a, psf_b,
            (psf_a[::-1,::-1,::-1] if backproj_a is None else backproj_a),
            (psf_b[::-1,::-1,::-1] if backproj_b is None else backproj_b),
            num_iter=num_iter, epsilon=epsilon,
            req_both=req_both, verbose=verbose
        )
    if crop is None:
        return out
    else:
        return out[crop:-crop,crop:-crop,crop:-crop]
