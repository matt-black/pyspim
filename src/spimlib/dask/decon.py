from contextlib import nullcontext
from typing import Callable, Optional

import cupy
import numpy
import dask.array
from tqdm.dask import TqdmCallback

from ..decon.dualview import joint_rl_dispim, additive_joint_rl, \
    efficient_bayesian
from ..typing import NDArray, PadType


def __effbayes_rl_dask_call(view_a : NDArray, view_b : NDArray,
                            psf_a  : NDArray, psf_b  : NDArray,
                            backproj_a          : Optional[NDArray]=None,
                            backproj_b          : Optional[NDArray]=None,
                            num_iter            : int=10,
                            epsilon             : float=1e-5,
                            boundary_correction : bool=True,
                            req_both            : bool=False,
                            zero_padding        : Optional[PadType]=None,
                            boundary_sigma_a    : float=1e-2,
                            boundary_sigma_b    : float=1e-2,
                            init_constant       : bool=False,
                            verbose             : bool=False) -> NDArray:
    """convenience method that ignores the backprojector argument so that
    the function's call method looks the same as those for additive & joint rl
    """
    return efficient_bayesian(view_a, view_b, psf_a, psf_b,
                              num_iter, epsilon, boundary_correction, req_both,
                              zero_padding, boundary_sigma_a, boundary_sigma_b,
                              init_constant, verbose)


def deconvolve_dask(decon_fun : str, device : str,
                    view_a : dask.array.Array, view_b : dask.array.Array,
                    psf_a : NDArray, psf_b : NDArray,
                    backproj_a : Optional[NDArray]=None,
                    backproj_b : Optional[NDArray]=None,
                    num_iter   : int=10,
                    epsilon    : float=1e-5,
                    boundary_correction : bool=False,
                    req_both   : bool=False,
                    zero_padding : Optional[PadType]=None,
                    boundary_sigma_a : float=1e-2,
                    boundary_sigma_b : float=1e-2,
                    init_constant : bool=False,
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
        fun = __effbayes_rl_dask_call
    elif decon_fun == 'additive':
        fun = additive_joint_rl
    else:
        raise ValueError('invalid decon function')
    if device == 'gpu':
        psf_a = cupy.asarray(psf_a).astype(cupy.float32)
        psf_b = cupy.asarray(psf_b).astype(cupy.float32)
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
            # keyword-arguments for the deconvolution function
            psf_a=psf_a, psf_b=psf_b,
            decon_fun=fun, device=device,
            backproj_a=backproj_a, backproj_b=backproj_b,
            num_iter=num_iter, epsilon=epsilon,
            boundary_correction=boundary_correction,
            req_both=req_both, 
            zero_padding=zero_padding,
            boundary_sigma_a=boundary_sigma_a,
            boundary_sigma_b=boundary_sigma_b,
            init_constant=init_constant,
            verbose=False,
            # keywords for _decon_dask
            mempool=mempool, crop=overlap,
            # keyword-arguments for map_overlap
            trim=False, depth=overlap,
            meta=view_a._meta,
            boundary='none'
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
                boundary_correction : bool=False,
                req_both : bool=False,
                zero_padding : Optional[PadType]=None,
                boundary_sigma_a : float=1e-2,
                boundary_sigma_b : float=1e-2,
                init_constant : bool=False,
                verbose : bool=False,
                mempool=None, crop : Optional[int]=None) -> numpy.ndarray:
    if device == 'gpu':
        if cupy.get_array_module(psf_a) != cupy:
            psf_a = cupy.asarray(psf_a)
        if cupy.get_array_module(psf_b) != cupy:
            psf_b = cupy.asarray(psf_b)
        out = decon_fun(
            cupy.asarray(view_a).astype(cupy.float32), 
            cupy.asarray(view_b).astype(cupy.float32), 
            psf_a, psf_b,
            (psf_a[::-1,::-1,::-1] if backproj_a is None else backproj_a),
            (psf_b[::-1,::-1,::-1] if backproj_b is None else backproj_b),
            num_iter=num_iter, epsilon=epsilon,
            boundary_correction=boundary_correction,
            req_both=req_both, 
            zero_padding=zero_padding,
            boundary_sigma_a=boundary_sigma_a,
            boundary_sigma_b=boundary_sigma_b,
            init_constant=init_constant,
            verbose=verbose
        ).get()
        if mempool is not None:
            mempool.free_all_blocks()
    else:  # cpu
        out = decon_fun(
            view_a, view_b, psf_a, psf_b,
            (psf_a[::-1,::-1,::-1] if backproj_a is None else backproj_a),
            (psf_b[::-1,::-1,::-1] if backproj_b is None else backproj_b),
            num_iter=num_iter, epsilon=epsilon,
            boundary_correction=boundary_correction,
            req_both=req_both, 
            zero_padding=zero_padding,
            boundary_sigma_a=boundary_sigma_a,
            boundary_sigma_b=boundary_sigma_b,
            init_constant=init_constant,
            verbose=verbose
        )
    if crop is None:
        return out
    else:
        return out[crop:-crop,crop:-crop,crop:-crop]