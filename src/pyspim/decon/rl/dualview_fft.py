"""Deconvolution algorithms for jointly deconvolving dual-view microscopy data.

References
---
[1] Wu, et. al., "Spatially isotropic four-dimensional...", doi:10.1038/nbt.2713
[2] Preibsch, et al. "Efficient Bayesian-based...", doi:10.1038/nmeth.2929
[3] Guo, et al. "Rapid image deconvolution..." doi:10.1038/s41587-020-0560-x
[4] Anconelli, B., et al. "Reduction of boundary effects...", doi:10.1051/0004-6361:20053848
"""
import multiprocessing
import concurrent.futures
from itertools import repeat
from functools import partial
from contextlib import nullcontext
from typing import Iterable, Optional, Tuple

import zarr
import cupy
import numpy
from scipy.signal import fftconvolve as fftconv_cpu
from cupyx.scipy.signal import fftconvolve as fftconv_gpu
from tqdm.auto import tqdm, trange

from ...typing import NDArray, PadType
from ..._util import supported_float_type
from .._util import div_stable, calculate_conv_chunks, ChunkProps


def _deconvolve_single_channel(view_a : NDArray, view_b : NDArray, 
                               est_i : NDArray|None,
                               psf_a : NDArray, psf_b : NDArray,
                               backproj_a : NDArray, backproj_b : NDArray,
                               decon_function : str,
                               num_iter : int,
                               epsilon : float,
                               boundary_correction : bool,
                               req_both : bool,
                               zero_padding : Optional[PadType],
                               boundary_sigma_a : float,
                               boundary_sigma_b : float,
                               verbose : bool) -> NDArray:
    if est_i is None:
        est_i = estimate_initialization(view_a, view_b, False)
    if decon_function == 'additive':
        return additive_joint_rl(view_a, view_b, est_i, psf_a, psf_b,
                                 backproj_a, backproj_b,
                                 num_iter, epsilon, boundary_correction,
                                 req_both, zero_padding,
                                 boundary_sigma_a, boundary_sigma_b,
                                 verbose)
    elif decon_function == 'dispim':
        return joint_rl_dispim(view_a, view_b, est_i, psf_a, psf_b,
                               backproj_a, backproj_b, num_iter, epsilon,
                               boundary_correction, req_both, zero_padding,
                               boundary_sigma_a, boundary_sigma_b, verbose)
    elif decon_function == 'efficient':
        return efficient_bayesian(view_a, view_b, est_i, psf_a, psf_b,
                                  num_iter, epsilon, boundary_correction, 
                                  req_both, zero_padding, boundary_sigma_a,
                                  boundary_sigma_b, verbose)
    else:
        raise ValueError('invalid deconvolution function')
    

def _deconvolve_multichannel(view_a : NDArray, view_b : NDArray, 
                             est_i : NDArray,
                             psf_a : NDArray|Iterable[NDArray],
                             psf_b : NDArray|Iterable[NDArray],
                             backproj_a : NDArray|Iterable[NDArray],
                             backproj_b : NDArray|Iterable[NDArray],
                             decon_function : str,
                             num_iter : int,
                             epsilon : float,
                             boundary_correction : bool,
                             req_both : bool,
                             zero_padding : Optional[PadType],
                             boundary_sigma_a : float,
                             boundary_sigma_b : float,
                             verbose : bool) -> NDArray:
    xp = cupy.get_array_module(view_a)
    n_chan = view_a.shape[0]
    psf_a = repeat(psf_a) if isinstance(psf_a, NDArray) else psf_a
    psf_b = repeat(psf_b) if isinstance(psf_b, NDArray) else psf_b
    if isinstance(backproj_a, NDArray):
        backproj_a = repeat(backproj_a)
    if isinstance(backproj_b, NDArray):
        backproj_b = repeat(backproj_b)
    pfun = partial(_deconvolve_single_channel,
                   decon_function=decon_function,
                   num_iter=num_iter, epsilon=epsilon,
                   boundary_correction=boundary_correction,
                   req_both=req_both,
                   zero_padding=zero_padding,
                   boundary_sigma_a=boundary_sigma_a,
                   boundary_sigma_b=boundary_sigma_b,
                   verbose=verbose)
    return xp.stack([
        pfun(view_a[i,...], view_b[i,...], 
             (est_i[i,...] if est_i is not None else None), pa, pb, bpa, bpb)
        for i, pa, pb, bpa, bpb 
        in zip(range(n_chan), psf_a, psf_b, backproj_a, backproj_b)
    ], axis=0)


def deconvolve(view_a : NDArray, view_b : NDArray, est_i : NDArray|None,
               psf_a : NDArray|Iterable[NDArray], 
               psf_b : NDArray|Iterable[NDArray],
               backproj_a : NDArray|Iterable[NDArray], 
               backproj_b : NDArray|Iterable[NDArray],
               decon_function : str,
               num_iter : int,
               epsilon : float,
               req_both : bool,
               boundary_correction : bool,
               zero_padding : Optional[PadType],
               boundary_sigma_a : float,
               boundary_sigma_b : float,
               verbose : bool) -> NDArray:
    """deconvolve Joint deconvolution of 2 views into a single one.

    Args:
        view_a (NDArray): array from 1st view
        view_b (NDArray): array from 2nd view
        est_i (NDArray | None): initial estimate
        psf_a (NDArray | Iterable[NDArray]): psf arrays for view A (single one, or 1 per channel)
        psf_b (NDArray | Iterable[NDArray]): psf arrays for view B (single, or 1 per channel)
        backproj_a (NDArray | Iterable[NDArray]): backprojector array(s) for view A
        backproj_b (NDArray | Iterable[NDArray]): backprojector array(s) for view B
        decon_function (str): style of deconvolution to use
        num_iter (int): number of iterations
        epsilon (float): small parameter to prevent division by zero
        req_both (bool): set all areas where either view doesn't have data to 0
        boundary_correction (bool): do boundary correction
        zero_padding (Optional[PadType]): zero padding for boundary correction
        boundary_sigma_a (float): significance level for pixels, view a
        boundary_sigma_b (float): significance level for pixels, view b
        verbose (bool): show progressbar for iterations

    Raises:
        ValueError: input array is not 3 or 4D

    Returns:
        NDArray
    """
    if len(view_a.shape) == 4:
        return _deconvolve_multichannel(view_a, view_b, est_i, psf_a, psf_b,
                                        backproj_a, backproj_b, decon_function,
                                        num_iter, epsilon, boundary_correction,
                                        req_both, zero_padding,
                                        boundary_sigma_a, boundary_sigma_b,
                                        verbose)
    elif len(view_a.shape) == 3:
        return _deconvolve_single_channel(view_a, view_b, est_i, psf_a, psf_b,
                                          backproj_a, backproj_b, decon_function, 
                                          num_iter, epsilon, boundary_correction, 
                                          req_both, zero_padding,
                                          boundary_sigma_a, boundary_sigma_b, 
                                          verbose)
    else:
        raise ValueError('invalid shape, must be 3- or 4D')


def joint_rl_dispim(view_a : NDArray, view_b : NDArray, est_i : NDArray,
                    psf_a  : NDArray, psf_b  : NDArray,
                    backproj_a          : Optional[NDArray] = None,
                    backproj_b          : Optional[NDArray] = None,
                    num_iter            : int = 10,
                    epsilon             : float = 1e-5,
                    boundary_correction : bool = True,
                    req_both            : bool = False,
                    zero_padding        : Optional[PadType] = None,
                    boundary_sigma_a    : float = 1e-2,
                    boundary_sigma_b    : float = 1e-2,
                    verbose             : bool = False) -> NDArray:
    """joint_rl_dispim Modified Richardson-Lucy joint deconvolution, as described in [1]. 
    
    originally developed by the Shroff group at the NIH for use in diSPIM imaging.
    NOTE: boundary correction is not implemented for this method, as at the time of writing, there is no known way of doing this

    Args:
        view_a (NDArray): data array corresponding to first view (A)
        view_b (NDArray): data array corresponding to second view (B)
        est_i (NDArray): initial estimate for deconvolution. If ``None``, will use (A+B)/2.
        psf_a (NDArray): array with real-space point spread function for view A
        psf_b (NDArray): array with real-space point spread function for view B
        backproj_a (NDArray, optional): array with real-space backprojector for view A. If `None`, the mirrored point spread function is used.
        backproj_b (NDArray, optional): array with real-space backprojector for view B. If `None`, the mirrored point spread function is used.
        num_iter (int, optional): number of iterations to deconvolve for. Defaults to 10.
        epsilon (float, optional): small parameter to prevent division by zero errors. Defaults to 1e-5.
        boundary_correction (bool, optional): correct boundary effects, defaults to True.
        req_both (bool, optional): only deconvolve areas where views a & b have data, defaults to False.
        zero_padding (PadType, optional): amount of zero-padding to add to each axis, defaults to None. if None, each axis of size N is padded on each side by N/2 so that the padded image has dimension 2N in that axis.
        boundary_sigma_a (float, optional): threshold for determining significant pixels in view A, defaults to 1e-2.
        boundary_sigma_b (float, optional): threshold for determining significant pixels in view B, defaults to 1e-2.
        verbose (bool, optional): display progress bar, defaults to False

    Returns:
        NDArray
    """
    if boundary_correction:
        return _additive_joint_rl_boundcorr(
            view_a, view_b, est_i, psf_a, psf_b, backproj_a, backproj_b,
            num_iter, epsilon, zero_padding, 
            boundary_sigma_a, boundary_sigma_b,
            req_both, verbose
        )
    else:
        return _joint_rl_dispim_uncorr(
            view_a, view_b, est_i, psf_a, psf_b, backproj_a, backproj_b,
            num_iter, epsilon, req_both, verbose
        )


def _joint_rl_dispim_uncorr(view_a : NDArray, view_b : NDArray, est_i : NDArray,
                            psf_a  : NDArray, psf_b  : NDArray,
                            backproj_a : Optional[NDArray] = None,
                            backproj_b : Optional[NDArray] = None,
                            num_iter   : int = 10,
                            epsilon    : float = 1e-5,
                            req_both   : bool = False,
                            verbose    : bool = False) -> NDArray:
    """joint RL deconvolution without boundary correction.
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
    # utility function for doing convolution
    # NOTE: by default `fftconvolve` wants the larger array as first arg
    # but for clarity, I like specifying the PSF first
    if xp == cupy:
        conv = lambda k, i: fftconv_gpu(i, k, mode='same')
    else:
        conv = lambda k, i: fftconv_cpu(i, k, mode='same')
    for _ in (trange(num_iter) if verbose else range(num_iter)):
        cona = conv(psf_a, est_i)
        est_a = xp.multiply(
            est_i, conv(backproj_a, div_stable(view_a, cona, epsilon))
        )
        conb = conv(psf_b, est_a)
        est_i = xp.multiply(
            est_a, conv(backproj_b, div_stable(view_b, conb, epsilon))
        )
    if req_both:
        est_i[xp.logical_or(view_a==0, view_b==0)] = 0
    return est_i


def efficient_bayesian_backprojectors(psf_a : NDArray, psf_b : NDArray) -> \
    Tuple[NDArray, NDArray]:
    """efficient_bayesian_backprojectors Calculate proper backprojectors for "Efficient Bayesian" deconvolution.

    Args:
        psf_a (NDArray): point spread function for view A
        psf_b (NDArray): point spread function for view B

    Returns:
        Tuple[NDArray,NDArray]: tuple of backprojectors, one for each view.
    """
    xp = cupy.get_array_module(psf_a)
    if xp == cupy:
        conv = lambda k, i: fftconv_gpu(i, k, mode='same')
    else:
        conv = lambda k, i: fftconv_cpu(i, k, mode='same')
    float_type = supported_float_type(psf_a.dtype)
    psf_a = psf_a.astype(float_type, copy=False)
    psf_b = psf_b.astype(float_type, copy=False)
    # formulate backprojectors for "virtual view" updates
    # TODO: write this s.t. it's valid for 2D images, too
    flp_a = xp.ascontiguousarray(psf_a[::-1,::-1,::-1])
    flp_b = xp.ascontiguousarray(psf_b[::-1,::-1,::-1])
    # calculate backprojectors
    bp_a = flp_a * conv(conv(flp_a, psf_b), flp_b)
    bp_b = flp_b * conv(conv(flp_b, psf_a), flp_a)
    return bp_a, bp_b


def efficient_bayesian(view_a : NDArray, view_b : NDArray, 
                       est_i : NDArray,    
                       psf_a  : NDArray, psf_b  : NDArray,
                       num_iter            : int = 10,
                       epsilon             : float = 1e-5,
                       boundary_correction : bool = True,
                       req_both            : bool = False,
                       zero_padding        : Optional[PadType] = None,
                       boundary_sigma_a    : float = 1e-2,
                       boundary_sigma_b    : float = 1e-2,
                       verbose             : bool = False) -> NDArray:
    """efficient_bayesian Efficient bayesian multiview deconvolution.

    Originally described in [2], this is just additive joint deconvolution with backprojectors calculated as described in the "quadruple-view deconvolution" section of [3]

    Args:
        view_a (NDArray): data array corresponding to first view (A)
        view_b (NDArray): data array corresponding to second view (B)
        est_i (NDArray): initial estimate for deconvolution. If ``None``, will use (A+B)/2.
        psf_a (NDArray): array with real-space point spread function for view A
        psf_b (NDArray): array with real-space point spread function for view B
        backproj_a (NDArray, optional): array with real-space backprojector for view A. If `None`, the mirrored point spread function is used.
        backproj_b (NDArray, optional): array with real-space backprojector for view B. If `None`, the mirrored point spread function is used.
        num_iter (int, optional): number of iterations to deconvolve for. Defaults to 10.
        epsilon (float, optional): small parameter to prevent division by zero errors. Defaults to 1e-5.
        boundary_correction (bool, optional): correct boundary effects, defaults to True.
        req_both (bool, optional): only deconvolve areas where views a & b have data, defaults to False.
        zero_padding (PadType, optional): amount of zero-padding to add to each axis, defaults to None. if None, each axis of size N is padded on each side by N/2 so that the padded image has dimension 2N in that axis.
        boundary_sigma_a (float, optional): threshold for determining significant pixels in view A, defaults to 1e-2.
        boundary_sigma_b (float, optional): threshold for determining significant pixels in view B, defaults to 1e-2.
        verbose (bool, optional): display progress bar, defaults to False

    Returns:
        NDArray
    """
    # compute backprojectors
    backproj_a, backproj_b = efficient_bayesian_backprojectors(psf_a, psf_b)
    return additive_joint_rl(
        view_a, view_b, est_i, psf_a, psf_b, backproj_a, backproj_b,
        num_iter, epsilon, boundary_correction, req_both, zero_padding,
        boundary_sigma_a, boundary_sigma_b, verbose
    )


def additive_joint_rl(view_a : NDArray, view_b : NDArray, est_i : NDArray,
                      psf_a  : NDArray, psf_b  : NDArray,
                      backproj_a          : Optional[NDArray] = None,
                      backproj_b          : Optional[NDArray] = None,
                      num_iter            : int = 10,
                      epsilon             : float = 1e-5,
                      boundary_correction : bool = True,
                      req_both            : bool = False,
                      zero_padding        : Optional[PadType] = None,
                      boundary_sigma_a    : float = 1e-2,
                      boundary_sigma_b    : float = 1e-2,
                      verbose             : bool = False) -> NDArray:
    """additive_joint_rl Additive joint deconvolution.
    
    With boundary correction turned off, each view is deconvolved separately and then averaged at each iteration. 
    With boundary correction turned on, an ordered subset EM algorithm from [4] is used. 
    
    Args:
        view_a (NDArray): data array corresponding to first view (A)
        view_b (NDArray): data array corresponding to second view (B)
        est_i (NDArray): initial estimate for deconvolution. If ``None``, will use (A+B)/2.
        psf_a (NDArray): array with real-space point spread function for view A
        psf_b (NDArray): array with real-space point spread function for view B
        backproj_a (NDArray, optional): array with real-space backprojector for view A. If `None`, the mirrored point spread function is used.
        backproj_b (NDArray, optional): array with real-space backprojector for view B. If `None`, the mirrored point spread function is used.
        num_iter (int, optional): number of iterations to deconvolve for. Defaults to 10.
        epsilon (float, optional): small parameter to prevent division by zero errors. Defaults to 1e-5.
        boundary_correction (bool, optional): correct boundary effects, defaults to True.
        req_both (bool, optional): only deconvolve areas where views a & b have data, defaults to False.
        zero_padding (PadType, optional): amount of zero-padding to add to each axis, defaults to None. if None, each axis of size N is padded on each side by N/2 so that the padded image has dimension 2N in that axis.
        boundary_sigma_a (float, optional): threshold for determining significant pixels in view A, defaults to 1e-2.
        boundary_sigma_b (float, optional): threshold for determining significant pixels in view B, defaults to 1e-2.
        verbose (bool, optional): display progress bar, defaults to False

    Returns:
        NDArray
    """
    if boundary_correction:
        return _additive_joint_rl_boundcorr(
            view_a, view_b, est_i, psf_a, psf_b, backproj_a, backproj_b,
            num_iter, epsilon,
            zero_padding, boundary_sigma_a, boundary_sigma_b,
            req_both, verbose
        )
    else:
        return _additive_joint_rl_uncorr(
            view_a, view_b, est_i, psf_a, psf_b, backproj_a, backproj_b,
            num_iter, epsilon, req_both, verbose
        )


def _additive_joint_rl_boundcorr(view_a : NDArray, view_b : NDArray,
                                 est_i : NDArray,
                                 psf_a  : NDArray, psf_b  : NDArray,
                                 backproj_a : Optional[NDArray] = None,
                                 backproj_b : Optional[NDArray] = None,
                                 num_iter : int = 10,
                                 epsilon  : float = 1e-5,
                                 zero_padding : Optional[PadType] = None,
                                 boundary_sigma_a : float = 1e-2,
                                 boundary_sigma_b : float = 1e-2,
                                 req_both : bool = False,
                                 verbose  : bool = False) -> NDArray:
    """_additive_joint_rl_boundcorr Dual-view additive joint Richardson-Lucy deconvolution with boundary correction.

    NOTE: this implementation uses ordered subset expectation maximization and so doesn't iterate in the same way as the boundary-uncorrected version. 
    """
    xp = cupy.get_array_module(view_a)
    # make sure input views have same dimension/size
    assert len(view_a.shape) == len(view_b.shape), \
        "both views must have same dimensionality"
    assert all([sa == sb for sa, sb in zip(view_a.shape, view_b.shape)]), \
        "both views must have same shape"
    # make sure all inputs are floats
    float_type = supported_float_type(view_a.dtype)
    view_a = view_a.astype(float_type, copy=False)
    view_b = view_b.astype(float_type, copy=False)
    psf_a = psf_a.astype(float_type, copy=False)
    psf_b = psf_b.astype(float_type, copy=False)
    if est_i is None:
        est_i = (view_a + view_b) / 2.
    # utility function for convolution
    if xp == cupy:
        conv = lambda k, i: fftconv_gpu(i, k, mode='same')
    else:
        conv = lambda k, i: fftconv_cpu(i, k, mode='same')
    # default back-projectors are mirrored PSFs
    if backproj_a is None:
        backproj_a = xp.ascontiguousarray(psf_a[::-1,::-1,::-1])
    if backproj_b is None:
        backproj_b = xp.ascontiguousarray(psf_b[::-1,::-1,::-1])
    # figure out padding
    if zero_padding is None:  # use default from paper (2N)
        # default is to pad s.t. NxN image is 2Nx2N
        pad = tuple([(d//2, d//2) for d in view_a.shape])
    else:
        if isinstance(zero_padding, int):
            pad = tuple(
                [(zero_padding, zero_padding) for _ in view_a.shape]
            )
        else:
            assert len(zero_padding) == len(view_a.shape), \
                "zero padding must be specified for all dimensions of input"
            pad = tuple([(p,p) for p in zero_padding])
    # compute the flux constant
    c = xp.sum((view_a + view_b) / 2.)
    # generate mask that we'll use to determine $\overbar{alpha}$
    M_a, M_b = xp.ones_like(view_a), xp.ones_like(view_b)
    # pad the views and mask with zeros
    view_a = xp.pad(view_a, pad, mode='constant', constant_values=0)
    view_b = xp.pad(view_b, pad, mode='constant', constant_values=0)
    M_a = xp.pad(M_a, pad, mode='constant', constant_values=0) 
    M_b = xp.pad(M_b, pad, mode='constant', constant_values=0)
    # calculate the window, $\overbar{w}(n')$ (Eqn. 17)
    # NOTE: after calculation, don't need masks anymore so `del` them
    alpha_a, alpha_b = conv(backproj_a, M_a), conv(backproj_b, M_b)  # Eqn. 9
    window_a = xp.where(alpha_a > boundary_sigma_a, 1. / alpha_a, 0.)
    window_b = xp.where(alpha_b > boundary_sigma_b, 1. / alpha_b, 0.)
    del M_a, M_b
    # compute $\overbar{\alpha}(n)$ (Eqn. 8)
    # NOTE: after we compute this, don't need $\alpha_{a,b}$ anymore, so `del`
    alpha = alpha_a + alpha_b
    del alpha_a, alpha_b
    # do the RL deconvolution
    # NOTE: by convention, and to match the API of the uncorrected functions 
    # we do a full OSEM cycle for each "iteration" (this is a different
    # convention from that used in the original paper)
    for _ in (trange(num_iter) if verbose else range(num_iter)):
        for k in range(2):
            if k == 0:  # use view_a
                con = conv(psf_a, est)
                est = xp.multiply(
                    xp.multiply(window_a, est), #conv(window_a, est),
                    conv(backproj_a, div_stable(view_a, con, epsilon))
                )
            else:  # use view_b
                con = conv(psf_b, est)
                est = xp.multiply(
                    xp.multiply(window_b, est), #conv(window_b, est),
                    conv(backproj_b, div_stable(view_b, con, epsilon))
                )
            # recompute the flux constant
            c_tilde = xp.sum(alpha * est) / 2.0
            # recalculate estimate based on shifting flux constant
            est = (c / c_tilde) * est
    if req_both:
        est[xp.logical_or(view_a==0, view_b==0)] = 0
    # trim out padding
    if len(est) == 2:
        return est[pad[0][0]:-pad[0][1],pad[1][0]:-pad[1][1]]
    else:
        return est[pad[0][0]:-pad[0][1],
                   pad[1][0]:-pad[1][1],
                   pad[2][0]:-pad[2][1]]


def _additive_joint_rl_uncorr(view_a : NDArray, view_b : NDArray, 
                              est_i : NDArray,
                              psf_a  : NDArray, psf_b  : NDArray,
                              backproj_a : Optional[NDArray] = None,
                              backproj_b : Optional[NDArray] = None,
                              num_iter : int = 10,
                              epsilon  : float = 1e-5,
                              req_both : bool = False,
                              verbose  : bool = False) -> NDArray:
    """Dual-view additive joint Richardson-Lucy deconvolution without boundary correction.
    """
    xp = cupy.get_array_module(view_a)
    # make sure all inputs are floats
    float_type = supported_float_type(view_a.dtype)
    view_a = view_a.astype(float_type, copy=False)
    view_b = view_b.astype(float_type, copy=False)
    psf_a = psf_a.astype(float_type, copy=False)
    psf_b = psf_b.astype(float_type, copy=False)
    # utility function for convolution
    if xp == cupy:
        conv = lambda k, i: fftconv_gpu(i, k, mode='same')
    else:
        conv = lambda k, i: fftconv_cpu(i, k, mode='same')
    if est_i is None:
        est_i = (view_a + view_b) / 2
    # default back-projectors are mirrored PSFs
    if backproj_a is None:
        backproj_a = xp.ascontiguousarray(psf_a[::-1,::-1,::-1])
    if backproj_b is None:
        backproj_b = xp.ascontiguousarray(psf_b[::-1,::-1,::-1])
    # deconvolution loop
    for _ in (trange(num_iter) if verbose else range(num_iter)):
        cona = conv(psf_a, est_i)
        est_a = xp.multiply(
            est_i, conv(backproj_a, div_stable(view_a, cona, epsilon))
        )
        conb = conv(psf_b, est_i)
        est_b = xp.multiply(
            est_i, conv(backproj_b, div_stable(view_b, conb, epsilon))
        )
        est_i = (est_a + est_b) / 2
    if req_both:
        est_i[xp.logical_or(view_a==0, view_b==0)] = 0
    return est_i


def estimate_initialization(view_a : NDArray, view_b : NDArray, 
                            init_constant : bool) -> NDArray:
    """estimate_initialization Initialize estimate for deconvolution

    Args:
        view_a (NDArray): input volume, view A
        view_b (NDArray): input volume, view B
        init_constant (bool): initialize as constant matrix, otherwise (A + B)/2

    Returns:
        NDArray
    """
    xp = cupy.get_array_module(view_a)
    if init_constant:
        # initialize the estimate as a constant value that can satisfy
        # the flux constraint, (Eqn. 16 of [4])
        c = xp.sum((view_a + view_b) / 2)
        return xp.ones_like(view_a) * (c / xp.prod(xp.asarray(view_a.shape)))
    else:
        # do (view_a + view_b) / 2. as is done typically in RL decon. 
        # (NOTE: this still satisfies flux constraint, so ok)
        return (view_a + view_b) / 2
    

def deconvolve_chunkwise(
    view_a : zarr.Array, view_b : zarr.Array, out : zarr.Array,
    chunk_size : int|Tuple[int,int,int], overlap : int|Tuple[int,int,int],
    psf_a : numpy.ndarray, psf_b : numpy.ndarray, bp_a : numpy.ndarray, bp_b : numpy.ndarray,
    decon_function : str, num_iter : int, epsilon : float,
    boundary_correction : bool,
    zero_padding : Optional[PadType],
    boundary_sigma_a : float, boundary_sigma_b : float,
    verbose : bool,
):
    if len(view_a.shape) == 4:
        channel_slice = slice(None)
        ch_a, z_a, r_a, c_a = view_a.shape
        ch_b, z_b, r_b, c_b = view_b.shape
        assert ch_a == ch_b, "input volumes must have same # channels"
    else:
        channel_slice = None
        z_a, r_a, c_a = view_a.shape
        z_b, r_b, c_b = view_b.shape
    assert all([a==b for a, b in zip([z_a, r_a, c_a],[z_b, r_b, c_b])]), \
        "input volumes must be same shape"
    chunks = calculate_conv_chunks(z_a, r_a, c_a,
                                    chunk_size, overlap, channel_slice)
    n_gpu = cupy.cuda.runtime.getDeviceCount()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_gpu, mp_context=multiprocessing.get_context('spawn')
        ) as executor, multiprocessing.Manager() as manager:
        gpu_queue = manager.Queue()
        for gpu_id in range(n_gpu):
            gpu_queue.put(gpu_id)
        fun = partial(_decon_chunk,
                      out, view_a, view_b, psf_a, psf_b, bp_a, bp_b,
                      decon_function, num_iter, epsilon, 
                      boundary_correction, zero_padding,
                      boundary_sigma_a, boundary_sigma_b)
        futures = []
        for _, chunk in chunks.items():
            future = executor.submit(fun, chunk, gpu_queue)
            futures.append(future)
        if verbose:
            pbar = tqdm(total=len(futures), desc="Deconvolving Chunks")
        else:
            pbar = nullcontext
        with pbar:
            for future in concurrent.futures.as_completed(futures):
                val = future.result()
                if verbose:
                    pbar.update(1)
                    pbar.set_postfix({'data' : val>0})
    

def _decon_chunk(
    # shared arguments across chunks
    out : zarr.Array, 
    view_a : zarr.Array, view_b : zarr.Array,
    psf_a : numpy.ndarray, psf_b : numpy.ndarray,
    bp_a : numpy.ndarray, bp_b : numpy.ndarray,
    decon_function : str, num_iter : int, epsilon : float,
    boundary_correction : bool, 
    zero_padding : Optional[PadType],
    boundary_sigma_a : float, boundary_sigma_b : float,
    # iterate over these
    chunk_props : ChunkProps, gpu_queue,
):
    gpu_id = gpu_queue.get()
    # load data
    a = view_a.oindex[chunk_props.read_window]
    b = view_b.oindex[chunk_props.read_window]
    # in many large volumes, there's large chunks of black-space
    # (e.g. using the 'shear' or 'dispim' deskewing)
    # so just do a quick check and see if we're in one of those, and if we
    # are, return immediately (skip doing deconvolution)
    if numpy.all(a == 0) and numpy.all(b == 0): 
        gpu_queue.put(gpu_id)
        return 0
    # pad the inputs according to the chunking properties
    a = numpy.pad(a, chunk_props.paddings)
    b = numpy.pad(b, chunk_props.paddings)
    with cupy.cuda.Device(gpu_id):
        psf_a = cupy.asarray(psf_a, dtype=cupy.float32)
        psf_b = cupy.asarray(psf_b, dtype=cupy.float32)
        bp_a  = cupy.asarray(bp_a, dtype=cupy.float32)
        bp_b  = cupy.asarray(bp_b, dtype=cupy.float32)
        dec = deconvolve(cupy.asarray(a, dtype=cupy.float32),
                          cupy.asarray(b, dtype=cupy.float32),
                          None, psf_a, psf_b, bp_a, bp_b,
                          decon_function, num_iter, epsilon, True, 
                          boundary_correction, zero_padding, 
                          boundary_sigma_a, boundary_sigma_b,
                          verbose=False).get()[chunk_props.out_window]
    gpu_queue.put(gpu_id)
    out.set_orthogonal_selection(chunk_props.data_window, dec)
    return 1
