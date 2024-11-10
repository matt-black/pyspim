"""deconvolution algorithms for jointly deconvolving dual-view microscopy data
"""
from contextlib import nullcontext
from typing import Callable, Optional, Tuple

import cupy
import numpy

from scipy.signal import fftconvolve as fftconv_cpu
from cupyx.scipy.signal import fftconvolve as fftconv_gpu
from tqdm.auto import trange

from ..typing import NDArray, PadType
from .._util import supported_float_type


def joint_rl_dispim(view_a : NDArray, view_b : NDArray,
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
                    init_constant       : bool = False,
                    verbose             : bool = False) -> NDArray:
    """modified Richardson-Lucy joint deconvolution, as described in [1]
        originally developed by the Shroff group at the NIH
        for use in diSPIM imaging.
        NOTE: boundary correction is not implemented for this method, as at
        the time of writing, there is no known way of doing this

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
    :param boundary_correction: correct boundary effects, defaults to True
    :type boundary_correction: bool, optional
    :param req_both: only deconvolve areas where views a & b have data,
        defaults to False
    :type req_both: bool, optional
    :param zero_padding: amount of zero-padding to add to each axis,
        defaults to None. if None, each axis of size N is padded on each side
        by N/2 so that the padded image has dimension 2N in that axis
    :type zero_padding: Optional[PadType], optional
    :param boundary_sigma_a: threshold for determining significant pixels 
        in view A, defaults to 1e-2
    :type boundary_sigma_a: float, optional
    :param boundary_sigma_b: threshold for determining significant pixels 
        in view B, defaults to 1e-2
    :type boundary_sigma_b: float, optional
    :param init_constant: initialize iterations with constant array, 
        defaults to False
    :type init_constant: bool, optional
    :param verbose: display progress bar, defaults to False
    :type verbose: bool, optional
    :returns: deconvolved volume
    :rtype: NDArray
    """
    if boundary_correction:
        return _additive_joint_rl_boundcorr(
            view_a, view_b, psf_a, psf_b, backproj_a, backproj_b,
            num_iter, epsilon, 
            zero_padding, boundary_sigma_a, boundary_sigma_b, init_constant,
            req_both, verbose
        )
    else:
        return _joint_rl_dispim_uncorr(
            view_a, view_b, psf_a, psf_b, backproj_a, backproj_b,
            num_iter, epsilon, req_both, verbose
        )


def _joint_rl_dispim_uncorr(view_a : NDArray, view_b : NDArray,
                            psf_a  : NDArray, psf_b  : NDArray,
                            backproj_a : Optional[NDArray] = None,
                            backproj_b : Optional[NDArray] = None,
                            num_iter   : int = 10,
                            epsilon    : float = 1e-5,
                            req_both   : bool = False,
                            verbose    : bool = False) -> NDArray:
    """
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
        conv = lambda k, i: fftconv_gpu(i, k, mode='same')
    else:
        conv = lambda k, i: fftconv_cpu(i, k, mode='same')
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


def efficient_bayesian_backprojectors(psf_a : NDArray, psf_b : NDArray) -> \
    Tuple[NDArray, NDArray]:
    """calculate proper backprojectors for "Efficient Bayesian" deconvolution

    :param psf_a: point spread function for view A
    :type psf_a: NDArray
    :param psf_b: point spread function for view B
    :type psf_b: NDArray
    :return: (backprojector_a, backprojector_b), 
        tuple of backprojectors, one for each view
    :rtype: Tuple[NDArray, NDArray]
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
                       psf_a  : NDArray, psf_b  : NDArray,
                       num_iter            : int = 10,
                       epsilon             : float = 1e-5,
                       boundary_correction : bool = True,
                       req_both            : bool = False,
                       zero_padding        : Optional[PadType] = None,
                       boundary_sigma_a    : float = 1e-2,
                       boundary_sigma_b    : float = 1e-2,
                       init_constant       : bool = False,
                       verbose             : bool = False) -> NDArray:
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
    :param boundary_correction: correct boundary effects, defaults to True
    :type boundary_correction: bool, optional
    :param req_both: only deconvolve areas where views a & b have data,
        defaults to False
    :type req_both: bool, optional
    :param zero_padding: amount of zero-padding to add to each axis,
        defaults to None. if None, each axis of size N is padded on each side
        by N/2 so that the padded image has dimension 2N in that axis
    :type zero_padding: Optional[PadType], optional
    :param boundary_sigma_a: threshold for determining significant pixels 
        in view A, defaults to 1e-2
    :type boundary_sigma_a: float, optional
    :param boundary_sigma_b: threshold for determining significant pixels 
        in view B, defaults to 1e-2
    :type boundary_sigma_b: float, optional
    :param init_constant: initialize iterations with constant array, 
        defaults to False
    :type init_constant: bool, optional
    :param verbose: display progress bar, defaults to False
    :type verbose: bool, optional
    :returns: deconvolved volume
    :rtype: NDArray
    """
    # compute backprojectors
    backproj_a, backproj_b = efficient_bayesian_backprojectors(psf_a, psf_b)
    return additive_joint_rl(
        view_a, view_b, psf_a, psf_b, backproj_a, backproj_b,
        num_iter, epsilon, boundary_correction, req_both, zero_padding,
        boundary_sigma_a, boundary_sigma_b, init_constant, verbose
    )


def additive_joint_rl(view_a : NDArray, view_b : NDArray,
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
                      init_constant       : bool = False,
                      verbose             : bool = False) -> NDArray:
    """additive joint deconvolution.
        with boundary correction turned off, each view is deconvolved
        separately and then averaged at each iteration. 
        with boundary correction turned on, an ordered subset EM algorithm
        from [1] is used. 

    References
    ---
    [1] Anconelli, B., et al. "Reduction of boundary effects...",
        doi:10.1051/0004-6361:20053848

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
    :param boundary_correction: correct boundary effects, defaults to True
    :type boundary_correction: bool, optional
    :param req_both: only deconvolve areas where views a & b have data,
        defaults to False
    :type req_both: bool, optional
    :param zero_padding: amount of zero-padding to add to each axis,
        defaults to None. if None, each axis of size N is padded on each side
        by N/2 so that the padded image has dimension 2N in that axis
    :type zero_padding: Optional[PadType], optional
    :param boundary_sigma_a: threshold for determining significant pixels 
        in view A, defaults to 1e-2
    :type boundary_sigma_a: float, optional
    :param boundary_sigma_b: threshold for determining significant pixels 
        in view B, defaults to 1e-2
    :type boundary_sigma_b: float, optional
    :param init_constant: initialize iterations with constant array, 
        defaults to False
    :type init_constant: bool, optional
    :param verbose: display progress bar, defaults to False
    :type verbose: bool, optional
    :returns: deconvolved volume
    :rtype: NDArray
    """
    if boundary_correction:
        return _additive_joint_rl_boundcorr(
            view_a, view_b, psf_a, psf_b, backproj_a, backproj_b,
            num_iter, epsilon,
            zero_padding, boundary_sigma_a, boundary_sigma_b, init_constant,
            req_both, verbose
        )
    else:
        return _additive_joint_rl_uncorr(
            view_a, view_b, psf_a, psf_b, backproj_a, backproj_b,
            num_iter, epsilon, req_both, verbose
        )


def _additive_joint_rl_boundcorr(view_a : NDArray, view_b : NDArray,
                                 psf_a  : NDArray, psf_b  : NDArray,
                                 backproj_a : Optional[NDArray] = None,
                                 backproj_b : Optional[NDArray] = None,
                                 num_iter : int = 10,
                                 epsilon  : float = 1e-5,
                                 zero_padding : Optional[PadType] = None,
                                 boundary_sigma_a : float = 1e-2,
                                 boundary_sigma_b : float = 1e-2,
                                 init_constant : bool = False,
                                 req_both : bool = False,
                                 verbose  : bool = False) -> NDArray:
    """Dual-view additive joint Richardson-Lucy deconvolution 
        with boundary correction
        NOTE: this implementation uses ordered subset expectation maximization
        and so doesn't iterate in the same way as the boundary-uncorrected 
        version. 

    References
    ---
    [1] Anconelli, B., et al. "Reduction of boundary effects...",
        doi:10.1051/0004-6361:20053848

    :param view_a: array of data corresp. to view A
    :type view_a: NDArray
    :param view_b: array of data corresp. to view B
    :type view_b: NDArray
    :param psf_a: point spread function for view A
    :type psf_a: NDArray
    :param psf_b: point spread function for view B
    :type psf_b: NDArray
    :param backproj_a: backprojector for view A, defaults to None
    :type backproj_a: Optional[NDArray], optional
    :param backproj_b: backprojector for view B, defaults to None
    :type backproj_b: Optional[NDArray], optional
    :param num_iter: number of Richardson-Lucy iterations, defaults to 10
    :type num_iter: int, optional
    :param epsilon: small parameter to prevent division by small numbers, 
        values below which intermediate results become 0, defaults to 1e-5
    :type epsilon: float, optional
    :param zero_padding: amount of zero-padding to add to each axis,
        defaults to None. if None, each axis of size N is padded on each side
        by N/2 so that the padded image has dimension 2N in that axis
    :type zero_padding: Optional[PadType], optional
    :param boundary_sigma_a: threshold for determining significant pixels 
        in view A, defaults to 1e-2
    :type boundary_sigma_a: float, optional
    :param boundary_sigma_b: threshold for determining significant pixels 
        in view B, defaults to 1e-2
    :type boundary_sigma_b: float, optional
    :param init_constant: initialize iterations with constant array, 
        defaults to False
    :type init_constant: bool, optional
    :param req_both: set all pixels that dont have data from both views to 0, 
        defaults to False
    :type req_both: bool, optional
    :param verbose: show progress bar, defaults to False
    :type verbose: bool, optional
    :return: deconvolved image/volume
    :rtype: NDArray
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
    if init_constant:
        # initialize the estimate as a constant value that can satisfy
        # the constraint, (Eqn. 16 of [1])
        est = xp.ones_like(view_a) * (c / xp.prod(xp.asarray(alpha.shape)))
    else:
        # otherwise, do (view_a + view_b) / 2. as is done typically in
        # microscopy RL decon. 
        # (NOTE: this still satisfies flux constraint, so ok)
        est = (view_a + view_b) / 2.
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
                    conv(backproj_a, xp.where(con < epsilon, 0, view_a / con))
                )
            else:  # use view_b
                con = conv(psf_b, est)
                est = xp.multiply(
                    xp.multiply(window_b, est), #conv(window_b, est),
                    conv(backproj_b, xp.where(con < epsilon, 0, view_b / con))
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
                              psf_a  : NDArray, psf_b  : NDArray,
                              backproj_a : Optional[NDArray] = None,
                              backproj_b : Optional[NDArray] = None,
                              num_iter : int = 10,
                              epsilon  : float = 1e-5,
                              req_both : bool = False,
                              verbose  : bool = False) -> NDArray:
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