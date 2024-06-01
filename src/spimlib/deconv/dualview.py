"""joint deconvolution algorithms

all algorithms are looped over a finite, fixed number of iterations
(as opposed to looking for convergence and terminating when converged)
"""
import cupy
import numpy
import cupyx.scipy
from scipy import signal as cpusig
from cupyx.scipy import signal as cusig
from tqdm.auto import trange

from .._util import supported_float_type


def joint_rl_dispim(view_a, view_b, psf_a, psf_b,
                    backproj_a=None, backproj_b=None,
                    num_iter : int=10, epsilon : float=1e-5,
                    req_both : bool=False, verbose : bool=False):
    """modified Richardson-Lucy joint deconvolution, as described in [1]

    this is the deconvolution algorithm developed by the Shroff group for
    use in diSPIM imaging

    inputs `view_a` and `view_b` must be pre-registered with one another
    `psf_a` and `psf_b` are the corresp. point spread functions for views a & b
    `backproj_a(b)` are back projectors for views a & b, if not specified these
    will default to the mirrored point spread functions
    `epsilon` is a small parameter that prevents division by zero errors
    
    References
    ---
    [1] Wu, et. al., "Spatially isotropic four-dimensional...", doi:10.1038/nbt.2713
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


def efficient_bayesian(view_a, view_b, psf_a, psf_b,
                       num_iter : int=10, epsilon : float=1e-5,
                       req_both : bool=False, verbose : bool=False):
    """efficient bayesian multiview deconvolution

    NOTE: this is just the additive joint deconvolution with backprojectors
    calculated as described in the "quadruple-view deconvolution" section of [1]
    
    References
    ---
    [1] Guo, et al. "Rapid image deconvolution..." doi:10.1038/s41587-020-0560-x
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


def additive_joint_rl(view_a, view_b, psf_a, psf_b,
                      backproj_a=None, backproj_b=None,
                      num_iter=10, epsilon=1e-5,
                      req_both=False, verbose=False):
    """additive joint Richardson-Lucy deconvolution
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
