import math
from typing import List, Optional, Tuple, Union

import cupy
from scipy.signal import fftconvolve as fftconv_cpu
from cuypx.scipy.signal import fftconvolve as fftconv_gpu
from tqdm.auto import trange

from ..typing import NDArray, PadType
from .._util import get_skimage_module, supported_float_type


def richardson_lucy(image : NDArray, psf : NDArray,
                    num_iter : int=50, 
                    boundary_correction : bool=True,
                    clip : bool=False,
                    filter_epsilon : Optional[float]=None,
                    boundary_padding : Optional[int]=None,
                    boundary_sigma : float=1e-2) -> NDArray:
    if boundary_correction:
        raise NotImplementedError('to finish')
    else:
        return richardson_lucy_skimage(
            image, psf, num_iter, clip, filter_epsilon
        )


def richardson_lucy_boundcorr(image : NDArray, psf : NDArray,
                              bp : Optional[NDArray]=None,
                              num_iter : int=50, 
                              filter_epsilon : float=1e-4,
                              zero_padding : Optional[PadType]=None,
                              boundary_sigma : float=1e-2,
                              init_constant : bool=False,
                              verbose : bool=False) -> NDArray:
    """Richardson-Lucy deconvolution with boundary correction
        described in [1], this method reduces Gibbs oscillations that occur
        due to boundary effects when doing RL deconvolution. 
        
    References
    ---
    [1] Bertero & Boccacci, "A simple method...", 
        doi:10.1051/0004-6361:20052717

    :param image: input image or volume to be deconvolved
    :type image: NDArray
    :param psf: point spread function
    :type psf: NDArray
    :param bp: backprojector, defaults to None
    :type bp: Optional[NDArray], optional
    :param num_iter: number of iterations, defaults to 50
    :type num_iter: int, optional
    :param filter_epsilon: values below which intermediate results become 0, 
        defaults to 1e-4
    :type filter_epsilon: float, optional
    :param zero_padding: amount of zero-padding to add to each axis, 
        defaults to None. if None, each axis of size N is padded on each side
        by N/2 so that the padded image has dimension 2N in that axis
    :type zero_padding: Optional[PadType], optional
    :param boundary_sigma: threshold for determining significant pixels 
        to include in window, defaults to 1e-2. 
    :type boundary_sigma: float, optional
    :param init_constant: initialize iterations with constant array, defaults to False
    :type init_constant: bool, optional
    :param verbose: show progress, defaults to False
    :type verbose: bool, optional
    :return: deconvolved image/volume
    :rtype: NDArray
    """
    assert len(image.shape) == 2 or len(image.shape) == 3, \
        "RL deconvolution only implemented for 2- and 3d inputs"
    xp = cupy.get_array_module(image)
    # make sure all inputs are floats
    float_type = supported_float_type(image.dtype)
    image = image.astype(float_type, copy=False)
    psf   = psf.astype(float_type, copy=False)
    # default back-projector is mirrored PSF
    if bp is None:
        bp = xp.ascontiguousarray(psf[::-1,::-1,::-1])
    # figure out padding
    if zero_padding is None: # use default value from paper (2N)
        # default is to pad such that an NxN image is 2Nx2N
        pad = tuple([(d//2, d//2) for d in image.shape])
    else:  
        if isinstance(zero_padding, int):
            pad = tuple([(zero_padding, zero_padding) for _ in image.shape])
        else:
            assert len(zero_padding) == len(image.shape), \
                "zero padding must be specified for all dimensions of input"
            pad = tuple([(p,p) for p in zero_padding])
    # define utility function for doing convolution
    # NOTE: by default `fftconvolve` wants the larger array as first arg
    # but for clarity, I like specifying the PSF first
    if xp == cupy:
        conv = lambda k, i: fftconv_gpu(i, k, mode='same')
    else:
        conv = lambda k, i: fftconv_cpu(i, k, mode='same')
    # generate the mask that we'll use to determine $\overbar{alpha}$
    mask = xp.ones_like(image)
    # pad the image and the mask with zeros
    image = xp.pad(image, padding=pad, mode='constant', constant_values=0)
    mask  = xp.pad(mask, padding=pad, mode='constant', constant_values=0)
    # calculate the window, $\overbar{w}(n')
    alpha = conv(bp, mask)
    window = xp.where(alpha > boundary_sigma, 1. / alpha, 0.)
    # initialization conditions
    # if *not* init_constant, just start with the data
    if init_constant:
        est = xp.ones_like(image) * xp.mean(image)
    else:
        est = image
    # Lucy-Richardson iterations
    for _ in (trange(num_iter) if verbose else range(num_iter)):
        con = conv(psf, est)
        est = xp.multiply(
            xp.multiply(window, est),  # conv(window, est)
            conv(bp, xp.where(con < filter_epsilon, 0, image / con))
        )
    # trim out padding
    if len(est) == 2:
        return est[pad[0][0]:-pad[0][1],pad[1][0]:-pad[1][1]]
    else:
        return est[pad[0][0]:-pad[0][1],
                   pad[1][0]:-pad[1][1],
                   pad[2][0]:-pad[2][1]]


def richardson_lucy_skimage(image : NDArray, psf : NDArray,
                            num_iter : int=50, clip : bool=False,
                            filter_epsilon : Optional[float]=None) -> NDArray:
    """Richardson-Lucy deconvolution
        this is just a thin wrapper around cucim/skimage that dispatches
        to the correct library based on the input being already on the
        cpu (`skimage`) or gpu (`cucim`)

    :param image: input image/volume
    :type image: NDArray
    :param psf: point spread function
    :type psf: NDArray
    :param num_iter: number of iterations
    :type num_iter: int
    :param clip: if `True`, pixel values >1 or <-1 are thresholded
        (for skimage pipeline compatability)
    :type clip: bool
    :param filter_epsilon: values below which intermediate results become 0
        (avoids division by small numbers errors)
    :type filter_epsilon: Optional[float]
    :returns: deconvolved image
    :rtype: NDArray
    """
    skimage = get_skimage_module(image)
    return skimage.restoration.richardson_lucy(
        image, psf, num_iter, clip, filter_epsilon
    )