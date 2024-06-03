import math

import cupy

from .typing import NDArray
from .._util import get_skimage_module, get_fft_module


def translation(ref : NDArray, mov : NDArray,
                    upsample_factor : int=1):
    """determine translation between `ref` and `mov`

    :param ref: reference image
    :type ref: NDArray
    :param mov: image to register. must be same dimensionality as `ref`
    :type mov: NDArray
    :param upsample_factor: upsampling factor, images are registered to
        `1/upsample_factor`. defaults to 1.
    :type upsample_factor: int
    :returns: translation along each axis
    :rtype: NDArray
    """
    # get relevant modules for cpu/gpu
    xp = cupy.get_array_module(ref)
    skimage = get_skimage_module(xp)
    fft = get_fft_module(xp)
    # compute fft's and then do phase cross-correlation
    ref_fft = fft.fft2(ref)
    mov_fft = fft.fft2(mov)
    trans, _, _ = skimage.registration.phase_cross_correlation(
        ref_fft, mov_fft,
        space='fourier',
        upsample_factor=upsample_factor,
    )
    return trans


def scale_rotation(ref : NDArray, mov : NDArray,
                   upsample_factor : int=1):
    """determine scaling & rotation between `ref` & `mov` using phase cross
        correlation of the log-polar transformed inputs

    :param ref: reference image
    :type ref: NDArray
    :param mov: image to register. must be same dimensionality as `ref`
    :type mov: NDArray
    :param upsample_factor: upsampling factor, images are registered to
        `1/upsample_factor`. defaults to 1.
    :type upsample_factor: int
    :returns: rotation (in degrees) and scale difference between `ref` & `mov` 
    :rtype: Tuple[float,float]
    """
    xp = cupy.get_array_module(ref)
    skimage = get_skimage_module(xp)
    fft = get_fft_module(xp)
    radius = min([min(ref.shape), min(mov.shape)])
    ref_polar = skimage.transform.warp_polar(ref, radius=radius,
                                             scaling='log')
    mov_polar = skimage.transform.warp_polar(mov, radius=radius,
                                             scaling='log')
    shift, err, _ = skimage.registration.phase_cross_correlation(
        ref_polar, mov_polar,
        upsample_factor=upsample_factor
    )
    cc_rot = shift[0]
    scale = 1 / (math.exp(shift[1] / (radius/math.log(radius))))
    return scale, cc_rot
