import math

import cupy
from numpy.typing import ArrayLike

from .._util import get_skimage_module, get_fft_module


def translation(ref : ArrayLike, mov : ArrayLike,
                    upsample_factor : int=1):
    """determine translation between `ref` and `mov` using phase cross correlation

    Args:
        ref (ArrayLike): reference image
        mov (ArrayLike): image to register. must be same dimensionality as `ref`
        upsample_factor (int, optional): Upsampling factor, images are registered to `1/upsample_factor`. Defaults to 1.

    Returns:
        ArrayLike: translation (in pixels) to register `mov`  with `ref`
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


def scale_rotation(ref : ArrayLike, mov : ArrayLike,
                       upsample_factor : int=1):
    """determine scaling & rotation between `ref` & `mov` using phase cross correlation of the log-polar transformed inputs

    Args:
        ref (ArrayLike): reference image
        mov (ArrayLike): image to register. must be same dimensionality as `ref`
        upsample_factor (int, optional): _description_. Defaults to 1.

    Returns:
        float : rotation (in degrees) to register `mov` with `ref`
        float : scale difference between `ref` and `mov`
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
