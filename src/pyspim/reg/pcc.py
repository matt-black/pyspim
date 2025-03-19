import math
from typing import Tuple

import cupy
import numpy
import skimage

from ..typing import NDArray
from .._util import get_fft_module, pad_to_same_size


def translation(ref : NDArray, mov : NDArray,
                upsample_factor : float=1) -> NDArray:
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
    fft = get_fft_module(xp)
    if any([rd!=md for rd, md in zip(ref.shape, mov.shape)]):
        ref, mov = pad_to_same_size(ref, mov, style='right')
    # compute fft's and then do phase cross-correlation
    ref_fft = fft.fft2(ref)
    mov_fft = fft.fft2(mov)
    if xp == cupy:
        ref_fft, mov_fft = ref_fft.get(), mov_fft.get()
    trans, _, _ = skimage.registration.phase_cross_correlation(
        ref_fft, mov_fft,
        space='fourier',
        upsample_factor=upsample_factor,
    )
    return trans


def scale_rotation(ref : NDArray, mov : NDArray,
                   upsample_factor : float=1) -> Tuple[float,float]:
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
    if xp == cupy:
        ref, mov = ref.get(), mov.get()
    if any([rd!=md for rd, md in zip(ref.shape, mov.shape)]):
        ref, mov = pad_to_same_size(ref, mov, style='right')
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


def rotation_scale_for_volumes(ref : NDArray, mov : NDArray,
                               upsample_factor : float=1.0):
    """calculate rotation, and scaling between input volumes by 
        phase cross correlation of the maximum projections in each axis

    :param ref: reference volume
    :type ref: NDArray
    :param mov: 'moving' volume to be registered to `ref`
    :type mov: NDArray
    :param upsample_factor: upsampling factor. images are registered to
        `1/upsample_factor`. defaults to 1
    :type upsample_factor: float
    :returns: rotation and scaling for each axis
    :rtype: Tuple[NDArray,NDArray]
    """
    xp = cupy.get_array_module(ref)
    # calculate max-projections for XY
    ref_xy, mov_xy = xp.amax(ref, axis=0), xp.amax(mov, axis=0)
    scale_xy, rot_xy = scale_rotation(ref_xy, mov_xy, upsample_factor)
    # repeat for ZX
    ref_zx, mov_zx = xp.amax(ref, axis=1), xp.amax(mov, axis=1)
    scale_zx, rot_zx = scale_rotation(ref_zx, mov_zx, upsample_factor)
    # and again for ZY
    ref_zy, mov_zy = xp.amax(ref, axis=2), xp.amax(mov, axis=2)
    scale_zy, rot_zy = scale_rotation(ref_zy, mov_zy, upsample_factor)
    # rotations & scalings are scalars, just concat them
    rot = numpy.asarray([rot_zy, rot_zx, rot_xy])
    scale  = numpy.asarray([scale_zy, scale_zx, scale_xy])
    return [float(r) for r in rot[::-1]], [float(s) for s in scale[::-1]]


def translation_for_volumes(ref : NDArray, mov : NDArray,
                            upsample_factor : float=1.0):
    """calculate relative translation, rotation, and scaling between input
        volumes by phase cross correlation of the maximum projections
        in each axis

    :param ref: reference volume
    :type ref: NDArray
    :param mov: 'moving' volume to be registered to `ref`
    :type mov: NDArray
    :param upsample_factor: upsampling factor. images are registered to
        `1/upsample_factor`. defaults to 1
    :type upsample_factor: float
    :returns: translation, rotation, and scaling for each axis
    :rtype: Tuple[NDArray,NDArray,NDArray]
    """
    xp = cupy.get_array_module(ref)
    # calculate max-projections for XY
    ref_xy = xp.amax(ref, axis=0)
    mov_xy = xp.amax(mov, axis=0)
    trans_xy = translation(ref_xy, mov_xy, upsample_factor)
    trans_xy = numpy.concatenate(
        [numpy.asarray([numpy.inf]), numpy.asarray(trans_xy)]
    )
    # repeat for ZX
    ref_zx = xp.amax(ref, axis=1)
    mov_zx = xp.amax(mov, axis=1)
    trans_zx = translation(ref_zx, mov_zx, upsample_factor)
    trans_zx = numpy.concatenate([numpy.asarray([trans_zx[0]]),
                                  numpy.asarray([numpy.inf]),
                                  numpy.asarray([trans_zx[1]])])
    # and again for ZY
    ref_zy = xp.amax(ref, axis=2)
    mov_zy = xp.amax(mov, axis=2)
    trans_zy = translation(ref_zy, mov_zy, upsample_factor)
    trans_zy = numpy.concatenate(
        [numpy.asarray(trans_zy), numpy.asarray([numpy.inf])]
    )
    # assemble outputs
    # starting translation is just whichever the smallest one
    # calculated for each dimension
    trans = numpy.amin(
        numpy.vstack([trans_xy, trans_zx, trans_zy]), axis=0
    )
    # NOTE: _ab corresponds to the translation for the 'c' axis so right
    # now its formatted like ZRC
    return [float(t) for t in trans[::-1]]  # this makes output XYZ


def translation_rotation_scale_for_volumes(ref : NDArray, mov : NDArray,
                                           upsample_factor : float=1.0):
    """calculate relative translation, rotation, and scaling between input
        volumes by phase cross correlation of the maximum projections
        in each axis

    :param ref: reference volume
    :type ref: NDArray
    :param mov: 'moving' volume to be registered to `ref`
    :type mov: NDArray
    :param upsample_factor: upsampling factor. images are registered to
        `1/upsample_factor`. defaults to 1
    :type upsample_factor: float
    :returns: translation, rotation, and scaling for each axis
    :rtype: Tuple[NDArray,NDArray,NDArray]
    """
    xp = cupy.get_array_module(ref)
    # calculate max-projections for XY
    ref_xy = xp.amax(ref, axis=0)
    mov_xy = xp.amax(mov, axis=0)
    trans_xy = translation(ref_xy, mov_xy, upsample_factor)
    trans_xy = numpy.concatenate(
        [numpy.asarray([numpy.inf]), numpy.asarray(trans_xy)]
    )
    scale_xy, rot_xy = scale_rotation(ref_xy, mov_xy, upsample_factor)
    # repeat for ZX
    ref_zx = xp.amax(ref, axis=1)
    mov_zx = xp.amax(mov, axis=1)
    trans_zx = translation(ref_zx, mov_zx, upsample_factor)
    trans_zx = numpy.concatenate([numpy.asarray([trans_zx[0]]),
                                  numpy.asarray([numpy.inf]),
                                  numpy.asarray([trans_zx[1]])])
    scale_zx, rot_zx = scale_rotation(ref_zx, mov_zx, upsample_factor)
    # and again for ZY
    ref_zy = xp.amax(ref, axis=2)
    mov_zy = xp.amax(mov, axis=2)
    trans_zy = translation(ref_zy, mov_zy, upsample_factor)
    trans_zy = numpy.concatenate(
        [numpy.asarray(trans_zy), numpy.asarray([numpy.inf])]
    )
    scale_zy, rot_zy = scale_rotation(ref_zy, mov_zy, upsample_factor)
    # assemble outputs
    # starting translation is just whichever the smallest one
    # calculated for each dimension
    trans = numpy.amin(
        numpy.vstack([trans_xy, trans_zx, trans_zy]), axis=0
    )[::-1]
    # rotations & scalings are scalars, just concat them
    rot = numpy.asarray([rot_zy, rot_zx, rot_xy])[::-1]
    scale  = numpy.asarray([scale_zy, scale_zx, scale_xy])[::-1]
    return trans, rot, scale