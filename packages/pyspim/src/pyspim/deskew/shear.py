"""Deskew by shear-warp algorithm.

Deskew the input volume using an affine transformation. Should return the same result as ``dispim`` but using shear-warp algorithm and (possibly higher-order) interpolation instead of texture-based interpolation.
"""

from typing import Tuple

import cupy
import numpy

from .._matrix import rotation_about_point_matrix, translation_matrix
from ..interp.affine import output_shape_for_transform, transform


def inv_deskew_matrix(
    pixel_size: float, step_size: float, direction: int
) -> numpy.ndarray:
    """inv_deskew_matrix Inverse deskewing matrix

    Args:
        pixel_size (float): size of pixels, in real space
        step_size (float): spacing between sheets, in real units
        direction (int): scan direction (+/-1)

    Returns:
        numpy.ndarray
    """
    return numpy.asarray(
        [
            [1, 0, direction * step_size / pixel_size, 0],
            [0, 1, 0, 0],
            [0, 0, step_size / pixel_size, 0],
            [0, 0, 0, 1],
        ]
    )


def output_shape(
    z: int,
    r: int,
    c: int,
    pixel_size: float,
    step_size: float,
    direction: int,
    auto_crop: bool,
) -> Tuple[int, int, int]:
    """output_shape Compute output shape of deskewing transform.

    Args:
        z (int): shape of input volume, z-direction
        r (int): shape of input volume, r-direction (# rows)
        c (int): shape of input volume, c-direction (# cols)
        pixel_size (float): size of pixels, in real units
        step_size (float): spacing between steps, in real units
        direction (int): scan direction (+/- 1)
        auto_crop (bool): automatically crop the output to account for rotation of B-view

    Returns:
        Tuple[int,int,int]
    """
    full_shp = output_shape_for_transform(
        inv_deskew_matrix(pixel_size, step_size, direction), [z, r, c]
    )
    if auto_crop:
        return tuple([full_shp[0], full_shp[1], full_shp[0]])
    else:
        return tuple(full_shp)


def deskewing_transform(
    z: int,
    r: int,
    c: int,
    pixel_size: float,
    step_size: float,
    direction: int,
    auto_crop: bool,
    rotation_thetas: Tuple[int, int, int] | None,
):
    """deskewing_transform Compute the deskewing transform for the input volume.

    Args:
        z (int): size of input volume, z-direction (pixels)
        r (int): size of input volume, r-direction (pixels, # rows)
        c (int): size of input volume, c-direction (pixels, # cols)
        pixel_size (float): size of pixels, in real units
        step_size (float): step size, in real units
        direction (int): scan direction (+/-1)
        auto_crop (bool): auto-crop outputs to account for rotation of B into coordinate system of A.
        rotation_thetas (Tuple[int,int,int] | None): rotation angles for each axis. If ``None``, no rotation is done.

    Returns:
        numpy.ndarray, List[int]: deskewing transform & the output shape
    """
    D = inv_deskew_matrix(pixel_size, step_size, direction)
    full_shp = output_shape_for_transform(D, [z, r, c])
    if direction < 0:
        t = translation_matrix(full_shp[0], 0, 0)
        D = D @ t
    if auto_crop:
        # rotating B into view of A implies that some of the B view
        # this will take care of automatically cropping the outputs
        # to this overlap region (helps save memory)
        out_shp = [full_shp[0], full_shp[1], full_shp[0]]
        t = translation_matrix(-(full_shp[2] - full_shp[0]) / 2, 0, 0)
        D = D @ t
    else:
        out_shp = full_shp
    if rotation_thetas is None:
        T = numpy.linalg.inv(D).astype(numpy.float32)
    else:
        R = rotation_about_point_matrix(
            *rotation_thetas, *[s / 2 for s in out_shp[::-1]]
        )
        T = (numpy.linalg.inv(D) @ R).astype(numpy.float32)
    return T, out_shp


def deskew_stage_scan(
    im: cupy.ndarray,
    pixel_size: float,
    step_size: float,
    direction: int,
    rotation_thetas: Tuple[int, int, int] | None,
    interp_method: str,
    auto_crop: bool,
    preserve_dtype: bool,
    block_size: Tuple[int, int, int],
) -> cupy.ndarray:
    """deskew_stage_scan Deskew the input volume using the shear-warp algorithm.

    Args:
        im (cupy.ndarray): input volume to be deskewed.
        pixel_size (float): pixel size, in real units
        step_size (float): step size, in real units
        direction (int): scan direction (+/- 1)
        rotation_thetas (Tuple[int,int,int] | None): rotation angles for each axis. If ``None``, no rotation is done.
        interp_method (str): interpolation method to use
        auto_crop (bool): auto-crop output volumes to account for rotation of view B
        preserve_dtype (bool): make output volume have same datatype as input volume (probably uint16).
        block_size (Tuple[int,int,int]): size of blocks for CUDA kernel launch.

    Returns:
        cupy.ndarray
    """
    T, out_shape = deskewing_transform(
        im.shape[0],
        im.shape[1],
        im.shape[2],
        pixel_size,
        step_size,
        direction,
        auto_crop,
        rotation_thetas,
    )
    T = cupy.asarray(T).astype(cupy.float32)
    dsk = transform(im, T, interp_method, preserve_dtype, out_shape, *block_size)
    return dsk
