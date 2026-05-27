"""Deskew by shear-warp algorithm.

Deskew the input volume using an affine transformation. Should return the same result as ``dispim`` but using shear-warp algorithm and (possibly higher-order) interpolation instead of texture-based interpolation.
"""
from math import sqrt
from typing import Tuple

import cupy
import numpy

from ..typing import NDArray
from ..interp.affine import output_shape_for_transform, transform
from .._matrix import translation_matrix


def fwd_deskew_matrix(
    pixel_size: float, step_size: float, direction: int,
    zed: int, row: int, col: int
) -> numpy.ndarray:
    """Forward deskewing matrix

    Args:
        pixel_size (float): size of pixels, in real space
        step_size (float): spacing between sheets, in real units
        direction (int): scan direction (+/-1)
        z: input volume size in z
        r: input volume size in r
        c: input volume size in c

    Returns:
        numpy.ndarray
    """
    sq2 = sqrt(2)
    a = 1 / sqrt(2)
    r = step_size / pixel_size
    u_c = (col - 1) / 2
    v_c = (row - 1) / 2
    w_c = (zed - 1) / 2

    if direction > 0:
        return numpy.asarray([
            [ a, 0, 0, -a * u_c],
            [ 0, 1, 0, -v_c],
            [ a, 0, r * sq2, -(a * u_c + r * sq2 * w_c)],
            [ 0, 0, 0, 1],
        ])
    else:
        return numpy.asarray([
            [-a, 0, 0, a * u_c],
            [ 0, 1, 0, -v_c],
            [ a, 0, -r * sq2, -(a * u_c - r * sq2 * w_c)],
            [ 0, 0, 0, 1],
        ])


def output_shape(
    z: int,
    r: int,
    c: int,
    pixel_size: float,
    step_size: float,
    direction: int,
) -> Tuple[int, int, int]:
    """Compute output shape of deskewing transform.

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
        fwd_deskew_matrix(pixel_size, step_size, direction, z, r, c), 
        (z, r, c)
    )
    return full_shp


def deskewing_transform(
    z: int,
    r: int,
    c: int,
    pixel_size: float,
    step_size: float,
    direction: int,
) -> Tuple[NDArray, Tuple[int,int,int]]:
    """Compute the deskewing transform for the input volume.

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
    T = fwd_deskew_matrix(pixel_size, step_size, direction, z, r, c)
    shp = output_shape_for_transform(T, (z, r, c))
    
    # Translate output volume coordinates (0 to shape-1) to origin-centered coordinates
    u_c_out = (shp[2] - 1) / 2
    v_c_out = (shp[1] - 1) / 2
    w_c_out = (shp[0] - 1) / 2
    
    t = translation_matrix(-u_c_out, -v_c_out, -w_c_out)
    D = numpy.linalg.inv(T) @ t
    return D, shp


def deskew_stage_scan(
    im: cupy.ndarray,
    pixel_size: float,
    step_size: float,
    direction: int,
    interp_method: str,
    preserve_dtype: bool,
    block_size: Tuple[int, int, int],
) -> cupy.ndarray:
    """Deskew the input volume using the shear-warp algorithm.

    Args:
        im (cupy.ndarray): input volume to be deskewed.
        pixel_size (float): pixel size, in real units
        step_size (float): step size, in real units
        direction (int): scan direction (+/- 1)
        rotation_thetas (Tuple[float,float,float] | None): rotation angles for each axis. If ``None``, no rotation is done.
        interp_method (str): interpolation method to use
        auto_crop (bool): auto-crop output volumes to account for rotation of view B
        preserve_dtype (bool): make output volume have same datatype as input volume (probably uint16).
        block_size (Tuple[int,int,int]): size of blocks for CUDA kernel launch.

    Returns:
        cupy.ndarray
    """
    T_inv, out_shape = deskewing_transform(
        im.shape[0],
        im.shape[1],
        im.shape[2],
        pixel_size,
        step_size,
        direction,
    )
    T_inv = cupy.asarray(T_inv).astype(cupy.float32)
    dsk = transform(
        im, T_inv, interp_method, preserve_dtype, 
        out_shape, *block_size
    )
    return dsk.swapaxes(0, 2)
