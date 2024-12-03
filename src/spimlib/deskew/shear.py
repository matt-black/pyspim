from typing import Tuple

import cupy
import numpy

from ..interp.affine import transform, output_shape_for_transform
from .._matrix import translation_matrix, rotation_about_point_matrix


def inv_deskew_matrix(pixel_size : float, step_size : float, 
                      direction : int) -> numpy.ndarray:
    return numpy.asarray(
        [[1, 0, direction * step_size/pixel_size, 0],
         [0, 1, 0, 0],
         [0, 0, step_size/pixel_size, 0],
         [0, 0, 0, 1]]
    )


def deskewing_transform(z : int, r : int, c : int, 
                        pixel_size : float, step_size : float,
                        direction : int, auto_crop : bool,
                        rotation_thetas : Tuple[int,int,int]|None):
    D = inv_deskew_matrix(pixel_size, step_size, direction)
    full_shp = output_shape_for_transform(D, [z, r, c])
    if direction < 0:
        t = translation_matrix(full_shp[0], 0, 0)
        D = D @ t
    if auto_crop:
        # rotating B into view of A implies that some of the B view
        # this will take care of automatically cropping the outputs
        # to this overlap region (helps save memory)
        out_shp = [full_shp[0],full_shp[1], full_shp[0]]
        t = translation_matrix(-(full_shp[2]-full_shp[0])/2, 0, 0)
        D = D @ t
    else:
        out_shp = full_shp
    if rotation_thetas is None:
        T = numpy.linalg.inv(D).astype(numpy.float32)
    else:
        R = rotation_about_point_matrix(*rotation_thetas,
                                        *[s/2 for s in out_shp[::-1]])
        T = (numpy.linalg.inv(D) @ R).astype(numpy.float32)
    return T, out_shp


def deskew_stage_scan(im : cupy.ndarray, pixel_size : float, 
                      step_size : float, direction : int,
                      rotation_thetas : Tuple[int,int,int]|None,
                      interp_method : str, auto_crop : bool,
                      block_size : Tuple[int,int,int]) -> cupy.ndarray:
    T, out_shape = deskewing_transform(im.shape[0], im.shape[1], im.shape[2],
                                       pixel_size, step_size, direction,
                                       auto_crop, rotation_thetas)
    T = cupy.asarray(T).astype(cupy.float32)
    dsk = cupy.zeros(out_shape, dtype=cupy.uint16)
    dsk = transform(im, T, interp_method, True, out_shape, *block_size)
    return dsk