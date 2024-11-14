import math
from typing import Iterable, Tuple

import cupy
import numpy

from ..typing import NDArray
from .._util import launch_params_for_volume
from .._matrix import rotation_about_point_matrix
from ..interp.affine import maxblend_into_existing


def inv_deskew_matrix(pixel_size : float, step_size : float, 
                      direction : int) -> numpy.ndarray:
    return numpy.asarray(
        [[1, 0, direction * step_size/pixel_size],
         [0, 1, 0],
         [0, 0, step_size/pixel_size]]
    )


def full_shape(z : int, r : int, c : int, pixel_size : float, 
               step_size : float, theta : float) -> Tuple[int,int,int]:
    z_f = math.ceil(z * (step_size / pixel_size))
    c_f = math.ceil(c * (step_size / pixel_size * math.cos(theta)))
    return z_f, r, c_f


def translation_full(r : int, c : int, direction : int):
    if direction == 1:
        return numpy.asarray([0, 0, 0])[:,numpy.newaxis]
    elif direction == -1:
        return numpy.asarray([c-r, 0, 0])[:,numpy.newaxis]
    else:
        raise ValueError('invalid direction')


def centercrop(im : NDArray):
    xp = cupy.get_array_module(im)
    zdim, xdim = im.shape[0], im.shape[2]
    if xdim > zdim:  # crop x
        dif = (xdim - zdim) // 2
        # cropping like this will break contiguity of array, so re-enforce
        # it after cropping (b/c most downstream stuff expects it)
        return xp.ascontiguousarray(im[...,dif:-dif])
    else:
        raise NotImplementedError('todo')


def deskew_stage_scan(im : cupy.ndarray, pixel_size : float, 
                      step_size : float, direction : int, theta : float, 
                      rotation_thetas : Tuple[int,int,int]|None,
                      center_crop : bool, interp_method : str,
                      block_size_z : int,
                      block_size_y : int, 
                      block_size_x : int):
    full_shp = full_shape(im.shape[0], im.shape[1], im.shape[2],
                          pixel_size, step_size, theta)
    D = inv_deskew_matrix(pixel_size, step_size, direction)
    t = translation_full(full_shp[1], full_shp[2], direction)
    M = numpy.concatenate([D, t], axis=1)
    M = numpy.concatenate([M, numpy.asarray([0,0,0,1])[numpy.newaxis,:]],
                          axis=0)
    if rotation_thetas is None:
        T = cupy.asarray(numpy.linalg.inv(M)).astype(cupy.float32)
    else:
        R = rotation_about_point_matrix(*rotation_thetas,
                                        *[s/2 for s in full_shp[::-1]])
        T = cupy.asarray(numpy.linalg.inv(M) @ R).astype(cupy.float32)
    dsk = cupy.zeros(full_shp, dtype=cupy.uint16)
    launch_params = launch_params_for_volume(
        dsk.shape, block_size_z, block_size_y, block_size_x
    )
    maxblend_into_existing(dsk, im, T, interp_method, launch_params)
    if center_crop:
        return centercrop(dsk)
    else:
        return dsk