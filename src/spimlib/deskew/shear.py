import math
from typing import Tuple

import cupy
import numpy

from ..typing import NDArray
from .._matrix import rotation_about_point_matrix
from ..interp.affine import transform


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


def _crop(im : NDArray, ctype : str):
    zdim, xdim = im.shape[0], im.shape[2]
    flag = ctype[0].lower()
    if xdim > zdim:
        dif = (xdim - zdim) // 2
        if flag == 'l':
            return im[...,dif:]
        elif flag == 'r':
            return im[...,:-dif]
        elif flag == 'c':
            return im[...,dif:-dif]
        else:
            raise ValueError('invalid crop type')
    else:
        raise NotImplementedError('todo')


def deskewing_transform(z : int, r : int, c : int, 
                        pixel_size : float, step_size : float,
                        theta : float, direction : int,
                        rotation_thetas : Tuple[int,int,int]|None):
    full_shp = full_shape(z, r, c, pixel_size, step_size, theta)
    D = inv_deskew_matrix(pixel_size, step_size, direction)
    t = translation_full(full_shp[1], full_shp[2], direction)
    M = numpy.concatenate([D, t], axis=1)
    M = numpy.concatenate([M, numpy.asarray([0,0,0,1])[numpy.newaxis,:]],
                          axis=0)
    if rotation_thetas is None:
        T = numpy.linalg.inv(M).astype(numpy.float32)
    else:
        R = rotation_about_point_matrix(*rotation_thetas,
                                        *[s/2 for s in full_shp[::-1]])
        T = (numpy.linalg.inv(M) @ R).astype(numpy.float32)
    return T, full_shp


def deskew_stage_scan(im : cupy.ndarray, pixel_size : float, 
                      step_size : float, direction : int, theta : float, 
                      rotation_thetas : Tuple[int,int,int]|None,
                      interp_method : str, auto_crop : str|None, 
                      block_size_z : int,
                      block_size_y : int, 
                      block_size_x : int):
    T, out_shape = deskewing_transform(im.shape[0], im.shape[1], im.shape[2],
                                       pixel_size, step_size, theta, direction,
                                       rotation_thetas)
    T = cupy.asarray(T).astype(cupy.float32)
    dsk = cupy.zeros(out_shape, dtype=cupy.uint16)
    dsk = transform(im, T, interp_method, True, out_shape, 
                    block_size_z, block_size_y, block_size_x)
    if auto_crop is not None:
        return _crop(dsk, auto_crop)
    else:
        return dsk