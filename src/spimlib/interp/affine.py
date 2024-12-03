import os
import math
from itertools import product
from typing import Iterable, List, Tuple

import cupy
import numpy
from numba import njit, prange

from ..typing import NDArray, CuLaunchParameters
from .._util import launch_params_for_volume

## CUDA kernel setup and module compilation
# setup raw modules
"""with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'nearest.cu'), 'r') as f:
    __nearest_module_txt = f.read()

__nearest_ker_names = (
    'affineTransformNearest<unsigned short>'
    'affineTransformNearest<float>'
)
__cuda_module_nearest = cupy.RawModule(code=__nearest_module_txt,
                                       name_expressions=__nearest_ker_names)
__cuda_module_nearest.compile()
"""

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                       'linear.cu'), 'r') as f:
    __linear_module_txt = f.read()
__linear_ker_names = (
    'affineTransformLerp<unsigned short>',
    'affineTransformLerp<float>',
    'affineTransformLerpUShort',
    'affineTransformMaxBlend',
    'affineTransformMeanBlend'
)
__cuda_module_linear = cupy.RawModule(code=__linear_module_txt, 
                                      name_expressions=__linear_ker_names)
__cuda_module_linear.compile()

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                       'cubspl.cu'), 'r') as f:
    __cubspl_module_txt = f.read()
__cubspl_ker_names = (
    'affineTransformCubSpl<unsigned short>',
    'affineTransformCubSpl<float>',
    'affineTransformCubSplUShort',
    'affineTransformMaxBlend',
    'affineTransformMeanBlend'
)
__cuda_module_cubspl = cupy.RawModule(code=__cubspl_module_txt,
                                      name_expressions=__cubspl_ker_names)
__cuda_module_cubspl.compile()


def _get_kernel(dtype, method : str = 'linear', preserve_dtype : bool = False):
    if method == 'linear':
        if dtype == cupy.uint16:
            if preserve_dtype:
                return __cuda_module_linear.get_function(__linear_ker_names[2])
            else:
                return __cuda_module_linear.get_function(__linear_ker_names[0])
        elif dtype == cupy.float32:
            return __cuda_module_linear.get_function(__linear_ker_names[1])
        else:
            raise ValueError('invalid datatype')
    elif method == 'cubspl':
        if dtype == cupy.uint16:
            if preserve_dtype:
                return __cuda_module_cubspl.get_function(__cubspl_ker_names[2])
            else:
                return __cuda_module_cubspl.get_function(__cubspl_ker_names[0])
        elif dtype == cupy.float32:
            return __cuda_module_cubspl.get_function(__cubspl_ker_names[1])
        else:
            raise ValueError('invalid datatype')
    elif method == 'nearest':
        raise NotImplementedError('to finish')
        #if dtype == cupy.uint16:
        #    return __cuda_module_nearest.get_function(__nearest_ker_names[0])
        #elif dtype == cupy.float32:
        #    return __cuda_module_nearest.get_function(__nearest_ker_names[1])
    else:
        raise ValueError('invalid interpolation method')


## decompositions of affine transformation matrices
def decompose_transform(A : NDArray) -> Tuple[NDArray,NDArray,NDArray,NDArray]:
    xp = cupy.get_array_module(A)
    T = A[:-1,-1]
    RZS = A[:-1,:-1]
    ZS = xp.linalg.cholesky(xp.dot(RZS.T, RZS)).T
    Z = numpy.diag(ZS).copy()
    shears = ZS / Z[:,xp.newaxis]
    n = len(Z)
    S = shears[xp.triu(xp.ones((n,n)), 1).astype(bool)]
    R = xp.dot(RZS, xp.linalg.inv(ZS))
    if xp.linalg.det(R) < 0:
        Z[0] *= -1
        ZS[0] *= -1
        R = xp.dot(RZS, xp.linalg.inv(ZS))
    return T, R, Z, S


def output_shape_for_transform(T : NDArray, 
                               input_shape : Iterable) -> List[int]:
    t = T.get() if cupy.get_array_module(T) == cupy else T
    coord = list(product(*[(0,s) for s in input_shape[::-1]]))
    coord = numpy.asarray(coord).T
    coord = numpy.vstack([coord, numpy.zeros_like(coord[0,:])])
    coordT = (t @ coord)[:-1,:]
    ptp = numpy.ceil(numpy.ptp(coordT, axis=1))
    return [int(v) for v in ptp[::-1]]


def output_shape_for_inv_transform(T : NDArray, 
                                   input_shape : Iterable) -> List[int]:
    xp = cupy.get_array_module(T)
    fwd = xp.linalg.inv(T).get() if xp == cupy else xp.linalg.inv(T)
    return output_shape_for_transform(fwd, input_shape)


def transform(A : NDArray, T : NDArray, interp_method : str, 
              preserve_dtype : bool, out_shp : Tuple[int,int,int]|None,
              block_size_z : int, block_size_y : int, block_size_x : int):
    if cupy.get_array_module(A) == cupy:
        kernel = _get_kernel(A.dtype, interp_method, preserve_dtype)
        if out_shp is None:
            out_shp = output_shape_for_transform(T, A.shape)
        launch_params = launch_params_for_volume(
            out_shp, block_size_z, block_size_y, block_size_x
        )
        T = cupy.asarray(T).astype(cupy.float32)
        # preallocate output and call kernel
        out_dtype = A.dtype if preserve_dtype else cupy.float32
        out = cupy.zeros(out_shp, dtype=out_dtype)
        kernel(
            launch_params[0], launch_params[1],
            (out, A, T, *out_shp, *A.shape)
        )
        return out
    else:
        raise ValueError('only works on cupy arrays')
        #if interp_method == 'linear':
        #    return linear_interp(A, T)
        #elif interp_method == 'cubspl':
        #    return cubspl_interp(A, T)
        #else:
        #    raise ValueError('invalid interpolation method')


def maxblend_into_existing(E : cupy.ndarray, N : cupy.ndarray, T : NDArray,
                           interp_method : str,
                           launch_params : CuLaunchParameters):
    if interp_method == 'linear':
        kernel = __cuda_module_linear.get_function('affineTransformMaxBlend')
    else:
        kernel = __cuda_module_cubspl.get_function('affineTransformMaxBlend')
    kernel(
        launch_params[0], launch_params[1],
        (E, N, T, *E.shape, *N.shape)
    )


def meanblend_into_existing(S : cupy.ndarray, C : cupy.ndarray,
                            N : cupy.ndarray, T : NDArray,
                            interp_method : str,
                            launch_params : CuLaunchParameters):
    if interp_method == 'linear':
        kernel = __cuda_module_linear.get_function('affineTransformMeanBlend')
    elif interp_method == 'cubspl':
        kernel = __cuda_module_cubspl.get_function('affineTransformMeanBlend')
    else:
        raise ValueError('invalid interpolation method')
    kernel(
        launch_params[0], launch_params[1],
        (S, C, N, T, *S.shape, *N.shape)
    )


## NUMBA-optimized interpolation functions for CPU
"""
@njit
def __lerp(v0, v1, t):
    return (1-t)*v0 + t*v1

@njit
def __lerp3(A : numpy.ndarray, 
            x : int, y : int, z : int,
            dx : float, dy : float, dz : float):
    return __lerp(__lerp(__lerp(A[z,  y,  x], A[z,  y,  x+1], dx),
                         __lerp(A[z,  y+1,x], A[z,  y+1,x+1], dx), dy),
                  __lerp(__lerp(A[z+1,y,  x], A[z+1,y,  x+1], dx),
                         __lerp(A[z+1,y+1,x], A[z+1,y+1,x+1], dx), dy),
                  dz)


@njit(parallel=True)
def linear_interp(A : numpy.ndarray, T : numpy.ndarray) -> numpy.ndarray:
    sz_i, sy_i, sx_i = A.shape
    sz_o, sy_o, sx_o = output_shape_for_transform(T, [sz_i, sy_i, sx_i])
    out_size_vec = numpy.array([sz_o, sy_o, sx_o])
    out = numpy.zeros((sz_o, sy_o, sx_o), dtype=A.dtype)
    for z in prange(0, sz_o):
        for y in prange(0, sy_o):
            for x in prange(0, sx_o):
                v = numpy.asarray([x,y,z])[:,numpy.newaxis]
                v_t = T[:-1,:] @ v
                v_td = numpy.floor(v_t).astype(int)
                dv = v_t - v_td
                if numpy.all(v_td>=0) and numpy.all(v_td < out_size_vec):
                    out[z,y,x] = __lerp3(A, v_td[0], v_td[1], v_td[2],
                                         dv[0], dv[1], dv[2])
    return out


@njit
def bspline_weights(fraction : float) -> \
    Tuple[numpy.ndarray,numpy.ndarray,numpy.ndarray,numpy.ndarray]:
    one_frac = 1.0 - fraction
    squared = numpy.square(fraction)
    one_sqd = numpy.square(one_frac)
    w0 = 1.0/6.0 * one_sqd * one_frac
    w1 = 2.0/3.0 - 0.5 * squared * (2.0 - fraction)
    w2 = 2.0/3.0 - 0.5 * one_sqd * (2.0 - one_frac)
    w3 = 1.0/6.0 * squared * fraction
    return w0, w1, w2, w3


@njit(parallel=True)
def cubspl_interp(A : numpy.ndarray, T : numpy.ndarray) -> numpy.ndarray:
    sz_i, sy_i, sx_i = A.shape
    sz_o, sy_o, sx_o = output_shape_for_transform(T, [sz_i, sy_i, sx_i])
    osv = numpy.array([sz_o, sy_o, sx_o])
    out = numpy.zeros((sz_o, sy_o, sx_o), dtype=A.dtype)
    for z in prange(0, sz_o):
        for y in prange(0, sy_o):
            for x in prange(0, sx_o):
                v = numpy.asarray([x,y,z])[:,numpy.newaxis]
                v_t = T[:-1,:] @ v
                v_td = numpy.floor(v_t).astype(int)
                fraction = v_t - v_td
                w0, w1, w2, w3 = bspline_weights(fraction)
                g0, g1 = w0 + w1, w2 + w3
                h0 = w1 / g0 - 1 + v_td
                h0i = numpy.floor(h0).astype(int)
                h0f = h0 - h0i
                h1 = w3 / g1 + 1 + v_td
                h1i = numpy.floor(h1)
                h1f = h1 - h1i
                if (numpy.all(h0i > 0) and numpy.all(h1i > 0) and
                    numpy.all(h0i < osv) and numpy.all(h1i < osv)):
                    data000 = __lerp3(
                        A, h0i[0], h0i[1], h0i[2], h0f[0], h0f[1], h0f[2]
                    )
                    data100 = __lerp3(
                        A, h1i[0], h0i[1], h0i[2], h1f[0], h0f[1], h0f[2]
                    )
                    data000 = g0[0] * data000 + g1[0] * data100
                    data010 = __lerp3(
                        A, h0i[0], h1i[1], h0i[2], h0f[0], h1f[1], h0f[2]
                    )
                    data110 = __lerp3(
                        A, h1i[0], h1i[1], h0i[2], h1f[0], h1f[1], h0f[2]
                    )
                    data010 = g0[0] * data010 + g1[0] * data110
                    data000 = g0[1] * data000 + g1[1] * data010
                    data001 = __lerp3(
                        A, h0i[0], h0i[1], h1i[2], h0f[0], h0f[1], h1f[2]
                    )
                    data101 = __lerp3(
                        A, h1i[0], h0i[1], h1i[2], h1f[0], h0f[1], h1f[2]
                    )
                    data001 = g0[0] * data001 + g1[0] * data101
                    data011 = __lerp3(
                        A, h0i[0], h1i[1], h1i[2], h0f[0], h1f[1], h1f[2]
                    )
                    data111 = __lerp3(
                        A, h1i[0], h1i[1], h1i[2], h1f[0], h1f[1], h1f[2]
                    )
                    data011 = g0[0] * data011 + g1[0] * data111
                    data001 = g0[1] * data001 + g1[1] * data011
                    out[z,y,x] = g0[2] * data000 + g1[2] * data001
    return out
"""