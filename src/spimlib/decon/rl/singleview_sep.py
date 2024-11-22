from contextlib import nullcontext
from typing import Optional

import cupy
from tqdm.auto import tqdm

from .._util import div_stable
from ..conv._cuda import convolve_3d
from ...typing import NDArray


def deconvolve(volume : NDArray, 
               psf_z : NDArray, psf_y : NDArray, psf_x : NDArray,
               bp_z  : NDArray, bp_y  : NDArray, bp_x  : NDArray,
               num_iter : int,
               boundary_correction : bool,
               epsilon : Optional[float],
               init_constant : bool,
               boundary_padding : Optional[int],
               boundary_sigma : float,
               verbose : bool) -> NDArray:
    volume = cupy.asarray(volume, dtype=cupy.float32, order='F')
    psf_z  = cupy.asarray(psf_z,  dtype=cupy.float32)
    psf_y  = cupy.asarray(psf_y,  dtype=cupy.float32)
    psf_x  = cupy.asarray(psf_x,  dtype=cupy.float32)
    bp_z   = cupy.asarray(bp_z,   dtype=cupy.float32)
    bp_y   = cupy.asarray(bp_y,   dtype=cupy.float32)
    bp_x   = cupy.asarray(bp_x,   dtype=cupy.float32)
    if init_constant:
        raise NotImplementedError('todo')
    else:
        est_i = volume
    if boundary_correction:
        raise NotImplementedError('todo')
    else:
        return _deconvolve_uncorrected(volume, psf_z, psf_y, psf_x,
                                       bp_z, bp_y, bp_x, num_iter,
                                       epsilon, verbose)


def _deconvolve_corrected(
    volume : cupy.ndarray, est_i : cupy.ndarray,
    psf_z : cupy.ndarray, psf_y : cupy.ndarray, psf_x : cupy.ndarray,
    bp_z  : cupy.ndarray, bp_y  : cupy.ndarray, bp_x  : cupy.ndarray,
    num_iter : int,
    epsilon : float,
    boundary_pad : int,
    boundary_sigma : float,
    verbose : bool
) -> cupy.ndarray:
    raise NotImplementedError('todo')


def _deconvolve_uncorrected(
    volume : cupy.ndarray, est_i : cupy.ndarray,
    psf_z : cupy.ndarray, psf_y : cupy.ndarray, psf_x : cupy.ndarray,
    bp_z  : cupy.ndarray, bp_y  : cupy.ndarray, bp_x  : cupy.ndarray,
    num_iter : int,
    epsilon : float,
    verbose : bool
) -> cupy.ndarray:
    if verbose:
        pbar = tqdm(total=num_iter, desc="Deconvolution")
    else:
        pbar = nullcontext
    with pbar:
        for _ in range(num_iter):
            con = convolve_3d(est_i, psf_z, psf_y, psf_x)
            est_i = cupy.multiply(
                est_i, convolve_3d(div_stable(volume, con, epsilon),
                                 bp_z, bp_y, bp_x)
            )
    return est_i


def _deconvolve_corrected(volume : cupy.ndarray, 
                          psf_z : cupy.ndarray, psf_y : cupy.ndarray, psf_x : cupy.ndarray,
                          bp_z  : cupy.ndarray, bp_y  : cupy.ndarray, bp_x  : cupy.ndarray,
                          num_iter : int,
                          boundary_correction : bool,
                          epsilon : Optional[float],
                          init_constant : bool,
                          boundary_padding : Optional[int],
                          boundary_sigma : float,
                          verbose : bool):
    pass