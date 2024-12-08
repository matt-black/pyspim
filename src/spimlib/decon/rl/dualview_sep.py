import time
import multiprocessing
import concurrent.futures
from warnings import warn
from itertools import repeat
from functools import partial
from contextlib import nullcontext
from typing import Iterable, Optional, Tuple

import numpy
import cupy
import zarr
from tqdm.auto import tqdm, trange

from .._util import calculate_decon_chunks, ChunkProps
from .._util import gaussian_kernel_1d, div_stable, initialize_estimate
from ...conv._cuda import make_conv_module, calc_launch_params, convolve_3d
from ...typing import NDArray, PadType, CuLaunchParameters


def _deconvolve_single_channel(view_a : NDArray, view_b : NDArray, 
                               est_i : NDArray|None,
                               psf_az : NDArray, psf_ay : NDArray, 
                               psf_ax : NDArray, psf_bz : NDArray,
                               psf_by : NDArray, psf_bx : NDArray,
                               bp_az  : NDArray, bp_ay  : NDArray, 
                               bp_ax  : NDArray, bp_bz  : NDArray, 
                               bp_by  : NDArray, bp_bx  : NDArray,
                               decon_function : str,
                               num_iter : int,
                               epsilon : float,
                               boundary_correction : bool,
                               zero_padding : Optional[PadType],
                               boundary_sigma_a : float,
                               boundary_sigma_b : float,
                               verbose : bool) -> NDArray:
    if decon_function == 'additive':
        fun = additive_joint_rl
    elif decon_function == 'dispim':
        fun = joint_rl_dispim
    else:
        raise ValueError('invalid deconvolution function')
    view_a = cupy.asarray(view_a, dtype=cupy.float32, order='F')
    view_b = cupy.asarray(view_b, dtype=cupy.float32, order='F')
    if est_i is None:
        est_i = initialize_estimate(view_a, view_b, order='F')
    else:
        est_i  = cupy.asarray(est_i,  dtype=cupy.float32, order='F')
    psf_az = cupy.asarray(psf_az, dtype=cupy.float32)
    psf_ay = cupy.asarray(psf_ay, dtype=cupy.float32)
    psf_ax = cupy.asarray(psf_ax, dtype=cupy.float32)
    psf_bz = cupy.asarray(psf_bz, dtype=cupy.float32)
    psf_by = cupy.asarray(psf_by, dtype=cupy.float32)
    psf_bx = cupy.asarray(psf_bx, dtype=cupy.float32)
    bp_az  = cupy.asarray(bp_az , dtype=cupy.float32)
    bp_ay  = cupy.asarray(bp_ay , dtype=cupy.float32)
    bp_ax  = cupy.asarray(bp_ax , dtype=cupy.float32)
    bp_bz  = cupy.asarray(bp_bz , dtype=cupy.float32)
    bp_by  = cupy.asarray(bp_by , dtype=cupy.float32)
    bp_bx  = cupy.asarray(bp_bx , dtype=cupy.float32)
    return fun(view_a, view_b, est_i,
               psf_az, psf_ay, psf_ax,
               psf_bz, psf_by, psf_bx,
               bp_az, bp_ay, bp_ax,
               bp_bz, bp_by, bp_bx,
               num_iter,
               epsilon,
               boundary_correction,
               zero_padding,
               boundary_sigma_a,
               boundary_sigma_b,
               None, None,
               verbose)


def _deconvolve_multichannel(
    view_a : NDArray, view_b : NDArray, est_i : NDArray|None, 
    psf_az : NDArray|Iterable[NDArray], psf_ay : NDArray|Iterable[NDArray], 
    psf_ax : NDArray|Iterable[NDArray], psf_bz : NDArray|Iterable[NDArray],
    psf_by : NDArray|Iterable[NDArray], psf_bx : NDArray|Iterable[NDArray],
    bp_az  : NDArray|Iterable[NDArray], bp_ay  : NDArray|Iterable[NDArray], 
    bp_ax  : NDArray|Iterable[NDArray], bp_bz  : NDArray|Iterable[NDArray], 
    bp_by  : NDArray|Iterable[NDArray], bp_bx  : NDArray|Iterable[NDArray],
    decon_function : str,
    num_iter : int,
    epsilon : float,
    boundary_correction : bool,
    zero_padding : Optional[PadType],
    boundary_sigma_a : float,
    boundary_sigma_b : float,
    verbose : bool
) -> NDArray:
    n_chan = view_a.shape[0]
    psf_az = psf_az if isinstance(psf_az, list) else repeat(psf_az)
    psf_ay = psf_ay if isinstance(psf_ay, list) else repeat(psf_ay)
    psf_ax = psf_ax if isinstance(psf_ax, list) else repeat(psf_ax)
    psf_bz = psf_bz if isinstance(psf_bz, list) else repeat(psf_bz)
    psf_by = psf_by if isinstance(psf_by, list) else repeat(psf_by)
    psf_bx = psf_bx if isinstance(psf_bx, list) else repeat(psf_bx)
    bp_az  = bp_az if isinstance(bp_az, list) else repeat(bp_az)
    bp_ay  = bp_ay if isinstance(bp_ay, list) else repeat(bp_ay)
    bp_ax  = bp_ax if isinstance(bp_ax, list) else repeat(bp_ax)
    bp_bz  = bp_bz if isinstance(bp_bz, list) else repeat(bp_bz)
    bp_by  = bp_by if isinstance(bp_by, list) else repeat(bp_by)
    bp_bx  = bp_bx if isinstance(bp_bx, list) else repeat(bp_bx)
    pfun = partial(_deconvolve_single_channel,
                   decon_function=decon_function,
                   num_iter=num_iter, epsilon=epsilon, 
                   boundary_correction=boundary_correction,
                   zero_padding=zero_padding,
                   boundary_sigma_a=boundary_sigma_a,
                   boundary_sigma_b=boundary_sigma_b,
                   verbose=verbose)
    return cupy.stack([
        pfun(view_a[i,...], view_b[i,...], 
             (est_i[i,...] if est_i is not None else None),
             paz, pay, pax, pbz, pby, pbx, baz, bay, bax, bbz, bby, bbx)
        for i, paz, pay, pax, pbz, pby, pbx, baz, bay, bax, bbz, bby, bbx
        in zip(range(n_chan), psf_az, psf_ay, psf_ax, psf_bz, psf_by, psf_bx,
               bp_az, bp_ay, bp_ax, bp_bz, bp_by, bp_bx)
    ], axis=0)


def deconvolve(
    view_a : NDArray, view_b : NDArray, est_i : NDArray|None,
    psf_az : NDArray|Iterable[NDArray], psf_ay : NDArray|Iterable[NDArray], 
    psf_ax : NDArray|Iterable[NDArray], psf_bz : NDArray|Iterable[NDArray],
    psf_by : NDArray|Iterable[NDArray], psf_bx : NDArray|Iterable[NDArray],
    bp_az  : NDArray|Iterable[NDArray], bp_ay  : NDArray|Iterable[NDArray], 
    bp_ax  : NDArray|Iterable[NDArray], bp_bz  : NDArray|Iterable[NDArray], 
    bp_by  : NDArray|Iterable[NDArray], bp_bx  : NDArray|Iterable[NDArray],
    decon_function : str,
    num_iter : int,
    epsilon : float,
    boundary_correction : bool,
    zero_padding : Optional[PadType],
    boundary_sigma_a : float,
    boundary_sigma_b : float,
    verbose : bool
) -> NDArray:
    if len(view_a.shape) == 4:
        fun = _deconvolve_multichannel
    elif len(view_a.shape) == 3:
        fun = _deconvolve_single_channel
    else:
        raise ValueError('invalid shape, must be 3 or 4D')
    return fun(view_a, view_b, est_i,
               psf_az, psf_ay, psf_ax, psf_bz, psf_by, psf_bx,
               bp_az, bp_ay, bp_ax, bp_bz, bp_by, bp_bx,
               decon_function, num_iter, epsilon, 
               boundary_correction, zero_padding, 
               boundary_sigma_a, boundary_sigma_b,
               verbose)


def additive_joint_rl(
    view_a : cupy.ndarray, view_b : cupy.ndarray, est_i  : cupy.ndarray,
    psf_az : cupy.ndarray, psf_ay : cupy.ndarray, psf_ax : cupy.ndarray,
    psf_bz : cupy.ndarray, psf_by : cupy.ndarray, psf_bx : cupy.ndarray,
    bp_az  : cupy.ndarray, bp_ay  : cupy.ndarray, bp_ax  : cupy.ndarray,
    bp_bz  : cupy.ndarray, bp_by  : cupy.ndarray, bp_bx  : cupy.ndarray,
    num_iter : int,
    epsilon : float,
    boundary_correction : bool,
    zero_padding : Optional[PadType],
    boundary_sigma_a : float,
    boundary_sigma_b : float,
    conv_module : Optional[cupy.RawModule],
    launch_pars : Optional[CuLaunchParameters],
    verbose : bool,
) -> cupy.ndarray:
    # compile the convolution module
    if conv_module is None:
        conv_module = make_conv_module(len(psf_az)//2)
    if launch_pars is None:
        launch_pars = calc_launch_params(len(psf_az)//2, view_a.shape)
    if boundary_correction:
        # determine zero padding
        if zero_padding is None:
            pad = tuple([(d//2, d//2) for d in view_a.shape])
        else:
            if isinstance(zero_padding, int):
                pad = tuple(
                    [(zero_padding, zero_padding) for _ in view_a.shape]
                )
            else:
                assert len(zero_padding) == len(view_a.shape), \
                    "zero-padding must be specified for all dimensions of input"
                pad = tuple([(p,p) for p in zero_padding])
        return _additive_joint_rl_boundcorr(view_a, view_b, est_i,
                                            psf_az, psf_ay, psf_ax,
                                            psf_bz, psf_by, psf_bx,
                                            bp_az, bp_ay, bp_ax,
                                            bp_bz, bp_by, bp_bx,
                                            num_iter, epsilon, pad,
                                            boundary_sigma_a,
                                            boundary_sigma_b,
                                            conv_module, launch_pars,
                                            verbose)
    else:
        return _additive_joint_rl_uncorr(view_a, view_b, est_i,
                                         psf_az, psf_ay, psf_ax, 
                                         psf_bz, psf_by, psf_bx,
                                         bp_az, bp_ay, bp_ax,
                                         bp_bz, bp_by, bp_bx,
                                         num_iter, epsilon, 
                                         conv_module, launch_pars,
                                         False)


def _additive_joint_rl_boundcorr(
    view_a : cupy.ndarray, view_b : cupy.ndarray, est_i  : cupy.ndarray,
    psf_az : cupy.ndarray, psf_ay : cupy.ndarray, psf_ax : cupy.ndarray,
    psf_bz : cupy.ndarray, psf_by : cupy.ndarray, psf_bx : cupy.ndarray,
    bp_az  : cupy.ndarray, bp_ay  : cupy.ndarray, bp_ax  : cupy.ndarray,
    bp_bz  : cupy.ndarray, bp_by  : cupy.ndarray, bp_bx  : cupy.ndarray,
    num_iter : iter,
    epsilon : float,
    zero_padding : PadType,
    boundary_sigma_a : float,
    boundary_sigma_b : float,
    cuda_conv_module : cupy.RawModule, 
    launch_params : CuLaunchParameters,
    verbose : bool
) -> cupy.ndarray:
    conv = partial(convolve_3d, 
                   cuda_module=cuda_conv_module,
                   launch_pars=launch_params)
    c = cupy.sum(est_i)  # compute flux constant
    mask_a, mask_b = cupy.ones_like(view_a), cupy.ones_like(view_b)
    # pad masks with zeros, compute window
    mask_a = cupy.pad(view_a, zero_padding, mode='constant', constant_values=0)
    alpha_a = conv(mask_a, bp_az, bp_ay, bp_ax)
    wind_a = cupy.where(alpha_a > boundary_sigma_a, 1 / alpha_a, 0)
    del mask_a
    mask_b = cupy.pad(view_b, zero_padding, mode='constant', constant_values=0)
    alpha_b = conv(mask_b, bp_bz, bp_by, bp_bx)
    wind_b = cupy.where(alpha_b > boundary_sigma_b, 1 / alpha_b, 0)
    del mask_b
    # compute $\overbar{\alpha}(n) (Eqn. 8)
    alpha = alpha_a + alpha_b
    del alpha_a, alpha_b
    for _ in (trange(num_iter) if verbose else range(num_iter)):
        for k in range(2):
            if not k:  # k == 0, use view a
                con = conv(est_i, psf_az, psf_ay, psf_ax)
                est_i = cupy.multiply(
                    cupy.multiply(wind_a, est_i),
                    conv(div_stable(view_a, con, epsilon), bp_az, bp_ay, bp_ax)
                )
            else:
                con = conv(est_i, psf_bz, psf_by, psf_bx)
                est_i = cupy.multiply(
                    cupy.multiply(wind_b, est_i),
                    conv(div_stable(view_b, con, epsilon), bp_bz, bp_by, bp_bx)
                )
            c_tilde = cupy.sum(alpha * est_i) / 2.0
            est_i = (c / c_tilde) * est_i
    return est_i[zero_padding[0][0]:-zero_padding[0][1],
                 zero_padding[1][0]:-zero_padding[1][1],
                 zero_padding[2][0]:-zero_padding[2][1]]


def _additive_joint_rl_uncorr(
    view_a : cupy.ndarray, view_b : cupy.ndarray, est_i  : cupy.ndarray,
    psf_az : cupy.ndarray, psf_ay : cupy.ndarray, psf_ax : cupy.ndarray,
    psf_bz : cupy.ndarray, psf_by : cupy.ndarray, psf_bx : cupy.ndarray,
    bp_az  : cupy.ndarray, bp_ay  : cupy.ndarray, bp_ax  : cupy.ndarray,
    bp_bz  : cupy.ndarray, bp_by  : cupy.ndarray, bp_bx  : cupy.ndarray,
    num_iter : int,
    epsilon : float,
    cuda_conv_module : cupy.RawModule,
    launch_params : CuLaunchParameters,
    verbose : bool
) -> cupy.ndarray:
    conv = partial(convolve_3d, 
                   cuda_module=cuda_conv_module, 
                   launch_pars=launch_params)
    for _ in (trange(num_iter) if verbose else range(num_iter)):
        con_a = conv(est_i, psf_az, psf_ay, psf_ax)
        est_a = cupy.multiply(
            est_i, 
            conv(div_stable(view_a, con_a, epsilon), bp_az, bp_ay, bp_ax)
        )
        con_b = conv(est_i, psf_bz, psf_by, psf_bx)
        est_b = cupy.multiply(
            est_i,
            conv(div_stable(view_b, con_b, epsilon), bp_bz, bp_by, bp_bx)
        )
        est_i = (est_a + est_b) / 2.0
    return est_i


def _additive_joint_rl_boundcorr_chunk(
    view_a : numpy.ndarray, view_b : numpy.ndarray, est_i : numpy.ndarray|None,
    psf_az : numpy.ndarray, psf_ay : numpy.ndarray, psf_ax : numpy.ndarray,
    psf_bz : numpy.ndarray, psf_by : numpy.ndarray, psf_bx : numpy.ndarray,
    bp_az  : numpy.ndarray, bp_ay  : numpy.ndarray, bp_ax  : numpy.ndarray,
    bp_bz  : numpy.ndarray, bp_by  : numpy.ndarray, bp_bx  : numpy.ndarray,
    num_iter : int, epsilon : float,
    boundary_sigma_a : float, boundary_sigma_b : float,
    out_window : Tuple[slice,slice,slice],
    conv_module : Optional[cupy.RawModule],
    launch_pars : Optional[CuLaunchParameters]
) -> numpy.ndarray:
    raise NotImplementedError('TOFIX: only returns zeros')
    # compile the convolution module
    if conv_module is None:
        conv_module = make_conv_module(len(psf_az)//2)
    if launch_pars is None:
        launch_pars = calc_launch_params(len(psf_az)//2, view_a.shape)
    conv = partial(convolve_3d, 
                   cuda_module=conv_module, 
                   launch_pars=launch_pars)
    # move everything to the gpu
    view_a = cupy.asarray(view_a, dtype=cupy.float32, order='F')
    view_b = cupy.asarray(view_b, dtype=cupy.float32, order='F')
    psf_az = cupy.asarray(psf_az, dtype=cupy.float32)
    psf_ay = cupy.asarray(psf_bz, dtype=cupy.float32)
    psf_ax = cupy.asarray(psf_ax, dtype=cupy.float32)
    psf_bz = cupy.asarray(psf_bz, dtype=cupy.float32)
    psf_by = cupy.asarray(psf_by, dtype=cupy.float32)
    psf_bx = cupy.asarray(psf_bx, dtype=cupy.float32)
    bp_az = cupy.asarray(bp_az, dtype=cupy.float32)
    bp_ay = cupy.asarray(bp_bz, dtype=cupy.float32)
    bp_ax = cupy.asarray(bp_ax, dtype=cupy.float32)
    bp_bz = cupy.asarray(bp_bz, dtype=cupy.float32)
    bp_by = cupy.asarray(bp_by, dtype=cupy.float32)
    bp_bx = cupy.asarray(bp_bx, dtype=cupy.float32)
    # initialize estimate, if not already done
    if est_i is None:
        est_i = initialize_estimate(view_a, view_b, order='F')
    # compute masks for where the relevant data is
    mask_a = cupy.zeros_like(view_a)
    mask_a[out_window] = 1
    alpha_a = conv(mask_a, bp_az, bp_ay, bp_ax)
    wind_a = cupy.where(alpha_a > boundary_sigma_a, 1 / alpha_a, 0)
    del mask_a
    mask_b = cupy.zeros_like(view_b)
    mask_b[out_window] = 1
    alpha_b = conv(mask_b, bp_bz, bp_by, bp_bx)
    wind_b = cupy.where(alpha_b > boundary_sigma_b, 1 / alpha_b, 0)
    del mask_b
    # compute flux constant, initial \alpha
    c = cupy.sum(est_i[out_window])
    alpha = alpha_a + alpha_b
    del alpha_a, alpha_b
    for _ in range(num_iter):
        for k in range(2):
            if not k:
                con = conv(est_i, psf_az, psf_ay, psf_ax)
                est_i = cupy.multiply(
                    cupy.multiply(wind_a, est_i),
                    conv(div_stable(view_a, con, epsilon), bp_az, bp_ay, bp_ax)
                )
            else:
                con = conv(est_i, psf_bz, bp_by, psf_bx)
                est_i = cupy.multiply(
                    cupy.multiply(wind_b, est_i),
                    conv(div_stable(view_b, con, epsilon), bp_bz, bp_by, bp_bx)
                )
            c_tilde = cupy.sum(alpha * est_i) / 2.0
            est_i = (c / c_tilde) * est_i
    return est_i[out_window].get()


def joint_rl_dispim(
    view_a : cupy.ndarray, view_b : cupy.ndarray, est_i  : cupy.ndarray,
    psf_az : cupy.ndarray, psf_ay : cupy.ndarray, psf_ax : cupy.ndarray,
    psf_bz : cupy.ndarray, psf_by : cupy.ndarray, psf_bx : cupy.ndarray,
    bp_az  : cupy.ndarray, bp_ay  : cupy.ndarray, bp_ax  : cupy.ndarray,
    bp_bz  : cupy.ndarray, bp_by  : cupy.ndarray, bp_bx  : cupy.ndarray,
    num_iter : int,
    epsilon : float,
    boundary_correction : bool,
    zero_padding : Optional[PadType],
    boundary_sigma_a : float,
    boundary_sigma_b : float,
    conv_module : Optional[cupy.RawModule],
    launch_pars : Optional[CuLaunchParameters],
    verbose : bool
) -> cupy.ndarray:
    # compile the convolution module
    if conv_module is None:
        conv_module = make_conv_module(len(psf_az)//2)
    if launch_pars is None:
        launch_pars = calc_launch_params(len(psf_az)//2, view_a.shape)
    # compile the 
    if boundary_correction:
        warn('not implemented, falling back to additive algorithm')
        return _joint_rl_dispim_corr(view_a, view_b, est_i,
                                     psf_az, psf_ay, psf_ax,
                                     psf_bz, psf_by, psf_bx,
                                     bp_az, bp_ay, bp_ax,
                                     bp_bz, bp_by, bp_bx,
                                     num_iter, epsilon, zero_padding,
                                     boundary_sigma_a, boundary_sigma_b,
                                     conv_module, launch_pars,
                                     verbose)
    else:
        return _joint_rl_dispim_uncorr(view_a, view_b, est_i,
                                       psf_az, psf_ay, psf_ax,
                                       psf_bz, psf_by, psf_bx,
                                       bp_az, bp_ay, bp_ax,
                                       bp_bz, bp_by, bp_bx,
                                       num_iter, epsilon,
                                       conv_module, launch_pars,
                                       verbose)


def _joint_rl_dispim_corr(
    view_a : cupy.ndarray, view_b : cupy.ndarray, est_i  : cupy.ndarray,
    psf_az : cupy.ndarray, psf_ay : cupy.ndarray, psf_ax : cupy.ndarray,
    psf_bz : cupy.ndarray, psf_by : cupy.ndarray, psf_bx : cupy.ndarray,
    bp_az  : cupy.ndarray, bp_ay  : cupy.ndarray, bp_ax  : cupy.ndarray,
    bp_bz  : cupy.ndarray, bp_by  : cupy.ndarray, bp_bx  : cupy.ndarray,
    num_iter : iter,
    epsilon : float,
    zero_padding : PadType,
    boundary_sigma_a : float,
    boundary_sigma_b : float,
    cuda_conv_module : cupy.RawModule, 
    launch_params : CuLaunchParameters,
    verbose : bool
) -> cupy.ndarray:
    return _additive_joint_rl_boundcorr(view_a, view_b, est_i,
                                        psf_az, psf_ay, psf_ax,
                                        psf_bz, psf_by, psf_bx,
                                        bp_az, bp_ay, bp_ax,
                                        bp_bz, bp_by, bp_bx,
                                        num_iter, epsilon, zero_padding,
                                        boundary_sigma_a, boundary_sigma_b,
                                        cuda_conv_module, launch_params,
                                        verbose)


def _joint_rl_dispim_uncorr(
    view_a : cupy.ndarray, view_b : cupy.ndarray, est_i  : cupy.ndarray,
    psf_az : cupy.ndarray, psf_ay : cupy.ndarray, psf_ax : cupy.ndarray,
    psf_bz : cupy.ndarray, psf_by : cupy.ndarray, psf_bx : cupy.ndarray,
    bp_az  : cupy.ndarray, bp_ay  : cupy.ndarray, bp_ax  : cupy.ndarray,
    bp_bz  : cupy.ndarray, bp_by  : cupy.ndarray, bp_bx  : cupy.ndarray,
    num_iter : int,
    epsilon : float,
    cuda_conv_module : cupy.RawModule,
    launch_params : CuLaunchParameters,
    verbose : bool
) -> cupy.ndarray:
    conv = partial(convolve_3d,
                   cuda_module=cuda_conv_module,
                   launch_pars=launch_params)
    for _ in (trange(num_iter) if verbose else range(num_iter)):
        cona = conv(est_i, psf_az, psf_ay, psf_ax)
        est_a = cupy.multiply(
            est_i, conv(div_stable(view_a, cona, epsilon), bp_az, bp_ay, bp_ax)
        )
        conb = conv(est_a, psf_bz, psf_by, psf_bx)
        est_i = cupy.multiply(
            est_a, conv(div_stable(view_b, conb, epsilon), bp_bz, bp_by, bp_bx)
        )
    return est_i


## chunk-wise deconvolution
def deconvolve_chunkwise(
    view_a : zarr.Array, view_b : zarr.Array, out : zarr.Array,
    chunk_size : int|Tuple[int,int,int], overlap : int|Tuple[int,int,int],
    sigma_az : float, sigma_ay : float, sigma_ax : float,
    sigma_bz : float, sigma_by : float, sigma_bx : float,
    kernel_radius_z : int, kernel_radius_y : int, kernel_radius_x : int,
    decon_function : str, num_iter : int, epsilon : float,
    boundary_correction : bool,
    zero_padding : Optional[PadType],
    boundary_sigma_a : float, boundary_sigma_b : float,
    verbose : bool,
    ):
    if len(view_a.shape) == 4:
        channel_slice = slice(None)
        ch_a, z_a, r_a, c_a = view_a.shape
        ch_b, z_b, r_b, c_b = view_b.shape
        assert ch_a == ch_b, "input volumes must have same # channels"
    else:
        channel_slice = None
        z_a, r_a, c_a = view_a.shape
        z_b, r_b, c_b = view_b.shape
    assert all([a==b for a, b in zip([z_a, r_a, c_a],[z_b, r_b, c_b])]), \
        "input volumes must be same shape"
    chunks = calculate_decon_chunks(z_a, r_a, c_a,
                                    chunk_size, overlap, channel_slice)
    n_gpu = cupy.cuda.runtime.getDeviceCount()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_gpu, mp_context=multiprocessing.get_context('spawn')
        ) as executor, multiprocessing.Manager() as manager:
        gpu_queue = manager.Queue()
        for gpu_id in range(n_gpu):
            gpu_queue.put(gpu_id)
        fun = partial(_decon_chunk,
                      out, view_a, view_b, 
                      sigma_az, sigma_ay, sigma_ax,
                      sigma_bz, sigma_by, sigma_bx,
                      kernel_radius_z, kernel_radius_y, kernel_radius_x,
                      decon_function, num_iter, epsilon, 
                      boundary_correction,
                      zero_padding,
                      boundary_sigma_a, boundary_sigma_b)
        futures = []
        for _, chunk in chunks.items():
            future = executor.submit(fun, chunk, gpu_queue)
            futures.append(future)
        if verbose:
            pbar = tqdm(total=len(futures), desc="Deconvolving Chunks")
        else:
            pbar = nullcontext
        with pbar:
            for future in concurrent.futures.as_completed(futures):
                val = future.result()
                if verbose: 
                    pbar.update(1)
                    pbar.set_postfix({'ind' : val})


def _decon_chunk(
    # shared arguments across chunks
    out : zarr.Array, 
    view_a : zarr.Array, view_b : zarr.Array,
    sigma_az : float, sigma_ay : float, sigma_ax : float,
    sigma_bz : float, sigma_by : float, sigma_bx : float,
    kernel_radius_z : int, kernel_radius_y : int, kernel_radius_x : int,
    decon_function : str, num_iter : int, epsilon : float,
    boundary_correction : bool, 
    zero_padding : Optional[PadType],
    boundary_sigma_a : float, boundary_sigma_b : float,
    # iterate over these
    chunk_props : ChunkProps, gpu_queue,
):
    gpu_id = gpu_queue.get()
    # load data
    a = view_a.oindex[chunk_props.read_window]
    b = view_b.oindex[chunk_props.read_window]
    # in many large volumes, there's large chunks of black-space
    # (e.g. using the 'shear' or 'dispim' deskewing)
    # so just do a quick check and see if we're in one of those, and if we
    # are, return immediately (skip doing deconvolution)
    if numpy.all(a == 0) and numpy.all(b == 0): 
        gpu_queue.put(gpu_id)
        return 0
    # pad the inputs according to the chunking properties
    a = numpy.pad(a, chunk_props.paddings)
    b = numpy.pad(b, chunk_props.paddings)
    # make PSFs
    psf_az = gaussian_kernel_1d(sigma_az, kernel_radius_z); 
    bp_az  = psf_az[::-1].copy()
    psf_ay = gaussian_kernel_1d(sigma_ay, kernel_radius_y); 
    bp_ay  = psf_ay[::-1].copy()
    psf_ax = gaussian_kernel_1d(sigma_ax, kernel_radius_x); 
    bp_ax  = psf_ax[::-1].copy()
    psf_bz = gaussian_kernel_1d(sigma_bz, kernel_radius_z); 
    bp_bz  = psf_bz[::-1].copy()
    psf_by = gaussian_kernel_1d(sigma_by, kernel_radius_y); 
    bp_by  = psf_by[::-1].copy()
    psf_bx = gaussian_kernel_1d(sigma_bx, kernel_radius_x);
    bp_bx  = psf_bx[::-1].copy()
    with cupy.cuda.Device(gpu_id):
        if boundary_correction:
            # for boundary correction, implement 'special' version of 
            conv_module = make_conv_module(len(psf_az)//2)
            if len(a.shape) == 3:
                launch_pars = calc_launch_params(len(psf_az)//2, a.shape)
                d = _additive_joint_rl_boundcorr_chunk(
                    a, b, None,
                    psf_az, psf_ay, psf_ax, psf_bz, psf_by, psf_bx,
                    bp_az, bp_ay, bp_ax, bp_bz, bp_by, bp_bx,
                    num_iter, epsilon, 
                    boundary_sigma_a, boundary_sigma_b,
                    chunk_props.out_window, 
                    conv_module, launch_pars
                )
            else:
                launch_pars = calc_launch_params(len(psf_az)//2, a.shape[1:])
                fun = partial(_additive_joint_rl_boundcorr_chunk,
                              psf_az=psf_az, psf_ay=psf_ay, psf_ax=psf_ax,
                              psf_bz=psf_bz, psf_by=psf_by, psf_bx=psf_bx,
                              bp_az=bp_az, bp_ay=bp_ay, bp_ax=bp_ax,
                              bp_bz=bp_bz, bp_by=bp_by, bp_bx=bp_bx,
                              num_iter=num_iter, epsilon=epsilon,
                              boundary_sigma_a=boundary_sigma_a,
                              boundary_sigma_b=boundary_sigma_b,
                              out_window=chunk_props.out_window[1:],
                              conv_module=conv_module,
                              launch_pars=launch_pars)
                d = numpy.stack([fun(a[i,...], b[i,...], None) 
                                 for i in range(a.shape[0])], axis=0)
        else:
            d = deconvolve(
                a, b, None,
                psf_az, psf_ay, psf_ax, psf_bz, psf_by, psf_bx,
                bp_az, bp_ay, bp_ax, bp_bz, bp_by, bp_bx,
                decon_function, num_iter, epsilon, boundary_correction,
                zero_padding,
                boundary_sigma_a, boundary_sigma_b,
                False
            ).get()[chunk_props.out_window]
    gpu_queue.put(gpu_id)
    # crop down and write
    #print("", flush=True)
    #print(chunk_props.data_window, flush=True)
    #print(chunk_props.read_window, flush=True)
    #print(chunk_props.out_window, flush=True)
    out.set_orthogonal_selection(chunk_props.data_window, d)
    return 1