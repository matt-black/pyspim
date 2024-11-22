from itertools import repeat
from typing import Iterable, Optional
from functools import partial
from warnings import warn

import cupy
from tqdm.auto import trange

from .._util import div_stable, initialize_estimate
from ...conv._cuda import make_conv_module, calc_launch_params, convolve_3d
from ...typing import NDArray, PadType, CuLaunchParameters


def _deconvolve_single_channel(view_a : NDArray, view_b : NDArray, 
                               est_i : NDArray, 
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
    view_a : NDArray, view_b : NDArray, est_i : NDArray, 
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
    assert len(view_a) == len(view_b) and len(view_a) == len(est_i), \
        "all input image lists must be same length"
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
        pfun(view_a[i,...], view_b[i,...], est_i[i,...],
             paz, pay, pax, pbz, pby, pbx, baz, bay, bax, bbz, bby, bbx)
        for i, paz, pay, pax, pbz, pby, pbx, baz, bay, bax, bbz, bby, bbx
        in zip(range(n_chan), psf_az, psf_ay, psf_ax, psf_bz, psf_by, psf_bx,
               bp_az, bp_ay, bp_ax, bp_bz, bp_by, bp_bx)
    ], axis=0)


def deconvolve(
    view_a : NDArray, view_b : NDArray, est_i : NDArray,
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


def additive_joint_rl(view_a : NDArray, view_b : NDArray, est_i  : NDArray,
                      psf_az : NDArray, psf_ay : NDArray, psf_ax : NDArray,
                      psf_bz : NDArray, psf_by : NDArray, psf_bx : NDArray,
                      bp_az  : NDArray, bp_ay  : NDArray, bp_ax  : NDArray,
                      bp_bz  : NDArray, bp_by  : NDArray, bp_bx  : NDArray,
                      num_iter : int,
                      epsilon : float,
                      boundary_correction : bool,
                      zero_padding : Optional[PadType],
                      boundary_sigma_a : float,
                      boundary_sigma_b : float,
                      verbose : bool) -> NDArray:
    view_a = cupy.asarray(view_a, dtype=cupy.float32, order='F')
    view_b = cupy.asarray(view_b, dtype=cupy.float32, order='F')
    est_i  = initialize_estimate(view_a, view_b, order='F')
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


def joint_rl_dispim(view_a : NDArray, view_b : NDArray, est_i  : NDArray|None,
                    psf_az : NDArray, psf_ay : NDArray, psf_ax : NDArray,
                    psf_bz : NDArray, psf_by : NDArray, psf_bx : NDArray,
                    bp_az  : NDArray, bp_ay  : NDArray, bp_ax  : NDArray,
                    bp_bz  : NDArray, bp_by  : NDArray, bp_bx  : NDArray,
                    num_iter : int,
                    epsilon : float,
                    boundary_correction : bool,
                    zero_padding : Optional[PadType],
                    boundary_sigma_a : float,
                    boundary_sigma_b : float,
                    conv_module : Optional[cupy.RawModule],
                    launch_pars : Optional[CuLaunchParameters],
                    verbose : bool) -> NDArray:
    view_a = cupy.asarray(view_a, dtype=cupy.float32, order='F')
    view_b = cupy.asarray(view_b, dtype=cupy.float32, order='F')
    est_i  = initialize_estimate(view_a, view_b, order='F')
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