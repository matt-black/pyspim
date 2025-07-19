import os
import tempfile
import itertools
from typing import List
from functools import partial

import cupy
import numpy
from tqdm.auto import tqdm

from .powell import powell, powell_record
from ..typing import CuLaunchParameters
from ..reg.powell import parse_transform_string
from ..reg.met import _cub as met_cub
from ..reg.met import linear as met_linear
from ..reg.met import cubspl as met_cubspl
from ..reg.met import nearest as met_nearest
from ..util import launch_params_for_volume


def enumerate(ref : cupy.ndarray, mov : cupy.ndarray,
              metric : str, transform : str, interp_method : str,
              par_ranges : List[List[float]],
              kernel_launch_params : CuLaunchParameters|None=None,
              landscape_out_path : os.PathLike|None=None,
              verbose : bool=False):
    assert cupy.get_array_module(ref) == cupy, \
        "reference image must be on the GPU"
    assert cupy.get_array_module(mov) == cupy, \
        "moving image must be on the GPU"
    # figure out coords of centroid of image
    msze_z, msze_y, msze_x = mov.shape
    cx, cy, cz = msze_x / 2., msze_y / 2., msze_z / 2.
    # make function that will generate the transform matrix
    # also get # of params req'd
    mat_fun, _, _ = parse_transform_string(transform)
    par_fun = lambda p: mat_fun(p, cx, cy, cz).astype(float).flatten()[:12]
    # get shape of reference and moving images
    sz_ref, sy_ref, sx_ref = ref.shape
    sz_mov, sy_mov, sx_mov = mov.shape
    # figure out launch parameters
    if kernel_launch_params is None:
        block_size = 8
        kernel_launch_params = launch_params_for_volume(
            [sz_mov, sy_mov, sx_mov], block_size, block_size, block_size
        )
    # formulate optimization function
    # NOTE: optimization is done for minimization, so during optimization
    # we flip this by subtracting 1 (all fns used are bounded [0,1])
    if metric[-1] == '*':  # flag for CUB-reduction kernels
        _kern = met_cub.make_kernel(metric[:-1], interp_method,
                                    *kernel_launch_params[1])
        if metric[:-1] == 'nip':
            mu_ref, mu_mov = 0., 0.
        elif metric[:-1] == 'cr':
            mu_ref, mu_mov = cupy.mean(ref), 0.
        elif metric[:-1] == 'ncc':
            mu_ref, mu_mov = cupy.mean(ref), cupy.mean(mov)
        else:
            raise ValueError('invalid metric')
        met_fun = partial(met_cub.compute_kernel, _kern,
                          reference=ref, moving=mov,
                          mu_reference=mu_ref, mu_moving=mu_mov,
                          sz_r=sz_ref, sy_r=sy_ref, sx_r=sx_ref,
                          sz_m=sz_mov, sy_m=sy_mov, sx_m=sx_mov,
                          launch_params=kernel_launch_params)
        opt_fun = lambda p: met_fun(par_fun(p))
    elif metric == 'nip':
        if interp_method == 'linear':
            _nip_fun = met_linear.normalized_inner_product
        elif interp_method == 'cubspl':
            _nip_fun = met_cubspl.normalized_inner_product
        elif interp_method == 'nearest':
            _nip_fun = met_nearest.normalized_inner_product
        else:
            raise ValueError('invalid interpolation method')
        met_fun = partial(_nip_fun,
                          reference=ref, moving=mov, 
                          sz_r=sz_ref, sy_r=sy_ref, sx_r=sx_ref,
                          sz_m=sz_mov, sy_m=sy_mov, sx_m=sx_mov,
                          launch_params=kernel_launch_params)
        opt_fun = lambda p: met_fun(par_fun(p))
    elif metric == 'cr':
        mu_ref = cupy.mean(ref)
        if interp_method == 'linear':
            _cr_fun = met_linear.correlation_ratio
        elif interp_method == 'cubspl':
            _cr_fun = met_cubspl.correlation_ratio
        elif interp_method == 'nearest':
            _cr_fun = met_nearest.correlation_ratio
        else:
            raise ValueError('invalid interpolation method')
        met_fun = partial(_cr_fun,
                          reference=ref, moving=mov,
                          mu_reference=mu_ref,
                          sz_r=sz_ref, sy_r=sy_ref, sx_r=sx_ref,
                          sz_m=sz_mov, sy_m=sy_mov, sx_m=sx_mov,
                          launch_params=kernel_launch_params)
        opt_fun = lambda p: met_fun(par_fun(p))
    elif metric == 'ncc':
        mu_ref, mu_mov = cupy.mean(ref), cupy.mean(mov)
        mu_ref = cupy.mean(ref)
        if interp_method == 'linear':
            _ncc_fun = met_linear.normalized_cross_correlation
        elif interp_method == 'cubspl':
            _ncc_fun = met_cubspl.normalized_cross_correlation
        elif interp_method == 'nearest':
            _ncc_fun = met_nearest.normalized_cross_correlation
        else:
            raise ValueError('invalid interpolation method')
        met_fun = partial(_ncc_fun,
                          reference=ref, moving=mov,
                          mu_reference=mu_ref, mu_moving=mu_mov,
                          sz_r=sz_ref, sy_r=sy_ref, sx_r=sx_ref,
                          sz_m=sz_mov, sy_m=sy_mov, sx_m=sx_mov,
                          launch_params=kernel_launch_params)
        opt_fun = lambda p: met_fun(par_fun(p))
    n_cond = numpy.prod([len(x) for x in par_ranges])
    if verbose:
        iterator = tqdm(itertools.product(*par_ranges), total=n_cond)
    else:
        iterator = itertools.product(*par_ranges)
    opt_val, opt_par = 0, []
    with (open(landscape_out_path, 'w') if landscape_out_path is None else 
          tempfile.TemporaryFile(mode='w')) as f_out:
        for state in iterator:
            val = opt_fun(state)
            line = ",".join([f"{val:.4}" for val in state+[val]]) + "\n"
            f_out.write(line)
            if val > opt_val:
                opt_val = val
                opt_par = state
    T = mat_fun(opt_par, cx, cy, cz).reshape(4, 4)
    return numpy.linalg.inv(T), opt_val


def optimize(ref : cupy.ndarray, mov : cupy.ndarray,
             metric : str, transform : str, interp_method : str,
             par0 : List[float], max_iter : int, 
             search_incr : float, tolerance : float,
             kernel_launch_params : CuLaunchParameters|None=None,
             landscape_out_path : os.PathLike|None=None,
             verbose : bool=False):
    # make sure both input images are already on the GPU
    assert cupy.get_array_module(ref) == cupy, \
        "reference image must be on the GPU"
    assert cupy.get_array_module(mov) == cupy, \
        "moving image must be on the GPU"
    # figure out coords of centroid of image
    msze_z, msze_y, msze_x = mov.shape
    cx, cy, cz = msze_x / 2., msze_y / 2., msze_z / 2.
    # make function that will generate the transform matrix
    # also get # of params req'd
    mat_fun, _, ipar0 = parse_transform_string(transform)
    if par0 is None:
        par0 = ipar0
    if len(par0) != len(ipar0):
        raise ValueError('invalid # of initial params for transform')
    par_fun = lambda p: mat_fun(p, cx, cy, cz).astype(float).flatten()[:12]
    # get shape of reference and moving images
    sz_ref, sy_ref, sx_ref = ref.shape
    sz_mov, sy_mov, sx_mov = mov.shape
    # figure out launch parameters
    if kernel_launch_params is None:
        block_size = 8
        kernel_launch_params = launch_params_for_volume(
            [sz_mov, sy_mov, sx_mov], block_size, block_size, block_size
        )
    # formulate optimization function
    # NOTE: optimization is done for minimization, so during optimization
    # we flip this by subtracting 1 (all fns used are bounded [0,1])
    if metric[-1] == '*':  # flag for CUB-reduction kernels
        _kern = met_cub.make_kernel(metric[:-1], interp_method,
                                    *kernel_launch_params[1])
        if metric[:-1] == 'nip':
            mu_ref, mu_mov = 0., 0.
        elif metric[:-1] == 'cr':
            mu_ref, mu_mov = cupy.mean(ref), 0.
        elif metric[:-1] == 'ncc':
            mu_ref, mu_mov = cupy.mean(ref), cupy.mean(mov)
        else:
            raise ValueError('invalid metric')
        met_fun = partial(met_cub.compute_kernel, _kern,
                          reference=ref, moving=mov,
                          mu_reference=mu_ref, mu_moving=mu_mov,
                          sz_r=sz_ref, sy_r=sy_ref, sx_r=sx_ref,
                          sz_m=sz_mov, sy_m=sy_mov, sx_m=sx_mov,
                          launch_params=kernel_launch_params)
        opt_fun = lambda p: 1.0 - met_fun(par_fun(p))
    elif metric == 'nip':
        if interp_method == 'linear':
            _nip_fun = met_linear.normalized_inner_product
        elif interp_method == 'cubspl':
            _nip_fun = met_cubspl.normalized_inner_product
        elif interp_method == 'nearest':
            _nip_fun = met_nearest.normalized_inner_product
        else:
            raise ValueError('invalid interpolation method')
        met_fun = partial(_nip_fun,
                          reference=ref, moving=mov, 
                          sz_r=sz_ref, sy_r=sy_ref, sx_r=sx_ref,
                          sz_m=sz_mov, sy_m=sy_mov, sx_m=sx_mov,
                          launch_params=kernel_launch_params)
        opt_fun = lambda p: 1.0 - met_fun(par_fun(p))
    elif metric == 'cr':
        mu_ref = cupy.mean(ref)
        if interp_method == 'linear':
            _cr_fun = met_linear.correlation_ratio
        elif interp_method == 'cubspl':
            _cr_fun = met_cubspl.correlation_ratio
        elif interp_method == 'nearest':
            _cr_fun = met_nearest.correlation_ratio
        else:
            raise ValueError('invalid interpolation method')
        met_fun = partial(_cr_fun,
                          reference=ref, moving=mov,
                          mu_reference=mu_ref,
                          sz_r=sz_ref, sy_r=sy_ref, sx_r=sx_ref,
                          sz_m=sz_mov, sy_m=sy_mov, sx_m=sx_mov,
                          launch_params=kernel_launch_params)
        opt_fun = lambda p: 1.0 - met_fun(par_fun(p))
    elif metric == 'ncc':
        mu_ref, mu_mov = cupy.mean(ref), cupy.mean(mov)
        mu_ref = cupy.mean(ref)
        if interp_method == 'linear':
            _ncc_fun = met_linear.normalized_cross_correlation
        elif interp_method == 'cubspl':
            _ncc_fun = met_cubspl.normalized_cross_correlation
        elif interp_method == 'nearest':
            _ncc_fun = met_nearest.normalized_cross_correlation
        else:
            raise ValueError('invalid interpolation method')
        met_fun = partial(_ncc_fun,
                          reference=ref, moving=mov,
                          mu_reference=mu_ref, mu_moving=mu_mov,
                          sz_r=sz_ref, sy_r=sy_ref, sx_r=sx_ref,
                          sz_m=sz_mov, sy_m=sy_mov, sx_m=sx_mov,
                          launch_params=kernel_launch_params)
        opt_fun = lambda p: 1.0 - met_fun(par_fun(p))
    else:
        raise ValueError('invalid metric')
    if landscape_out_path is None:
        opt_par, _ = powell(opt_fun, par0, max_iter, search_incr,
                            tolerance, None, verbose)
    else:
        opt_par, _ = powell_record(opt_fun, par0, landscape_out_path,
                                   max_iter, search_incr, tolerance,
                                   verbose)
    val = opt_fun(opt_par)
    T = mat_fun(opt_par, cx, cy, cz).reshape(4, 4)
    return numpy.linalg.inv(T), val
    