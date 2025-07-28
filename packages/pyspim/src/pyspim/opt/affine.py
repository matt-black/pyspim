import itertools
import os
import tempfile
from functools import partial
from typing import List, Optional

import cupy
import numpy
from tqdm.auto import tqdm

from ..reg import kernels as kern
from ..reg.powell import parse_transform_string
from ..typing import CuLaunchParameters
from ..util import launch_params_for_volume
from .powell import powell, powell_record


def enumerate(
    ref: cupy.ndarray,
    mov: cupy.ndarray,
    metric: str,
    transform: str,
    interp_method: str,
    par_ranges: List[List[float]],
    kernel_launch_params: Optional[CuLaunchParameters] = None,
    landscape_out_path: Optional[os.PathLike] = None,
    verbose: bool = False,
):
    if cupy.get_array_module(ref) != cupy: # type: ignore
        raise ValueError("reference image must be on the GPU")
    if cupy.get_array_module(mov) != cupy: # type: ignore
        raise ValueError("moving image must be on the GPU")
        
    # figure out coords of centroid of image
    msze_z, msze_y, msze_x = mov.shape
    cx, cy, cz = msze_x / 2.0, msze_y / 2.0, msze_z / 2.0
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
    
    kernels, streams, args = kern.initialize_computation(
        metric, interp_method, ref, mov, 
        cupy.mean(ref).astype(cupy.float32), cupy.mean(mov).astype(cupy.float32),
        sz_ref, sy_ref, sx_ref, sz_mov, sy_mov, sx_mov
    )
    def opt_fun(pars: tuple[float,...]) -> float:
        T = par_fun(pars)
        return 1.0 - kern.compute(
            T, kernels, streams, args, kernel_launch_params
        )

    n_cond = int(numpy.prod([len(x) for x in par_ranges]))
    if verbose:
        iterator = tqdm(
            itertools.product(*par_ranges), 
            total=n_cond
        )
    else:
        iterator = itertools.product(*par_ranges)
    opt_val, opt_par = 0, []
    with (
        open(landscape_out_path, "w")
        if landscape_out_path is not None
        else tempfile.TemporaryFile(mode="w")
    ) as f_out:
        for state in iterator:
            val = opt_fun(state)
            line = ",".join([f"{val:.4}" for val in list(state) + [val]]) + "\n"
            f_out.write(line)
            if val > opt_val:
                opt_val = val
                opt_par = state
    T = mat_fun(opt_par, cx, cy, cz).reshape(4, 4)
    return numpy.linalg.inv(T), opt_val


def optimize(
    ref: cupy.ndarray,
    mov: cupy.ndarray,
    metric: str,
    transform: str,
    interp_method: str,
    par0: List[float],
    max_iter: int,
    search_incr: float,
    tolerance: float,
    kernel_launch_params: Optional[CuLaunchParameters] = None,
    landscape_out_path: Optional[os.PathLike] = None,
    verbose: bool = False,
):
    # make sure both input images are already on the GPU
    if cupy.get_array_module(ref) != cupy: # type: ignore
        raise ValueError("reference image must be on the GPU")
    if cupy.get_array_module(mov) != cupy: # type: ignore
        raise ValueError("moving image must be on the GPU")
        
    # figure out coords of centroid of image
    msze_z, msze_y, msze_x = mov.shape
    cx, cy, cz = msze_x / 2.0, msze_y / 2.0, msze_z / 2.0
    # make function that will generate the transform matrix
    # also get # of params req'd
    mat_fun, _, ipar0 = parse_transform_string(transform)
    if par0 is None:
        par0 = ipar0
    if len(par0) != len(ipar0):
        raise ValueError("invalid # of initial params for transform")
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
    
    kernels, streams, args = kern.initialize_computation(
        metric, interp_method, ref, mov, 
        cupy.mean(ref).astype(cupy.float32), cupy.mean(mov).astype(cupy.float32),
        sz_ref, sy_ref, sx_ref, sz_mov, sy_mov, sx_mov
    )    
    def opt_fun(pars: numpy.ndarray) -> float:
        T = par_fun(pars)
        return 1.0 - kern.compute(
            T, kernels, streams, args, kernel_launch_params
        )

    if landscape_out_path is None:
        opt_par, _ = powell(
            opt_fun, par0, max_iter, search_incr, tolerance, None, verbose
        )
    else:
        opt_par, _ = powell_record(
            opt_fun, par0, landscape_out_path, max_iter, search_incr, tolerance, verbose
        )
    val = opt_fun(opt_par)
    T = mat_fun(opt_par, cx, cy, cz).reshape(4, 4)
    return numpy.linalg.inv(T), val
