from functools import partial
from typing import List, Optional, Union

import cupy
import numpy
from scipy.optimize import minimize
from tqdm.auto import tqdm

from . import kernels as kern
from ._common import parse_transform_string
from ..typing import CuLaunchParameters, OptBoundMargins, OptBounds
from ..util import launch_params_for_volume


def optimize_affine(
    ref: cupy.ndarray,
    mov: cupy.ndarray,
    metric: str,
    transform: str,
    interp_method: str,
    par0: List[float],
    bounds: Optional[Union[OptBounds, OptBoundMargins]] = None,
    kernel_launch_params: Optional[CuLaunchParameters] = None,
    opt_method: str = "powell",
    verbose: bool = False,
    **opt_kwargs,
):
    """Find optimal affine transform to register `mov` with `ref`.
         
    `**opt_kwargs` are passed to `scipy.optimize.minimize`

    :param ref: reference volume
    :type ref: cupy.ndarray
    :param mov: moving volume, to be registered
    :type mov: cupy.ndarray
    :param metric: metric to optimize for registration
        one of ('nip', 'cr', 'ncc').
        'nip' : normalized inner product
        'cr'  : correlation ratio
        'ncc' : normalized cross correlation
    :type metric: str
    :param transform: transform to optimize
    :type transform: str
    :param interp: interpolation method to use during transformation
        one of ('linear', 'cubspl')
        'linear' : trilinear interpolation
        'cubspl' : cubic b-spline interpolation
    :type interp: str
    :param par0: initial guess for parameters
    :type par0: list
    :param bounds: bounds for parameters
    :type bounds: list
    :param opt_method: optimization method for scipy.optimize.minimize
        supported: 'powell' (default), 'L-BFGS-B', 'Nelder-Mead', 'TNC', 'COBYLA'
    :type opt_method: str
    :param verbose: show intermediate results with tqdm progress bar
    :type verbose: bool
    :returns: transform and optimization results
    :rtype: Tuple[NDArray,OptimizeResult]
    """
    # make sure both input images are already on the GPU and are floating point
    # TODO: update so that this works on CPU, too
    assert cupy.get_array_module(ref) == cupy, "reference image must be on the GPU" # type: ignore
    assert cupy.get_array_module(mov) == cupy, "moving image must be on the GPU" # type: ignore
    # figure out coords of centroid of image
    msze_z, msze_y, msze_x = mov.shape
    cx, cy, cz = msze_x / 2.0, msze_y / 2.0, msze_z / 2.0
    # make function that will generate the transform matrix
    # also get # of params req'd
    mat_fun, postfix_fun, ipar0 = parse_transform_string(transform)
    if par0 is None:
        par0 = ipar0
    if len(par0) != len(ipar0):
        raise ValueError("invalid # of initial params for transform")
    if bounds is not None:
        if len(bounds) != len(ipar0):
            raise ValueError("invalid # of bounds for transform")
        if isinstance(bounds[0], float) or isinstance(bounds[0], int):
            bounds = [(p - b, p + b) for p, b in zip(par0, bounds)] # type: ignore
        elif isinstance(bounds[0], tuple):
            pass  # already correctly formatted
        else:
            raise ValueError("invalid bound formatting")
    # par_fun takes in the parameter vector and generates the matrix that then gets passed to the kernel
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
    # initialize the computation
    # this makes all the kernels, CUDA streams, and per-device arguments that will get used per-device.
    kernels, streams, kernel_args = kern.initialize_computation(
        metric, interp_method, ref, mov, cupy.mean(ref), cupy.mean(mov),
        sz_ref, sy_ref, sx_ref, sz_mov, sy_mov, sx_mov
    )
    # formulate optimization function
    # NOTE: scipy only has a `minimize` function and these metrics should be maximized. 
    # They are all confined to [0,1] so to minimize, do 1.0 - kernel_value.
    def opt_fun(pars: numpy.ndarray) -> float:
        T = par_fun(pars) # generate transform matrix
        val = kern.compute(
            T, kernels, streams, kernel_args, kernel_launch_params
        )
        return 1.0 - val.get()

    # optimize
    opt_call = partial(
        minimize, opt_fun, x0=par0, bounds=bounds, method=opt_method, options=opt_kwargs
    )
    if verbose:
        def cback(x, pbar, postfix_fun):
            pbar.update(1)
            pbar.set_postfix(postfix_fun(x))
        with tqdm(desc="Registration", leave=False) as pbar:
            _cback = partial(cback, pbar=pbar, postfix_fun=postfix_fun)
            res = opt_call(callback=_cback)
    else:
        res = opt_call()
    # calculate transform
    T = mat_fun(res.x, cx, cy, cz).reshape(4, 4)
    return T, res


def _split_substrings(transform_str: str):
    indivs = transform_str.split('+')
    return ['+'.join(indivs[:i+1]) for i in range(len(indivs))]


def optimize_affine_piecewise(
    ref: cupy.ndarray,
    mov: cupy.ndarray,
    metric: str,
    transform: str,
    interp_method: str,
    par0: List[float],
    bounds: Optional[Union[OptBounds, OptBoundMargins]] = None,
    kernel_launch_params: Optional[CuLaunchParameters] = None,
    opt_method: str = "powell",
    verbose: bool = False,
    **opt_kwargs,
):
    # split at the '+' in the transform string to generate sub-problems
    sub_transforms = _split_substrings(transform)
    iterator = tqdm(sub_transforms) if verbose else sub_transforms
    T, res = numpy.eye(4), None
    for subt in iterator:
        if verbose:
            iterator.set_description_str(f"Transform ({subt:s})") # type: ignore
        # get number of parameters for this transform
        _, _, ipar0 = parse_transform_string(subt)
        _idx = len(ipar0)
        _par0 = par0[:_idx]
        bnds = bounds[:_idx] if bounds is not None else None
        T, res = optimize_affine(
            ref,
            mov,
            metric,
            subt,
            interp_method,
            _par0,
            bnds,
            kernel_launch_params,
            opt_method,
            verbose,
            **opt_kwargs,
        )
        par0[:_idx] = res.x
        if verbose:
            iterator.set_postfix_str( # type: ignore
                f"Prev ({subt:s}): met={1.0 - res.fun:.3f}"
            )
    return T, res
