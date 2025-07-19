from functools import partial
from itertools import accumulate
from numbers import Number
from typing import Callable, List, Tuple

import cupy
import numpy
from scipy.optimize import minimize
from tqdm.auto import tqdm

from .._matrix import (
    rotation_about_point_matrix,
    rotation_matrix,
    scale_matrix,
    shear_about_point_matrix,
    symmetric_shear_about_point_matrix,
    translation_matrix,
)
from ..typing import CuLaunchParameters, OptBoundMargins, OptBounds
from ..util import launch_params_for_volume

# other imports
from .met import _cub as met_cub
from .met import cubspl as met_cubspl
from .met import linear as met_linear
from .met import nearest as met_nearest


def _trans_matrix(params, x, y, z):
    return translation_matrix(*params)


def _trans_scale_matrix(params, x, y, z):
    T = translation_matrix(*params[:3])
    T[0, 0], T[1, 1], T[2, 2] = params[3], params[4], params[5]
    return T


def _rot_trans_matrix(params, x, y, z):
    R = rotation_about_point_matrix(*(params[3:] * numpy.pi / 180), x, y, z)
    T = translation_matrix(*params[:3])
    return T @ R


def _rot0_trans_matrix(params, x, y, z):
    R = rotation_matrix(*(params[3:] * numpy.pi / 180))
    T = translation_matrix(*params[:3])
    return T @ R


def _rot_trans_scale_matrix(params, x, y, z):
    R = rotation_about_point_matrix(*(params[3:6] * numpy.pi / 180), x, y, z)
    S = scale_matrix(*params[6:])
    T = translation_matrix(*params[:3])
    return T @ R @ S


def _rot0_trans_scale_matrix(params, x, y, z):
    R = rotation_matrix(*(params[3:6] * numpy.pi / 180))
    S = scale_matrix(*params[6:])
    T = translation_matrix(*params[:3])
    return T @ R @ S


def _symmshear_trans_matrix(params, x, y, z):
    S = symmetric_shear_about_point_matrix(
        *(numpy.tan(params[3:] * numpy.pi / 180)), x, y, z
    )
    T = translation_matrix(*params[:3])
    return T @ S


def _shear_trans_matrix(params, x, y, z):
    S = shear_about_point_matrix(*(numpy.tan(params[3:] * numpy.pi / 180)), x, y, z)
    T = translation_matrix(*params[:3])
    return T @ S


def _shear_trans_scale_matrix(params, x, y, z):
    Sh = shear_about_point_matrix(*(numpy.tan(params[3:9] * numpy.pi / 180)), x, y, z)
    T = translation_matrix(*params[:3])
    S = scale_matrix(*params[9:])
    return T @ S @ Sh


def parse_transform_string(string: str) -> Tuple[Callable, Callable, List[float]]:
    if string in ("t", "trans", "translation"):
        postfix = lambda x: {"t": [f"{v:.2f}" for v in x[:3]]}
        return (
            _trans_matrix,
            postfix,
            [
                0,
            ]
            * 3,
        )
    elif string in ("t+s", "transscale", "translation+scale"):
        postfix = lambda x: {
            "t": [f"{v:.2f}" for v in x[:3]],
            "s": [f"{v:.2f}" for v in x[3:]],
        }
        return _trans_scale_matrix, postfix, [0, 0, 0, 1, 1, 1]
    # rotation + translation (+ scaling)
    elif string in ("t+r", "transrot", "translation+rotation"):
        postfix = lambda x: {
            "t": [f"{v:.2f}" for v in x[:3]],
            "a": [f"{v:.2f}" for v in x[3:]],
        }
        par0 = [
            0,
        ] * 6
        return (
            _rot_trans_matrix,
            postfix,
            [
                0,
            ]
            * 6,
        )
    elif string in ("t+r0", "transrot0", "translation+rotation0"):
        postfix = lambda x: {
            "t": [f"{v:.2f}" for v in x[:3]],
            "a": [f"{v:.2f}" for v in x[3:]],
        }
        par0 = [
            0,
        ] * 6
        return (
            _rot0_trans_matrix,
            postfix,
            [
                0,
            ]
            * 6,
        )
    elif string in ("t+r+s", "transrotscale", "translation+rotation+scale"):
        postfix = lambda x: {
            "t": [f"{v:.2f}" for v in x[:3]],
            "a": [f"{v:.2f}" for v in x[3:6]],
            "s": [f"{v:.2f}" for v in x[6:]],
        }
        par0 = [
            0,
        ] * 6 + [
            1,
        ] * 3
        return _rot_trans_scale_matrix, postfix, par0
    elif string in ("t+r0+s", "transrot0scale", "translation+rotation0+scale"):
        postfix = lambda x: {
            "t": [f"{v:.2f}" for v in x[:3]],
            "a": [f"{v:.2f}" for v in x[3:6]],
            "s": [f"{v:.2f}" for v in x[6:]],
        }
        par0 = [
            0,
        ] * 6 + [
            1,
        ] * 3
        return _rot0_trans_scale_matrix, postfix, par0
    # shear + translation
    elif string in ("t+ssh", "transsymmshear", "translation+symmetricshear"):
        postfix = lambda x: {
            "t": [f"{v:.2f}" for v in x[:3]],
            "sh": [f"{v:.2f}" for v in x[3:]],
        }
        return (
            _symmshear_trans_matrix,
            postfix,
            [
                0,
            ]
            * 6,
        )
    elif string in ("t+sh", "transshear", "translation+shear"):
        postfix = lambda x: {
            "t": [f"{v:.2f}" for v in x[:3]],
            "sh": [f"{v:.2f}" for v in x[3:]],
        }
        return (
            _shear_trans_matrix,
            postfix,
            [
                0,
            ]
            * 9,
        )
    elif string in ("t+sh+s", "transshearscale", "translation+shear+scale"):
        postfix = lambda x: {
            "t": [f"{v:.2f}" for v in x[:3]],
            "sh": [f"{v:.2f}" for v in x[3:9]],
            "s": [f"{v:.2f}" for v in x[9:]],
        }
        par0 = [
            0,
        ] * 9 + [
            1,
        ] * 3
        return _shear_trans_scale_matrix, postfix, par0
    else:
        raise ValueError("invalid transform string")


def optimize_affine(
    ref: cupy.ndarray,
    mov: cupy.ndarray,
    metric: str,
    transform: str,
    interp_method: str,
    par0: List[float],
    bounds: OptBounds | OptBoundMargins | None,
    kernel_launch_params: CuLaunchParameters | None = None,
    verbose: bool = False,
    **opt_kwargs,
):
    """find optimal affine transform to register `mov` with `ref`
        by Powell's method
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
    :param verbose: show intermediate results with tqdm progress bar
    :type verbose: bool
    :returns: transform and optimization results
    :rtype: Tuple[NDArray,OptimizeResult]
    """
    # make sure both input images are already on the GPU and are floating point
    # TODO: update so that this works on CPU, too
    assert cupy.get_array_module(ref) == cupy, "reference image must be on the GPU"
    assert cupy.get_array_module(mov) == cupy, "moving image must be on the GPU"
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
        if isinstance(bounds[0], Number):
            bounds = [(p - b, p + b) for p, b in zip(par0, bounds)]
        elif len(bounds[0]) == 2:  # do nothing
            pass  # already correctly formatted
        else:
            raise ValueError("input bounds should be scalar or tuple")
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
    # NOTE: scipy only has a `minimize` function and most of these metrics
    # we want to maximize, so for metrics confined to [0,1] we do
    # 1 - metric and for unbound metrics, take its negative
    if metric[-1] == "*":  # flag for CUB-reduction kernels
        _kern = met_cub.make_kernel(
            metric[:-1], interp_method, *kernel_launch_params[1]
        )
        if metric[:-1] == "nip":
            mu_ref, mu_mov = 0.0, 0.0
        elif metric[:-1] == "cr":
            mu_ref, mu_mov = cupy.mean(ref), 0.0
        elif metric[:-1] == "ncc":
            mu_ref, mu_mov = cupy.mean(ref), cupy.mean(mov)
        else:
            raise ValueError("invalid metric")
        met_fun = partial(
            met_cub.compute_kernel,
            _kern,
            reference=ref,
            moving=mov,
            mu_reference=mu_ref,
            mu_moving=mu_mov,
            sz_r=sz_ref,
            sy_r=sy_ref,
            sx_r=sx_ref,
            sz_m=sz_mov,
            sy_m=sy_mov,
            sx_m=sx_mov,
            launch_params=kernel_launch_params,
        )
        opt_fun = lambda p: 1.0 - met_fun(par_fun(p))
    elif metric == "nip":
        if interp_method == "linear":
            _nip_fun = met_linear.normalized_inner_product
        elif interp_method == "cubspl":
            _nip_fun = met_cubspl.normalized_inner_product
        elif interp_method == "nearest":
            _nip_fun = met_nearest.normalized_inner_product
        else:
            raise ValueError("invalid interpolation method")
        met_fun = partial(
            _nip_fun,
            reference=ref,
            moving=mov,
            sz_r=sz_ref,
            sy_r=sy_ref,
            sx_r=sx_ref,
            sz_m=sz_mov,
            sy_m=sy_mov,
            sx_m=sx_mov,
            launch_params=kernel_launch_params,
        )
        opt_fun = lambda p: 1.0 - met_fun(par_fun(p))
    elif metric == "cr":
        mu_ref = cupy.mean(ref)
        if interp_method == "linear":
            _cr_fun = met_linear.correlation_ratio
        elif interp_method == "cubspl":
            _cr_fun = met_cubspl.correlation_ratio
        elif interp_method == "nearest":
            _cr_fun = met_nearest.correlation_ratio
        else:
            raise ValueError("invalid interpolation method")
        met_fun = partial(
            _cr_fun,
            reference=ref,
            moving=mov,
            mu_reference=mu_ref,
            sz_r=sz_ref,
            sy_r=sy_ref,
            sx_r=sx_ref,
            sz_m=sz_mov,
            sy_m=sy_mov,
            sx_m=sx_mov,
            launch_params=kernel_launch_params,
        )
        opt_fun = lambda p: 1.0 - met_fun(par_fun(p))
    elif metric == "ncc":
        mu_ref, mu_mov = cupy.mean(ref), cupy.mean(mov)
        mu_ref = cupy.mean(ref)
        if interp_method == "linear":
            _ncc_fun = met_linear.normalized_cross_correlation
        elif interp_method == "cubspl":
            _ncc_fun = met_cubspl.normalized_cross_correlation
        elif interp_method == "nearest":
            _ncc_fun = met_nearest.normalized_cross_correlation
        else:
            raise ValueError("invalid interpolation method")
        met_fun = partial(
            _ncc_fun,
            reference=ref,
            moving=mov,
            mu_reference=mu_ref,
            mu_moving=mu_mov,
            sz_r=sz_ref,
            sy_r=sy_ref,
            sx_r=sx_ref,
            sz_m=sz_mov,
            sy_m=sy_mov,
            sx_m=sx_mov,
            launch_params=kernel_launch_params,
        )
        opt_fun = lambda p: 1.0 - met_fun(par_fun(p))
    else:
        raise ValueError("invalid metric")
    # do powell's registration
    opt_call = partial(
        minimize, opt_fun, x0=par0, bounds=bounds, method="powell", options=opt_kwargs
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
    # the transform calculated here is mapping mov into reference (the inverse)
    # so spit out the inverse of it, which gives the forward transform
    return numpy.linalg.inv(T), res


def _split_substrings(transform_str: str):
    return list(accumulate(transform_str))[::2]


def optimize_affine_piecewise(
    ref: cupy.ndarray,
    mov: cupy.ndarray,
    metric: str,
    transform: str,
    interp_method: str,
    par0: List[float],
    bounds: OptBounds | OptBoundMargins | None,
    kernel_launch_params: CuLaunchParameters | None = None,
    verbose: bool = False,
    **opt_kwargs,
):
    # split at the '+' in the transform string to generate sub-problems
    sub_transforms = _split_substrings(transform)
    iterator = tqdm(sub_transforms) if verbose else sub_transforms
    for subt in iterator:
        if verbose:
            iterator.set_description_str(f"Transform ({subt:s})")
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
            verbose,
            **opt_kwargs,
        )
        par0[:_idx] = res.x
        if verbose:
            iterator.set_postfix_str(
                f"Prev ({subt:s}): met={1.0 - res.fun:.3f}"
            )
    return T, res
