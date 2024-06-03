import math
import functools
import itertools

import numpy
import cupy
from scipy.optimize import minimize
from tqdm.auto import tqdm

from .._matrix import translation_matrix, \
    rotation_about_point_matrix, \
    scale_matrix, \
    symmetric_shear_about_point_matrix, shear_about_point_matrix
from ..affine.interp import transform as affine_transform
# metric imports
from .metrics import discrete_entropy
from .metrics import _corr_ratio_reftex_ewk
from .metrics import _mutual_information_precomp_target
from .metrics import _entropy_correlation_coeff_precomp_target
from .._util import create_texture_object
from .._util import get_skimage_module, get_fft_module


def _trans_matrix(params, x, y, z):
    return translation_matrix(*params)


def _rot_trans_matrix(params, x, y, z):
    R = rotation_about_point_matrix(*(params[3:] * numpy.pi/180), x, y, z)
    T = translation_matrix(*params[:3])
    return R @ T


def _rot_trans_scale_matrix(params, x, y, z):
    R = rotation_about_point_matrix(*(params[3:6] * numpy.pi/180), x, y, z)
    S = scale_matrix(*params[6:])
    T = translation_matrix(*params[:3])
    return S @ R @ T


def _symmshear_trans_matrix(params, x, y, z):
    S = symmetric_shear_about_point_matrix(
        *(numpy.tan(params[3:] * numpy.pi/180)), x, y, z
    )
    T = translation_matrix(*params[:3])
    return S @ T


def _shear_trans_matrix(params, x, y, z):
    S = shear_about_point_matrix(
        *(numpy.tan(params[3:] * numpy.pi/180)), x, y, z
    )
    T = translation_matrix(*params[:3])
    return S @ T


def _shear_trans_scale_matrix(params, x, y, z):
    Sh = shear_about_point_matrix(
        *(numpy.tan(params[3:9] * numpy.pi/180)), x, y, z
    )
    T  = translation_matrix(*params[:3])
    S  = scale_matrix(*params[9:])
    return Sh @ S @ T


def __is_floating_point(x):
    xp = cupy.get_array_module(x)
    return x.dtype in (xp.float16, xp.float32, xp.float64)


def _parse_transform_string(string):
    if string in ('t', 'trans', 'translation'):
        return _trans_matrix, 3
    # rotation + translation (+ scaling)
    elif string in ('t+r', 'transrot', 'translation+rotation'):
        return _rot_trans_matrix, 6
    elif string in ('t+r+s', 'transrotscale', 'translation+rotation+scale'):
        return _rot_trans_scale_matrix, 9
    # shear + translation
    elif string in ('t+ssh', 'transsymmshear', 'translation+symmetricshear'):
        return _symmshear_trans_matrix, 6
    elif string in ('t+sh', 'transshear', 'translation+shear'):
        return _shear_trans_matrix, 9
    elif string in ('t+sh+s', 'transshearscale', 'translation+shear+scale'):
        return _shear_trans_scale_matrix, 12
    else:
        raise ValueError('invalid transform string')


def optimize(ref : NDArray, mov : NDArray, metric : str, transform : str,
             par0 : list=[0,]*6, nbin_ref : int=64, nbin_mov : int=64,
             bounds : list=None, verbose : bool=False,
             **opt_kwargs):
    """find optimal transform to register `mov` with `ref` by Powell's method
        `**opt_kwargs` are passed to `scipy.optimize.minimize`

    :param ref: reference volume
    :type ref: NDArray
    :param mov: moving volume, to be registered
    :type mov: NDArray
    :param metric: metric to optimize for registration
        one of ('cr', 'mi', 'ec').
        'cr' : correlation ratio
        'mi' : mutual information
        'ec' : entropy correlation coefficient
    :type metric: str
    :param transform: transform to optimize
    :type transform: str
    :param par0: initial guess for parameters
    :type par0: list
    :param nbin_ref: number of bins in histogram for `ref`
       only used for metric=='mi' or 'ec'
    :type nbin_ref: int
    :param nbin_mov: number of bins in histogram for `mov`
       only used for metric=='mi' or 'ec'
    :type nbin_mov: int
    :param bounds: bounds for parameters
    :type bounds: list
    :param verbose: show intermediate results with tqdm progress bar
    :type verbose: bool
    :returns: transform and optimization results
    :rtype: Tuple[NDArray,OptimizeResult]
    """

    # make sure both input images are already on the GPU and are floating point
    # TODO: update so that this works on CPU, too
    assert cupy.get_array_module(ref) == cupy and __is_floating_point(ref), \
        "reference image must be on the GPU and floating point"
    assert cupy.get_array_module(mov) == cupy and __is_floating_point(mov), \
        "moving image must be on the GPU and floating point"

    # figure out coords of centroid of image
    msze_z, msze_y, msze_x = mov.shape
    cx, cy, cz = msze_x / 2., msze_y / 2., msze_z / 2.

    # make function that will generate the transform matrix
    # also get # of params req'd
    mat_fun, n_par_req = _parse_transform_string(transform)
    if len(par0) != n_par_req:
        raise ValueError('invalid # of initial params for transform')
    if bounds is not None and len(bounds) != n_par_req:
        raise ValueError('invalid # of bounds for transform')
    par_fun = lambda p: mat_fun(p, cx, cy, cz).astype(float).flatten()[:12]

    # move `ref` to texture memory
    ref_tex, tex_arr = create_texture_object(ref, 'border', 'linear',
                                             'element_type')

    # formulate optimization function
    # NOTE: scipy only has a `minimize` function and most of these metrics
    # we want to maximize, so for metrics confined to [0,1] we do
    # 1 - metric and for unbound metrics, take its negative
    if metric == 'cr':
        met_fun = functools.partial(_corr_ratio_reftex_ewk,
                                    ref_tex, mov)
        opt_fun = lambda p: 1.0 - float(met_fun(par_fun(p)))
    elif metric == 'mi' or metric == 'ec':
        # need to normalize input images to [0, 1] for algo. to work
        max_val = max([cupy.amax(mov), cupy.amax(ref)])
        mov /= max_val
        ref /= max_val
        # precompute entropy of moving image
        P_mov, _ = cupy.histogram(mov, nbin_mov)
        P_mov = P_mov / cupy.sum(P_mov) + cupy.finfo(float).eps
        H_mov = discrete_entropy(P_mov) / cupy.log2(nbin_mov)
        # formulate function
        if metric == 'mi':
            met_fun = functools.partial(
                _mutual_information_precomp_target,
                ref_tex, mov, nbin_ref=nbin_ref, nbin_tar=nbin_mov,
                H_tar=H_mov
            )
            opt_fun = lambda p: -float(met_fun(par_fun(p)))
        elif metric == 'ec':
            met_fun = functools.partial(
                _entropy_correlation_coeff_precomp_target,
                ref_tex, mov, nbin_ref=nbin_ref, nbin_tar=nbin_mov,
                H_tar=H_mov
            )
            opt_fun = lambda p: 1.0 - float(met_fun(par_fun(p)))
        else:
            raise ValueError(
                'invalid metric, must be one of [\'cr\', \'mi\', \'ec\']'
            )
    else:
        raise ValueError('invalid metric')
    opt_call = functools.partial(minimize, opt_fun,
                                 x0=par0, bounds=bounds,
                                 method='powell', options=opt_kwargs)
    if verbose:
        def cback(x, transform, pbar):
            pbar.update(1)
            a = ["{:.1f}".format(p*180/numpy.pi) for p in x[3:6]]
            if len(x) == 6:
                t = ["{:.1f}".format(p) for p in x[:3]]
                pbar.set_postfix({'t' : t, 'r' : a})
            elif len(x) == 9:
                t = ["{:.1f}".format(p) for p in x[:3]]
                s = ["{:.1f}".format(p) for p in x[6:]]
                pbar.set_postfix({'t':t, 'r':a, 's':s})
            elif len(x) == 12:
                pass
            else:
                pbar.set_postfix({'t' : t})
        with tqdm(desc="Registration") as pbar:
            _cback = functools.partial(cback, transform=transform, pbar=pbar)
            res = opt_call(callback=_cback)
    else:
        res = opt_call()
    # get rid of texture objects (ensures they get GC'd)
    del ref_tex, tex_arr
    # calculate transform
    T = mat_fun(res.x, cx, cy, cz).reshape(4, 4)
    return numpy.linalg.inv(T), res
