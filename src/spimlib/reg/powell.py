import math
import functools
import itertools

import numpy
import cupy
from scipy.optimize import minimize
from tqdm.auto import tqdm

from .._matrix import translation_matrix, \
    rotation_about_point_matrix, \
    scale_matrix
from ..affine.interp import transform as affine_transform
# metric imports
from .metrics import discrete_entropy
from .metrics import _corr_ratio_reftex_ewk
from .metrics import _mutual_information_precomp_target
from .metrics import _entropy_correlation_coeff_precomp_target
from .._util import create_texture_object
from .._util import get_skimage_module, get_fft_module


def _dof3matrix(params):
    return translation_matrix(*params)


def _dof6matrix(params, x, y, z):
    R = rotation_about_point_matrix(*params[3:], x, y, z)
    T = translation_matrix(*params[:3])
    return R @ T


def _dof9matrix(params, x, y, z):
    R = rotation_about_point_matrix(*params[3:6], x, y, z)
    S = scale_matrix(*params[6:])
    T = translation_matrix(*params[:3])
    return S @ R @ T


def __is_floating_point(x):
    xp = cupy.get_array_module(x)
    return x.dtype in (xp.float16, xp.float32, xp.float64)


def optimize(ref, mov, metric,
             par0=[0,]*6, nbin_ref=64, nbin_mov=64,
             bounds=None, verbose=False,
             **opt_kwargs):
    """find optimal transformation s.t. `mov` is registered to `ref`
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
    # move `ref` to texture memory
    ref_tex, tex_arr = create_texture_object(ref, 'border', 'linear',
                                             'element_type')
    # partial function for making proper parameter vector
    if len(par0) == 3:
        mat_fun = lambda p: _dof3matrix(p).astype(float).flatten()[:12]
    elif len(par0) == 6:
        mat_fun = lambda p: \
            _dof6matrix(p, cx, cy, cz).astype(float).flatten()[:12]
    elif len(par0) == 9:
        mat_fun = lambda p: \
            _dof9matrix(p, cx, cy, cz).astype(float).flatten()[:12]
    elif len(par0) == 12:
        mat_fun = lambda p: p.astype(float)
    else:
        raise ValueError(
            'can only optimize 3/6/9/12 parameters, passed {:d}'.format(
                len(par0)
            )
        )
    # formulate optimization function
    # NOTE: scipy only has a `minimize` function and most of these metrics
    # we want to maximize, so for metrics confined to [0,1] we do
    # 1 - metric and for unbound metrics, take its negative
    if metric == 'cr':
        met_fun = functools.partial(_corr_ratio_reftex_ewk,
                                    ref_tex, mov)
        opt_fun = lambda p: 1.0 - float(met_fun(mat_fun(p)))
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
            opt_fun = lambda p: -float(met_fun(mat_fun(p)))
        elif metric == 'ec':
            met_fun = functools.partial(
                _entropy_correlation_coeff_precomp_target,
                ref_tex, mov, nbin_ref=nbin_ref, nbin_tar=nbin_mov,
                H_tar=H_mov
            )
            opt_fun = lambda p: 1.0 - float(met_fun(mat_fun(p)))
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
        def cback(x, pbar):
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
            _cback = functools.partial(cback, pbar=pbar)
            res = opt_call(callback=_cback)
    else:
        res = opt_call()
    # get rid of texture objects (ensures they get GC'd)
    del ref_tex, tex_arr
    # calculate transform
    T = mat_fun(res.x).reshape(3, 4)
    T = numpy.vstack([T, [0,0,0,1]])
    return numpy.linalg.inv(T), res
