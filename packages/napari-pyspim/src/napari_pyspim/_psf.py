"""
PSF generation functions for deconvolution.

Vendored from pyspim.psf.gauss (pure numpy/scipy, no cupy dependency).
Source: packages/pyspim/src/pyspim/psf/gauss.py

These functions are needed at module level by _deconvolution.py but should
not require pyspim (and its cupy dependency) to be installed for remote-only
usage.
"""

from functools import partial
from typing import Iterable, Tuple

import numpy
from scipy import optimize


def _gauss_term(x0: float, sx: float, x: float) -> float:
    return numpy.square(x - x0) / (2 * numpy.square(sx))


def _gauss_cross_term(x0: float, y0: float, s: float, x: float, y: float) -> float:
    return ((x - x0) * (y - y0)) / (2 * numpy.square(s))


def spherical_gaussian_3d(
    x0: float,
    y0: float,
    z0: float,
    sx: float,
    sy: float,
    sz: float,
    ampl: float,
    bkgrnd: float,
    x: float,
    y: float,
    z: float,
) -> float:
    gtx = partial(_gauss_term, x0, sx)
    gty = partial(_gauss_term, y0, sy)
    gtz = partial(_gauss_term, z0, sz)
    return ampl * numpy.exp(-(gtx(x) + gty(y) + gtz(z))) + bkgrnd


def elliptical_gaussian_3d(
    x0: float,
    y0: float,
    z0: float,
    sx: float,
    sy: float,
    sz: float,
    sxy: float,
    sxz: float,
    syz: float,
    ampl: float,
    bkgrnd: float,
    x: float,
    y: float,
    z: float,
) -> float:
    stx = partial(_gauss_term, x0, sx)
    sty = partial(_gauss_term, y0, sy)
    stz = partial(_gauss_term, z0, sz)
    ctxy = partial(_gauss_cross_term, x0, y0, sxy)
    ctxz = partial(_gauss_cross_term, x0, z0, sxz)
    ctyz = partial(_gauss_cross_term, y0, z0, syz)
    exp_term = numpy.exp(
        -(stx(x) + sty(y) + stz(z) + ctxy(x, y) + ctxz(x, z) + ctyz(y, z))
    )
    return ampl * exp_term + bkgrnd


def spherical_gaussian_3d_fit(pars, x: float, y: float, z: float) -> float:
    x0, y0, z0, sx, sy, sz, ampl, bkgrnd = pars
    return spherical_gaussian_3d(x0, y0, z0, sx, sy, sz, ampl, bkgrnd, x, y, z)


def elliptical_gaussian_3d_fit(pars, x: float, y: float, z: float) -> float:
    x0, y0, z0, sx, sy, sz, sxy, sxz, syz, ampl, bkgrnd = pars
    return elliptical_gaussian_3d(
        x0, y0, z0, sx, sy, sz, sxy, sxz, syz, ampl, bkgrnd, x, y, z
    )


def _min_fun_spherical(ref, x, y, z, pars) -> float:
    err = numpy.sqrt(
        numpy.sum(
            numpy.square(
                spherical_gaussian_3d_fit(pars, x, y, z).flatten() - ref.flatten()
            )
        )
    )
    N = len(ref.flatten())
    return float(err / N)


def _min_fun_elliptical(ref, x, y, z, pars) -> float:
    err = numpy.sqrt(
        numpy.sum(
            numpy.square(
                elliptical_gaussian_3d_fit(pars, x, y, z).flatten() - ref.flatten()
            )
        )
    )
    N = len(ref.flatten())
    return float(err / N)


def fit_3d_psf_gaussian(emp_psf: numpy.ndarray, par0=None, gtype: str = "spherical"):
    assert gtype in ("spherical", "elliptical"), (
        "valid `gtype`s are 'spherical' and 'elliptical' only"
    )
    if par0 is None:
        x0, y0, z0 = _init_guess_3d_psf_loc(emp_psf)
        sx, sy, sz = 3, 3, 3
        # background and amplitude estimations
        max_val = numpy.amax(emp_psf)
        bkgrnd = numpy.quantile(emp_psf, 0.2)
        ampl = (max_val - bkgrnd) / 2
        if gtype == "spherical":
            par0 = [x0, y0, z0, sx, sy, sz, ampl, bkgrnd]
        else:
            par0 = [x0, y0, z0, sx, sy, sz, sx, sz, sz, ampl, bkgrnd]
        par0 = list([float(f) for f in par0])
    else:
        if gtype == "spherical":
            assert par0.shape[0] == 8, "par0 should have 8 parameters"
        else:
            assert par0.shape[0] == 11, "par0 should have 11 parameters"
    # generate dummy coordinates for z, y, x dimensions
    z, y, x = numpy.meshgrid(*[numpy.arange(s) for s in emp_psf.shape], indexing="ij")
    if gtype == "spherical":
        f_min = partial(_min_fun_spherical, emp_psf, x, y, z)
    else:
        f_min = partial(_min_fun_elliptical, emp_psf, x, y, z)

    return optimize.minimize(f_min, par0)


def generate_psf_im(pars, im_shape: Iterable[int], gtype: str) -> numpy.ndarray:
    assert gtype in ("spherical", "elliptical"), (
        "valid `gtype`s are 'spherical' and 'elliptical' only"
    )
    z, y, x = numpy.meshgrid(*[numpy.arange(s) for s in im_shape], indexing="ij")
    if gtype == "spherical":
        return spherical_gaussian_3d_fit(pars, x, y, z)
    else:
        return elliptical_gaussian_3d_fit(pars, x, y, z)


def normalize_psf_im(psf_img: numpy.ndarray) -> numpy.ndarray:
    return psf_img.copy() / numpy.sum(psf_img)


def gaussian_fwhm(sigma: float) -> float:
    """Compute full-width at half-max for Gaussian from std. dev.

    Args:
        sigma: standard deviation of Gaussian

    Returns:
        Full-width at half-maximum
    """
    return 2 * numpy.sqrt(2 * numpy.log(2)) * sigma


def _init_guess_3d_psf_loc(emp_psf: numpy.ndarray) -> Tuple[float, float, float]:
    """Make an approximate guess as to where the PSF is centered in the image.

    This is done by just looking for the maximum value in the image and saying
    its probably centered there.
    """
    max_val = numpy.amax(emp_psf)
    z0, y0, x0 = numpy.where(emp_psf == max_val)
    x0, y0, z0 = x0[0], y0[0], z0[0]
    return x0, y0, z0
