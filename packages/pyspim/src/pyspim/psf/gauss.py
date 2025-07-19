from typing import Iterable, Tuple
from functools import partial

import numpy
from scipy import optimize

from ..typing import NDArray


def __gauss_term(x0 : float, sx : float, x : float) -> float:
    return numpy.square(x - x0) / (2 * numpy.square(sx))


def __gauss_cross_term(x0 : float, y0: float, s: float, x: float, y: float) -> float:
    return ((x - x0) * (y - y0)) / (2 * numpy.square(s))


def spherical_gaussian_3d(x0: float, y0: float, z0: float, 
                          sx: float, sy: float, sz: float, 
                          ampl: float, bkgrnd: float, 
                          x: float, y: float, z: float) -> float:
    gtx = partial(__gauss_term, x0, sx)
    gty = partial(__gauss_term, y0, sy)
    gtz = partial(__gauss_term, z0, sz)
    return ampl * numpy.exp(-(gtx(x) + gty(y) + gtz(z))) + bkgrnd


def elliptical_gaussian_3d(x0: float, y0: float, z0: float, 
                           sx: float, sy: float, sz: float, 
                           sxy: float, sxz: float, syz: float,
                           ampl: float, bkgrnd: float, 
                           x: float, y: float, z: float) -> float:
    stx = partial(__gauss_term, x0, sx)
    sty = partial(__gauss_term, y0, sy)
    stz = partial(__gauss_term, z0, sz)
    ctxy = partial(__gauss_cross_term, x0, y0, sxy)
    ctxz = partial(__gauss_cross_term, x0, z0, sxz)
    ctyz = partial(__gauss_cross_term, y0, z0, syz)
    exp_term = numpy.exp(
        -(stx(x) + sty(y) + stz(z) + ctxy(x, y) + ctxz(x, z) + ctyz(y, z))
    )
    return ampl * exp_term + bkgrnd


def spherical_gaussian_3d_fit(pars, x : float, y : float, z : float) -> float:
    x0, y0, z0, sx, sy, sz, ampl, bkgrnd = pars
    return spherical_gaussian_3d(x0, y0, z0,
                                 sx, sy, sz,
                                 ampl, bkgrnd,
                                 x, y, z)


def elliptical_gaussian_3d_fit(pars, x: float, y: float, z: float) -> float:
    x0, y0, z0, sx, sy, sz, sxy, sxz, syz, ampl, bkgrnd = pars
    return elliptical_gaussian_3d(x0, y0, z0,
                                  sx, sy, sz, sxy, sxz, syz,
                                  ampl, bkgrnd, x, y, z)


def __min_fun_spherical(ref, x, y, z, pars) -> float:
    err = numpy.sqrt(numpy.sum(numpy.square(
        spherical_gaussian_3d_fit(pars, x, y, z).flatten() - ref.flatten()
    )))
    N = len(ref.flatten())
    return float(err/N)


def __min_fun_elliptical(ref, x, y, z, pars) -> float:
    err = numpy.sqrt(numpy.sum(numpy.square(
        elliptical_gaussian_3d_fit(pars, x, y, z).flatten() - ref.flatten()
    )))
    N = len(ref.flatten())
    return float(err/N)


def fit_3d_psf_gaussian(emp_psf: numpy.ndarray, par0=None, gtype: str='spherical'):
    assert gtype in ('spherical', 'elliptical'), \
        "valid `gtype`s are 'spherical' and 'elliptical' only"
    if par0 is None:
        x0, y0, z0 = __init_guess_3d_psf_loc(emp_psf)
        sx, sy, sz = 3, 3, 3
        # background and amplitude estimations
        max_val = numpy.amax(emp_psf)
        bkgrnd = numpy.quantile(emp_psf, 0.2)
        ampl = (max_val - bkgrnd) / 2
        if gtype == 'spherical':
            par0 = [x0, y0, z0, sx, sy, sz, ampl, bkgrnd]
        else:
            par0 = [x0, y0, z0, sx, sy, sz, sx, sz, sz, ampl, bkgrnd]
        par0 = list([float(f) for f in par0])
    else:
        if gtype == 'spherical':
            assert par0.shape[0] == 8, "par0 should have 8 parameters"
        else:
            assert par0.shape[0] == 11, "par0 should have 11 parameters"
    # generate dummy coordinates for z, y, x dimensions
    z, y, x = numpy.meshgrid(*[numpy.arange(s) for s in emp_psf.shape],
                          indexing='ij')
    if gtype == 'spherical':
        f_min = partial(__min_fun_spherical, emp_psf, x, y, z)
    else:
        f_min = partial(__min_fun_elliptical, emp_psf, x, y, z)
    
    return optimize.minimize(f_min, par0)


def generate_psf_im(pars, im_shape : Iterable[int], gtype : str) -> numpy.ndarray:
    assert gtype in ('spherical', 'elliptical'), \
        "valid `gtype`s are 'spherical' and 'elliptical' only"
    z, y, x = numpy.meshgrid(*[numpy.arange(s) for s in im_shape],
                          indexing='ij')
    if gtype == 'spherical':
        return spherical_gaussian_3d_fit(pars, x, y, z)
    else:
        return elliptical_gaussian_3d_fit(pars, x, y, z)


def normalize_psf_im(psf_img : NDArray) -> NDArray:
    return psf_img.copy() / numpy.sum(psf_img)


def gaussian_fwhm(sigma : float) -> float:
    """gaussian_fwhm Compute full-width at half-max for Gaussian from std. dev.
    
    Args:
        sigma (float): standard deviation of Gaussian

    Returns:
        float
    """
    return 2 * numpy.sqrt(2 * numpy.log(2)) * sigma
    

def __init_guess_3d_psf_loc(emp_psf : NDArray) -> Tuple[float,float,float]:
    """make an approximate guess as to where the PSF is centered in the image

    this is done by just looking for the maximum value in the image and saying
    its probably centered there
    """
    max_val = numpy.amax(emp_psf)
    z0, y0, x0 = numpy.where(emp_psf == max_val)
    x0, y0, z0 = x0[0], y0[0], z0[0]
    return x0, y0, z0
