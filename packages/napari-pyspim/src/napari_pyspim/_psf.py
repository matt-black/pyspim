"""Vendored PSF generation functions from pyspim.psf.gauss.

These functions are pure numpy/scipy and are vendored here to allow
the napari-pyspim app to run without requiring pyspim installed locally.
"""

from functools import partial
from typing import Iterable

import numpy


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


def generate_psf_im(pars, im_shape: Iterable[int], gtype: str) -> numpy.ndarray:
    """Generate a Gaussian PSF volume.

    Args:
        pars: Parameters for the Gaussian. For spherical:
            [x0, y0, z0, sx, sy, sz, ampl, bkgrnd].
            For elliptical:
            [x0, y0, z0, sx, sy, sz, sxy, sxz, syz, ampl, bkgrnd].
        im_shape: Shape of the output volume (z, y, x).
        gtype: Type of Gaussian, either "spherical" or "elliptical".

    Returns:
        A numpy.ndarray of shape im_shape containing the PSF volume.
    """
    assert gtype in ("spherical", "elliptical"), (
        "valid `gtype`s are 'spherical' and 'elliptical' only"
    )
    z, y, x = numpy.meshgrid(*[numpy.arange(s) for s in im_shape], indexing="ij")
    if gtype == "spherical":
        return spherical_gaussian_3d_fit(pars, x, y, z)  # type: ignore (ufunc generates array)
    else:
        return elliptical_gaussian_3d_fit(pars, x, y, z)  # type: ignore (ufunc generates array)


def normalize_psf_im(psf_img: numpy.ndarray) -> numpy.ndarray:
    """Normalize PSF to unit total intensity.

    Args:
        psf_img: Input PSF volume.

    Returns:
        A copy of psf_img normalized so that the sum equals 1.
    """
    return psf_img.copy() / numpy.sum(psf_img)
