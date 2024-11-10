import math

import cupy
import numpy
from numba import njit, prange

from ..typing import NDArray

def deskew_stage_scan(im : NDArray, pixel_size : float, step_size : float,
                      direction : int, theta : float=math.pi/4) -> NDArray:
    """deskew stage scan data into xyz coordinate system
        output format is zyx where z is normal to the coverslip,
        and x & y are the coverslip
        y is along the long axis of the sheet, x along is the scan direction

        uses the 'orthogonal interpolation' scheme where interpolation is done
        orthogonally to the direction of the light sheet. this was originally
        proposed/implemented in V. Maioli's Ph.D. thesis [1]. our code is based
        heavily on the implementation by D. Shepherd's group [2].

    References
    ---
    [1] Vincent Miaioli's PhD Thesis doi: 10.25560/68022
    [2] github.com/QI2lab/OPM

    :param im: input volume to be deskewed
    :type im: NDArray
    :param pixel_size: size of pixels, in real (space) units
    :type pixel_size: float
    :param step_size: real space spacing between sheets
    :type step_size: float
    :param direction: scan direction (+/- 1)
    :type direction: int
    :param theta: angle of objective w.r.t coverslip
    :type theta: float
    :returns: deskewed volume
    :rtype NDArray
    """

    xp = cupy.get_array_module(im)
    if xp == numpy:
        return _deskew_orthogonal_cpu(
            im, pixel_size, step_size, direction, theta
        )
    else:
        return _deskew_orthogonal_gpu(
            im, pixel_size, step_size, direction, theta
        )


@njit(parallel=True)
def _deskew_orthogonal_cpu(im, pixel_size, step_size, direction, 
                           theta=numpy.pi/4):
    direction = numpy.sign(direction)
    # convert step to pixel units
    step_pix = step_size / pixel_size
    # precompute useful trig quantities
    sin_theta  = numpy.float64(numpy.sin(theta))
    cos_theta  = numpy.float64(numpy.cos(theta))
    tan_theta  = numpy.float64(numpy.tan(theta))
    tan_otheta = numpy.float64(numpy.tan(numpy.pi/2 - theta))
    cos_otheta = numpy.float64(numpy.cos(numpy.pi/2 - theta))
    # determine output shape & preallocate
    n_planes, n_y, h = im.shape  # unpack shape
    n_x = numpy.int64(numpy.ceil(n_planes * step_pix + h * cos_theta))
    n_y = numpy.int64(n_y)
    n_z = numpy.int64(numpy.ceil(h * sin_theta))
    dsk = numpy.zeros((n_z, n_y, n_x), dtype=numpy.float64)
    for z in prange(0, n_z):
        for x in prange(0, n_x):
            # compute where we are in "raw" coordinates (x'=xpr, z'=zpr)
            if direction > 0:
                xpr = z / sin_theta
                zpr = (x - z / tan_theta) / step_pix
            else:  # direction < 0
                # NOTE: reverse direction heavily exploits reverse indexing
                # into arrays in Python, so many of these will be negative
                xpr = z / sin_theta - h
                zpr = (x + z / tan_theta - h * cos_theta) / step_pix
            # get plane in raw data before z'
            zpb = numpy.int64(numpy.floor(zpr))
            if numpy.abs(zpb+1) < n_planes:
                # beta_p = distance to raw plane in z' axis
                beta_p = step_pix * (zpr - zpb)
                # find x' in before & after planes
                xib  = xpr + direction * beta_p * tan_otheta
                xibp = numpy.int64(numpy.floor(xib))
                if numpy.abs(xibp+1) < h:
                    dxib = xib - xibp
                    xia  = xpr - direction * (step_pix - beta_p) * tan_otheta
                    xiap = numpy.int64(numpy.floor(xia))
                    if numpy.abs(xiap+1) < h:
                        dxia = xia - xiap
                        # calculate distances along the x-axis from point
                        # (x',z') to before/after planes in raw data
                        l_b = beta_p / cos_otheta
                        l_a = (step_pix - beta_p) / cos_otheta
                        val = l_b * (dxia * im[zpb+1,:,xiap+1] + (1-dxia) * im[zpb+1,:,xiap]) + \
                            l_a * (dxib * im[zpb,:,xibp+1] + (1-dxib) * im[zpb,:,xibp])
                        dsk[z,:,x] = val / step_pix
    # zero out triangle in left-hand side of image where data was
    # (falsely) interpolated due to wrapping
    if direction < 0:
        dsk = numpy.flipud(dsk)
    for z in prange(n_z):
        for x in prange(n_x):
            if x <= z:
                dsk[z,:,x] = 0
    dsk = numpy.flipud(dsk)
    return dsk[...,::direction]


def _deskew_orthogonal_gpu(im, pixel_size, step_size, direction, 
                           theta=numpy.pi/4):
    direction = cupy.sign(direction)
    # convert step to pixel units
    step_pix = step_size / pixel_size
    # precompute useful trig quantities
    sin_theta  = cupy.float64(cupy.sin(theta))
    cos_theta  = cupy.float64(cupy.cos(theta))
    tan_theta  = cupy.float64(cupy.tan(theta))
    tan_otheta = cupy.float64(cupy.tan(cupy.pi/2 - theta))
    cos_otheta = cupy.float64(cupy.cos(cupy.pi/2 - theta))
    # determine output shape & preallocate
    n_planes, n_y, h = im.shape  # unpack shape
    n_x = cupy.int64(cupy.ceil(n_planes * step_pix + h * cos_theta))
    n_y = cupy.int64(n_y)
    n_z = cupy.int64(cupy.ceil(h * sin_theta))
    dsk = cupy.zeros((n_z, n_y, n_x), dtype=cupy.float64)
    for z in range(0, n_z):
        for x in range(0, n_x):
            # compute where we are in "raw" coordinates (x'=xpr, z'=zpr)
            if direction > 0:
                xpr = z / sin_theta
                zpr = (x - z / tan_theta) / step_pix
            else:  # direction < 0
                # NOTE: reverse direction heavily exploits reverse indexing
                # into arrays in Python, so many of these will be negative
                xpr = z / sin_theta - h
                zpr = (x + z / tan_theta - h * cos_theta) / step_pix
            # get plane in raw data before z'
            zpb = cupy.int64(cupy.floor(zpr))
            if cupy.abs(zpb+1) < n_planes:
                # beta_p = distance to raw plane in z' axis
                beta_p = step_pix * (zpr - zpb)
                # find x' in before & after planes
                xib  = xpr + direction * beta_p * tan_otheta
                xibp = cupy.int64(cupy.floor(xib))
                if cupy.abs(xibp+1) < h:
                    dxib = xib - xibp
                    xia  = xpr - direction * (step_pix - beta_p) * tan_otheta
                    xiap = cupy.int64(cupy.floor(xia))
                    if cupy.abs(xiap+1) < h:
                        dxia = xia - xiap
                        # calculate distances along the x-axis from point
                        # (x',z') to before/after planes in raw data
                        l_b = beta_p / cos_otheta
                        l_a = (step_pix - beta_p) / cos_otheta
                        val = l_b * (dxia * im[zpb+1,:,xiap+1] + (1-dxia) * im[zpb+1,:,xiap]) + \
                            l_a * (dxib * im[zpb,:,xibp+1] + (1-dxib) * im[zpb,:,xibp])
                        dsk[z,:,x] = val / step_pix
    # zero out triangle in left-hand side of image where data was
    # (falsely) interpolated due to wrapping
    if direction < 0:
        dsk = cupy.flipud(dsk)
    for z in range(n_z):
        for x in range(n_x):
            if x <= z:
                dsk[z,:,x] = 0
    dsk = cupy.flipud(dsk)
    return dsk[...,::direction]