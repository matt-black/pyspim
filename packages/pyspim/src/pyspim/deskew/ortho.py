""" Deskewing by orthogonal interpolation.

Uses the 'orthogonal interpolation' scheme where interpolation is done orthogonally to the direction of the light sheet. this was originally proposed/implemented in V. Maioli's Ph.D. thesis [1]. Our code is based heavily on the implementation by D. Shepherd's group [2].

References
---
[1] Vincent Miaioli's PhD Thesis doi: 10.25560/68022
[2] github.com/QI2lab/OPM
"""
import math
from typing import Tuple

import cupy
import numpy
from numba import njit, prange

from ..typing import NDArray


def deskew_stage_scan(im : NDArray, pixel_size : float, step_size : float,
                      direction : int, theta : float=math.pi/4,
                      preserve_dtype : bool = False) -> NDArray:
    """Deskew stage scan data into xyz coordinate system.
    
    Output format is zyx where z is normal to the coverslip, and x & y are the coverslip, y is along the long axis of the sheet, x along is the scan direction.

    Args:
        im (NDArray): input volume to be deskewed
        pixel_size (float): size of pixels, in real (space) units
        step_size (float): real space spacing between sheets
        direction (int): scan direction (+/-1)
        theta (float): angle of objective w.r.t coverslip (in radians)
        preserve_dtype (bool): preserve input datatype in output. If ``False``, will return single-precision float. 
    Returns:
        NDArray
    """
    xp = cupy.get_array_module(im)
    if xp == numpy:
        dsk = _deskew_orthogonal_cpu(
            im, pixel_size, step_size, direction, theta, preserve_dtype
        )
    else:
        dsk = _deskew_orthogonal_gpu(
            im, pixel_size, step_size, direction, theta, preserve_dtype
        )
    return dsk
    

@njit(parallel=True)
def _deskew_orthogonal_cpu(im : numpy.ndarray, 
                           pixel_size : float, step_size : float, 
                           direction :int, theta : float = numpy.pi/4, 
                           preserve_dtype : bool = False):
    direction = numpy.sign(direction)
    step_size_lat = step_size / math.cos(theta)
    # convert step to pixel units
    step_pix = step_size_lat / pixel_size
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
    dsk = numpy.zeros((n_z, n_y, n_x), dtype=im.dtype)
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
                        if preserve_dtype:
                            dsk[z,:,x] = numpy.clip(numpy.round(
                                val / step_pix), 0, 2**16-1
                            ).astype(im.dtype)
                        else:
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


def output_shape(z : int, r : int, c : int, 
                 pixel_size : float, step_size : float, 
                 theta : float=math.pi/4) -> Tuple[int,int,int]:
    """output_shape Calculate output shape of deskewing a volume.

    Args:
        z (int): size of input volume in z-direction
        r (int): size of input volume in r-direction (# rows)
        c (int): size of input volume in c-direction (# cols)
        pixel_size (float): pixel size, in real units
        step_size (float): step size, in real units
        theta (float, optional): angle of objective w.r.t coverslip. Defaults to math.pi/4.

    Returns:
        Tuple[int,int,int]: (n_z, n_y, n_x) shape of deskewed volume
    """
    step_size_lat = step_size / math.cos(theta)
    step_pix = step_size_lat / pixel_size
    # precompute useful trig quantities
    # determine output shape & preallocate
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    n_x = numpy.int64(numpy.ceil(z * step_pix + c * cos_theta))
    n_y = numpy.int64(r)
    n_z = numpy.int64(numpy.ceil(c * sin_theta))
    return (n_z, n_y, n_x)


def _deskew_orthogonal_gpu(im : cupy.ndarray, 
                           pixel_size : float, step_size : float, 
                           direction : int, theta : float = numpy.pi/4, 
                           preserve_dtype : bool = False):
    direction = cupy.sign(direction)
    # convert step to pixel units
    step_size_lat = step_size / math.cos(theta)
    step_pix = step_size_lat / pixel_size
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
    dsk = cupy.zeros((n_z, n_y, n_x), 
                     dtype=(im.dtype if preserve_dtype else cupy.float32))
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
                        if preserve_dtype:
                            dsk[z,:,x] = cupy.clip(numpy.round(
                                val / step_pix), 0, 2**16-1
                            ).astype(im.dtype)
                        else:
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