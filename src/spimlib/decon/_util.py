from typing import Optional, Tuple

import cupy
import numpy

from ..typing import NDArray, BBox3D

__tup3 = Tuple[int,int,int]  # convenience type for output


def gaussian_kernel_1d(sigma : float, radius : int):
    sigma2 = sigma * sigma
    x = numpy.arange(-radius, radius+1)
    phi = numpy.exp(-0.5 / sigma2 * x**2)
    return phi / phi.sum()


def crop_and_pad_for_deconv(vol : NDArray,
                            bbox : Optional[BBox3D], pad : int) -> NDArray:
    """crop and pad input volume in preparation for deconvolution
        deconvolution will produce artifacts at the edges of images (volumes)
        to help remedy this, the input can be padded, deconvolved, and then
        the padded regions cropped out post-deconvolution.
        this function will take the input volume, crop it, and then pad the
        edges such that after deconvolution, one can re-crop it to get
        rid of the aforementioned artifacts. instead of just cropping and
        padding, if the bbox and padding are still contained in the original
        volume, this function will under-crop the appropriate amount such that
        original parts of the volume are kept (instead of padding)
    
    :param vol: input volume to be deconvolved
    :type vol: NDArray
    :param bbox: bounding box to crop volume down to
        if `None`, cropping is not done
    :type bbox: Optional[BBox3D]
    :param pad: amount of padding past bbox (same for all axes)
    :type pad: int
    :returns: (under)cropped and possibly-padded volume ready for deconvolution
    :rtype: NDArray
    """
    if bbox is None:
        bbox = tuple(zip([0,0,0], list(vol.shape)))
    (padl, padr), (lbd, ubd) = _crop_bounds_and_padding(vol, bbox, pad)
    return _crop_and_pad(vol, padl, padr, lbd, ubd)



def _crop_bounds_and_padding(vol : NDArray, bbox : BBox3D, pad : int) -> \
        Tuple[Tuple[__tup3,__tup3], Tuple[__tup3, __tup3]]:
    """determine amount of padding and upper/lower bounds of crop box to
        use when cropping/padding the input volume in preparation
        for deconvolution

    :param vol: input volume
    :type vol: NDArray
    :param bbox: bounding box to crop volume down to
    :type bbox: BBox3D
    :param pad: amount of padding past bbox (same for all axes)
    :type pad: int
    :returns: paddings and upper/lower bounds
    """
    shp = vol.shape
    # figure out lower bound of (maybe) padded image
    lbd = [x[0] - pad for x, s in zip(bbox, shp)]
    # if the bound is negative, we have to pad to the left
    # otherwise, no padding
    pdl = tuple([abs(l) if l < 0 else 0 for l in lbd])
    # correct for negative bounds which are now 0 (b/c of padding)
    lbd = tuple([0 if l < 0 else l for l in lbd])
    ## now we follow the same logic for the upper bound except
    ## minuses are pluses and we have to check >shape instead of <0
    ubd = [x[1] + pad for x, _ in zip(bbox, shp)]
    pdr = tuple([v - s if v > s else 0 for v, s in zip(ubd, shp)])
    ubd = tuple([s if v > s else v for v, s in zip(ubd, shp)])
    return (pdl, pdr), (lbd, ubd)


def _crop_and_pad(vol : NDArray,
                  padl : __tup3, padr : __tup3,
                  lowb : __tup3, uppb : __tup3):
    """crop and pad the input volume w. specified parameters
        padding is done with option 'symmetric' for (see : `numpy.pad`)

    :param vol: input volume to be deconvolved
    :type vol: NDArray
    :param padl: left paddings for each axis
    :type padl: Tuple[int,int,int]
    :param padr: right paddings for each axis
    :type padr: Tuple[int,int,int]
    :param lowb: lower crop bounds for each axis
    :type lowb: Tuple[int,int,int]
    :param uppb: upper crop bounds for each axis
    :type uppb: Tuple[int,int,int]
    :returns: cropped and padded volume suitable for deconvolution
    :rtype: NDArray
    """
    xp = cupy.get_array_module(vol)
    return xp.pad(vol[lowb[0]:uppb[0],lowb[1]:uppb[1],lowb[2]:uppb[2]],
                  [(l, r) for l, r in zip(padl, padr)],
                  'symmetric')


## stable division: only divide `a / b` if denominator, `b`, isn't too small
div_stable = cupy.ElementwiseKernel(
    'T a, T b, float32 eps',
    'T o',
    '''
    o = (b > eps) ? a / b : 0;
    ''',
    'div_stable_kernel',
)


def initialize_estimate(a : cupy.ndarray, b : cupy.ndarray, order : str = 'F'):
    out = cupy.zeros(a.shape, dtype=cupy.float32, order=order)
    _initialize_estimate_kernel(a, b, out)
    return out


_initialize_estimate_kernel = cupy.ElementwiseKernel(
    'T a, T b', 'float32 o',
    '''
    o = (a + b) / 2.0f
    ''',
    'initialize_estimate_kernel'
)