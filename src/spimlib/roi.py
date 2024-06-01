import cupy
from numpy.typing import ArrayLike

from .util import threshold_triangle
from ._util import get_scipy_module, get_skimage_module


def detect_roi_2d(im : ArrayLike, method : str='triangle', quantile : float=0,
                  **kwargs):
    assert method in ('triangle', 'otsu', 'threshold'), \
        "invalid segmentation method specified"
    if method == 'threshold':
        thresh = kwargs['threshold']
    xp = cupy.get_array_module(im)
    sp = get_scipy_module(xp)
    sk = get_skimage_module(xp)
    # get rid of line artifacts
    vert_line = xp.zeros((5, 5))
    vert_line[:,2] = 1
    thrim = sp.ndimage.grey_erosion(
        sp.ndimage.grey_erosion(im, structure=vert_line),
        structure=vert_line.T
    )
    # threshold the image by triangle method
    if method == 'triangle':
        thresh = threshold_triangle(im)
    elif method == 'otsu':
        thresh = sk.filters.threshold_otsu(im)
    else:  # method == 'threshold'
        pass  # already defined this, above when we grabbed from **kwargs
    
    # dilate 3 times to remove small particles
    thrim = sp.ndimage.binary_dilation(thrim > thresh, iterations=3,
                                       brute_force=True)
    # figure out how to draw bounding box/ROI
    row, col = xp.nonzero(thrim)
    if quantile > 0:
        rq = xp.quantile(row, [quantile, 1-quantile])
        assert len(rq) == 2, 'should only return 2 values'
        rs, re = int(rq[0]), int(rq[1])
        cq = xp.quantile(col, [quantile, 1-quantile])
        cs, ce = int(cq[0]), int(cq[1])
    else:
        rs, re = int(xp.amin(row)), int(xp.amax(row)+1)
        cs, ce = int(xp.amin(col)), int(xp.amax(col)+1)
    return (rs, re), (cs, ce)


def detect_roi_3d(im : ArrayLike, method : str='triangle',
                  quantile_rc : float=0, quantile_zc : float=0,
                  **kwargs):
    xp = cupy.get_array_module(im)
    rc = xp.amax(im, axis=0)
    (rs, re), (cs, ce) = detect_roi_2d(rc, method, quantile_rc, **kwargs)
    zc = xp.amax(im, axis=1)
    (zs, ze), _ = detect_roi_2d(zc, method, quantile_zc, **kwargs)
    return (zs, ze), (rs, re), (cs, ce)


def combine_rois(a, b, ensure_even=False):
    (azs, aze), (ars, are), (acs, ace) = a
    (bzs, bze), (brs, bre), (bcs, bce) = b
    zs, ze = min([azs, bzs]), max([aze, bze])
    rs, re = min([ars, brs]), max([are, bre])
    cs, ce = min([acs, bcs]), max([ace, bce])
    if ensure_even:
        if (ze - zs) % 2 > 0:
            zs += 1
        if (re - rs) % 2 > 0:
            rs += 1
        if (ce - cs) % 2 > 0:
            cs += 1
    return (zs, ze), (rs, re), (cs, ce)
