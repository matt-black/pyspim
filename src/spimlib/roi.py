import cupy

import skimage
from .typing import NDArray, BBox2D, BBox3D
from .util import threshold_triangle
from ._util import get_ndimage_module


def detect_roi_2d(im : NDArray, method : str='triangle', quantile : float=0,
                  **kwargs) -> BBox2D:
    """determine ROI in 2d image by thresholding
        if using `method=threshold` then `threshold=[value]` must be passed as
        a keyword argument
        
    :param im: image to find ROI in
    :type im: NDArray
    :param method: thresholding method
       one of ('triangle', 'otsu', 'threshold')
    :type method: str
    :param quantile: quantile to truncate coordinates with
        this helps get rid of outlier values
    :type quantile: float
    :returns: bounding box ROI
    :rtype: Tuple[Tuple[int,int],Tuple[int,int]]
    """
    assert method in ('triangle', 'otsu', 'threshold'), \
        "invalid segmentation method specified"
    if method == 'threshold':
        thresh = kwargs['threshold']
    xp = cupy.get_array_module(im)
    ndi = get_ndimage_module(xp)
    # get rid of line artifacts
    vert_line = xp.zeros((5, 5))
    vert_line[:,2] = 1
    thrim = ndi.grey_erosion(
        ndi.grey_erosion(im, structure=vert_line),
        structure=vert_line.T
    )
    # threshold the image by triangle method
    if method == 'triangle':
        thresh = threshold_triangle(im)
    elif method == 'otsu':
        thresh = skimage.filters.threshold_otsu(im)
    else:  # method == 'threshold'
        pass  # already defined this, above when we grabbed from **kwargs
    
    # dilate 3 times to remove small particles
    thrim = ndi.binary_dilation(thrim > thresh, iterations=3,
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


def detect_roi_3d(im : NDArray, method : str='triangle',
                  quantile_rc : float=0, quantile_zc : float=0,
                  **kwargs) -> BBox3D:
    """determine 3d bounding box by combining ROIs of 2D max-projections
        if using `method=threshold` then `threshold=[value]` must be passed as
        a keyword argument
        
    :param im: image to find ROI in
    :type im: NDArray
    :param method: thresholding method
       one of ('triangle', 'otsu', 'threshold')
    :type method: str
    :param quantile_rc: quantile used for determining row-column bounding box
    :type quantile_rc: float
    :param quantile_zc: quantile used for determining z-row bounding box
    :type quantile_zc: float
    :returns: bounding box ROI
       format is `((lb_z, ub_z), (lb_r,ub_r), (lb_c,ub_c))`
    :rtype: Tuple[Tuple[int,int],Tuple[int,int],Tuple[int,int]]
    """
    xp = cupy.get_array_module(im)
    rc = xp.amax(im, axis=0)
    (rs, re), (cs, ce) = detect_roi_2d(rc, method, quantile_rc, **kwargs)
    zc = xp.amax(im, axis=1)
    (zs, ze), _ = detect_roi_2d(zc, method, quantile_zc, **kwargs)
    return (zs, ze), (rs, re), (cs, ce)


def combine_rois(a : BBox3D, b : BBox3D, ensure_even : bool=False) -> BBox3D:
    """make the smallest ROI that encompasses both ROI's `a` & `b`

    :param a: bounding box for first ROI
    :type a: BBox3D
    :param b: bounding box for second ROI
    :type b: BBox3D
    :param ensure_even: make all dimensions of output ROI even
    :type ensure_even: bool
    :returns: bounding box ROI
    :rtype: Tuple[Tuple[int,int],Tuple[int,int],Tuple[int,int]]
    """
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
