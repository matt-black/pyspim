import cupy

import skimage
from .typing import NDArray, BBox2D, BBox3D
from .util import threshold_triangle
from ._util import get_ndimage_module


def detect_roi_2d(im : NDArray, method : str='triangle', quantile : float=0,
                  **kwargs) -> BBox2D:
    """detect_roi_2d Determine ROI in 2D image by thresholding. If using `method=threshold`, the `threshold=[value]` keyword arg must be passed into function.
    Args:
        im (NDArray): image to find ROI in
        method (str, optional): thresholding method, one of ('triangle', 'otsu', 'threshold'). Defaults to 'triangle'.
        quantile (float, optional): quantile to truncate coordinates with (helps get rid of outliers). Defaults to 0.

    Returns:
        BBox2D: bounding box ROI
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
    """detect_roi_3d Detect ROI of 3D volume by combining ROIs of 2D max-projections.

    Args:
        im (NDArray): volume to find ROI of
        method (str, optional): thresholding method. Defaults to 'triangle'.
        quantile_rc (float, optional): qunatile used for determining row-col bounding box. Defaults to 0.
        quantile_zc (float, optional): quantile used for determine z-row bounding box. Defaults to 0.

    Returns:
        BBox3D: format is `((lb_z, ub_z), (lb_r,ub_r), (lb_c,ub_c))`
    """
    xp = cupy.get_array_module(im)
    rc = xp.amax(im, axis=0)
    (rs, re), (cs, ce) = detect_roi_2d(rc, method, quantile_rc, **kwargs)
    zc = xp.amax(im, axis=1)
    (zs, ze), _ = detect_roi_2d(zc, method, quantile_zc, **kwargs)
    return (zs, ze), (rs, re), (cs, ce)


def combine_rois(a : BBox3D, b : BBox3D, ensure_even : bool=False) -> BBox3D:
    """combine_rois Combine 2 ROIs into smallest possible one that encompasses both.

    Args:
        a (BBox3D): 3d bounding box for first ROI
        b (BBox3D): 3d bounding box for second ROI
        ensure_even (bool, optional): make all dimensions of output ROI even. Defaults to False.

    Returns:
        BBox3D
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


def crop_with_roi(a : NDArray, roi : BBox3D|BBox2D) -> NDArray:
    """crop_with_roi Utility function to crop input using specified bounding box ROI.

    Args:
        a (NDArray): input to be cropped
        roi (BBox3D | BBox2D): bounding box ROI

    Raises:
        ValueError: if input isn't 2 or 3D

    Returns:
        NDArray: cropped input
    """
    if len(a.shape) == 3:
        return a[roi[0][0]:roi[0][1],
                 roi[1][0]:roi[1][1],
                 roi[2][0]:roi[2][1]]
    elif len(a.shape) == 2:
        return a[roi[0][0]:roi[0][1],
                 roi[1][0]:roi[1][1]]
    else:
        raise ValueError('invalid shape, must be 2 or 3D input')