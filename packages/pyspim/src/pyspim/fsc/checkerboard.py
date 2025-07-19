"""image/volume splitting by checkerboard subsampling
"""

from typing import Tuple

import cupy

from ..typing import NDArray


def checkerboard_split(arr : NDArray, split : int) -> Tuple[NDArray,NDArray]:
    """checkerboard split images or volumes for FSC analysis
        generate pairs of images for FS(R)C analysis by subsampling
        them, as described in Fig. 2(a) of [1]

    References
    ---
    [1] Koho, et al. "Fourier ring correlation..." doi:10.1038/s41467-019-11024-z

    :param arr: input image/volume
    :type arr: NDArray
    :param split: type of split to do
    :type split: int
    :returns: checkerboard-subsampled halves of input array
    :rtype: Tuple[NDArray,NDArray]
    """
    if len(arr.shape) == 3:
        return _checkerboard_split_vol(arr, split)
    elif len(arr.shape) == 2:
        return _checkerboard_split_img(arr, split)
    else:
        raise ValueError('input arr must be 2- or 3-dimensional')


def _checkerboard_split_vol(vol : NDArray, split : int) -> \
        Tuple[NDArray,NDArray]:
    """
    """
    assert split > 0 and split < 5, "split must be in [1,4]"
    depth, height, width = vol.shape
    xp = cupy.get_array_module(vol)
    # pre-compute indices
    eidxd = xp.arange(0, depth, 2)
    eidxh = xp.arange(0, height, 2)
    eidxw = xp.arange(0, width, 2)
    oidxd = xp.arange(1, depth, 2)
    oidxh = xp.arange(1, height, 2)
    oidxw = xp.arange(1, width, 2)
    if split == 1:  # even/even/even & odd/odd/odd
        v1 = vol[eidxd,...][:,eidxh,:][...,eidxw]
        v2 = vol[oidxd,...][:,oidxh,:][...,oidxw]
    elif split == 2:  # odd/odd/even, even/even/odd
        v1 = vol[oidxd,...][:,oidxh,:][...,eidxw]
        v2 = vol[eidxd,...][:,eidxh,:][...,oidxw]
    elif split == 3:  # odd/even/even, even/odd/odd
        v1 = vol[oidxd,...][:,eidxh,:][...,eidxw]
        v2 = vol[eidxd,...][:,oidxh,:][...,oidxw]
    else:  # (split == 4), odd/even/odd, even/odd/even
        v1 = vol[oidxd,...][:,eidxh,:][...,oidxw]
        v2 = vol[eidxd,...][:,oidxh,:][...,eidxw]
    return v1, v2


def _checkerboard_split_img(img : NDArray, split : int) -> \
        Tuple[NDArray,NDArray]:
    """
    """
    assert split == 1 or split == 2, "split must be [1,2]"
    height, width = img.shape
    xp = cupy.get_array_module(img)
    # pre-compute indices for selecting
    eidxh = xp.arange(0, height, 2)
    eidxw = xp.arange(0, width, 2)
    oidxh = xp.arange(1, height, 2)
    oidxw = xp.arange(1, width, 2)
    if split == 1:  # split 1 is even/even & odd/odd (rows/cols)    
        im1 = img[eidxh,:][:,eidxw]
        im2 = img[oidxh,:][:,oidxw]
    else:  # split 2 is even/odd & odd/even (rows/cols)
        im1 = img[eidxh,:][:,oidxw]
        im2 = img[oidxh,:][:,eidxw]
    return im1, im2
