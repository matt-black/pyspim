"""utilities for loading data acquired in micro-manager
with the ASI diSPIM plugin
"""
import os
import typing
import operator
import warnings
from types import ModuleType
from collections.abc import Iterable
from functools import partial
from itertools import accumulate

import zarr
import cupy
import numpy
import tifffile as tiff
from tqdm.auto import tqdm

from ..typing import NDArray


# NOTE: these lists are hard-coded to accomodate up to 10 channels/head
# if you need more, increase the range upper bound
__a_chans = list(range(0, 20, 2))
__b_chans = list(range(1, 20, 2))
def _page_calculator(shape : typing.Iterable[int], head : str,
                     time : int=0, channel : int=0) -> typing.Tuple[int,int]:
    """determine which pages to read in a single z-stack
        taken at a given time, with the specified head, and in the
        specified channel

    :param shape: shape of data series being read
    :type shape: Iterable[int]
    :param head: which head to read data from
    :type head: str
    :param time: timepoint to read data from
    :type time: int
    :param channel: channel to read
    :type channel: int
    :returns: start and end pages to read
    :rtype: Tuple[int,int]
    """
    z_sze = shape[-3]
    n_chn = shape[-4] // 2
    n_per_time = z_sze * n_chn * 2
    if head == 'a':
        start = n_per_time * time + __a_chans[channel] * z_sze
    else:  # head == 'b'
        start = n_per_time * time + __b_chans[channel] * z_sze
    end = start + z_sze
    return start, end


def uManagerAcquisition(path : str, xp : ModuleType=numpy):
    """create a class for loading a micro-manager acquisition

    :param path: path to acquisition folder or file
    :type path: str
    :param xp: one of `numpy` or `cupy` to return output arrays with
    :type xp: ModuleType
    :returns: class to load acquisition
    """
    if os.path.isdir(path):
        return uManagerAcquisitionFolderZarr(path, xp)
    else:
        return uManagerAcquisitionFile(path, xp)


class uManagerAcquisitionFile(object):
    def __init__(self, path : str, xp : ModuleType=numpy):
        if not os.path.exists(path):
            raise ValueError("specified path doesn't exist")
        # is it a folder or a file?
        self._path = path
        self._xp = xp
        # open the tif file, determine if there are empty frames
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._f = tiff.TiffFile(self._path)
        self._page_calc = partial(_page_calculator, self._f.series[0].shape)
        
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._f.close()

    @property
    def shape(self):
        return self._f.series[0].shape

    def get(self, head, time, channel):
        sp, ep = self._page_calc(head, time, channel)
        return self._xp.stack(
            [self._xp.asarray(self._f.pages[i].asarray())
             for i in range(sp, ep)], axis=0
        )


class uManagerAcquisitionFolderZarr(object):
    def __init__(self, path : str, xp : ModuleType=numpy):
        if not os.path.isdir(path):
            raise ValueError('input path must be a folder')
        tif_paths = sorted(
            [os.path.join(path, f) for f in os.listdir(path)
             if f.endswith('.ome.tif')]
        )
        self._tifs = tif_paths
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._f = tiff.TiffFile(self._tifs[0])
        self._zarr = zarr.open(
            self._f.series[0].aszarr(), mode='r'
        )
        self._xp = xp
        
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._f.close()

    @property
    def shape(self):
        return self._zarr.shape

    def get(self, head, time, channel):
        if head == 'a':
            _ch = list(range(0, 20, 2))[channel]
        else:
            _ch = list(range(1, 20, 2))[channel]
        if len(self._zarr.shape) == 4:
            return self._xp.asarray(self._zarr[_ch,...])
        else:
            assert len(self._zarr.shape) == 5, \
                "expected 5d array, was {:d}".format(len(self._zarr.shape))
            return self._xp.asarray(self._zarr[time,_ch,...])


def subtract_constant_uint16arr(arr : NDArray, const : int) -> NDArray:
    """subtract constant value from uint16 input array
        useful when handling unsigned integer data as it typically is formatted
        by the cameras used in diSPIM imaging

    :param arr: input array
    :type arr: NDArray
    :param const: constant to subtract
    :type const: int
    :returns: input array less the constant value
    :rtype: NDArray
    """
    xp = cupy.get_array_module(arr)
    return (arr.astype(xp.int32) - const).clip(0, 2**16).astype(xp.uint16)
