"""utilities for loading data acquired in micro-manager
with the ASI diSPIM plugin
"""
import os
import json
import warnings
from typing import List, Optional, Sequence, Tuple
from types import ModuleType
from collections.abc import Iterable

import zarr
try:
    import cupy
except ImportError as e:
    warnings.warn('cupy not available')
import numpy
import tifffile

from ..typing import NDArray, SliceWindow3D


# NOTE: these lists are hard-coded to accomodate up to 10 channels/head
# if you need more, increase the range upper bound
__a_chans = list(range(0, 20, 2))
__b_chans = list(range(1, 20, 2))
def _page_calculator(shape : Iterable[int], head : str,
                     time : int=0, channel : int=0) -> Tuple[int,int]:
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


def uManagerAcquisition(path : os.PathLike, multi_pos : bool = False,
                        xp : ModuleType=numpy):
    """uManagerAcquisition Create new acquisition object for fetching raw acquisition data.

    Args:
        path (os.PathLike): path to root directory of acquisition
        multi_pos (bool, optional): flag for if this is a multi-position acquisition. Defaults to False.
        xp (ModuleType, optional): array module to load data into (``numpy`` or ``cupy``). Defaults to numpy.
    """
    if multi_pos:
        return uManagerAcquisitionMultiPos(path, xp)
    else:
        return uManagerAcquisitionOnePos(path, xp)


class _uManagerAcquision(object):
    # the acquisition file is structured such that the "channel" dimension
    # has both the actual channel (e.g. GFP) and the view and the listing
    # alternates such that for the first channel, indices [0,1] correspond
    # to view A and B, resp., for that channel.
    _channel_lookup = [(0,1), (2,3), (4,5), (6,7), (8,9), (10, 11), (12, 13)]

    def __init__(self, path : os.PathLike, xp : ModuleType=numpy):
        self._f = None
        self._z = None

        return self
    
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._f.close()

    @property
    def shape(self) -> Sequence[int]:
        return self._z.shape
    
    @property
    def image_shape(self) -> Tuple[int,int,int]:
        return self._z.shape[-3:]


class uManagerAcquisitionOnePos(_uManagerAcquision):
    
    def __init__(self, path : os.PathLike, xp : ModuleType=numpy):
        if not os.path.exists(path):
            raise ValueError("specified path doesn't exist")
        # is it a folder or a file?
        self._xp = xp
        if os.path.isdir(path):
            tif_fnames = sorted([f for f in os.listdir(path)
                                 if f.endswith('.ome.tif')])
            self._path = os.path.join(path, tif_fnames[0])
        else:  # it's a file
            self._path = path
        # open the tif file, determine if there are empty frames
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._f = tifffile.imread(self._path, aszarr=True)
            self._z = zarr.open(self._f, mode='r')

    def _get_single_chan(self, head : str, channel : int, time : int = 0,
                         window : Optional[SliceWindow3D] = None) -> NDArray:
        head_idx = int(head == 'b')
        chan = self._channel_lookup[channel][head_idx]
        if len(self._z.shape) == 4:
            if window is None:
                return self._xp.asarray(self._z[chan,...])
            else:
                window = tuple([slice(chan,chan+1)] + list(window))
                return self._xp.asarray(self._z[window]).squeeze()
        else:
            if window is None:
                return self._xp.asarray(self._z[time,chan,...])
            else:
                window = tuple(
                    [slice(time,time+1), slice(chan,chan+1)] + list(window)
                )
                return self._xp.asarray(self._z[window]).squeeze()

    def get(self, head : str, channels : Iterable[int]|int, time : int = 0, 
            window : Optional[SliceWindow3D]=None) -> NDArray:
        if isinstance(channels, int):
            return self._get_single_chan(head, channels, time, window)
        else:
            return self._xp.stack(
                [self._get_single_chan(head, c, time, window) 
                 for c in channels], axis=0
            )


class uManagerAcquisitionMultiPos(_uManagerAcquision):

    def __init__(self, path : os.PathLike, xp : ModuleType = numpy):
        if not os.path.exists(path):
            raise ValueError('specified path does not exist')
        assert os.path.isdir(path), "multiposition datasets must be folders"
        self._path = path
        self._xp = xp
        # grab the folder name -- will need this to parse file names
        _, folder_name = os.path.split(path)
        self._folder_name = folder_name
        # load the Pos0 file and keep it in a zarr format so that it doesnt
        # go into memory
        # NOTE: these are *.ome.tif files but the OME metadata doesnt get
        # parsed correctly by `tifffile` and conflicts with how the library
        # thinks it should be handled according to the micro-manager metadata
        # by loading this way, we get a [#pos,2*#chans,Z,R,C] shaped
        # array that can be indexed directly (indexing will load into memory)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._f = tifffile.imread(
                os.path.join(self._path, 
                             self._folder_name+'_MMStack_Pos0.ome.tif'),
                aszarr=True, is_ome=False, is_mmstack=True, is_imagej=False
            )
            self._z = zarr.open(self._f, mode='r')

    @property
    def num_positions(self):
        if len(self._z.shape) > 4:
            return self._z.shape[0]
        else:
            return 1

    def _get_single_chan(self, position : int, head : str, 
                         channel : int, time : int = 0, 
                         window : Optional[SliceWindow3D] = None) -> NDArray:
        head_idx = int(head == 'b')
        chan = self._channel_lookup[channel][head_idx]
        if len(self._z.shape) == 5:  # most common
            if window is None:
                return self._xp.asarray(self._z[position,chan,...])
            else:
                window = tuple(
                    [slice(position,position+1), slice(chan,chan+1)] + 
                    list(window)
                )
                return self._xp.asarray(self._z[window]).squeeze()
        elif len(self._z.shape) == 4:
            if window is None:
                return self._xp.asarray(self._z[chan,...])
            else:
                window = tuple([slice(chan,chan+1)] + list(window))
                return self._xp.asarray(self._z[window]).squeeze()
        else:
            raise NotImplementedError('multipos/multi-time is TODO')

    def get(self, position : int, head : str, channels : Iterable[int]|int, 
            time : int=0, window : Optional[SliceWindow3D]=None) -> NDArray:
        if isinstance(channels, int):
            return self._get_single_chan(position, head, channels, time, window)
        else:
            return self._xp.stack(
                [self._get_single_chan(position, head, c, time, window)
                 for c in channels], axis=0
            )


class PositionList(object):
    
    def __init__(self, path : os.PathLike, um_per_pix : float = 0.1625):
        self._arr = parse_position_list(path)
        self._um_per_pix = um_per_pix
    
    def neighbors(self, idx : int) -> numpy.ndarray:
        neig_idx = self._grid_neighbors(idx)
        shifts = []
        for nidx in neig_idx:
            shift = self._arr[idx,-2:] - self._arr[nidx,-2:]
            shifts.append(shift)
        shifts = _round_from_zero(
            numpy.vstack(shifts) / self._um_per_pix
        ).astype(int)
        return numpy.hstack([numpy.asarray(neig_idx)[:,None],
                             shifts]).astype(int)
    
    def _grid_neighbors(self, idx : int) -> List[int]:
        gz, gy = self._arr[idx,1], self._arr[idx,2]
        neig = numpy.where(
            numpy.logical_and(numpy.abs(self._arr[:,1]-gz)<=1,
                              numpy.abs(self._arr[:,2]-gy)<=1)
        )[0]
        return neig[numpy.where(neig!=idx)]

    @property
    def grid_size(self) -> Tuple[int,int]:
        nz = len(numpy.unique(self._arr[:,1]))
        ny = len(numpy.unique(self._arr[:,2]))
        return nz, ny
    
    @property
    def num_pos(self):
        return self._arr.shape[0]
    
    def overlap_props(self):
        rnd = numpy.ceil(numpy.abs(
            numpy.mean(numpy.diff(self._arr[:,-3:], axis=0), 0)
        ) / self._um_per_pix)
        overlap_axes = list(map(int, numpy.where(rnd>0)[0]))
        overlap_pix = [int(rnd[o]) for o in overlap_axes]
        return overlap_axes, overlap_pix
    
    def yaxis_stitch_pairs(self, z : int=0):
        filt = self._arr[:,1] == z
        ys = self._arr[filt,2]
        srt = numpy.argsort(ys)
        idxs = self._arr[filt,0].astype(int)[srt]
        idxs = [int(i) for i in idxs]
        return list(zip(idxs, idxs[1:]))

    def zaxis_stitch_pairs(self, y : int=0):
        filt = self._arr[:,2] == y
        zs = self._arr[filt,1]
        srt = numpy.argsort(zs)
        idxs = self._arr[filt,0].astype(int)[srt]
        idxs = [int (i) for i in idxs]
        return list(zip(idxs, idxs[1:]))


class StitchedDataset(object):
    def __init__(self, path : os.PathLike, xp : ModuleType = numpy):
        if not os.path.exists(path):
            raise ValueError('specified path does not exist')
        assert os.path.isdir(path), 'dataset path must be a folder'
        self._path = path
        self._xp = xp

    @staticmethod
    def make_fname(head : str, channel : int, position : int|None):
        if position is None:
            return "{:s}_c{:02d}.tif".format(head, channel)
        else:
            return "{:s}_p{:}_c{:02d}.tif".format(head, position, channel)
        
    def get(self, position : int|None, head : str, channel : int, 
            window : Optional[SliceWindow3D] = None) -> NDArray:
        fname = StitchedDataset.make_fname(head, channel, position)
        fpath = os.path.join(self._path, fname)
        if window is None:
            return self._xp.asarray(tifffile.imread(fpath))
        else:
            return self._xp.asarray(tifffile.imread(fpath)[window])


def _round_from_zero(x):
    return numpy.sign(x) * numpy.ceil(numpy.abs(x))


def parse_position_list(path : os.PathLike) -> numpy.ndarray:
    """parse_position_list Parse uManager position list into numpy array

    Args:
        path (os.PathLike): path to the PositionList.pos file

    Returns:
        numpy.ndarray: 6-column array [position_index, grid_z, grid_y, z, y, x]
    """
    with open(path, 'r') as f:
        pos_dict = json.load(f)
    positions = pos_dict['POSITIONS']
    rows = []
    for idx, position in enumerate(positions):
        gz, gy = __label_to_grid_location(position['LABEL'])
        x, y, z = __physical_position_from_devices(position['DEVICES'])
        rows.append([idx, gz, gy, z, y, x])
    return numpy.vstack(rows)


def __physical_position_from_devices(devices : List) -> \
    Tuple[float,float,float]:
    x, y, z = 0., 0., 0.
    for device in devices:
        if device['AXES'] == 2:
            x, y = device['X'], device['Y']
        elif device['AXES'] == 1:
            z = device['X']
        else:
            raise Exception('found device without 1 or 2 axes')
    return x, y, z


def __label_to_grid_location(label : str) -> Tuple[int, int]:
    rc = [int(x) for x in label.split('_')[1:]]
    return rc[0], rc[1]


def subtract_constant_uint16arr(arr : NDArray, const : int) -> NDArray:
    """subtract_constant_uint16arr Subtract constant value from uint16 array, preventing underflow.

    Args:
        arr (NDArray): input array
        const (int): value to subtract

    Returns:
        NDArray
    """
    xp = cupy.get_array_module(arr)
    return (arr.astype(xp.int32) - const).clip(0, 2**16).astype(xp.uint16)