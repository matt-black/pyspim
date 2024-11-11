import os
import functools
import concurrent.futures
from typing import Callable, Iterable, Tuple

import cupy
import numpy
from tqdm.auto import tqdm

from .grid import Grid1D, Grid2D
from ..typing import CuLaunchParameters, SliceWindow3D, \
    OptBounds, OptBoundMargins
from ..data.dispim import uManagerAcquisitionMultiPos
from ..data.dispim import PositionList, subtract_constant_uint16arr


def _load_function(data_path : os.PathLike, head : str, channel : int,
                   position : int, window : SliceWindow3D):
    with uManagerAcquisitionMultiPos(data_path, xp=cupy) as acq:
        im = subtract_constant_uint16arr(
            acq.get(position, head, channel, 0, window), 100
        )
        return im


class OneDimensionalStitcher(Grid1D):
    def __init__(self, 
                 root_fldr : os.PathLike, head : str, channel : int,
                 position_list : os.PathLike,
                 pct_overlap : float,
                 overlap_axis : int):
        self._root_fldr = root_fldr
        self._pos_list = PositionList(position_list)
        idxs = list(range(self._pos_list.num_pos))
        self._pair_idxs = [(i, j) for i, j in zip(idxs[:-1], idxs[1:])]
        with uManagerAcquisitionMultiPos(self._root_fldr) as f:
            tile_shape = f.image_shape
        load_fun = functools.partial(_load_function, root_fldr, head, channel)
        super().__init__(load_fun, tile_shape, pct_overlap, overlap_axis)


class DiSPIM1DStitcher(object):

    def __init__(self,
                 root_fldr : os.PathLike, channel : int,
                 position_list : os.PathLike,
                 pct_overlap : float,
                 overlap_axis : int):
        self._grid_a = OneDimensionalStitcher(
            root_fldr, 'a', channel, position_list, pct_overlap, overlap_axis
        )
        self._grid_b = OneDimensionalStitcher(
            root_fldr, 'b', channel, position_list, pct_overlap, overlap_axis
        )
        
    def register_pairs_parallel(self, pair_idxs : Iterable[Tuple[int,int]],
                                executor : concurrent.futures.Executor,
                                as_completed : Callable,
                                pcc_prereg : bool, piecewise : bool,
                                metric : str, transform : str, 
                                interp_method : str, 
                                bounds : OptBounds|OptBoundMargins|None,
                                kernel_launch_params : CuLaunchParameters,
                                verbose : bool = False,
                                **opt_kwargs):
        reg_a = functools.partial(self._grid_a.register_pair,
                                  pcc_prereg=pcc_prereg, piecewise=piecewise,
                                  metric=metric, transform=transform,
                                  interp_method=interp_method,
                                  bound_margins=bounds,
                                  kernel_launch_params=kernel_launch_params,
                                  **opt_kwargs)
        reg_b = functools.partial(self._grid_a.register_pair,
                                  pcc_prereg=pcc_prereg, piecewise=piecewise,
                                  metric=metric, transform=transform,
                                  interp_method=interp_method,
                                  bound_margins=bounds,
                                  kernel_launch_params=kernel_launch_params,
                                  **opt_kwargs)
        futures_to_pairs_a = {
            executor.submit(reg_a, idx0, idx1) : ('a', idx0, idx1)
            for idx0, idx1 in pair_idxs
        }
        futures_to_pairs_b = {
            executor.submit(reg_b, idx0, idx1) : ('b', idx0, idx1)
            for idx0, idx1 in pair_idxs
        }
        futures_to_pairs = {**futures_to_pairs_a, **futures_to_pairs_b}
        iterator = as_completed(futures_to_pairs)
        if verbose:
            iterator = tqdm(iterator, desc="Registration Pairs")
        out_list = [('a', pair_idxs[0][0], numpy.eye(4)), 
                    ('b', pair_idxs[0][0], numpy.eye(4))]
        for future in iterator:
            head, idx0, idx1 = futures_to_pairs[future]
            T, res = future.result()
            if verbose:
                iterator.set_postfix_str(
                    '{:s} {:d}->{:d}, met={:.3f}'.format(head, idx0, idx1,
                                                         1-res.fun)
                )
            out_list.append((head, idx1, T))
        out_a = [t for t in out_list if t[0] == 'a']
        out_b = [t for t in out_list if t[0] == 'b']
        return out_a, out_b
    

class TwoDimensionalStitcher(Grid2D):
    def __init__(self,
                 root_fldr : os.PathLike, head : str, channel : int,
                 position_list : os.PathLike,
                 pct_overlap : Tuple[float,float],
                 overlap_axes : Tuple[int,int]):
        self._root_fldr = root_fldr
        self._pos_list = PositionList(position_list)
        with uManagerAcquisitionMultiPos(self._root_fldr) as f:
            tile_shape = f.image_shape
        load_fun = functools.partial(_load_function, root_fldr, head, channel)
        super().__init__(load_fun, tile_shape, pct_overlap, overlap_axes)