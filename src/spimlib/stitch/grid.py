import math
import functools
import concurrent.futures
from typing import Callable, Iterable, List, Tuple

import cupy
import numpy
from tqdm.auto import tqdm

from ..reg import pcc, powell
from ..interp.affine import maxblend_into_existing, meanblend_into_existing
from ..interp.affine import output_shape_for_transform
from ..typing import CuLaunchParameters, NDArray, SliceWindow3D, \
    OptBounds, OptBoundMargins


class Grid(object):
    
    def __init__(self,
                 load_fun : Callable[[int,SliceWindow3D], NDArray],
                 tile_shape : Tuple[int,int,int]):
        self._load_fun = load_fun
        self._tile_shape = tile_shape
        self._fwd_slice, self._rev_slice = self.build_slices()
    
    def build_slices(self) -> Tuple[SliceWindow3D,SliceWindow3D]:
        raise NotImplementedError('to be overriden by inheriting class')
        
    def register_pair(self, idx0 : int, idx1 : int,
                      pcc_prereg : bool,
                      metric : str, transform : str, interp_method : str,
                      bounds : OptBounds|OptBoundMargins|None,
                      kernel_launch_params : CuLaunchParameters|None,
                      **opt_kwargs):
        vol0 = self._load_fun(idx0, self._fwd_slice)
        vol1 = self._load_fun(idx1, self._rev_slice)
        _, _, par0 = powell.parse_transform_string(transform)
        if pcc_prereg:
            t0 = pcc.translation_for_volumes(vol0, vol1)
            par0 = list(t0)
            print(par0)
        # do the registration
        T, res = powell.optimize_affine(vol0, vol1, metric, transform,
                                        interp_method, par0, bounds,
                                        kernel_launch_params,
                                        verbose=True,
                                        **opt_kwargs)
        return T, res


class Grid1D(Grid):

    def __init__(self, 
                 load_fun : Callable[[int,SliceWindow3D], NDArray], 
                 tile_shape : Tuple[int,int,int],
                 pct_overlap : float,
                 overlap_axis : int):
        self._pct_overlap = pct_overlap
        assert self._pct_overlap > 0 and self._pct_overlap < 1, \
            "percent overlap must be in range (0,1)"
        self._overlap_axis = overlap_axis
        # build slices by figuring out how much the pixel overlap is
        # and constructing slice objects
        self._dim = int(tile_shape[self._overlap_axis])
        self._len_overlap = int(math.ceil(self._dim * self._pct_overlap))
        self._len_rest = self._dim - self._len_overlap
        super().__init__(load_fun, tile_shape)

    def build_slices(self) -> Tuple[SliceWindow3D,SliceWindow3D]:
        fwd_slice = [slice(None),]*3
        fwd_slice[self._overlap_axis] = slice(0, self._len_overlap)
        rev_slice = [slice(None),]*3
        rev_slice[self._overlap_axis] = slice(self._len_rest, self._dim)
        return fwd_slice, rev_slice

    def accumulate_transforms(self, *mats):
        M = [mats[0]]
        for idx, m_curr in enumerate(mats[1:]):
            if idx == 0:
                m_curr[self._overlap_axis,-1] += self._len_rest
            M.append(m_curr @ M[idx])
        M = numpy.stack(M, axis=0)  # (#tiles x 4 x 4)
        ## shift all translations into global frame
        # accumulate translations
        G = numpy.cumsum(M[:,:,-1], axis=0)[:,:-1]
        # shift into relative frame such that in each axis, all
        # translations are positive with respect to the global coordinate
        # frame of the stitch
        midx = numpy.argmin(G, axis=0)
        rel_shift = [G[midx[i],i] for i in range(3)]
        rel_shift = [numpy.floor(g) if g < 0 else numpy.ceil(g)
                     for g in rel_shift]
        rel_shift = numpy.asarray([int(abs(r)) for r in rel_shift])
        G = G + rel_shift[None,:]
        for row in range(G.shape[0]):
            M[row,:-1,-1] = G[row,:]
        # split back out to per-image matrices
        new_mats = list(map(numpy.squeeze, numpy.split(M, G.shape[0])))
        ## figure out output shape
        # determine largest possible image shape
        out_shp_fun = functools.partial(output_shape_for_transform,
                                        input_shape=self._tile_shape)
        shp_lrg = numpy.amax(numpy.stack(
            list(map(out_shp_fun, new_mats)), axis=0), axis=0
        )
        # add largest possible translation
        out_shp = shp_lrg + numpy.ceil(numpy.amax(numpy.abs(G), axis=0))
        # flip coordinate system in y-direction
        sub_val = out_shp[self._overlap_axis] - \
            self._tile_shape[self._overlap_axis]
        for nm in new_mats:
            nm[self._overlap_axis,-1] -= sub_val
        return new_mats, [int(s) for s in out_shp.astype(int)]

    def register_pairs_serial(self, pair_idxs : Iterable[Tuple[int,int]],
                              pcc_prereg : bool, metric : str, transform : str, 
                              interp_method : str, 
                              bound_margins : OptBounds|OptBoundMargins|None,
                              kernel_launch_params : CuLaunchParameters,
                              verbose : bool = False,
                              **opt_kwargs):
        reg_fun = functools.partial(self.register_pair,
                                    pcc_prereg=pcc_prereg, metric=metric, 
                                    transform=transform, 
                                    interp_method=interp_method,
                                    bound_margins=bound_margins,
                                    kernel_launch_params=kernel_launch_params,
                                    **opt_kwargs)
        if verbose:
            iterator = tqdm(pair_idxs, desc='Registration Pairs')
        else:
            iterator = pair_idxs
        out_list = [(pair_idxs[0][0], numpy.eye(4))]
        for idx0, idx1 in iterator:
            T, res = reg_fun(idx0, idx1)
            if verbose:
                iterator.set_postfix_str(
                    '{:d}->{:d}, met={:.3f}'.format(idx0, idx1, 1 - res.fun)
                )
            out_list.append((idx1, T))
        return out_list

    def register_pairs_parallel(self, pair_idxs : Iterable[Tuple[int,int]],
                                executor : concurrent.futures.Executor, 
                                pcc_prereg : bool, metric : str, 
                                transform : str, interp_method : str, 
                                bound_margins : OptBounds|OptBoundMargins|None,
                                kernel_launch_params : CuLaunchParameters,
                                verbose : bool = False,
                                **opt_kwargs):
        reg_fun = functools.partial(self.register_pair,
                                    pcc_prereg=pcc_prereg, metric=metric, 
                                    transform=transform, 
                                    interp_method=interp_method,
                                    bound_margins=bound_margins,
                                    kernel_launch_params=kernel_launch_params,
                                    **opt_kwargs)
        futures_to_pairs = {
            executor.submit(reg_fun, idx0, idx1) : (idx0, idx1)
            for idx0, idx1 in pair_idxs
        }
        iterator = concurrent.futures.as_completed(futures_to_pairs)
        if verbose:
            iterator = tqdm(iterator, desc='Registration Pairs')
        idx0 = pair_idxs[0][0]
        out_list = [(idx0, numpy.eye(4))]
        for future in iterator:
            idx0, idx1 = futures_to_pairs[future]
            T, res = future.result()
            if verbose:
                iterator.set_postfix_str(
                    '{:d}->{:d}, met={:.3f}'.format(idx0, idx1, 1-res.fun)
                )
            out_list.append((idx1, T))
        out_list.sort(key=lambda x: x[0])
        return out_list
    
    def assemble_stitch(self, idxs : List[int], transforms : List[NDArray], 
                        out_shp : List[int],
                        blend_method : str, interp_method : str,
                        launch_params : CuLaunchParameters):
        # preallocate what is needed for blend method
        if blend_method == 'max':
            out = cupy.zeros(out_shp, dtype=cupy.uint16)
        elif blend_method == 'mean':
            sum = cupy.zeros(out_shp, dtype=cupy.float16)
            cnt = cupy.zeros(out_shp, dtype=cupy.uint8)
        else:
            raise ValueError('invalid blend method')
        for idx, mat in zip(idxs, transforms):
            T = cupy.asarray(mat).astype(cupy.float32)
            vol = self._load_fun(idx, [slice(None),]*3)
            if blend_method == 'max':
                maxblend_into_existing(out, vol, T, interp_method, 
                                       launch_params)
            elif blend_method == 'mean':
                meanblend_into_existing(sum, cnt, vol, T, interp_method,
                                        launch_params)
        if blend_method == 'max':
            return out
        else:
            return  cupy.clip(
                cupy.round(sum/cnt), 0, 2**16-1
            ).astype(cupy.uint16)


class Grid2D(Grid):

    def __init__(self, 
                 load_fun : Callable[[int,SliceWindow3D], NDArray], 
                 tile_shape : Tuple[int,int,int],
                 pct_overlap : Tuple[float,float],
                 overlap_axes : Tuple[int,int]):
        self._pct_overlap = pct_overlap
        assert self._pct_overlap > 0 and self._pct_overlap < 1, \
            "percent overlap must be in range (0,1)"
        self._overlap_axes = overlap_axes
        # build slices by figuring out how much the pixel overlap is
        # and constructing slice objects
        self._dims = [int(t) for t in tile_shape[self._overlap_axes]]
        self._len_overlaps = [int(math.ceil(d * po)) for d, po
                              in zip(self._dims, self._pct_overlap)]
        self._len_rests = [d-o for d, o in zip(self._dims, self._len_overlaps)]
        super().__init__(load_fun, tile_shape)

    def one_dimensionalify(self):
        raise NotImplementedError('todo')