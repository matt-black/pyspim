import os
import json
import math
import concurrent.futures
from numbers import Number
from functools import partial
from contextlib import nullcontext
from typing import Callable, Iterable, List, Tuple

import zarr
import cupy
import numpy
import tifffile
from tqdm.auto import tqdm

from ..reg import pcc, powell
from ..interp.affine import maxblend_into_existing, meanblend_into_existing
from ..interp.affine import output_shape_for_transform
from ..typing import CuLaunchParameters, NDArray, SliceWindow3D, \
    OptBounds, OptBoundMargins


LoadFunction = Callable[[SliceWindow3D,int], NDArray]


def register_pair(vol0 : NDArray, vol1 : NDArray,
                  pcc_prereg : bool, piecewise : bool,
                  metric : str, transform : str, interp_method : str,
                  bounds : OptBounds|OptBoundMargins|None,
                  kernel_launch_params : CuLaunchParameters|None,
                  verbose : bool,
                  **opt_kwargs):
    _, _, par0 = powell.parse_transform_string(transform)
    if pcc_prereg:
        t0 = pcc.translation_for_volumes(vol0, vol1)
        par0 = list(t0)
    # do the registration
    if piecewise:
        reg_fun = powell.optimize_affine_piecewise
    else:
        reg_fun = powell.optimize_affine
    return reg_fun(vol0, vol1, metric, transform, interp_method, 
                   par0, bounds, kernel_launch_params, verbose=verbose,
                   **opt_kwargs)


def register_pair_load(idx0 : int, idx1 : int, 
                       loadfun0 : Callable[[int], NDArray],
                       loadfun1 : Callable[[int], NDArray],
                       pcc_prereg : bool, piecewise : bool,
                       metric : str, transform : str, interp_method : str,
                       bounds : OptBounds|OptBoundMargins|None,
                       kernel_launch_params : CuLaunchParameters|None,
                       verbose : bool, **opt_kwargs):
    vol0, vol1 = loadfun0(idx0), loadfun1(idx1)
    return register_pair(vol0, vol1, pcc_prereg, piecewise,
                         metric, transform, interp_method,
                         bounds, kernel_launch_params, verbose,
                         **opt_kwargs)


def register_pair_load_gpuq(idx0 : int, idx1: int, 
                            queue,  # : multiprocessing.Manager().Queue
                            loadfun0 : Callable[[int], NDArray],
                            loadfun1 : Callable[[int], NDArray],
                            pcc_prereg : bool, piecewise : bool,
                            metric : str, transform : str, 
                            interp_method : str,
                            bounds : OptBounds|OptBoundMargins|None,
                            kernel_launch_params : CuLaunchParameters|None,
                            verbose : bool, **opt_kwargs):
    gpu_id = queue.get()
    with cupy.cuda.Device(gpu_id):
        vol0, vol1 = loadfun0(idx0), loadfun1(idx1)
        T, opt_res = register_pair(vol0, vol1, pcc_prereg, piecewise,
                                   metric, transform, interp_method,
                                   bounds, kernel_launch_params, verbose,
                                   **opt_kwargs)
        queue.put(gpu_id)
        return T, opt_res


def build_slices(tile_shape : Tuple[int,int,int],
                 overlap_axes : int|Iterable[int],
                 len_overlaps : int|Iterable[int]) -> \
                    Tuple[SliceWindow3D,SliceWindow3D]:
    fwd_slice, rev_slice = [slice(None),]*3, [slice(None),]*3
    len_rests = []
    if isinstance(overlap_axes, Number):
        overlap_axes = [overlap_axes]
        len_overlaps = [len_overlaps]
    len_rests = [tile_shape[o]-ol for o, ol in zip(overlap_axes, len_overlaps)]
    for oa, lo, lr in zip(overlap_axes, len_overlaps, len_rests):
        fwd_slice[oa] = slice(0, lo)
        rev_slice[oa] = slice(lr, tile_shape[oa])
    return fwd_slice, rev_slice
    

def assemble_stitch(load_fun : LoadFunction, out_shp : List[int],
                    idxs : List[int], transforms : List[NDArray], 
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
        vol = load_fun([slice(None),]*3, idx)
        if blend_method == 'max':
            maxblend_into_existing(out, vol, T, interp_method, 
                                    launch_params)
        elif blend_method == 'mean':
            meanblend_into_existing(sum, cnt, vol, T, interp_method,
                                    launch_params)
    if blend_method == 'max':
        return out
    else:
        return cupy.clip(cupy.round(sum/cnt), 0, 2**16-1).astype(cupy.uint16)


def assemble_stitch_and_save(out_path : os.PathLike, 
                             load_fun : LoadFunction, out_shp : List[int],
                             idxs : List[int], transforms : List[NDArray], 
                             blend_method : str, interp_method : str,
                             launch_params : CuLaunchParameters):
    stitch = assemble_stitch(
        load_fun, out_shp, idxs, transforms, 
        blend_method, interp_method, launch_params
    ).get().astype(numpy.uint16)
    if out_path[-4:] == 'zarr':
        zarr.save_array(out_path, stitch)
    elif out_path[-3:] == 'tif' or out_path[-4:] == 'tiff':
        tifffile.imwrite(out_path, stitch, 
                         bigtiff=True,
                         imagej=True,
                         metadata={'axes' : 'ZYX'})
    else:
        raise ValueError('invalid output extension')


def assemble_stitch_and_save_gpuq(out_path : os.PathLike, queue, 
                                  load_fun : LoadFunction, out_shp : List[int],
                                  idxs : List[int], transforms : List[NDArray], 
                                  blend_method : str, interp_method : str,
                                  launch_params : CuLaunchParameters):
    gpu_id = queue.get()
    with cupy.cuda.Device(gpu_id):
        stitch = assemble_stitch(
            load_fun, out_shp, idxs, transforms, 
            blend_method, interp_method, launch_params
        ).get().astype(numpy.uint16)
    if out_path[-4:] == 'zarr':
        zarr.save_array(out_path, stitch)
    elif out_path[-3:] == 'tif' or out_path[-4:] == 'tiff':
        tifffile.imwrite(out_path, stitch, 
                         bigtiff=True,
                         imagej=True,
                         metadata={'axes' : 'ZYX'})
    else:
        raise ValueError('invalid output extension')
    queue.put(gpu_id)


def accumulate_transforms_1d(tile_shape : Tuple[int,int,int], 
                             overlap_axis : int, len_rest : int, 
                             *mats):
    M = [mats[0]]
    for idx, m_curr in enumerate(mats[1:]):
        if idx == 0:
            m_curr[overlap_axis,-1] += len_rest
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
    out_shp_fun = partial(output_shape_for_transform,
                                    input_shape=tile_shape)
    shp_lrg = numpy.amax(numpy.stack(
        list(map(out_shp_fun, new_mats)), axis=0), axis=0
    )
    # add largest possible translation
    out_shp = shp_lrg + numpy.ceil(numpy.amax(numpy.abs(G), axis=0))
    # flip coordinate system in direction of overlap
    sub_val = out_shp[overlap_axis] - \
        tile_shape[overlap_axis]
    for nm in new_mats:
        nm[overlap_axis,-1] -= sub_val
    return new_mats, [int(s) for s in out_shp.astype(int)]


def order_pair_indices(pair_idxs : List[Tuple[int,int]]):
    firsts = set([t[0] for t in pair_idxs])
    secnds = set([t[1] for t in pair_idxs])
    first_idx = next(iter(firsts - secnds))
    last_idx  = next(iter(secnds - firsts))
    ordered_idxs = [first_idx]
    curr_idx = ordered_idxs[0]
    while curr_idx != last_idx:
        curr_idx = [t[1] for t in pair_idxs if t[0] == curr_idx][0]
        ordered_idxs.append(curr_idx)
    return ordered_idxs


class Grid2D(object):

    def __init__(self, 
                 load_fun : LoadFunction, 
                 tile_shape : Tuple[int,int,int],
                 pct_overlap : Tuple[float,float],
                 overlap_axes : Tuple[int,int]):
        self._load_fun = load_fun
        self._tile_shape = tile_shape
        self._pct_overlap = pct_overlap
        assert self._pct_overlap > 0 and self._pct_overlap < 1, \
            "percent overlap must be in range (0,1)"
        self._overlap_axes = overlap_axes
        self._dims = self._tile_shape[overlap_axes]
        self._len_overlaps = [int(math.ceil(d * po)) for d, po
                              in zip(self._dims, self._pct_overlap)]
        self._len_rests = [t-o for t, o in zip(self._dims, self._len_overlaps)]

    def one_dimensionalify(self, out_fldr : os.PathLike, reduction_index : int,
                           pair_idx_groups : Iterable[Tuple[int,int]],
                           executor : concurrent.futures.Executor|None,
                           as_completed : Callable|None,
                           pcc_prereg : bool, piecewise : bool,
                           metric : str, transform : str, interp_method : str,
                           bounds : OptBounds|OptBoundMargins|None,
                           launch_par_opt : CuLaunchParameters,
                           launch_par_bld : CuLaunchParameters,
                           blend_method : str,
                           verbose : bool = False,
                           **opt_kwargs):
        assert reduction_index == 0 or reduction_index == 1, \
            "reduction index must be 0 or 1"
        slice_f, slice_r = build_slices(self._tile_shape, 
                                        self._overlap_axes[reduction_index],
                                        self._len_overlaps[reduction_index])
        load_f = lambda i : self._load_fun(i, slice_f)
        load_r = lambda i : self._load_fun(i, slice_r)
        ## pair registration: take in all adjacent pairs (across all groups)
        ## and register them to one another
        new_grp_idxs = []
        for grp_idx, pair_group in enumerate(pair_idx_groups):
            # annotate all of the pairs with what group they're in
            new_group = [(grp_idx, pair[0], pair[1]) for pair in pair_group]
            new_grp_idxs.append(new_group)
        reg_fun = partial(register_pair,
                                    pcc_prereg=pcc_prereg, piecewise=piecewise,
                                    metric=metric, transform=transform,
                                    interp_method=interp_method,
                                    bounds=bounds,
                                    kernel_launch_params=launch_par_opt,
                                    verbose=verbose,
                                    **opt_kwargs)
        # initialize each group with a list containing idx -> transform
        # maps where idx0 is mapped to the identity transform
        out_lists = [[] for _ in pair_idx_groups]
        for gidx, grp in enumerate(new_grp_idxs):
            idx0 = grp[0][1]
            out_lists[gidx].append((idx0, numpy.eye(4)))
        # do the registration across all pairs
        # NOTE: this is done 
        if executor is None:  # serial
            if verbose:
                n_tot = sum([len(g) for g in new_grp_idxs])
                pbar = tqdm(total=n_tot, desc="Stitching Pairs")
            else:
                pbar = nullcontext
            with pbar:
                for grp in new_grp_idxs:
                    for grpid, idx0, idx1 in grp:
                        T, res = reg_fun(load_f(idx0), load_r(idx1))
                        out_lists[grpid].append(idx1, T)
                        if verbose:
                            pbar.update(1)
                            pbar.set_postfix_str(
                                'G{:d},{:d}->{:d}::met={:.3f}'.format(
                                    grpid, idx0, idx1, 1.0-res.fun
                                )
                            )
        else:  # parallel
            futures_to_pairs = {}
            # construct a bunch of futures
            for grp in new_grp_idxs:
                for grpid, idx0, idx1 in grp:
                    future = executor.submit(
                        reg_fun, self._load_fun(idx0, slice_f), 
                        self._load_fun(idx1, slice_r)
                    )
                    futures_to_pairs[future] = (grpid, idx0, idx1)
            if verbose:
                n_tot = sum([len(g) for g in new_grp_idxs])
                pbar = tqdm(total=n_tot, desc="Stitching Pairs")
            else:
                pbar = nullcontext
            with pbar:
                for future in as_completed(futures_to_pairs):
                    grpid, idx0, idx1 = futures_to_pairs[future]
                    T, res = future.result()
                    out_lists[grpid].append(idx1, T)
                    if verbose:
                        pbar.update(1)
                        pbar.set_postfix_str(
                            'G{:d},{:d}->{:d}::met={:.3f}'.format(
                                grpid, idx0, idx1, 1.0-res.fun
                            )
                        )
        # need to sure that out_lists have the right "order" so that when
        # we accumulate transforms, we do so correctly
        for grpid in range(len(pair_idx_groups)):
            _pidx_group = pair_idx_groups[grpid]
            idxo = order_pair_indices(_pidx_group)
            idxoe = list(enumerate(idxo))
            out_lists[grpid].sort(
                key=lambda t: [s for s,i in idxoe if i==t[0]][0]
            )
        # now do the accumulation and then stitch together the volumes
        load_all = lambda i : self._load_fun(i, [slice(None),]*3)
        if not os.path.exists(out_fldr):
            os.makedirs(out_fldr, exist_ok=True)
        futures = []
        for grpid, out_list in enumerate(out_lists):
            idxs = [id for id, _ in out_list]
            new_mats, out_shp = accumulate_transforms_1d(
                self._tile_shape, self._overlap_axes[reduction_index],
                self._len_rests, *[m for _, m in out_list]
            )
            out_path = os.path.join(out_fldr, 'tile{:03d}.zarr')
            if executor is None:
                assemble_stitch_and_save(out_path, load_all, out_shp, idxs, 
                                          new_mats, blend_method, interp_method, 
                                          launch_par_bld)
            else:
                futures.append(
                    executor.submit(assemble_stitch_and_save, 
                                    out_path, load_all, out_shp, idxs, 
                                    new_mats, blend_method, interp_method, 
                                    launch_par_bld)
                )
        if len(futures):
            for _ in as_completed(futures):
                pass
        # write out JSON of properties we'll need to stitch this 
        # dimension together in the future
        new_index = 1 if reduction_index == 0 else 0
        out_overlap = self._overlap_axes[new_index]
        out_pct_ov  = self._pct_overlap[new_index]
        with open(os.path.join(out_fldr, 'props.json'), 'w') as f:
            json.dump({'overlap_axis' : out_overlap, 
                       'pct_overlap'  : out_pct_ov}, f)
        return out_fldr