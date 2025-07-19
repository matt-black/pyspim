import concurrent.futures
import json
import math
import multiprocessing
import os
from contextlib import nullcontext
from functools import partial
from typing import List, Tuple
from warnings import warn

import cupy
import numpy
from tqdm.auto import tqdm

from .._util import NumpyArrayEncoder, launch_params_for_volume
from ..data.dispim import (
    PositionList,
    StitchedDataset,
    subtract_constant_uint16arr,
    uManagerAcquisitionMultiPos,
)
from ..typing import OptBoundMargins, OptBounds, SliceWindow3D
from .grid import (
    accumulate_transforms_1d,
    assemble_stitch_and_save_gpuq,
    build_slices,
    order_pair_indices,
    register_pair_load_gpuq,
)


def _load_function(
    data_path: os.PathLike,
    head: str,
    channel: int,
    window: SliceWindow3D,
    position: int,
):
    with uManagerAcquisitionMultiPos(data_path, xp=cupy) as acq:
        im = subtract_constant_uint16arr(
            acq.get(position, head, channel, 0, window), 100
        )
        return im


def _make_2dto1d_fname(head: str, position: int, channel: int):
    return StitchedDataset.make_fname(head, channel, position)


def _load_function_2dto1d(
    fldr_path: os.PathLike,
    head: str,
    channel: int,
    window: SliceWindow3D,
    position: int,
):
    dset = StitchedDataset(fldr_path, xp=cupy)
    return subtract_constant_uint16arr(dset.get(position, head, channel, window), 100)


class DiSPIM1DStitcher:
    def __init__(
        self,
        root_fldr: os.PathLike,
        position_list: os.PathLike | None,
        overlap_axis: int | None,
        pct_overlap: float | None,
        reduced_from_2d: bool = False,
    ):
        self._root_fldr = root_fldr
        with uManagerAcquisitionMultiPos(self._root_fldr) as f:
            tile_shape = f.image_shape
            self.tile_shape = list(tile_shape)
        if position_list is None:
            self.position_list = None
            assert pct_overlap is not None and overlap_axis is not None, (
                "position list not given, must supply % overlap & axis"
            )
            self.overlap_axis = overlap_axis
            self.pct_overlap = pct_overlap
            self.len_overlap = math.ceil(self.tile_shape[overlap_axis] * pct_overlap)
            self.len_rest = self.tile_shape[overlap_axis] - self.len_overlap
        else:
            pos_list = PositionList(position_list)
            self.position_list = pos_list
            overlap_axes, overlap_pix = pos_list.overlap_props()
            overlap_axis = overlap_axes[0]
            overlap_pix = overlap_pix[0]
            pct_overlap = overlap_pix / tile_shape[overlap_axis]
            self.overlap_axis = overlap_axis
            self.pct_overlap = pct_overlap
            self.len_overlap = overlap_pix
            self.len_rest = self.tile_shape[overlap_axis] - overlap_pix
        self._reduced_from_2d = reduced_from_2d

    def determine_pair_idxs(self) -> List[Tuple[int, int]]:
        if self._reduced_from_2d:
            with open(os.path.join(self._root_fldr, "params.json")) as f:
                num_pos = json.load(f)["num_pos"]
                return list(zip(range(num_pos), range(1, num_pos)))
        else:
            if self.position_list is None:
                with uManagerAcquisitionMultiPos(self._root_fldr) as acq:
                    pair_idxs = list(
                        zip(range(acq.num_positions), range(1, acq.num_positions))
                    )
            else:
                if self.overlap_axis == 0:
                    pair_idxs = self.position_list.zaxis_stitch_pairs(0)
                elif self.overlap_axis == 1:
                    pair_idxs = self.position_list.yaxis_stitch_pairs(0)
                else:
                    raise ValueError("invalid overlap axis found")
                return pair_idxs

    def register_pairs(
        self,
        channel: int,
        pcc_prereg: bool,
        piecewise: bool,
        metric: str,
        transform: str,
        interp_method: str,
        bounds: OptBounds | OptBoundMargins | None,
        block_sizes: Tuple[int, int, int],
        verbose: bool = False,
        **opt_kwargs,
    ):
        if transform != "t":
            warn("accumulation only works for pure translation")
        if transform == "t+r" or transform == "t+r+s":
            warn("auto-changing r -> r0")
            transform = transform.replace("r", "r0")
        if self._reduced_from_2d:
            load_fun_base = _load_function_2dto1d
        else:
            load_fun_base = _load_function
        # calculate pair idxs and how we'll slice the volumes for registration
        pair_idxs = self.determine_pair_idxs()
        slice_f, slice_r = build_slices(
            self.tile_shape, self.overlap_axis, self.len_overlap
        )
        # determine launch parameters
        shp = self.tile_shape
        shp[self.overlap_axis] = self.len_overlap
        klp = launch_params_for_volume(shp, *block_sizes)
        # make partial functions so all shared params are passed in already
        load_af = partial(load_fun_base, self._root_fldr, "a", channel, slice_f)
        load_bf = partial(load_fun_base, self._root_fldr, "b", channel, slice_f)
        load_ar = partial(load_fun_base, self._root_fldr, "a", channel, slice_r)
        load_br = partial(load_fun_base, self._root_fldr, "b", channel, slice_r)
        out_list = [
            ("a", pair_idxs[0][0], numpy.eye(4)),
            ("b", pair_idxs[0][0], numpy.eye(4)),
        ]
        n_gpu = cupy.cuda.runtime.getDeviceCount()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_gpu, mp_context=multiprocessing.get_context("spawn")
        ) as executor, multiprocessing.Manager() as manager:
            # setup shared queue for maintaining track of which GPU(s)
            # are available for computation
            queue = manager.Queue()
            for gpu_id in range(n_gpu):
                queue.put(gpu_id)
            # load all the registrations for view "A" into the executor
            reg_a = partial(
                register_pair_load_gpuq,
                queue=queue,
                loadfun0=load_af,
                loadfun1=load_ar,
                pcc_prereg=pcc_prereg,
                piecewise=piecewise,
                metric=metric,
                transform=transform,
                interp_method=interp_method,
                bounds=bounds,
                kernel_launch_params=klp,
                verbose=verbose,
                **opt_kwargs,
            )
            futures_to_pairs_a = {
                executor.submit(reg_a, idx0, idx1): ("a", idx0, idx1)
                for idx0, idx1 in pair_idxs
            }
            # similar for view "B"
            reg_b = partial(
                register_pair_load_gpuq,
                queue=queue,
                loadfun0=load_bf,
                loadfun1=load_br,
                pcc_prereg=pcc_prereg,
                piecewise=piecewise,
                metric=metric,
                transform=transform,
                interp_method=interp_method,
                bounds=bounds,
                kernel_launch_params=klp,
                verbose=verbose,
                **opt_kwargs,
            )
            futures_to_pairs_b = {
                executor.submit(reg_b, idx0, idx1): ("b", idx0, idx1)
                for idx0, idx1 in pair_idxs
            }
            # construct dictionary mapping all futures to the relevant
            # head and the index pairs being registered
            futures_to_pairs = {**futures_to_pairs_a, **futures_to_pairs_b}
            # iterator returns futures as they are done, so we can keep track
            iterator = concurrent.futures.as_completed(futures_to_pairs)
            if verbose:
                n_task = len(futures_to_pairs)
                iterator = tqdm(iterator, total=n_task, desc="Registration Pairs")
            for future in iterator:
                head, idx0, idx1 = futures_to_pairs[future]
                T, res = future.result()
                if verbose:
                    string = f"{head:s} {idx0:d}->{idx1:d}, met={1 - res.fun:.3f}"
                    iterator.set_postfix_str(string)
                    print(string, flush=True)
                out_list.append((head, idx1, T))
        # split outputs to each head
        out_a = [(t[1], t[2]) for t in out_list if t[0] == "a"]
        out_b = [(t[1], t[2]) for t in out_list if t[0] == "b"]
        # sort the output lists
        ord = order_pair_indices(pair_idxs)
        idxoe = list(enumerate(ord))
        out_a.sort(key=lambda t: [s for s, i in idxoe if i == t[0]][0])
        out_b.sort(key=lambda t: [s for s, i in idxoe if i == t[0]][0])
        # do the accumulation
        idx_a, mat_a = [id for id, _ in out_a], [mat for _, mat in out_a]
        idx_b, mat_b = [id for id, _ in out_b], [mat for _, mat in out_b]
        M_a, out_shp_a = accumulate_transforms_1d(
            self.tile_shape, self.overlap_axis, self.len_rest, *mat_a
        )
        new_a = list(zip(idx_a, M_a))
        M_b, out_shp_b = accumulate_transforms_1d(
            self.tile_shape, self.overlap_axis, self.len_rest, *mat_b
        )
        new_b = list(zip(idx_b, M_b))
        return (new_a, out_shp_a), (new_b, out_shp_b)

    def save_registration(
        self,
        out_fldr: os.PathLike,
        reg_tforms_a: List[Tuple[int, numpy.ndarray]],
        out_shp_a: List[int],
        reg_tforms_b: List[Tuple[int, numpy.ndarray]],
        out_shp_b: List[int],
    ):
        path_a = os.path.join(out_fldr, "reg_tforms_a.json")
        self._save_registration(path_a, reg_tforms_a, out_shp_a)
        path_b = os.path.join(out_fldr, "reg_tforms_b.json")
        self._save_registration(path_b, reg_tforms_b, out_shp_b)

    def load_registration(self, out_fldr: os.PathLike):
        path_a = os.path.join(out_fldr, "reg_tforms_a.json")
        tlst_a, out_shp_a = self._load_registration(path_a)
        path_b = os.path.join(out_fldr, "reg_tforms_b.json")
        tlst_b, out_shp_b = self._load_registration(path_b)
        return (tlst_a, out_shp_a), (tlst_b, out_shp_b)

    def _load_registration(self, fpath: os.PathLike):
        with open(fpath) as f:
            d = json.load(f)
        out_shp = d["out_shp"]
        reg_tforms = [
            (int(idx), numpy.asarray(tform))
            for idx, tform in d.items()
            if idx != "out_shp"
        ]
        return reg_tforms, out_shp

    def _save_registration(
        self,
        out_path: os.PathLike,
        reg_tforms: List[Tuple[int, numpy.ndarray]],
        out_shp: List[int],
    ):
        reg_dict = dict(reg_tforms)
        reg_dict["out_shp"] = out_shp
        with open(out_path, "w") as f:
            json.dump(reg_dict, f, cls=NumpyArrayEncoder)

    def stitch_registered(
        self,
        out_fldr: os.PathLike,
        reg_tforms_a: List[Tuple[int, numpy.ndarray]],
        out_shp_a: List[int],
        reg_tforms_b: List[Tuple[int, numpy.ndarray]],
        out_shp_b: List[int],
        channels: List[int] | None,
        blend_method: str,
        interp_method: str,
        block_sizes: Tuple[int, int, int],
        verbose: bool,
    ):
        # determine launch parameters
        klp_a = launch_params_for_volume(out_shp_a, *block_sizes)
        klp_b = launch_params_for_volume(out_shp_b, *block_sizes)
        if channels is None:
            channels = [self._channel]
        if self._reduced_from_2d:
            load_fun_base = _load_function_2dto1d
        else:
            load_fun_base = _load_function
        # split out indices and corresp. affine transform matrices for heads
        idx_a = [i for i, _ in reg_tforms_a]
        mat_a = [m for _, m in reg_tforms_a]
        idx_b = [i for i, _ in reg_tforms_b]
        mat_b = [m for _, m in reg_tforms_b]
        # partial functions for assembly
        assemble_a = partial(
            assemble_stitch_and_save_gpuq,
            out_shp=out_shp_a,
            idxs=idx_a,
            transforms=mat_a,
            blend_method=blend_method,
            interp_method=interp_method,
            launch_params=klp_a,
        )
        assemble_b = partial(
            assemble_stitch_and_save_gpuq,
            out_shp=out_shp_b,
            idxs=idx_b,
            transforms=mat_b,
            blend_method=blend_method,
            interp_method=interp_method,
            launch_params=klp_b,
        )
        # setup the parallel executor
        n_gpu = cupy.cuda.runtime.getDeviceCount()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_gpu, mp_context=multiprocessing.get_context("spawn")
        ) as executor, multiprocessing.Manager() as manager:
            futures = []
            queue = manager.Queue()
            for gpu_id in range(n_gpu):
                queue.put(gpu_id)
            for channel in channels:
                load_a = partial(load_fun_base, self._root_fldr, "a", channel)
                load_b = partial(load_fun_base, self._root_fldr, "b", channel)
                outp_a = os.path.join(
                    out_fldr, StitchedDataset.make_fname("a", channel, None)
                )
                outp_b = os.path.join(
                    out_fldr, StitchedDataset.make_fname("b", channel, None)
                )
                fut_a = executor.submit(assemble_a, outp_a, queue, load_a)
                fut_b = executor.submit(assemble_b, outp_b, queue, load_b)
                futures.append(fut_a)
                futures.append(fut_b)
            if verbose:
                pbar = tqdm(total=len(futures), desc="Assembling Views")
            else:
                pbar = nullcontext
            with pbar:
                for _ in concurrent.futures.as_completed(futures):
                    if verbose:
                        pbar.update(1)


class DiSPIM2DStitcher:
    def __init__(
        self,
        root_fldr: os.PathLike,
        position_list: os.PathLike | None,
        overlap_axes: Tuple[int, int] | None,
        pct_overlaps: Tuple[float, float] | None,
    ):
        self._root_fldr = root_fldr
        with uManagerAcquisitionMultiPos(self._root_fldr) as f:
            tile_shape = f.image_shape
            self.tile_shape = list(tile_shape)
        # determine overlap parameters/tiling of positions
        if position_list is None:
            self.position_list = None
            assert overlap_axes is not None and pct_overlaps is not None, (
                "position list not given, must supply % overlap & axes"
            )
            self.overlap_axes = overlap_axes
            self.pct_overlaps = pct_overlaps
            self.len_overlaps = [
                math.ceil(self.tile_shape[oa] * po)
                for oa, po in zip(overlap_axes, pct_overlaps)
            ]
        else:
            self.position_list = PositionList(position_list)
            if overlap_axes is None:
                overlap_axes, overlap_pix = self.position_list.overlap_props()
                pct_overlaps = [
                    op / tile_shape[oa] for op, oa in zip(overlap_pix, overlap_axes)
                ]
                self.overlap_axes = overlap_axes
                self.pct_overlaps = pct_overlaps
                self.len_overlaps = overlap_pix
            else:
                self.overlap_axes = overlap_axes
                self.pct_overlaps = pct_overlaps
                self.len_overlaps = [
                    math.ceil(self.tile_shape[oa] * po)
                    for oa, po in zip(overlap_axes, pct_overlaps)
                ]
        self.len_rests = [
            self.tile_shape[oa] - lo
            for oa, lo in zip(self.overlap_axes, self.len_overlaps)
        ]

    def one_dimensionalify(
        self,
        out_fldr: os.PathLike,
        reduction_index: int,
        reg_channel: int,
        pcc_prereg: bool,
        reg_piecewise: bool,
        reg_metric: str,
        reg_transform: str,
        interp_method: str,
        reg_bounds: OptBounds | OptBoundMargins | None,
        blend_method: str,
        stitch_channels: List[int] | None,
        block_sizes: Tuple[int, int, int],
        verbose: bool = False,
        **opt_kwargs,
    ):
        assert reduction_index == 0 or reduction_index == 1, (
            "reduction_index must be 0 or 1 (b/c 2D)"
        )
        if self.position_list is None:
            raise NotImplementedError("need a PositionList for this to work")
        if stitch_channels is None:
            stitch_channels = [reg_channel]
        # get overlap axis, overlap length, and "rest" length
        overlap_axis = self.overlap_axes[reduction_index]
        len_overlap = self.len_overlaps[reduction_index]
        len_rest = self.len_rests[reduction_index]
        # figure out pairs that we'll need to register
        nz, ny = self.position_list.grid_size
        if overlap_axis == 0:  #
            pair_idx_groups = [
                self.position_list.zaxis_stitch_pairs(y) for y in range(ny)
            ]
        elif overlap_axis == 1:
            pair_idx_groups = [
                self.position_list.yaxis_stitch_pairs(z) for z in range(nz)
            ]
        else:
            raise ValueError("")
        # determine slicing for this overlap axis & length
        slice_f, slice_r = build_slices(self.tile_shape, overlap_axis, len_overlap)
        # determine launch parameters
        shp = self.tile_shape
        shp[overlap_axis] = len_overlap
        klp = launch_params_for_volume(shp, *block_sizes)
        # make partial functions so all shared params are already passed in
        load_af = partial(_load_function, self._root_fldr, "a", reg_channel, slice_f)
        load_bf = partial(_load_function, self._root_fldr, "b", reg_channel, slice_f)
        load_ar = partial(_load_function, self._root_fldr, "a", reg_channel, slice_r)
        load_br = partial(_load_function, self._root_fldr, "b", reg_channel, slice_r)
        # initialize the output list such that for each group of pair indices,
        # the first position is given the identity transform
        out_list = []
        for gidx, pair_idxs in enumerate(pair_idx_groups):
            out_list.append(("a", gidx, pair_idxs[0][0], numpy.eye(4)))
            out_list.append(("b", gidx, pair_idxs[0][0], numpy.eye(4)))
        # do parallel registration of all pairs for both heads
        # degree of parallelization is determined by the number of GPUs
        n_gpu = cupy.cuda.runtime.getDeviceCount()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_gpu, mp_context=multiprocessing.get_context("spawn")
        ) as executor, multiprocessing.Manager() as manager:
            # setup shared queue for keeping track (lock) of which GPU(s)
            # are available for computation
            gpu_queue = manager.Queue()
            for gpu_id in range(n_gpu):
                gpu_queue.put(gpu_id)
            # make partial functions for registering "A" & "B" views
            reg_a = partial(
                register_pair_load_gpuq,
                queue=gpu_queue,
                loadfun0=load_af,
                loadfun1=load_ar,
                pcc_prereg=pcc_prereg,
                piecewise=reg_piecewise,
                metric=reg_metric,
                transform=reg_transform,
                interp_method=interp_method,
                bounds=reg_bounds,
                kernel_launch_params=klp,
                verbose=verbose,
                **opt_kwargs,
            )
            reg_b = partial(
                register_pair_load_gpuq,
                queue=gpu_queue,
                loadfun0=load_bf,
                loadfun1=load_br,
                pcc_prereg=pcc_prereg,
                piecewise=reg_piecewise,
                metric=reg_metric,
                transform=reg_transform,
                interp_method=interp_method,
                bounds=reg_bounds,
                kernel_launch_params=klp,
                verbose=verbose,
                **opt_kwargs,
            )
            # submit all the jobs to the executor, keeping track of which
            # futures are responsible for mapping pairs at each head & group
            futures_to_pairs = dict()
            for gidx, pair_idxs in enumerate(pair_idx_groups):
                for idx0, idx1 in pair_idxs:
                    fut_a = executor.submit(reg_a, idx0, idx1)
                    futures_to_pairs[fut_a] = ("a", gidx, idx0, idx1)
                    fut_b = executor.submit(reg_b, idx0, idx1)
                    futures_to_pairs[fut_b] = ("b", gidx, idx0, idx1)
            iterator = concurrent.futures.as_completed(futures_to_pairs)
            if verbose:
                n_task = len(futures_to_pairs)
                iterator = tqdm(iterator, total=n_task, desc="Registration Pairs")
            for future in iterator:
                head, gidx, idx0, idx1 = futures_to_pairs[future]
                T, res = future.result()
                if verbose:
                    iterator.set_postfix_str(
                        f"{head:s}_{gidx:d} {idx0:d}->{idx1:d}, met={1 - res.fun:.3f}"
                    )
                out_list.append((head, gidx, idx1, T))
            # NOTE: once we hit here, all the pairs have been registered
            # and the `out_list` should be full of results
            # ---- now we do the stitching for each axis/head
            # do accumulation for each axis/head, then do the stitching
            stitch_futures = []
            for gidx, pair_idxs in enumerate(pair_idx_groups):
                out_a = [
                    (t[2], t[3]) for t in out_list if (t[0] == "a" and t[1] == gidx)
                ]
                out_b = [
                    (t[2], t[3]) for t in out_list if (t[0] == "b" and t[1] == gidx)
                ]
                # sort the output lists
                ord = order_pair_indices(pair_idxs)
                idxoe = list(enumerate(ord))
                out_a.sort(key=lambda t: [s for s, i in idxoe if i == t[0]][0])
                out_b.sort(key=lambda t: [s for s, i in idxoe if i == t[0]][0])
                # do the accumulation for each head
                idx_a = [id for id, _ in out_a]
                mat_a = [mat for _, mat in out_a]
                M_a, out_shp_a = accumulate_transforms_1d(
                    self.tile_shape, overlap_axis, len_rest, *mat_a
                )
                klp_a = launch_params_for_volume(out_shp_a, *block_sizes)
                assemble_a = partial(
                    assemble_stitch_and_save_gpuq,
                    out_shp=out_shp_a,
                    idxs=idx_a,
                    transforms=M_a,
                    blend_method=blend_method,
                    interp_method=interp_method,
                    launch_params=klp_a,
                )
                idx_b = [id for id, _ in out_b]
                mat_b = [mat for _, mat in out_b]
                M_b, out_shp_b = accumulate_transforms_1d(
                    self.tile_shape, overlap_axis, len_rest, *mat_b
                )
                klp_b = launch_params_for_volume(out_shp_b, *block_sizes)
                assemble_b = partial(
                    assemble_stitch_and_save_gpuq,
                    out_shp=out_shp_b,
                    idxs=idx_b,
                    transforms=M_b,
                    blend_method=blend_method,
                    interp_method=interp_method,
                    launch_params=klp_b,
                )
                # iterate over channels, submitting stitching jobs for the
                # accumulated transforms at each head for each channel
                for channel in stitch_channels:
                    load_a = partial(_load_function, self._root_fldr, "a", channel)
                    outp_a = os.path.join(
                        out_fldr, _make_2dto1d_fname("a", gidx, channel)
                    )
                    fut_a = executor.submit(assemble_a, outp_a, gpu_queue, load_a)
                    load_b = partial(_load_function, self._root_fldr, "b", channel)
                    outp_b = os.path.join(
                        out_fldr, _make_2dto1d_fname("b", gidx, channel)
                    )
                    fut_b = executor.submit(assemble_b, outp_b, gpu_queue, load_b)
                    stitch_futures.append(fut_a)
                    stitch_futures.append(fut_b)
            stitch_iterator = concurrent.futures.as_completed(stitch_futures)
            if verbose:
                stitch_iterator = tqdm(
                    stitch_iterator, total=len(stitch_futures), desc="Assembling Views"
                )
            for _ in stitch_iterator:
                pass
        # save out parameters of this registration
        out_index = 1 if reduction_index == 0 else 0
        with open(os.path.join(out_fldr, "grid1d_params.json"), "w") as f:
            json.dump(
                {
                    "overlap_axis": self.overlap_axes[out_index],
                    "pct_overlap": self.pct_overlaps[out_index],
                    "len_overlap": self.len_overlaps[out_index],
                    "len_rest": self.len_rests[out_index],
                    "orig_fldr": self._root_fldr,
                    "num_pos": len(pair_idx_groups),
                },
                f,
            )

    def reduce_blockwise(
        self,
        reg_channel: int,
        pcc_prereg: bool,
        reg_piecewise: bool,
        reg_metric: str,
        reg_transform: str,
        interp_method: str,
        blend_method: str,
        reg_bounds: OptBounds | OptBoundMargins | None,
        block_sizes: Tuple[int, int, int],
        verbose: bool = False,
        **opt_kwargs,
    ):
        raise NotImplementedError("todo")
