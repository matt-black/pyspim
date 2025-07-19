import itertools
import multiprocessing
import concurrent.futures
from functools import partial
from types import ModuleType
from typing import List, Tuple

import cupy
import numpy
import zarr
from tqdm.auto import tqdm

from ..typing import NDArray, SliceWindow3D


def _load_region(vol : zarr.Array, window : SliceWindow3D, 
                 xp : ModuleType = numpy):
    return xp.asarray(vol.get_orthogonal_selection(window))


def _variance_queued(vol : numpy.ndarray, queue):
    gpu_id = queue.get()
    with cupy.cuda.Device(gpu_id):
        var = cupy.var(cupy.asarray(vol))
        out = var.get()
    queue.put(gpu_id)
    return out


def _load_and_compute_variance_queued(window : SliceWindow3D, vol : zarr.Array, 
                                      queue):
    return _variance_queued(_load_region(vol, window, numpy), queue)


class ChunkwiseRegistration(object):
    def __init__(self, vol0 : zarr.Array, vol1 : zarr.Array,
                 chunk_size : int|Tuple[int,int,int]):
        self._vol0 = vol0
        self._vol1 = vol1
        if isinstance(chunk_size, int):
            self._chunk_size = tuple([chunk_size for _ in range(3)])
        else:
            self._chunk_size = chunk_size
        # determine how the insides of the array are going to get chunked
        # NOTE: this strategy will clip out edges of the array that don't
        # fit neatly into the chunk_size (instead of e.g. padding the array
        # so that everything would be seen)
        n_chunk = [s // c for s, c in zip(vol0.shape, self._chunk_size)]
        # inner_shp is the part of the volume that will be chunked and
        # actually used in the computation
        inner_shp = [n * s for n, s in zip(n_chunk, chunk_size)]
        offsets = [(s-i)//2 for s, i in zip(vol0.shape, inner_shp)]
        # build up chunk indices
        chunk_windows = []
        chunk_mults = itertools.product(*[range(n) for n in n_chunk])
        for cm in chunk_mults:
            idx0 = [m * cs + o for m, cs, o in zip(cm, chunk_size, offsets)]
            idx1 = [i0 + cs for i0, cs in zip(idx0, chunk_size)]
            window = [slice(i0, i1) for i0, i1 in zip(idx0, idx1)]
            chunk_windows.append(window)
        self._chunk_windows = chunk_windows
        self._chunk_indices = list(range(len(chunk_windows)))

    def variance_perchunk(self, use_vol1 : bool = False, 
                          verbose : bool = False):
        n_gpu = cupy.cuda.runtime.getDeviceCount()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_gpu, mp_context=multiprocessing.get_context('spawn')
        ) as executor, multiprocessing.Manager() as manager:
            # setup shared queue for keeping track (lock) of which GPU(s)
            # are available for computation
            gpu_queue = manager.Queue()
            for gpu_id in range(n_gpu):
                gpu_queue.put(gpu_id)
            variance = partial(_load_and_compute_variance_queued,
                               vol=(self._vol1 if use_vol1 else self._vol0),
                               queue=gpu_queue)
            future_to_idx = dict()
            for idx, window in zip(self._chunk_indices, self._chunk_windows):
                fut = executor.submit(variance, window)
                future_to_idx[fut] = idx
            iterator = concurrent.futures.as_completed(future_to_idx)
            if verbose:
                n_task = len(future_to_idx)
                iterator = tqdm(iterator, total=n_task, 
                                desc="Chunkwise Variance")
            vars = [0 for _ in self._chunk_indices]
            for future in iterator:
                idx = future_to_idx[future]
                var = future.result()
                if verbose:
                    iterator.set_postfix_str('{:d} -> {:.2f}'.format(
                        idx, var
                    ))
                vars[idx] = var
            return vars
        
    def evaluate(self, T : NDArray, metric : str, chunks : List[int]|None,
                 executor : concurrent.futures.Executor, gpu_queue):
        raise NotImplementedError('todo')