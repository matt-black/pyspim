import os
import json
import shutil
import multiprocessing
import concurrent.futures
from functools import partial
from typing import List, Tuple
from contextlib import nullcontext
from argparse import ArgumentParser

import zarr
import cupy
import numpy
from tqdm.auto import tqdm

from pyspim.interp import affine


def main(
    input_folder: os.PathLike,
    output_folder: os.PathLike,
    reg_params_path: os.PathLike,
    reg_transform_path: str,
    interp_method: str | None,
    block_size: Tuple[int, int, int],
    num_cpu_workers: int,
    verbose: bool,
):
    # load registration parameters
    with open(reg_params_path, "r") as fh:
        reg_params = json.load(fh)
    T = numpy.load(reg_transform_path)
    T = cupy.asarray(T).astype(cupy.float32)
    # open input zarr array
    a_input = zarr.open(os.path.join(input_folder, "a.zarr"))
    b_input = zarr.open(os.path.join(input_folder, "b.zarr"))
    n_chan = b_input.shape[0]
    # grab some constants
    if interp_method is None:
        interp_method = reg_params["interp_method"]
    vol_shape = reg_params["image_shape"]
    out_shape = [n_chan] + vol_shape
    # figure out if we have to crop a
    crop_a = False
    for vol_dim, (cba_l, cba_r) in zip(a_input.shape[1:], reg_params["roi_a"]):
        diff = cba_r - cba_l
        if vol_dim != diff:
            crop_a = True
            break
    if crop_a:
        if verbose:
            print("Cropping View A...", flush=True)
        a_trans = zarr.creation.open_array(
            os.path.join(output_folder, "a.zarr"),
            mode="w",
            shape=out_shape,
            dtype=numpy.uint16,
            fill_value=0,
        )
        # formulate mapping function
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_cpu_workers
        ) as executor:
            roi_a = [slice(*t) for t in reg_params["roi_a"]]
            crop = partial(crop_and_save, a_input, a_trans, roi_a)
            crop_futs = []
            for channel in range(n_chan):
                crop_futs.append(executor.submit(crop, channel))
            if verbose:
                pbar = tqdm(total=len(crop_futs), desc="Cropping View A")
            else:
                pbar = nullcontext
            with pbar:
                for future in concurrent.futures.as_completed(crop_futs):
                    _ = future.result()
                    if verbose:
                        pbar.update(1)
    else:
        if verbose:
            print("Don't have to crop view a, symlinking...", flush=True)
        os.symlink(
            os.path.join(input_folder, "a.zarr"),
            os.path.join(output_folder, "a.zarr"),
            target_is_directory=True,
        )
    roi_b = [slice(*t) for t in reg_params["roi_b"]]
    # open up input/output zarr arrays
    b_trans = zarr.creation.open_array(
        os.path.join(output_folder, "b.zarr"),
        mode="w",
        shape=out_shape,
        dtype=numpy.uint16,
        fill_value=0,
    )
    num_gpu = cupy.cuda.runtime.getDeviceCount()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_gpu, mp_context=multiprocessing.get_context("spawn")
    ) as executor, multiprocessing.Manager() as manager:
        gpu_queue = manager.Queue()
        for device_id in range(num_gpu):
            gpu_queue.put(device_id)
        transform = partial(
            transform_and_save,
            b_input,
            b_trans,
            T,
            roi_b,
            vol_shape,
            interp_method,
            block_size,
            gpu_queue,
        )
        futures = []
        for channel in range(n_chan):
            future = executor.submit(transform, channel)
            futures.append(future)
        if verbose:
            pbar = tqdm(total=len(futures), desc="Transforming View B")
        else:
            pbar = nullcontext
        with pbar:
            for future in concurrent.futures.as_completed(futures):
                _ = future.result()
                if verbose:
                    pbar.update(1)
    return 0


def crop_and_save(
    arr_in: zarr.Array, arr_out: zarr.Array, roi: List[slice], channel: int
):
    crop_window = tuple([channel] + roi)
    arr_out.set_orthogonal_selection(
        tuple([channel, slice(None), slice(None), slice(None)]),
        arr_in.oindex[crop_window],
    )


def transform_and_save(
    arr_in: zarr.Array,
    arr_out: zarr.Array,
    T: cupy.ndarray,
    roi: List[slice],
    out_shape: Tuple[int, int, int],
    interp_method: str,
    block_size: Tuple[int, int, int],
    gpu_queue,
    channel: int,
):
    device_id = gpu_queue.get()
    with cupy.cuda.Device(device_id):
        window = tuple([channel] + roi)
        x = cupy.asarray(arr_in.oindex[window])
        trans = affine.transform(
            x, T, interp_method, True, out_shape, *block_size
        ).get()
        print([numpy.amax(x), numpy.amax(trans)], flush=True)
        del x
        cupy.get_default_memory_pool().free_all_blocks()
    gpu_queue.put(device_id)
    arr_out.set_orthogonal_selection(
        tuple([[channel], slice(None), slice(None), slice(None)]),
        trans[None, ...],
    )


if __name__ == "__main__":
    parser = ArgumentParser("Deskewing & Affine Transformation")
    parser.add_argument("-i", "--input-folder", type=str, required=True)
    parser.add_argument("-o", "--output-folder", type=str, required=True)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    parser.add_argument("-rp", "--reg-params", type=str, required=True)
    parser.add_argument("-t", "--reg-transform", type=str, required=True)
    parser.add_argument(
        "-im", "--interp-method", type=str, required=False, default=None
    )
    parser.add_argument("-w", "--num-workers", type=int, required=True)
    parser.add_argument(
        "-bs", "--block-size", type=str, required=False, default="8,8,8"
    )
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    args = parser.parse_args()
    args.block_size = [int(b) for b in args.block_size.split(",")]
    if os.path.exists(args.output_folder):
        if args.force:
            shutil.rmtree(args.output_folder)
        else:
            print("won't overwrite existing output directory, existing...")
            exit(-1)
    os.makedirs(args.output_folder, exist_ok=True)
    exit_code = main(
        args.input_folder,
        args.output_folder,
        args.reg_params,
        args.reg_transform,
        args.interp_method,
        args.block_size,
        args.num_workers,
        args.verbose,
    )
    exit(exit_code)
