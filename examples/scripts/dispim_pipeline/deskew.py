import os
import json
import math
import time
import shutil
import multiprocessing
import concurrent.futures
from typing import List, Optional, Tuple
from functools import partial
from contextlib import nullcontext
from argparse import ArgumentParser

import zarr
import cupy
import numpy
from tqdm.auto import tqdm

from pyspim.deskew import shear, ortho
from pyspim.data.dispim import StitchedDataset
from pyspim.data.dispim import uManagerAcquisitionOnePos
from pyspim.data.dispim import subtract_constant_uint16arr


def load_raw_image(
    input_folder: os.PathLike,
    data_type: str,
    head: str,
    channel: int,
    camera_offset: int | None,
    crop_box: Tuple[int, int, int, int, int, int],
):
    cbox = tuple([
        slice(crop_box[0], crop_box[1]),
        slice(crop_box[2], crop_box[3]),
        slice(crop_box[4], crop_box[5]),
    ])
    # fetch the acquisition
    if data_type == "umasi":
        with uManagerAcquisitionOnePos(input_folder, numpy) as acq:
            raw = acq.get(head, channel, 0, cbox) # type: ignore
    else:
        dset = StitchedDataset(input_folder, numpy)
        raw = dset.get(None, head, channel, cbox) # type: ignore
    if camera_offset is not None:
        return subtract_constant_uint16arr(raw, camera_offset)
    else:
        return raw


def deskew_shear(
    step_size: float,
    pixel_size: float,
    direction: int,
    interp_method: str,
    block_size: Tuple[int, int, int],
    verbose,
    gpu_queue,
    raw_img: numpy.ndarray,
) -> numpy.ndarray:
    gpu_id = gpu_queue.get()
    if direction == -1:
        rot_thetas = (0., math.pi / 2, 0.)
        auto_crop = True
    else:
        rot_thetas, auto_crop = None, False
    # do the actual deskewing
    with cupy.cuda.Device(gpu_id):
        start = time.time()
        dsk = shear.deskew_stage_scan(
            cupy.asarray(raw_img),
            pixel_size,
            step_size,
            direction,
            rot_thetas,
            interp_method,
            auto_crop,
            True,
            block_size,
        ).get()
        end = time.time()
        if verbose:
            print("Deskewing Time: {:.2f} sec".format(end - start), flush=True)
    gpu_queue.put(gpu_id)
    return dsk


def deskew_ortho(
    step_size: float,
    pixel_size: float,
    direction: int,
    gpu_queue,
    raw_img: numpy.ndarray,
) -> numpy.ndarray:
    gpu_id = gpu_queue.get()
    with cupy.cuda.Device(gpu_id):
        dsk = ortho.deskew_stage_scan(
            raw_img, pixel_size, step_size, direction, math.pi / 4, True
        )
    gpu_queue.put(gpu_id)
    return dsk


def load_deskew_save(
    input_folder: os.PathLike,
    data_type: str,
    gpu_queue,
    camera_offset: int,
    crop_box_a: Tuple[int, int, int, int, int, int] | None,
    crop_box_b: Tuple[int, int, int, int, int, int] | None,
    deskew_method: str,
    step_size: float,
    pixel_size: float,
    interp_method: str,
    block_size: Tuple[int, int, int],
    out_a: zarr.Array,
    out_b: zarr.Array,
    verbose: bool,
    head: str,
    channel: int,
    out_channel: int,
):
    crop_box = crop_box_a if head == "a" else crop_box_b
    start = time.time()
    raw = load_raw_image(
        input_folder, data_type, head, channel, camera_offset, crop_box
    )
    end = time.time()
    if verbose:
        print("\nLoading data took {:.2f} sec".format(end - start), flush=True)
    direction = 1 if head == "a" else -1
    if deskew_method == "shear":
        dsk = deskew_shear(
            step_size,
            pixel_size,
            direction,
            interp_method,
            block_size,
            verbose,
            gpu_queue,
            raw,
        )
    else:
        dsk = deskew_ortho(step_size, pixel_size, direction, gpu_queue, raw)
    start = time.time()
    (out_a if head == "a" else out_b).set_orthogonal_selection(
        tuple([[out_channel], slice(None), slice(None), slice(None)]),
        dsk[None, ...],
    )
    end = time.time()
    if verbose:
        print("Writing data took {:.2f} sec".format(end - start), flush=True)


def input_validation(
    input_folder: os.PathLike, data_type: str, deskew_method: str
):
    if not os.path.exists(input_folder):
        raise Exception("input folder does not exist")
    if not os.path.isdir(input_folder):
        raise Exception("input folder is not a directory")
    assert data_type in ["stitch", "umasi"], "invalid `data_type`"
    assert deskew_method in ["ortho", "shear"], "invalid `deskew_method`"


def main(
    input_folder: os.PathLike,
    data_type: str,
    output_folder: str,
    camera_offset: int,
    crop_box_a: Optional[Tuple[int, int, int, int, int, int]],
    crop_box_b: Optional[Tuple[int, int, int, int, int, int]],
    # deskewing
    deskew_method: str,
    step_size: float,
    pixel_size: float,
    # registration
    channels: List[int],
    interp_method: str,
    # other parameters
    block_size: Tuple[int, int, int],
    verbose: bool,
):
    # input validation
    input_validation(input_folder, data_type, deskew_method)
    if crop_box_a is None:
        with uManagerAcquisitionOnePos(input_folder, numpy) as acq:
            shape = acq.image_shape
            crop_box_a = tuple([0, shape[0], 0, shape[1], 0, shape[2]])
    if crop_box_b is None:
        with uManagerAcquisitionOnePos(input_folder, numpy) as acq:
            shape = acq.image_shape
            crop_box_b = tuple([0, shape[0], 0, shape[1], 0, shape[2]])
    z_a, z_b = crop_box_a[1] - crop_box_a[0], crop_box_b[1] - crop_box_b[0]
    r_a, r_b = crop_box_a[3] - crop_box_a[2], crop_box_b[3] - crop_box_b[2]
    c_a, c_b = crop_box_a[5] - crop_box_a[4], crop_box_b[5] - crop_box_b[4]
    num_chan = len(channels)
    if deskew_method == "shear":
        out_shp_a = shear.output_shape(
            z_a, r_a, c_a, pixel_size, step_size, direction=1, auto_crop=False
        )
        out_shp_b = shear.output_shape(
            z_b, r_b, c_b, pixel_size, step_size, direction=-1, auto_crop=True
        )
    elif deskew_method == "ortho":
        out_shp_a = ortho.output_shape(
            z_a, r_a, c_a, pixel_size, step_size, math.pi / 4
        )
        out_shp_b = ortho.output_shape(
            z_b, r_b, c_b, pixel_size, step_size, math.pi / 4
        )
    else:
        raise ValueError("dispim not implemented")
    out_shp_a = list([int(a) for a in out_shp_a])
    out_shp_b = list([int(b) for b in out_shp_b])
    a = zarr.open_array(
        os.path.join(output_folder, "a.zarr"),
        mode="w",
        shape=[num_chan] + out_shp_a,
        dtype=numpy.uint16,
        fill_value=0,
    )

    b = zarr.open_array(
        os.path.join(output_folder, "b.zarr"),
        mode="w",
        shape=[num_chan] + out_shp_b,
        dtype=numpy.uint16,
        fill_value=0,
    )
    if verbose:
        print("Setup complete, zarr output arrays created", flush=True)
    # setup the multiprocessor/work queue
    n_gpu = cupy.cuda.runtime.getDeviceCount()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_gpu, mp_context=multiprocessing.get_context("spawn")
    ) as executor, multiprocessing.Manager() as manager:
        gpu_queue = manager.Queue()
        for gpu_id in range(n_gpu):
            gpu_queue.put(gpu_id)
        fun = partial(
            load_deskew_save,
            input_folder,
            data_type,
            gpu_queue,
            camera_offset,
            crop_box_a,
            crop_box_b,
            deskew_method,
            step_size,
            pixel_size,
            interp_method,
            block_size,
            a,
            b,
            verbose,
        )
        futures = []
        for out_channel, channel in enumerate(channels):
            fut_a = executor.submit(fun, "a", channel, out_channel)
            fut_b = executor.submit(fun, "b", channel, out_channel)
            futures.append(fut_a)
            futures.append(fut_b)
        if verbose:
            pbar = tqdm(total=len(futures), desc="Deskewing")
        else:
            pbar = nullcontext
        with pbar:
            for future in concurrent.futures.as_completed(futures):
                _ = future.result()
                if verbose:
                    pbar.update(1)
    # save deskewing parameters
    with open(os.path.join(output_folder, "dsk_params.json"), "w") as fh:
        json.dump(
            {
                "camera_offset": camera_offset,
                "deskew_method": deskew_method,
                "step_size": step_size,
                "pixel_size": pixel_size,
                "channels": channels,
                "interp_method": interp_method,
                "roi_a": crop_box_a,
                "roi_b": crop_box_b,
                "shape_a": out_shp_a,
                "shape_b": out_shp_b,
            },
            fh,
        )
    return 0


if __name__ == "__main__":
    parser = ArgumentParser("Deskewing of Small Dualview Acquisitions")
    parser.add_argument("-i", "--input-folder", type=str, required=True)
    parser.add_argument(
        "-dt",
        "--data-type",
        type=str,
        required=True,
        choices=["stitch", "umasi"],
    )
    parser.add_argument("-o", "--output-folder", type=str, required=True)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    # initial data loading arguments
    parser.add_argument(
        "-co", "--camera-offset", type=int, required=False, default=None
    )
    parser.add_argument(
        "-cba", "--crop-box-a", type=str, required=False, default=None
    )
    parser.add_argument(
        "-cbb", "--crop-box-b", type=str, required=False, default=None
    )
    parser.add_argument("-cbj", "--crop-box-json", type=str, default=None)
    # deskewing arguments
    parser.add_argument(
        "-dm",
        "--deskew-method",
        type=str,
        required=True,
        choices=["ortho", "shear"],
    )
    parser.add_argument("-ss", "--step-size", type=float, required=True)
    parser.add_argument(
        "-ps", "--pixel-size", type=float, required=False, default=0.1625
    )
    parser.add_argument("-c", "--channels", type=str, required=True)
    parser.add_argument(
        "-im",
        "--interp-method",
        type=str,
        required=True,
        choices=["nearest", "linear", "cubspl"],
    )
    # gpu arguments
    parser.add_argument("-bs", "--block-size", type=str, default="8,8,8")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    args = parser.parse_args()
    # parse list arguments that are passed in as strings
    args.block_size = [int(b) for b in args.block_size.split(",")]
    args.channels = [int(c) for c in args.channels.split(",")]
    if args.crop_box_a is not None:
        args.crop_box_a = [int(b) for b in args.crop_box_a.split(",")]
        args.crop_box_b = [int(b) for b in args.crop_box_b.split(",")]
        assert len(args.crop_box_a) == 6, "crop box must be 6-long"
        assert len(args.crop_box_b) == 6, "crop box must be 6-long"
    else:
        if args.crop_box_json is None:
            # no cropping
            args.crop_box_a = None
            args.crop_box_b = None
        else:
            with open(args.crop_box_json) as f:
                djson = json.load(f)
                args.crop_box_a = djson["a"]
                args.crop_box_b = djson["b"]
    # make sure output folder exists
    if os.path.exists(args.output_folder):
        if args.force:
            shutil.rmtree(args.output_folder)
        else:
            print("Output exists, won't overwrite, use --force to overwrite")
            exit(-1)
    os.makedirs(args.output_folder, exist_ok=True)
    with open(
        os.path.join(args.output_folder, "cmdline_args_dsk.json"), "w"
    ) as f:
        json.dump(vars(args), f)
    exit_code = main(
        args.input_folder,
        args.data_type,
        args.output_folder,
        args.camera_offset,
        args.crop_box_a,
        args.crop_box_b,
        args.deskew_method,
        args.step_size,
        args.pixel_size,
        args.channels,
        args.interp_method,
        args.block_size,
        args.verbose,
    )
