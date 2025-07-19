import os
import json
import pickle
from typing import List, Optional, Tuple
from argparse import ArgumentParser

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import zarr
import cupy
import numpy
from skimage.transform import downscale_local_mean

from pyspim.typing import NDArray
from pyspim.reg import pcc, powell
from pyspim.interp import affine_transform
from pyspim.util import launch_params_for_volume


def downscale_2x_cpu(im: numpy.ndarray) -> numpy.ndarray:
    return (
        numpy.round(downscale_local_mean(im, (2, 2, 2)))
        .clip(0, 2**16 - 1)
        .astype(numpy.uint16)
    )


def downscale_2x_gpu(
    im: cupy.ndarray, block_size: Tuple[int, int, int]
) -> cupy.ndarray:
    return affine_transform(
        im, numpy.diag([2, 2, 2, 1]), "linear", True, None, *block_size
    )


def downscale_2x(im: NDArray, block_size: Tuple[int, int, int]) -> NDArray:
    if cupy.get_array_module(im) == numpy:
        return downscale_2x_cpu(im)
    else:
        return downscale_2x_gpu(im, block_size)


def main(
    input_folder: os.PathLike,
    output_folder: str,
    crop_box_a: Optional[List[int]],
    crop_box_b: Optional[List[int]],
    # registration
    reg_channel: int,
    reg_downscale: bool,
    pcc_prereg: bool,
    reg_metric: str,
    reg_transform: str,
    reg_bounds: List[float],
    interp_method: str,
    # other parameters
    block_size: Tuple[int, int, int],
    verbose: bool,
    debug: bool,
):
    # input validation
    if not os.path.exists(input_folder):
        raise Exception("input folder does not exist")
    if not os.path.isdir(input_folder):
        raise Exception("input folder is not a directory")
    if verbose:
        print("Reading in deskewed views...", flush=True)
    a_zarr = zarr.open_array(os.path.join(input_folder, "a.zarr"))
    b_zarr = zarr.open_array(os.path.join(input_folder, "b.zarr"))
    if crop_box_a is not None:
        a_dsk = a_zarr.oindex[
            reg_channel,
            slice(crop_box_a[0], crop_box_a[1]),
            slice(crop_box_a[2], crop_box_a[3]),
            slice(crop_box_a[4], crop_box_a[5]),
        ]
        crop_box_a = [
            (crop_box_a[0], crop_box_a[1]),
            (crop_box_a[2], crop_box_a[3]),
            (crop_box_a[4], crop_box_a[5]),
        ]
    else:
        crop_box_a = [(0, a_zarr.shape[i]) for i in range(1, 4)]
        a_dsk = a_zarr.oindex[
            reg_channel, slice(None), slice(None), slice(None)
        ]
    if crop_box_b is not None:
        b_dsk = b_zarr.oindex[
            reg_channel,
            slice(crop_box_b[0], crop_box_b[1]),
            slice(crop_box_b[2], crop_box_b[3]),
            slice(crop_box_b[4], crop_box_b[5]),
        ]
        crop_box_b = [
            (crop_box_b[0], crop_box_b[1]),
            (crop_box_b[2], crop_box_b[3]),
            (crop_box_b[4], crop_box_b[5]),
        ]
    else:
        crop_box_b = [(0, b_zarr.shape[i]) for i in range(1, 4)]
        b_dsk = b_zarr.oindex[
            reg_channel, slice(None), slice(None), slice(None)
        ]
    if verbose:
        print(
            "\tA shape [{:d},{:d},{:d}], {:.2f} Gb".format(
                *a_dsk.shape, a_dsk.nbytes / 1e9
            )
        )
        print(
            "\tB shape [{:d},{:d},{:d}], {:.2f} Gb".format(
                *b_dsk.shape, b_dsk.nbytes / 1e9
            )
        )
    if debug:
        print("[DBG] A : {} | B : {}".format(a_dsk.dtype, b_dsk.dtype))
        for i in range(3):
            _, ax = plt.subplots()
            ax.imshow(numpy.amax(a_dsk, i), cmap="binary_r", vmax=2000)
            ax.imshow(
                numpy.amax(b_dsk, i), cmap="viridis", alpha=0.3, vmax=2000
            )
            plt.savefig(
                os.path.join(input_folder, "debug_dsk{:d}.png".format(i))
            )
    # do the registration
    _, _, par0 = powell.parse_transform_string(reg_transform)
    klp = launch_params_for_volume(a_dsk.shape, *block_size)
    if reg_downscale:
        if verbose:
            print("Downscaling views 2x...", flush=True)
        a_dskd = downscale_2x(a_dsk, block_size)
        b_dskd = downscale_2x(b_dsk, block_size)
        if verbose:
            print("\tDone downscaling", flush=True)
        if pcc_prereg:
            if verbose:
                print("Doing PCC pre-registration...", flush=True)
            t0 = pcc.translation_for_volumes(a_dskd, b_dskd)
            par0[:3] = t0
            if verbose:
                print("\tDone with PCC pre-registration", flush=True)
                print(
                    "\tTranslation: {:.2f} {:.2f} {:.2f}".format(
                        *t0, flush=True
                    )
                )
        klp_d = launch_params_for_volume(a_dskd.shape, *block_size)
        a_dskd = cupy.asarray(a_dskd)
        b_dskd = cupy.asarray(b_dskd)
        if debug:
            print(
                "[DBG] GPU mem usage used|total : {:.2f}|{:.2f}".format(
                    cupy.get_default_memory_pool().used_bytes() / 1e9,
                    cupy.get_default_memory_pool().total_bytes() / 1e9,
                )
            )
        _, res_d = powell.optimize_affine_piecewise(
            a_dskd,
            b_dskd,
            metric=reg_metric,
            transform=reg_transform,
            interp_method=interp_method,
            par0=par0,
            bounds=[r / 2 for r in reg_bounds[:3]] + reg_bounds[3:],
            kernel_launch_params=klp_d,
            verbose=verbose,
        )
        # use downscale to formulate initial parameters
        par0 = numpy.concatenate([res_d.x[:3] * 2, res_d.x[3:]])
        #reg_bounds = [(p - b, p + b) for p, b in zip(par0, reg_bounds)]
        opt_fun = powell.optimize_affine
        del a_dskd, b_dskd
        cupy.get_default_memory_pool().free_all_blocks()
        if debug:
            print(
                "[DBG] GPU mem usage used|total : {:.2f}|{:.2f}".format(
                    cupy.get_default_memory_pool().used_bytes() / 1e9,
                    cupy.get_default_memory_pool().total_bytes() / 1e9,
                )
            )
    else:
        if pcc_prereg:
            if verbose:
                print("Doing PCC pre-registration...", flush=True)
            t0 = pcc.translation_for_volumes(a_dsk, b_dsk)
            par0[:3] = t0
            if verbose:
                print("\tDone with PCC pre-registration", flush=True)
                print(
                    "\tTranslation: {:.2f} {:.2f} {:.2f}".format(
                        *t0, flush=True
                    )
                )
        opt_fun = powell.optimize_affine_piecewise
    if verbose:
        print("Moving data to GPU", flush=True)
    if debug:
        print(
            "[DBG] A {:.2f} Gb/{} | B {:.2f} Gb/{}".format(
                a_dsk.nbytes / 1e9, a_dsk.dtype, b_dsk.nbytes / 1e9, b_dsk.dtype
            )
        )
        print(
            "[DBG] GPU mem usage used|total : {:.2f}|{:.2f}".format(
                cupy.get_default_memory_pool().used_bytes() / 1e9,
                cupy.get_default_memory_pool().total_bytes() / 1e9,
            )
        )
    a_dsk, b_dsk = cupy.asarray(a_dsk), cupy.asarray(b_dsk)
    if verbose:
        print("\tDone, data is on GPU", flush=True)
    if debug:
        print(
            "[DBG] A {:.2f} Gb/{} | B {:.2f} Gb/{}".format(
                a_dsk.nbytes / 1e9, a_dsk.dtype, b_dsk.nbytes / 1e9, b_dsk.dtype
            )
        )
        print(
            "[DBG] GPU mem usage used|total : {:.2f}|{:.2f}".format(
                cupy.get_default_memory_pool().used_bytes() / 1e9,
                cupy.get_default_memory_pool().total_bytes() / 1e9,
            )
        )
    T, res = opt_fun(
        a_dsk,
        b_dsk,
        metric=reg_metric,
        transform=reg_transform,
        interp_method=interp_method,
        par0=par0,
        bounds=reg_bounds,
        kernel_launch_params=klp,
        verbose=verbose,
    )
    # save registration results
    image_shape = [int(s) for s in a_dsk.shape]
    with open(os.path.join(output_folder, "reg_params.json"), "w") as fh:
        json.dump(
            {
                "interp_method": interp_method,
                "roi_a": crop_box_a,
                "roi_b": crop_box_b,
                "image_shape": image_shape,
            },
            fh,
        )
    numpy.save(os.path.join(output_folder, "reg_transform.npy"), T)
    with open(os.path.join(output_folder, "opt_res.pkl"), "wb") as fh:
        pickle.dump(res, fh)
    return 0


if __name__ == "__main__":
    parser = ArgumentParser("Registration of Small Dualview Acquisitions")
    parser.add_argument("-i", "--input-folder", type=str, required=True)
    parser.add_argument("-o", "--output-folder", type=str, required=True)
    # cropping arguments
    parser.add_argument(
        "-cba", "--crop-box-a", type=str, required=False, default=None
    )
    parser.add_argument(
        "-cbb", "--crop-box-b", type=str, required=False, default=None
    )
    # registration arguments
    parser.add_argument("-c", "--channel", type=int, required=True)
    parser.add_argument("--downscale", action="store_true", default=False)
    parser.add_argument("--pcc-prereg", action="store_true", default=False)
    parser.add_argument(
        "-m", "--metric", type=str, required=True, choices=["ncc", "nip", "cr"]
    )
    parser.add_argument("-t", "--transform", type=str, required=True)
    parser.add_argument(
        "-b", "--bounds", type=str, required=False, default=None
    )
    parser.add_argument(
        "-im",
        "--interp-method",
        type=str,
        required=True,
        choices=["nearest", "linear", "cubspl"],
    )
    parser.add_argument("-bs", "--block-size", type=str, default="8,8,8")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    # parse list arguments that are passed in as strings
    if args.crop_box_a is not None:
        args.crop_box_a = [int(b) for b in args.crop_box_a.split(",")]
        assert len(args.crop_box_a) == 6, "crop boxes must be length 6"
    if args.crop_box_b is not None:
        args.crop_box_b = [int(b) for b in args.crop_box_b.split(",")]
        assert len(args.crop_box_b) == 6, "crop boxes must be length 6"
    args.bounds = [float(b) for b in args.bounds.split(",")]
    args.block_size = [int(b) for b in args.block_size.split(",")]
    # make sure output folder exists
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    with open(
        os.path.join(args.output_folder, "cmdline_args_reg.json"), "w"
    ) as f:
        json.dump(vars(args), f)
    if args.verbose:
        print("Registering : {:s}".format(args.input_folder), flush=True)
    exit_code = main(
        args.input_folder,
        args.output_folder,
        args.crop_box_a,
        args.crop_box_b,
        args.channel,
        args.downscale,
        args.pcc_prereg,
        args.metric,
        args.transform,
        args.bounds,
        args.interp_method,
        args.block_size,
        args.verbose,
        args.debug,
    )
