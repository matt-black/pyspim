import os
import shutil
import tempfile
from functools import partial
from typing import Optional, Tuple
from argparse import ArgumentParser

import zarr
import cupy
import numpy
import tifffile
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image

from pyspim.typing import PadType

# deconvolution
from pyspim.decon.util import gaussian_kernel_1d
import pyspim.decon.rl.dualview_sep as sep
import pyspim.decon.rl.dualview_fft as fft


def main_fft(
    a_zarr_path: os.PathLike,
    b_zarr_path: os.PathLike,
    output_path: os.PathLike,
    decon_function: str,
    a_psf_path: os.PathLike,
    b_psf_path: os.PathLike,
    a_bp_path: os.PathLike | None,
    b_bp_path: os.PathLike | None,
    num_iter: int,
    epsilon: float,
    boundary_correction: bool,
    zero_padding: Optional[PadType],
    boundary_sigma_a: float,
    boundary_sigma_b: float,
    distributed: bool,
    verbose: bool,
) -> int:
    a = zarr.open(a_zarr_path)
    b = zarr.open(b_zarr_path)
    # determine output type
    output_zarr = output_path[:-4] == "zarr"
    # delete file, if it exists
    if os.path.exists(output_path):
        if output_zarr:
            shutil.rmtree(output_path)
        else:
            os.remove(output_path)
    if distributed:
        _ = generate_decon_dist_fft(
            a,
            b,
            output_path,
            decon_function,
            num_iter,
            epsilon,
            boundary_correction,
            zero_padding,
            boundary_sigma_a,
            boundary_sigma_b,
            a_psf_path,
            b_psf_path,
            a_bp_path,
            b_bp_path,
            verbose,
        )
    else:
        pfun_decon = partial(
            generate_decon_fft,
            a,
            b,
            decon_function,
            num_iter,
            epsilon,
            boundary_correction,
            zero_padding,
            boundary_sigma_a,
            boundary_sigma_b,
            a_psf_path,
            b_psf_path,
            a_bp_path,
            b_bp_path,
            verbose,
        )
        if output_zarr:
            out_arr = zarr.creation.open_array(
                output_path,
                mode="w",
                shape=a.shape,
                dtype=numpy.float32,
                fill_value=0,
            )
            for cidx, dec in enumerate(pfun_decon()):
                out_arr[cidx, ...] = dec
        else:
            tifffile.imwrite(
                output_path,
                pfun_decon(),
                bigtiff=True,
                shape=a.shape,
                dtype=numpy.float32,
                resolution=(1 / 0.1625, 1 / 0.1625),
                metadata={"axes": "CZYX", "spacing": 0.1625, "units": "um"},
            )
    return 0


def main_sep(
    a_zarr_path: os.PathLike,
    b_zarr_path: os.PathLike,
    output_path: os.PathLike,
    decon_function: str,
    sigma_az: float,
    sigma_ay: float,
    sigma_ax: float,
    sigma_bz: float,
    sigma_by: float,
    sigma_bx: float,
    kernel_radius_z: int,
    kernel_radius_y: int,
    kernel_radius_x: int,
    num_iter: int,
    epsilon: float,
    boundary_correction: bool,
    zero_padding: Optional[PadType],
    boundary_sigma_a: float,
    boundary_sigma_b: float,
    distributed: bool,
    verbose: bool,
) -> int:
    a = zarr.open(a_zarr_path)
    b = zarr.open(b_zarr_path)
    # determine output type
    output_zarr = output_path[-4:] == "zarr"
    # delete file, if it exists
    if os.path.exists(output_path):
        if output_zarr:
            shutil.rmtree(output_path)
        else:
            os.remove(output_path)
    if distributed:
        _ = generate_decon_dist_sep(
            a,
            b,
            output_path,
            decon_function,
            num_iter,
            epsilon,
            boundary_correction,
            zero_padding,
            boundary_sigma_a,
            boundary_sigma_b,
            sigma_az,
            sigma_ay,
            sigma_ax,
            sigma_bz,
            sigma_by,
            sigma_bx,
            kernel_radius_z,
            kernel_radius_y,
            kernel_radius_x,
            verbose,
        )
    else:
        # generate psf's and backprojectors
        psf_az = make_kernel(sigma_az, kernel_radius_z)
        bp_az = psf_az[::-1].copy()
        psf_ay = make_kernel(sigma_ay, kernel_radius_y)
        bp_ay = psf_ay[::-1].copy()
        psf_ax = make_kernel(sigma_ax, kernel_radius_x)
        bp_ax = psf_ax[::-1].copy()
        psf_bz = make_kernel(sigma_bz, kernel_radius_z)
        bp_bz = psf_bz[::-1].copy()
        psf_by = make_kernel(sigma_by, kernel_radius_y)
        bp_by = psf_by[::-1].copy()
        psf_bx = make_kernel(sigma_bx, kernel_radius_x)
        bp_bx = psf_bx[::-1].copy()
        # make generator function for writing to the tifffile
        pfun_decon = partial(
            generate_decon_sep,
            a,
            b,
            decon_function,
            num_iter,
            epsilon,
            boundary_correction,
            zero_padding,
            boundary_sigma_a,
            boundary_sigma_b,
            psf_az,
            psf_ay,
            psf_ax,
            psf_bz,
            psf_by,
            psf_bx,
            bp_az,
            bp_ay,
            bp_ax,
            bp_bz,
            bp_by,
            bp_bx,
            verbose,
        )
        if output_zarr:
            out_arr = zarr.creation.open_array(
                output_path,
                mode="w",
                shape=a.shape,
                dtype=numpy.float32,
                fill_value=0,
            )
            for cidx, dec in enumerate(pfun_decon()):
                out_arr[cidx, ...] = dec
        else:
            tifffile.imwrite(
                output_path,
                pfun_decon(),
                bigtiff=True,
                shape=a.shape,
                dtype=numpy.float32,
                resolution=(1 / 0.1625, 1 / 0.1625),
                metadata={"axes": "CZYX", "spacing": 0.1625, "units": "um"},
            )
    return 0


def make_kernel(sigma: float, radius: int) -> cupy.ndarray:
    return cupy.asarray(gaussian_kernel_1d(sigma, radius).astype(numpy.float32))


def generate_decon_dist_sep(
    a: zarr.Array,
    b: zarr.Array,
    out_path: str,
    decon_function: str,
    num_iter: int,
    epsilon: float,
    boundary_correction: bool,
    zero_padding: Optional[PadType],
    boundary_sigma_a: float,
    boundary_sigma_b: float,
    sigma_az: float,
    sigma_ay: float,
    sigma_ax: float,
    sigma_bz: float,
    sigma_by: float,
    sigma_bx: float,
    kernel_radius_z: float,
    kernel_radius_y: float,
    kernel_radius_x: float,
    verbose: bool,
) -> str:
    is_multichannel = len(a.shape) > 3
    overlap = [2 * kernel_radius_z, 2 * kernel_radius_y, 2 * kernel_radius_x]
    n_gpu = cupy.cuda.runtime.getDeviceCount()
    output_zarr = out_path[-4:] == "zarr"
    if verbose:
        print("Using {:d} GPUs".format(n_gpu), flush=True)
    with tempfile.TemporaryDirectory(suffix=".zarr") as tmp_o:
        out = zarr.creation.open_array(
            tmp_o, mode="w", shape=a.shape, dtype=numpy.float32, fill_value=0
        )
        axes = "CZYX" if is_multichannel else "ZYX"
        sep.deconvolve_chunkwise(
            a,
            b,
            out,
            [500, 500, 500],
            overlap,
            sigma_az,
            sigma_ay,
            sigma_ax,
            sigma_bz,
            sigma_by,
            sigma_bx,
            kernel_radius_z,
            kernel_radius_y,
            kernel_radius_x,
            decon_function,
            num_iter,
            epsilon,
            boundary_correction,
            zero_padding,
            boundary_sigma_a,
            boundary_sigma_b,
            verbose,
        )
        # now copy this out to an OME-* file for display
        if output_zarr:
            if verbose:
                print("Transferring zarr -> ome-ngff", flush=True)
            store = parse_url(out_path, mode="w").store
            root = zarr.group(store=store)
            write_image(image=out, group=root, axes=axes.lower())
        else:
            if verbose:
                print("Transferring zarr -> ome.tif", flush=True)
            tifffile.imwrite(
                out_path,
                bigtiff=True,
                shape=out.shape,
                dtype="float32",
                photometric="minisblack",
                resolution=(1 / 0.1625, 1 / 0.1625),
                metadata={"axes": axes, "spacing": 0.1625, "units": "um"},
            )
            store = tifffile.imread(out_path, mode="r+", aszarr=True)
            z = zarr.open(store, mode="r+")
            # NOTE: if the output only has 1 channel, some weird "correction"
            # happens where it gets eliminated when opening the zarr store and
            # z is 3D instead of 4D, the below "if" statement is for this case
            if len(z.shape) == 3: 
                z[:] = out[0,...]
            else: 
                z[:] = out[:]
            store.close()
            if verbose:
                print("\tDone with transfer - DONE", flush=True)
    return out_path


def generate_decon_sep(
    a: zarr.Array,
    b: zarr.Array,
    decon_function: str,
    decon_iter: int,
    epsilon: float,
    boundary_correction: bool,
    zero_padding: Optional[PadType],
    boundary_sigma_a: float,
    boundary_sigma_b: float,
    psf_az: cupy.ndarray,
    psf_ay: cupy.ndarray,
    psf_ax: cupy.ndarray,
    psf_bz: cupy.ndarray,
    psf_by: cupy.ndarray,
    psf_bx: cupy.ndarray,
    bp_az: cupy.ndarray,
    bp_ay: cupy.ndarray,
    bp_ax: cupy.ndarray,
    bp_bz: cupy.ndarray,
    bp_by: cupy.ndarray,
    bp_bx: cupy.ndarray,
    verbose: bool,
) -> numpy.ndarray:
    n_chan = a.shape[0]
    for cidx in range(n_chan):
        _a = a.oindex[cidx, slice(None), slice(None), slice(None)]
        _b = b.oindex[cidx, slice(None), slice(None), slice(None)]
        dec = sep.deconvolve(
            _a,
            _b,
            None,
            psf_az,
            psf_ay,
            psf_ax,
            psf_bz,
            psf_by,
            psf_bx,
            bp_az,
            bp_ay,
            bp_ax,
            bp_bz,
            bp_by,
            bp_bx,
            decon_function=decon_function,
            num_iter=decon_iter,
            epsilon=epsilon,
            boundary_correction=boundary_correction,
            zero_padding=zero_padding,
            boundary_sigma_a=boundary_sigma_a,
            boundary_sigma_b=boundary_sigma_b,
            verbose=verbose,
        )
        yield dec.get().astype(numpy.float32)


def generate_decon_dist_fft(
    a: zarr.Array,
    b: zarr.Array,
    out_path: str,
    decon_function: str,
    num_iter: int,
    epsilon: float,
    boundary_correction: bool,
    zero_padding: Optional[PadType],
    boundary_sigma_a: float,
    boundary_sigma_b: float,
    a_psf_path: os.PathLike,
    b_psf_path: os.PathLike,
    a_bp_path: os.PathLike,
    b_bp_path: os.PathLike,
    verbose: bool,
):
    is_multichannel = len(a.shape) > 3
    psf_a, psf_b, bp_a, bp_b = load_psfs(
        a_psf_path, b_psf_path, a_bp_path, b_bp_path
    )
    overlap = max([int(a)//2 for a in psf_a.shape])
    n_gpu = cupy.cuda.runtime.getDeviceCount()
    output_zarr = out_path[-4:] == "zarr"
    if verbose:
        print("Using {:d} GPUs".format(n_gpu), flush=True)
    with tempfile.TemporaryDirectory(suffix=".zarr") as tmp_o:
        out = zarr.creation.open_array(
            tmp_o, mode="w", shape=a.shape, dtype=numpy.float32, fill_value=0
        )
        fft.deconvolve_chunkwise(
            a,
            b,
            out,
            [400,400,400],
            overlap,
            psf_a,
            psf_b,
            bp_a,
            bp_b,
            decon_function,
            num_iter,
            epsilon,
            boundary_correction,
            zero_padding,
            boundary_sigma_a,
            boundary_sigma_b,
            verbose,
        )
        axes = "CZYX" if is_multichannel else "ZYX"
        if output_zarr:
            if verbose:
                print("Transferring zarr -> ome-ngff", flush=True)
            store = parse_url(out_path, mode="w").store
            root = zarr.group(store=store)
            write_image(image=out, group=root, axes="czyx")
        else:
            if verbose:
                print("Transferring zarr -> ome.tif", flush=True)
            tifffile.imwrite(
                out_path,
                bigtiff=True,
                shape=out.shape,
                dtype="float32",
                photometric="minisblack",
                resolution=(1 / 0.1625, 1 / 0.1625),
                metadata={"axes": axes, "spacing": 0.1625, "units": "um"},
            )
            store = tifffile.imread(out_path, mode="r+", aszarr=True)
            z = zarr.open(store, mode="r+")
            z[:] = out[:]
            store.close()
        if verbose:
            print("\tDone with transfer - DONE", flush=True)
    return out_path


def generate_decon_fft(
    a: zarr.Array,
    b: zarr.Array,
    decon_function: str,
    decon_iter: int,
    epsilon: float,
    boundary_correction: bool,
    zero_padding: Optional[PadType],
    boundary_sigma_a: float,
    boundary_sigma_b: float,
    a_psf_path: os.PathLike,
    b_psf_path: os.PathLike,
    a_bp_path: os.PathLike | None,
    b_bp_path: os.PathLike | None,
    verbose: bool,
):
    psf_a, psf_b, bp_a, bp_b = load_psfs(
        a_psf_path, b_psf_path, a_bp_path, b_bp_path
    )
    psf_a = cupy.asarray(psf_a, dtype=cupy.float32)
    psf_b = cupy.asarray(psf_b, dtype=cupy.float32)
    bp_a = cupy.asarray(bp_a, dtype=cupy.float32)
    bp_b = cupy.asarray(bp_b, dtype=cupy.float32)
    n_chan = a.shape[0]
    for cidx in range(n_chan):
        _a = cupy.asarray(
            a.oindex[cidx, slice(None), slice(None), slice(None)],
            dtype=cupy.float32,
        )
        _b = cupy.asarray(
            b.oindex[cidx, slice(None), slice(None), slice(None)],
            dtype=cupy.float32,
        )
        dec = fft.deconvolve(
            _a,
            _b,
            None,
            psf_a,
            psf_b,
            bp_a,
            bp_b,
            decon_function,
            decon_iter,
            epsilon,
            True,
            boundary_correction,
            zero_padding,
            boundary_sigma_a,
            boundary_sigma_b,
            verbose,
        )
        yield dec.get().astype(numpy.float32)


def load_psfs(
    a_psf_path: os.PathLike,
    b_psf_path: os.PathLike,
    a_bp_path: os.PathLike | None,
    b_bp_path: os.PathLike | None,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    psf_a = _load_psf(a_psf_path)
    psf_b = _load_psf(b_psf_path)
    if a_bp_path is None:
        bp_a = psf_a.copy()[::-1, ::-1, ::-1]
    else:
        bp_a = _load_psf(a_bp_path)
    if b_bp_path is None:
        bp_b = psf_b.copy()[::-1, ::-1, ::-1]
    return psf_a, psf_b, bp_a, bp_b


def _load_psf(path: os.PathLike) -> numpy.ndarray:
    if path[-3:] == "npy":
        psf = numpy.load(path)
    else:
        psf = tifffile.imread(path)
    return psf


if __name__ == "__main__":
    parser = ArgumentParser("Deconvolution of Registered Dualview Acquisitions")
    parser.add_argument("-a", "--a-zarr-path", type=str, required=True)
    parser.add_argument("-b", "--b-zarr-path", type=str, required=True)
    parser.add_argument("-o", "--output-path", type=str, required=True)
    parser.add_argument(
        "-f",
        "--function",
        type=str,
        required=True,
        choices=["dispim", "additive", "efficient"],
    )
    parser.add_argument("--psf-a-path", type=str, required=False)
    parser.add_argument("--psf-b-path", type=str, required=False)
    parser.add_argument("--bp-a-path", type=str, required=False, default=None)
    parser.add_argument("--bp-b-path", type=str, required=False, default=None)
    parser.add_argument("-saz", "--sigma-az", type=float, required=False)
    parser.add_argument("-say", "--sigma-ay", type=float, required=False)
    parser.add_argument("-sax", "--sigma-ax", type=float, required=False)
    parser.add_argument("-sbz", "--sigma-bz", type=float, required=False)
    parser.add_argument("-sby", "--sigma-by", type=float, required=False)
    parser.add_argument("-sbx", "--sigma-bx", type=float, required=False)
    parser.add_argument("-krz", "--kernel-radius-z", type=int, required=False)
    parser.add_argument("-kry", "--kernel-radius-y", type=int, required=False)
    parser.add_argument("-krx", "--kernel-radius-x", type=int, required=False)
    parser.add_argument("-n", "--num-iter", type=int, required=False)
    parser.add_argument(
        "-eps", "--epsilon", type=float, required=False, default=1e-6
    )
    parser.add_argument(
        "-bc", "--boundary-correction", action="store_true", default=False
    )
    parser.add_argument(
        "-bsa", "--boundary-sigma-a", type=float, required=False, default=1e-2
    )
    parser.add_argument(
        "-bsb", "--boundary-sigma-b", type=float, required=False, default=1e-2
    )
    parser.add_argument(
        "-dist", "--distributed", action="store_true", default=False
    )
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    args = parser.parse_args()
    if args.psf_a_path is not None:
        assert (
            args.psf_b_path is not None
        ), "if using PSF images (FFT-decon), must specify for both views"
        exit_code = main_fft(
            args.a_zarr_path,
            args.b_zarr_path,
            args.output_path,
            args.function,
            args.psf_a_path,
            args.psf_b_path,
            args.bp_a_path,
            args.bp_b_path,
            args.num_iter,
            args.epsilon,
            args.boundary_correction,
            None,
            args.boundary_sigma_a,
            args.boundary_sigma_b,
            args.distributed,
            args.verbose,
        )
    else:
        exit_code = main_sep(
            args.a_zarr_path,
            args.b_zarr_path,
            args.output_path,
            args.function,
            args.sigma_az,
            args.sigma_ay,
            args.sigma_ax,
            args.sigma_bz,
            args.sigma_by,
            args.sigma_bx,
            args.kernel_radius_z,
            args.kernel_radius_y,
            args.kernel_radius_x,
            args.num_iter,
            args.epsilon,
            args.boundary_correction,
            None,
            args.boundary_sigma_a,
            args.boundary_sigma_b,
            args.distributed,
            args.verbose,
        )
    exit(exit_code)
