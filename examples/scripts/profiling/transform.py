import os
from argparse import ArgumentParser

import cupy
import zarr
from cupyx.profiler import benchmark

from pyspim.interp import affine
from pyspim.util import launch_params_for_volume


def main(
    arr_path: os.PathLike,
    interp_method: str,
    transform_str: str,
    adapt_output_size: bool,
    num_repeat: int,
    use_cupyx_benchmark: bool,
) -> int:
    arr = cupy.asarray(
        zarr.open_array(arr_path)
    )[0]
    kname = "affineTransform"
    if interp_method == "cubspl":
        kname += "CubSpl"
        mod : cupy.RawModule = affine.__cuda_module_cubspl
    elif interp_method == "linear":
        kname += "Lerp"
        mod : cupy.RawModule = affine.__cuda_module_linear
    else:  # nearest
        kname += "Nearest"
        mod : cupy.RawModule = affine.__cuda_module_nearest
    kname += "<unsigned short>" # TODO: check if array is floating point and swap
    kernel = mod.get_function(kname)
    launch_pars = launch_params_for_volume(arr.shape, 8, 8, 8)
    T = _str_to_mat(transform_str)
    # allocate output array
    if adapt_output_size:
        out_shp = affine.output_shape_for_transform(T, arr.shape)
    else:
        out_shp = arr.shape
    out = cupy.zeros((out_shp), dtype=cupy.float32)
    if use_cupyx_benchmark:
        # TODO: defaults to n_warmup=10 -- set to 0 since debugging with ncu anyway?
        perf_case = benchmark(
            kernel, 
            (launch_pars[0], launch_pars[1],
             (out, arr, T, *out_shp, *arr.shape)),
            n_repeat=num_repeat
        )
        print(perf_case, flush=True)
    else:
        for _ in range(num_repeat):
            kernel(
                launch_pars[0], launch_pars[1],
                (out, arr, T, *out_shp, *arr.shape)
            )
    return 0


def _str_to_mat(t_str: str) -> cupy.ndarray:
    T = cupy.array([float(t) for t in t_str.split(',')],
                      dtype=cupy.float32).reshape(3, 4)
    return cupy.concatenate([T, cupy.zeros((1, 4))], axis=0)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--input-path", type=str, required=True)
    parser.add_argument("-i", "--interp-method", type=str, required=True,
                        choices=["cubspl", "linear", "nearest"],
                        help="interpolation method to use")
    parser.add_argument("-t", "--transform", type=str, 
                        default="1,0,0,0,0,1,0,0,0,0,1,0",
                        help="flattened 3x4 matrix specifying the transform to apply")
    parser.add_argument("-ao", "--adapt-output-size", action="store_true",
                        help="")
    parser.add_argument("-n", "--num-repeat", type=int, required=True,
                        help="# of times kernel is called")
    parser.add_argument("-cb", "--cupyx-benchmark", action="store_true",
                        help="use cupyx benchmarking instead of python loop")
    args = parser.parse_args()
    ec = main(
        args.input_path, 
        args.interp_method,
        args.transform,
        args.adapt_output_size,
        args.num_repeat,
        args.cupyx_benchmark,
    )
    exit(ec)
