import os
from argparse import ArgumentParser

import cupy
import zarr
from cupyx.profiler import benchmark

from pyspim.reg.met import cub
from pyspim.reg.met import nearest
from pyspim.reg.met import cubspl
from pyspim.reg.met import linear
from pyspim.util import launch_params_for_volume


def main(
    view_a: os.PathLike,
    view_b: os.PathLike,
    metric: str,
    interp_method: str,
    num_repeat: int,
    output_type: str,
    use_cupyx_benchmark: bool,
    use_cub: bool
) -> int:
    # read in the volumes
    a = zarr.open_array(view_a)[0]
    b = zarr.open_array(view_b)[0]
    num_gpus = cupy.cuda.runtime.getDeviceCount()
    print(f"n_GPUs: {num_gpus:d}")
    # grab the kernel we want to test
    if use_cub:
        kernel = cub.make_kernel(metric, interp_method, 8, 8, 8)
    else:
        if interp_method == "cubspl":
            m : cupy.RawModule = cubspl.__cuda_module
        elif interp_method == "linear":
            m : cupy.RawModule = linear.__cuda_module
        else:
            m : cupy.RawModule = nearest.__cuda_module
        if metric == "ncc":
            kname = "normalizedCrossCorrelation"
        elif metric == "cr":
            kname = "correlationRatio"
        else:
            kname = "normInnerProduct"
        kernel_name = kname + f"<unsigned short,{output_type}>"
        kernel = m.get_function(kernel_name)
    
    # make launch parameters
    launch_pars = launch_params_for_volume(a.shape, 8, 8, 8)
    # allocate output array
    prods = cupy.zeros(
        (3,), dtype=(cupy.float64 if output_type == "double" else cupy.float32)
    )
    sz_r, sy_r, sx_r = a.shape
    sz_m, sy_m, sx_m = b.shape
    # move moving/ref images to GPU
    ref = cupy.asarray(a, dtype=cupy.uint16)
    if output_type == "double":
        mu_refp = cupy.mean(ref).astype(cupy.float64)
        mu_ref = cupy.float64(mu_refp.item())
    else:
        mu_refp = cupy.mean(ref).astype(cupy.float32)
        mu_ref = cupy.float32(mu_refp.item())
    
    mov = cupy.asarray(b, dtype=cupy.uint16)
    if use_cub:
        mu_mov = cupy.mean(mov).astype(cupy.float32)
    T = cupy.concatenate(
        [cupy.eye(3), cupy.zeros((3,1))], axis=1
    ).astype(cupy.float32)
    if use_cupyx_benchmark:
        if use_cub:
            perf_case = benchmark(
                kernel,
                (launch_pars[0], launch_pars[1],
                (prods, T, ref, mov, mu_ref, mu_mov,
                 sz_r, sy_r, sx_r, sz_m, sy_m, sx_m)),
                n_repeat=num_repeat
            )
        else:
            perf_case = benchmark(kernel, 
                (launch_pars[0], launch_pars[1], 
                (prods, T, ref, mov, mu_ref,
                sz_r, sy_r, sx_r, sz_m, sy_m, sx_m)),
                n_repeat=num_repeat
            )
        print(perf_case, flush=True)
    for _ in range(num_repeat):
        if use_cub:
            kernel(
                launch_pars[0], launch_pars[1],
                (prods, T, ref, mov, mu_ref, mu_mov,
                 sz_r, sy_r, sx_r, sz_m, sy_m, sx_m)
            )
        else:
            kernel(
                launch_pars[0], launch_pars[1],
                (prods, T, ref, mov, mu_ref,
                sz_r, sy_r, sx_r, sz_m, sy_m, sx_m)
            )

    met = float(prods[0] / (cupy.sqrt(prods[1]) * cupy.sqrt(prods[2])))
    print(f"met = {met}")
    
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-a", "--view-a", type=str, required=True)
    parser.add_argument("-b", "--view-b", type=str, required=True)
    parser.add_argument("-m", "--metric", type=str, required=True,
                        choices=["ncc", "cr", "nip"],
                        help="metric to compute")
    parser.add_argument("-i", "--interp-method", type=str, required=True,
                        choices=["cubspl", "linear", "nearest"],
                        help="interpolation method to use")
    parser.add_argument("-n", "--num-repeat", type=int, required=True,
                        help="# of times kernel is called")
    parser.add_argument("-t", "--output-type", type=str,
                        choices=["float","double"], default="double",
                        help="floating point datatype output by kernel")
    parser.add_argument("-c", "--cub", action="store_true",
                        help="use CUB-based reduction kernel")
    parser.add_argument("-cb", "--cupyx-benchmark", action="store_true",
                        help="use cupyx benchmarking instead of python loop")
    args = parser.parse_args()
    ec = main(
        args.view_a, args.view_b, 
        args.metric, args.interp_method, args.num_repeat, args.output_type,
        args.cupyx_benchmark, args.cub
    )
    exit(ec)