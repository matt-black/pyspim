import os
import time
from argparse import ArgumentParser

import cupy
import zarr
import numpy
from cupyx.profiler import benchmark

from pyspim.reg import kernels as K 
from pyspim.typing import NDArray
from pyspim.util import launch_params_for_volume


def main(
    view_a: os.PathLike,
    view_b: os.PathLike,
    metric: str,
    interp_method: str,
    num_repeat: int,
    output_type: str,
    use_cupyx_benchmark: bool,
) -> int:
    # read in the volumes
    a = zarr.open_array(view_a)[0]
    b = zarr.open_array(view_b)[0]
    num_gpus = cupy.cuda.runtime.getDeviceCount()
    print(f"n_GPUs: {num_gpus:d}")
    # make launch parameters
    launch_pars = launch_params_for_volume(a.shape, 8, 8, 8) # type: ignore

    sz_r, sy_r, sx_r = a.shape # type: ignore
    sz_m, sy_m, sx_m = b.shape # type: ignore
    # move moving/ref images to GPU (default to 0 device)
    with cupy.cuda.Device(0):
        ref = cupy.asarray(a, dtype=cupy.uint16)
        mu_ref = cupy.mean(ref).astype(cupy.float32)
        mov = cupy.asarray(b, dtype=cupy.uint16)
        mu_mov = cupy.mean(mov).astype(cupy.float32)
    # initialize the computation by generating kernels, streams, args
    comp_start = time.perf_counter()
    kernels, streams, kernel_args = K.initialize_computation(
        metric, interp_method, ref, mov, mu_ref, mu_mov,
        sz_r, sy_r, sx_r, sz_m, sy_m, sx_m
    )
    T_init = kernel_args[0][1].get()

    def compute(T: NDArray) -> float:
        return K.compute(T, kernels, streams, kernel_args, launch_pars)
    
    comp_end = time.perf_counter()
    comp_time = (comp_end - comp_start) * 1000.0
    print(f"Took {comp_time:.2f} ms to initialize kernels, arrays, etc.",
          flush=True)
    # do a single computation, just to test
    comp_start = time.perf_counter()
    met = compute(T_init)
    comp_end = time.perf_counter()
    comp_time = (comp_end - comp_start) * 1000.0
    print(f"Took {comp_time:.2f} ms to do first computation", flush=True)
    if use_cupyx_benchmark:
        perf_case = benchmark(
            compute,
            (T_init,),
            n_repeat=num_repeat
        )
        print(perf_case, flush=True)
    else:
        met = numpy.nan
        comp_times = numpy.empty((num_repeat,))
        for r in range(num_repeat):
            comp_start = time.perf_counter()
            met = compute(T_init)
            comp_end = time.perf_counter()
            comp_times[r] = comp_end - comp_start
        avg_time = numpy.mean(comp_times) * 1000.
        std_time = numpy.std(comp_times) * 1000.
        ste_time = std_time / numpy.sqrt(num_repeat)
        print(f"Computed value: {met:.2f}", flush=True)
        print(
            f"{num_repeat:d} iters, avg: {avg_time:.2f} +/- {ste_time:.2f} ms",
            flush=True
        )
    
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
    parser.add_argument("-cb", "--cupyx-benchmark", action="store_true",
                        help="use cupyx benchmarking instead of python loop")
    args = parser.parse_args()
    ec = main(
        args.view_a, args.view_b, 
        args.metric, args.interp_method, args.num_repeat, args.output_type,
        args.cupyx_benchmark
    )
    exit(ec)