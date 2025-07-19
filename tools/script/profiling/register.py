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
    b = zarr.open_array(view_a)[0]
    
    T = cupy.concatenate([cupy.eye(3), cupy.zeros((3,1))], axis=1).astype(cupy.float32)
    # launch_pars = launch_params_for_volume(a.shape, 8, 8, 8)
    launch_pars = launch_params_for_volume(a.shape, 32, 4, 4)
    ref = cupy.asarray(a, dtype=cupy.uint16)
    mov = cupy.asarray(b, dtype=cupy.uint16)
    mu_ref = cupy.mean(ref).astype(cupy.float32).item()
    for i in range(8):
        met = cubspl.correlation_ratio(
                T, ref, mov, mu_ref,
                *ref.shape, *mov.shape, launch_pars, output_type
            )
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