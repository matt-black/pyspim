import os
from argparse import ArgumentParser

import cupy
import zarr
import numpy

from pyspim.decon.rl import dualview_fft as dv


def main(
    view_a: os.PathLike,
    view_b: os.PathLike,
    psf_a_path: os.PathLike,
    psf_b_path: os.PathLike,
    algorithm: str,
    boundary_correction: bool,
    num_iter: int,
    num_repeat: int,
    use_cpu: bool
) -> int:
    xp = numpy if use_cpu else cupy
    # read in the volumes
    view_a = xp.asarray(zarr.open_array(view_a)[0]).astype(xp.float32)
    view_b = xp.asarray(zarr.open_array(view_b)[0]).astype(xp.float32)
    # read in psfs
    psf_a = xp.asarray(numpy.load(psf_a_path)).astype(xp.float32)
    psf_b = xp.asarray(numpy.load(psf_b_path)).astype(xp.float32)
    if algorithm == 'shroff':
        fun = dv.joint_rl_dispim
        bp_a = psf_a[::-1,::-1,::-1]
        bp_b = psf_b[::-1,::-1,::-1]
    elif algorithm == "efficient":
        fun = dv.efficient_bayesian
        bp_a, bp_b = dv.efficient_bayesian_backprojectors(psf_a, psf_b)
    else: # additive
        fun = dv.additive_joint_rl
        bp_a = psf_a[::-1,::-1,::-1]
        bp_b = psf_b[::-1,::-1,::-1]
    for _ in num_repeat:
        _ = fun(view_a, view_b, (view_a + view_b) / 2, 
                psf_a, psf_b, bp_a, bp_b,
                num_iter, 1e-6, boundary_correction, 
                req_both=True,
                verbose=False)
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-a", "--view-a", type=str, required=True)
    parser.add_argument("-b", "--view-b", type=str, required=True)
    parser.add_argument("-pa", "--psf-a", type=str, required=True)
    parser.add_argument("-pb", "--psf-b", type=str, required=True)
    parser.add_argument("-f", "--algorithm", type=str, required=True,
                        choices=["shroff","efficient","additive"])
    parser.add_argument("-c", "--boundary-correction", action="store_true")
    parser.add_argument("-i", "--num-iter", type=int, required=True)
    parser.add_argument("-n", "--num-repeat", type=int, required=True)
    parser.add_argument("--cpu", action="store_true", default=False,
                        help="use CPU (NumPy) arrays")
    args = parser.parse_args()
    ec = main(
        args.view_a, args.view_b,
        args.algorithm, args.boundary_correction, args.num_iter,
        args.num_repeat, args.cpu
    )
    exit(ec)