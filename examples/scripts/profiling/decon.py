import os
import time

from argparse import ArgumentParser

import cupy
import zarr
import numpy
from cupy.cuda import memory_hooks

from pyspim.decon.rl import dvfft_nvm as nvm
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
    use_nvm: bool,
    crop: bool,
    dist: bool,
) -> int:
    xp = numpy if use_nvm else cupy
    # read in the volumes
    if crop:
        view_a = zarr.open_array(view_a)[0][:500,:500,:500]
        view_b = zarr.open_array(view_b)[0][:500,:500,:500]
    else:
        view_a = zarr.open_array(view_a)[0]
        view_b = zarr.open_array(view_b)[0]
    if not use_nvm:
        view_a = xp.asarray(view_a, dtype=xp.float32)
        view_b = xp.asarray(view_b, dtype=xp.float32)
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
    print([view_a.shape, psf_a.shape], flush=True)
    times = []
    if use_nvm:        
        for nr in range(num_repeat):
            if nr == num_repeat - 1:
                with memory_hooks.LineProfileHook() as mem_hook:
                    t_start = time.perf_counter()
                    if dist:
                        nvm.additive_distributed(
                            view_a, view_b, 
                            nvm.prep_psf(psf_a, view_a.shape),
                            nvm.prep_psf(psf_b, view_a.shape),
                            nvm.prep_psf(bp_a, view_a.shape),
                            nvm.prep_psf(bp_b, view_a.shape),
                            num_iter, 1e-6, 
                            req_both=True, verbose=True
                        )
                    nvm.additive(view_a, view_b,
                                psf_a, psf_b, bp_a, bp_b,
                                num_iter, 1e-6, 
                                req_both=True, verbose=True)
                    t_end = time.perf_counter()
                    times.append(t_end-t_start)
                    mem_hook.print_report()
            else:
                t_start = time.perf_counter()
                nvm.additive(view_a, view_b,
                                psf_a, psf_b, bp_a, bp_b,
                                num_iter, 1e-6, 
                                req_both=True, verbose=True)
                t_end = time.perf_counter()
                times.append(t_end-t_start)
    else:
        for nr in range(num_repeat):
            if nr == num_repeat - 1:
                with memory_hooks.LineProfileHook() as mem_hook:
                    t_start = time.perf_counter()
                    _ = fun(view_a, view_b, (view_a + view_b) / 2, 
                            psf_a, psf_b, bp_a, bp_b,
                            num_iter, 1e-6, boundary_correction, 
                            req_both=True,
                            verbose=True)
                    t_end = time.perf_counter()
                    times.append(t_end-t_start)
                    mem_hook.print_report()
            else:
                t_start = time.perf_counter()
                fun(view_a, view_b, (view_a + view_b) / 2, 
                    psf_a, psf_b, bp_a, bp_b,
                    num_iter, 1e-6, boundary_correction, 
                    req_both=True,
                    verbose=True)
                t_end = time.perf_counter()
                times.append(t_end - t_start)
    print(numpy.asarray(times))
    print([numpy.mean(times), numpy.std(times)])
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-a", "--view-a", type=str, required=True)
    parser.add_argument("-b", "--view-b", type=str, required=True)
    parser.add_argument("-pa", "--psf-a", type=str, required=True)
    parser.add_argument("-pb", "--psf-b", type=str, required=True)
    parser.add_argument("-f", "--algorithm", type=str, default="additive",
                        choices=["shroff","efficient","additive"])
    parser.add_argument("-c", "--boundary-correction", action="store_true")
    parser.add_argument("-i", "--num-iter", type=int, required=True)
    parser.add_argument("-n", "--num-repeat", type=int, required=True)
    parser.add_argument("--nvm", action="store_true", default=False,
                        help="use nvm deconvolution")
    parser.add_argument("--crop", action="store_true", default=False)
    parser.add_argument("--dist", action="store_true")
    args = parser.parse_args()
    ec = main(
        args.view_a, args.view_b, args.psf_a, args.psf_b,
        args.algorithm, args.boundary_correction, args.num_iter,
        args.num_repeat, args.nvm, args.crop, args.dist
    )
    exit(ec)