#!/usr/bin/env python
"""Validate that dualview_distributed module imports correctly.

Run:
    mpiexec -n 1 python scripts/validate_01_imports.py
"""

import sys

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

passed = True

try:
    from pyspim.decon.rl.dualview_distributed import (
        initialize_distributed,
        finalize_distributed,
        is_distributed_initialized,
        _distributed_fft_convolve,
        _precompute_psf_ffts,
        _distributed_efficient_bayesian_backprojectors,
        _distributed_sum,
        _div_stable_distributed,
        _get_local_shape,
        _get_global_offset,
        get_slab,
    )
    if rank == 0:
        print("PASS: All symbols imported successfully")
except ImportError as e:
    if rank == 0:
        print(f"FAIL: Import error: {e}")
    passed = False

if not passed:
    sys.exit(1)
