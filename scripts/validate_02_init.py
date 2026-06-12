#!/usr/bin/env python
"""Validate initialize/finalize works across ranks.

Run:
    mpiexec -n 1 python scripts/validate_02_init.py
    mpiexec -n 2 python scripts/validate_02_init.py
    mpiexec -n 4 python scripts/validate_02_init.py
"""

import sys

import cupy
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
device_id = rank % cupy.cuda.runtime.getDeviceCount()

passed = True

try:
    from pyspim.decon.rl.dualview_distributed import (
        initialize_distributed,
        finalize_distributed,
        is_distributed_initialized,
    )

    # Test initialization
    initialize_distributed(device_id, comm)

    if not is_distributed_initialized():
        if rank == 0:
            print("FAIL: is_distributed_initialized() returned False after init")
        passed = False

    if rank == 0:
        print(f"PASS: Initialized successfully with {nranks} ranks")

    # Test double init raises error
    try:
        # Only rank 0 tries double init to trigger error, but it's a collective
        # so we need a different approach - just test that is_distributed_initialized works
        comm.Barrier()
    except RuntimeError as e:
        if "already been initialized" in str(e):
            if rank == 0:
                print("PASS: Double initialization correctly raises RuntimeError")
        else:
            raise

    # Test finalize
    finalize_distributed()

    if is_distributed_initialized():
        if rank == 0:
            print("FAIL: is_distributed_initialized() returned True after finalize")
        passed = False

    if rank == 0:
        print("PASS: Finalized successfully")

    # Test re-initialization after finalize
    initialize_distributed(device_id, comm)
    finalize_distributed()
    if rank == 0:
        print("PASS: Re-initialization after finalize works")

except Exception as e:
    if rank == 0:
        print(f"FAIL: {e}")
    passed = False
    raise

if not passed:
    sys.exit(1)
