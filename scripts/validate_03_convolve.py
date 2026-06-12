#!/usr/bin/env python
"""Validate distributed FFT convolution against scipy reference.

This test distributes a signal across ranks, performs FFT-based convolution
using the distributed helper, and compares the result with scipy's fftconvolve.

Run:
    mpiexec -n 1 python scripts/validate_03_convolve.py
    mpiexec -n 2 python scripts/validate_03_convolve.py
"""

import sys

import cupy as cp
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
device_id = rank % cp.cuda.runtime.getDeviceCount()

from pyspim.decon.rl.dualview_distributed import (
    initialize_distributed,
    finalize_distributed,
    _distributed_fft_convolve,
    _precompute_psf_ffts,
    get_slab,
    _get_local_shape,
    _get_global_offset,
)

Slab = get_slab()

# Test parameters
global_shape = (64, 64, 64)
psf_shape = (7, 7, 7)
np.random.seed(42 + rank)  # Same seed for reproducibility

# Generate test data
psf = np.random.randn(*psf_shape).astype(np.float32)

# Generate reference signal on rank 0 and broadcast
if rank == 0:
    np.random.seed(42)
    signal_full = np.random.randn(*global_shape).astype(np.float32)
else:
    signal_full = np.zeros(global_shape, dtype=np.float32)

signal_full = cp.asarray(signal_full)
comm.Bcast(signal_full)
signal_full = signal_full.get()

# Initialize distributed runtime
initialize_distributed(device_id, comm)

# Allocate distributed operand on symmetric heap
local_shape = _get_local_shape(global_shape, rank, nranks, partition_dim=0)
operand = nvmath.distributed.allocate_symmetric_memory(
    local_shape, cp, dtype=cp.float32
)

# Fill operand with the local portion of the signal
global_offset = _get_global_offset(global_shape, rank, nranks, partition_dim=0)
local_signal = signal_full[
    global_offset:global_offset + local_shape[0], :, :
]
operand[:] = cp.asarray(local_signal)

# Pre-compute PSF FFT (replicated on all ranks)
psf_fft, _, _, _ = _precompute_psf_ffts(
    psf, psf, None, None, global_shape, np.float32
)

# Perform distributed convolution
result = _distributed_fft_convolve(psf_fft, operand, Slab.X)

# Synchronize
with cp.cuda.Device(device_id):
    cp.cuda.get_current_stream().synchronize()

# Gather result to rank 0 for comparison
result_np = result.get()
result_full = np.zeros(global_shape, dtype=np.float32)

# All-gather the results back
for r in range(nranks):
    r_local_shape = _get_local_shape(global_shape, r, nranks, partition_dim=0)
    r_offset = _get_global_offset(global_shape, r, nranks, partition_dim=0)
    if r == rank:
        # Send my data
        data_to_send = result_np
    else:
        data_to_send = np.zeros(r_local_shape, dtype=np.float32)

    gathered = comm.allgather((data_to_send, r_local_shape, r_offset))

for data, lshape, offset in gathered:
    result_full[offset:offset + lshape[0], :, :] = data

# Compute reference convolution
from scipy.signal import fftconvolve

expected = fftconvolve(signal_full, psf, mode="same")

# Compare on all ranks (all ranks have the full result after allgather)
max_err = np.max(np.abs(expected - result_full))
rel_err = max_err / (np.max(np.abs(expected)) + 1e-16)

if rank == 0:
    print(f"Global shape: {global_shape}")
    print(f"PSF shape: {psf_shape}")
    print(f"Nranks: {nranks}")
    print(f"Max absolute error: {max_err:.6e}")
    print(f"Relative error: {rel_err:.6e}")
    if rel_err < 1e-4:
        print("PASS: Distributed convolution matches scipy reference")
    else:
        print(f"FAIL: Relative error {rel_err:.6e} exceeds threshold 1e-4")
        sys.exit(1)

# Cleanup
nvmath.distributed.free_symmetric_memory(operand)
finalize_distributed()

import nvmath.distributed
