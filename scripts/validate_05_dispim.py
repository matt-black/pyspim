#!/usr/bin/env python
"""Validate distributed diSPIM joint RL (uncorrected).

This test compares the distributed diSPIM implementation against the reference
single-GPU implementation from dualview_fft.joint_rl_dispim with
boundary_correction=False.

Run:
    mpiexec -n 1 python scripts/validate_05_dispim.py
    mpiexec -n 2 python scripts/validate_05_dispim.py
"""

import sys

import cupy as cp
import nvmath.distributed
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
device_id = rank % cp.cuda.runtime.getDeviceCount()

from pyspim.decon.rl.dualview_distributed import (
    initialize_distributed,
    finalize_distributed,
    distributed_joint_rl_dispim,
    _get_local_shape,
    _get_global_offset,
)
from pyspim.decon.rl.dualview_fft import joint_rl_dispim as reference_joint_rl_dispim

# Test parameters - small shape for faster validation
global_shape = (32, 32, 32)
psf_shape = (7, 7, 7)
num_iter = 3

# Generate test data on rank 0 and broadcast
np.random.seed(42)
view_a_np = np.abs(np.random.randn(*global_shape)).astype(np.float32)
view_b_np = np.abs(np.random.randn(*global_shape)).astype(np.float32)
psf_a_np = np.abs(np.random.randn(*psf_shape)).astype(np.float32)
psf_b_np = np.abs(np.random.randn(*psf_shape)).astype(np.float32)

# Broadcast to all ranks
for arr in [view_a_np, view_b_np, psf_a_np, psf_b_np]:
    if rank == 0:
        comm.Bcast(arr, root=0)
    else:
        comm.Bcast(arr, root=0)

# -----------------------------------------------------------
# Reference computation (single GPU, uncorrected)
# -----------------------------------------------------------
if rank == 0:
    print("=" * 60)
    print("Reference: joint_rl_dispim (uncorrected)")
    print("=" * 60)
    print(f"  Global shape: {global_shape}")
    print(f"  PSF shapes: {psf_a_np.shape}, {psf_b_np.shape}")
    print(f"  Num iterations: {num_iter}")

ref_result = reference_joint_rl_dispim(
    cp.asarray(view_a_np), cp.asarray(view_b_np), None,
    cp.asarray(psf_a_np), cp.asarray(psf_b_np),
    num_iter=num_iter, boundary_correction=False,
)
ref_result_np = ref_result.get()

if rank == 0:
    print(f"  Reference result shape: {ref_result_np.shape}")
    print(f"  Reference result range: [{ref_result_np.min():.6f}, {ref_result_np.max():.6f}]")

# -----------------------------------------------------------
# Distributed computation
# -----------------------------------------------------------
if rank == 0:
    print()
    print("=" * 60)
    print("Distributed: distributed_joint_rl_dispim (uncorrected)")
    print("=" * 60)

# Initialize distributed runtime
initialize_distributed(device_id, comm)

# Allocate distributed views on symmetric heap
local_shape = _get_local_shape(global_shape, rank, nranks, partition_dim=0)
global_offset = _get_global_offset(global_shape, rank, nranks, partition_dim=0)

view_a_dist = nvmath.distributed.allocate_symmetric_memory(
    local_shape, cp, dtype=cp.float32
)
view_b_dist = nvmath.distributed.allocate_symmetric_memory(
    local_shape, cp, dtype=cp.float32
)

# Fill with local portion of the data
view_a_dist[:] = cp.asarray(
    view_a_np[global_offset:global_offset + local_shape[0], :, :]
)
view_b_dist[:] = cp.asarray(
    view_b_np[global_offset:global_offset + local_shape[0], :, :]
)

# Run distributed deconvolution
dist_result = distributed_joint_rl_dispim(
    view_a_dist, view_b_dist, None,
    psf_a_np, psf_b_np,
    num_iter=num_iter, boundary_correction=False,
)

# Synchronize
with cp.cuda.Device(device_id):
    cp.cuda.get_current_stream().synchronize()

# -----------------------------------------------------------
# Gather result to rank 0 for comparison
# -----------------------------------------------------------
dist_result_np = dist_result.get()
dist_result_full = np.zeros(global_shape, dtype=np.float32)

# All-gather the results
for r in range(nranks):
    r_local_shape = _get_local_shape(global_shape, r, nranks, partition_dim=0)
    r_offset = _get_global_offset(global_shape, r, nranks, partition_dim=0)
    if r == rank:
        data_to_send = dist_result_np
    else:
        data_to_send = np.zeros(r_local_shape, dtype=np.float32)

    gathered = comm.allgather((data_to_send, r_local_shape, r_offset))

for data, lshape, offset in gathered:
    dist_result_full[offset:offset + lshape[0], :, :] = data

# -----------------------------------------------------------
# Compare results
# -----------------------------------------------------------
max_err = np.max(np.abs(ref_result_np - dist_result_full))
rel_err = max_err / (np.max(np.abs(ref_result_np)) + 1e-16)

if rank == 0:
    print(f"  Distributed result shape: {dist_result_full.shape}")
    print(f"  Distributed result range: [{dist_result_full.min():.6f}, {dist_result_full.max():.6f}]")
    print()
    print("=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"  Nranks: {nranks}")
    print(f"  Max absolute error: {max_err:.6e}")
    print(f"  Relative error: {rel_err:.6e}")

    # Threshold: 1e-4 relative error
    if rel_err < 1e-4:
        print("  PASS: Distributed diSPIM matches reference")
    else:
        print(f"  FAIL: Relative error {rel_err:.6e} exceeds threshold 1e-4")
        sys.exit(1)

# Cleanup
nvmath.distributed.free_symmetric_memory(view_a_dist)
nvmath.distributed.free_symmetric_memory(view_b_dist)
nvmath.distributed.free_symmetric_memory(dist_result)
finalize_distributed()
