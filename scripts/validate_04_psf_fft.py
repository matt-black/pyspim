#!/usr/bin/env python
"""Validate PSF FFT precomputation and efficient Bayesian backprojectors.

This test validates that the distributed PSF FFT precomputation and
Bayesian backprojector functions produce results matching the reference
implementations.

Run:
    mpiexec -n 1 python scripts/validate_04_psf_fft.py
    mpiexec -n 2 python scripts/validate_04_psf_fft.py
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
    _precompute_psf_ffts,
    _distributed_efficient_bayesian_backprojectors,
)

# Test parameters
global_shape = (64, 64, 64)
psf_shape = (7, 7, 7)

# Generate test data on rank 0 and broadcast
np.random.seed(42)
psf_a = np.abs(np.random.randn(*psf_shape)).astype(np.float32)
psf_b = np.abs(np.random.randn(*psf_shape)).astype(np.float32)

# Initialize distributed runtime
initialize_distributed(device_id, comm)

# Synchronize to ensure all ranks have initialized
comm.Barrier()

# -----------------------------------------------------------
# Test 1: Validate _precompute_psf_ffts against cupy.fft.fftn
# -----------------------------------------------------------
if rank == 0:
    print("=" * 60)
    print("Test 1: _precompute_psf_ffts")
    print("=" * 60)

ffts = _precompute_psf_ffts(psf_a, psf_b, None, None, global_shape, np.float32)
psf_a_fft, psf_b_fft, bp_a_fft, bp_b_fft = ffts

# Verify PSF FFT matches cupy.fft.fftn on zero-padded PSF
padded_a = np.zeros(global_shape, dtype=np.float32)
slices_a = tuple(slice(0, k) for k in psf_a.shape)
padded_a[slices_a] = psf_a
expected_psf_a_fft = cp.fft.fftn(cp.asarray(padded_a))
max_err_psf_a = cp.max(cp.abs(psf_a_fft - expected_psf_a_fft)).get()

padded_b = np.zeros(global_shape, dtype=np.float32)
slices_b = tuple(slice(0, k) for k in psf_b.shape)
padded_b[slices_b] = psf_b
expected_psf_b_fft = cp.fft.fftn(cp.asarray(padded_b))
max_err_psf_b = cp.max(cp.abs(psf_b_fft - expected_psf_b_fft)).get()

# Verify backprojector FFTs (default is mirrored PSF)
mirrored_a = np.ascontiguousarray(psf_a[::-1, ::-1, ::-1])
padded_mir_a = np.zeros(global_shape, dtype=np.float32)
slices_ma = tuple(slice(0, k) for k in mirrored_a.shape)
padded_mir_a[slices_ma] = mirrored_a
expected_bp_a_fft = cp.fft.fftn(cp.asarray(padded_mir_a))
max_err_bp_a = cp.max(cp.abs(bp_a_fft - expected_bp_a_fft)).get()

mirrored_b = np.ascontiguousarray(psf_b[::-1, ::-1, ::-1])
padded_mir_b = np.zeros(global_shape, dtype=np.float32)
slices_mb = tuple(slice(0, k) for k in mirrored_b.shape)
padded_mir_b[slices_mb] = mirrored_b
expected_bp_b_fft = cp.fft.fftn(cp.asarray(padded_mir_b))
max_err_bp_b = cp.max(cp.abs(bp_b_fft - expected_bp_b_fft)).get()

if rank == 0:
    print(f"  PSF_A FFT max error: {max_err_psf_a:.2e}")
    print(f"  PSF_B FFT max error: {max_err_psf_b:.2e}")
    print(f"  BP_A FFT max error:  {max_err_bp_a:.2e}")
    print(f"  BP_B FFT max error:  {max_err_bp_b:.2e}")

psf_fft_pass = max_err_psf_a < 1e-6 and max_err_psf_b < 1e-6
bp_fft_pass = max_err_bp_a < 1e-6 and max_err_bp_b < 1e-6

if rank == 0:
    if psf_fft_pass and bp_fft_pass:
        print("  PASS: All FFTs match reference")
    else:
        if not psf_fft_pass:
            print("  FAIL: PSF FFT mismatch")
        if not bp_fft_pass:
            print("  FAIL: Backprojector FFT mismatch")

# -----------------------------------------------------------
# Test 2: Validate custom backprojectors in _precompute_psf_ffts
# -----------------------------------------------------------
if rank == 0:
    print()
    print("=" * 60)
    print("Test 2: _precompute_psf_ffts with custom backprojectors")
    print("=" * 60)

# Use different backprojectors (not mirrored PSFs)
custom_bp_a = np.abs(np.random.randn(5, 5, 5)).astype(np.float32)
custom_bp_b = np.abs(np.random.randn(9, 9, 9)).astype(np.float32)

ffts_custom = _precompute_psf_ffts(
    psf_a, psf_b, custom_bp_a, custom_bp_b, global_shape, np.float32
)
_, _, custom_bp_a_fft, custom_bp_b_fft = ffts_custom

padded_custom_a = np.zeros(global_shape, dtype=np.float32)
slices_ca = tuple(slice(0, k) for k in custom_bp_a.shape)
padded_custom_a[slices_ca] = custom_bp_a
expected_custom_bp_a_fft = cp.fft.fftn(cp.asarray(padded_custom_a))
max_err_custom_a = cp.max(cp.abs(custom_bp_a_fft - expected_custom_bp_a_fft)).get()

padded_custom_b = np.zeros(global_shape, dtype=np.float32)
slices_cb = tuple(slice(0, k) for k in custom_bp_b.shape)
padded_custom_b[slices_cb] = custom_bp_b
expected_custom_bp_b_fft = cp.fft.fftn(cp.asarray(padded_custom_b))
max_err_custom_b = cp.max(cp.abs(custom_bp_b_fft - expected_custom_bp_b_fft)).get()

if rank == 0:
    print(f"  Custom BP_A FFT max error: {max_err_custom_a:.2e}")
    print(f"  Custom BP_B FFT max error: {max_err_custom_b:.2e}")

custom_bp_pass = max_err_custom_a < 1e-6 and max_err_custom_b < 1e-6

if rank == 0:
    if custom_bp_pass:
        print("  PASS: Custom backprojector FFTs match reference")
    else:
        print("  FAIL: Custom backprojector FFT mismatch")

# -----------------------------------------------------------
# Test 3: Validate _distributed_efficient_bayesian_backprojectors
# -----------------------------------------------------------
if rank == 0:
    print()
    print("=" * 60)
    print("Test 3: _distributed_efficient_bayesian_backprojectors")
    print("=" * 60)

bp_a, bp_b = _distributed_efficient_bayesian_backprojectors(
    psf_a, psf_b, np.float32
)

# Compare with reference implementation
from pyspim.decon.rl.dualview_fft import efficient_bayesian_backprojectors

ref_bp_a, ref_bp_b = efficient_bayesian_backprojectors(
    cp.asarray(psf_a), cp.asarray(psf_b)
)

max_err_ba = cp.max(cp.abs(bp_a - ref_bp_a)).get()
max_err_bb = cp.max(cp.abs(bp_b - ref_bp_b)).get()

if rank == 0:
    print(f"  BP_A max error: {max_err_ba:.2e}")
    print(f"  BP_B max error: {max_err_bb:.2e}")

bp_pass = max_err_ba < 1e-4 and max_err_bb < 1e-4

if rank == 0:
    if bp_pass:
        print("  PASS: Bayesian backprojectors match reference")
    else:
        print("  FAIL: Bayesian backprojectors mismatch")

# -----------------------------------------------------------
# Final Summary
# -----------------------------------------------------------
# Use allreduce to propagate pass/fail across all ranks
local_pass = 1 if (psf_fft_pass and bp_fft_pass and custom_bp_pass and bp_pass) else 0
global_pass = comm.allreduce(local_pass, op=MPI.SUM)

if rank == 0:
    print()
    print("=" * 60)
    print(f"Nranks: {nranks}")
    print(f"Global shape: {global_shape}")
    print(f"PSF shapes: {psf_a.shape}, {psf_b.shape}")
    if global_pass == nranks:
        print("PASS: All tests passed")
    else:
        print("FAIL: Some tests failed")
        sys.exit(1)

finalize_distributed()
