#!/usr/bin/env python
"""Diagnostic script to isolate the source of convolution error.

Tests each step of the distributed FFT convolution pipeline independently.

Run:
    mpiexec -n 1 python scripts/diagnose_convolve.py
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

# Generate reproducible test data
np.random.seed(42)
psf = np.random.randn(*psf_shape).astype(np.float32)
signal_full = np.random.randn(*global_shape).astype(np.float32)

# Initialize distributed runtime
initialize_distributed(device_id, comm)

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} {detail}")
        failed += 1

# ============================================================
# TEST 1: Simple FFT roundtrip (forward + inverse)
# ============================================================
print("\n=== TEST 1: FFT Roundtrip ===")

local_shape = _get_local_shape(global_shape, rank, nranks, partition_dim=0)
arr = nvmath.distributed.allocate_symmetric_memory(
    local_shape, cp, dtype=cp.float32
)
global_offset = _get_global_offset(global_shape, rank, nranks, partition_dim=0)
local_signal = signal_full[
    global_offset:global_offset + local_shape[0], :, :
]
arr[:] = cp.asarray(local_signal)

# Allocate complex symmetric buffer for FFT
if arr.dtype == cp.float32:
    complex_dtype = cp.complex64
else:
    complex_dtype = cp.complex128

temp_buf = nvmath.distributed.allocate_symmetric_memory(
    arr.shape, cp, dtype=complex_dtype
)
temp_buf[:] = arr

fft_options = {"reshape": True}
operand_fft = nvmath.distributed.fft.fft(
    temp_buf, distribution=Slab.X, options=fft_options
)
roundtrip = nvmath.distributed.fft.ifft(
    operand_fft, distribution=Slab.X, options=fft_options
)
roundtrip = cp.real(roundtrip)

with cp.cuda.Device(device_id):
    cp.cuda.get_current_stream().synchronize()

roundtrip_np = roundtrip.get()
local_err = np.max(np.abs(local_signal - roundtrip_np))
print(f"  Local max error after FFT roundtrip: {local_err:.2e}")
check("FFT roundtrip", local_err < 1e-5, f"(error={local_err:.2e})")

nvmath.distributed.free_symmetric_memory(temp_buf)
nvmath.distributed.free_symmetric_memory(arr)

# ============================================================
# TEST 2: Compare nvmath FFT vs cupy FFT
# ============================================================
print("\n=== TEST 2: nvmath FFT vs CuPy FFT ===")

arr2 = nvmath.distributed.allocate_symmetric_memory(
    local_shape, cp, dtype=cp.float32
)
arr2[:] = cp.asarray(local_signal)

temp2 = nvmath.distributed.allocate_symmetric_memory(
    arr2.shape, cp, dtype=complex_dtype
)
temp2[:] = arr2

nv_fft = nvmath.distributed.fft.fft(
    temp2, distribution=Slab.X, options=fft_options
)

# CuPy FFT on the same local data (but this is NOT a distributed FFT - just local)
cp_fft = cp.fft.fftn(cp.asarray(local_signal).astype(complex_dtype))

with cp.cuda.Device(device_id):
    cp.cuda.get_current_stream().synchronize()

nv_fft_np = nv_fft.get()
cp_fft_np = cp_fft.get()

# For nranks=1, these should match
if nranks == 1:
    fft_err = np.max(np.abs(nv_fft_np - cp_fft_np))
    fft_rel_err = fft_err / (np.max(np.abs(cp_fft_np)) + 1e-16)
    print(f"  Max absolute error (nvmath vs cupy FFT): {fft_err:.2e}")
    print(f"  Relative error: {fft_rel_err:.2e}")
    check("nvmath FFT matches cupy FFT", fft_rel_err < 1e-5,
          f"(rel_err={fft_rel_err:.2e})")

nvmath.distributed.free_symmetric_memory(temp2)
nvmath.distributed.free_symmetric_memory(arr2)

# ============================================================
# TEST 3: Test with reshape=False (default nvmath pattern)
# ============================================================
print("\n=== TEST 3: Convolution with reshape=False ===")

arr3 = nvmath.distributed.allocate_symmetric_memory(
    local_shape, cp, dtype=cp.float32
)
arr3[:] = cp.asarray(local_signal)

temp3 = nvmath.distributed.allocate_symmetric_memory(
    arr3.shape, cp, dtype=complex_dtype
)
temp3[:] = arr3

# Forward FFT with reshape=False (output is Slab.Y)
fft_options_false = {"reshape": False}
operand_fft_y = nvmath.distributed.fft.fft(
    temp3, distribution=Slab.X, options=fft_options_false
)

# Pre-compute PSF FFT
psf_fft, _, _, _ = _precompute_psf_ffts(
    psf, psf, None, None, global_shape, np.float32
)

# Element-wise multiply (result has Slab.Y strides)
product = operand_fft_y * psf_fft

# Inverse FFT with reshape=False (input is Slab.Y, output is Slab.X)
result_slabx = nvmath.distributed.fft.ifft(
    product, distribution=Slab.Y, options=fft_options_false
)
result_slabx = cp.real(result_slabx)

with cp.cuda.Device(device_id):
    cp.cuda.get_current_stream().synchronize()

# Gather and compare
result3_np = result_slabx.get()
result3_full = np.zeros(global_shape, dtype=np.float32)
for r in range(nranks):
    r_local_shape = _get_local_shape(global_shape, r, nranks, partition_dim=0)
    r_offset = _get_global_offset(global_shape, r, nranks, partition_dim=0)
    if r == rank:
        data_to_send = result3_np
    else:
        data_to_send = np.zeros(r_local_shape, dtype=np.float32)
    gathered = comm.allgather((data_to_send, r_local_shape, r_offset))
for data, lshape, offset in gathered:
    result3_full[offset:offset + lshape[0], :, :] = data

from scipy.signal import fftconvolve
expected = fftconvolve(signal_full, psf, mode="same")
max_err3 = np.max(np.abs(expected - result3_full))
rel_err3 = max_err3 / (np.max(np.abs(expected)) + 1e-16)
print(f"  Max absolute error (reshape=False): {max_err3:.6e}")
print(f"  Relative error: {rel_err3:.6e}")
check("Convolution with reshape=False", rel_err3 < 1e-4,
      f"(rel_err={rel_err3:.6e})")

nvmath.distributed.free_symmetric_memory(temp3)
nvmath.distributed.free_symmetric_memory(arr3)

# ============================================================
# TEST 4: Test current implementation (reshape=True)
# ============================================================
print("\n=== TEST 4: Convolution with reshape=True (current) ===")

arr4 = nvmath.distributed.allocate_symmetric_memory(
    local_shape, cp, dtype=cp.float32
)
arr4[:] = cp.asarray(local_signal)

result4 = _distributed_fft_convolve(psf_fft, arr4, Slab.X)

with cp.cuda.Device(device_id):
    cp.cuda.get_current_stream().synchronize()

result4_np = result4.get()
result4_full = np.zeros(global_shape, dtype=np.float32)
for r in range(nranks):
    r_local_shape = _get_local_shape(global_shape, r, nranks, partition_dim=0)
    r_offset = _get_global_offset(global_shape, r, nranks, partition_dim=0)
    if r == rank:
        data_to_send = result4_np
    else:
        data_to_send = np.zeros(r_local_shape, dtype=np.float32)
    gathered = comm.allgather((data_to_send, r_local_shape, r_offset))
for data, lshape, offset in gathered:
    result4_full[offset:offset + lshape[0], :, :] = data

max_err4 = np.max(np.abs(expected - result4_full))
rel_err4 = max_err4 / (np.max(np.abs(expected)) + 1e-16)
print(f"  Max absolute error (reshape=True): {max_err4:.6e}")
print(f"  Relative error: {rel_err4:.6e}")
check("Convolution with reshape=True", rel_err4 < 1e-4,
      f"(rel_err={rel_err4:.6e})")

nvmath.distributed.free_symmetric_memory(arr4)

# ============================================================
# TEST 5: Check normalization - compare with manual CuPy approach
# ============================================================
print("\n=== TEST 5: Manual CuPy convolution for reference ===")

signal_cp = cp.asarray(signal_full)
psf_cp = cp.asarray(psf)

# Manual FFT convolution using CuPy (circular, same size)
signal_fft_manual = cp.fft.fftn(signal_cp.astype(complex_dtype))
psf_padded = np.zeros(global_shape, dtype=np.float32)
slices = tuple(slice(0, k) for k in psf.shape)
psf_padded[slices] = psf
psf_fft_manual = cp.fft.fftn(cp.asarray(psf_padded))

# Convolution: IFFT(FFT(signal) * FFT(psf))
# CuPy's ifft applies 1/N automatically
result_manual = cp.real(cp.fft.iftn(signal_fft_manual * psf_fft_manual))

max_err5 = np.max(np.abs(expected - result_manual.get()))
rel_err5 = max_err5 / (np.max(np.abs(expected)) + 1e-16)
print(f"  Max abs error (cupy circular conv vs scipy linear conv): {max_err5:.6e}")
print(f"  Relative error: {rel_err5:.6e}")
print(f"  Note: Circular convolution differs from scipy's linear fftconvolve")
check("Circular conv reference captured", True)

# ============================================================
# TEST 6: Check if the normalization factor is correct
# ============================================================
print("\n=== TEST 6: Normalization check ===")

n_elements = np.prod(global_shape)
print(f"  Global shape: {global_shape}")
print(f"  N elements: {n_elements}")

# Test: ifft(fft(x)) should give back x
# With cuFFT: ifft applies 1/N, so roundtrip should work
# If we additionally divide by N, we get x/N

arr6 = nvmath.distributed.allocate_symmetric_memory(
    local_shape, cp, dtype=cp.float32
)
arr6[:] = cp.asarray(local_signal)

temp6 = nvmath.distributed.allocate_symmetric_memory(
    arr6.shape, cp, dtype=complex_dtype
)
temp6[:] = arr6

operand_fft6 = nvmath.distributed.fft.fft(
    temp6, distribution=Slab.X, options=fft_options
)
inv6 = nvmath.distributed.fft.ifft(
    operand_fft6, distribution=Slab.X, options=fft_options
)
# WITHOUT manual 1/N
raw_roundtrip = cp.real(inv6).get()
with cp.cuda.Device(device_id):
    cp.cuda.get_current_stream().synchronize()

raw_err = np.max(np.abs(local_signal - raw_roundtrip))
print(f"  Raw roundtrip error (no manual 1/N): {raw_err:.2e}")

inv6_scaled = inv6 / n_elements
scaled_roundtrip = cp.real(inv6_scaled).get()
scaled_err = np.max(np.abs(local_signal - scaled_roundtrip))
print(f"  Scaled roundtrip error (with manual 1/N): {scaled_err:.2e}")

if raw_err < 1e-5:
    print("  CONCLUSION: cuFFT already applies 1/N in ifft. Manual 1/N is WRONG.")
    check("cuFFT applies 1/N automatically", True)
elif scaled_err < 1e-5:
    print("  CONCLUSION: cuFFT does NOT apply 1/N. Manual 1/N is needed.")
    check("cuFFT does NOT apply 1/N", True)
else:
    print(f"  CONCLUSION: Neither works perfectly. raw_err={raw_err:.2e}, scaled_err={scaled_err:.2e}")
    check("Normalization ambiguous", False)

nvmath.distributed.free_symmetric_memory(temp6)
nvmath.distributed.free_symmetric_memory(arr6)

# ============================================================
# SUMMARY
# ============================================================
print(f"\n=== SUMMARY: {passed} passed, {failed} failed ===")

# Cleanup
finalize_distributed()

if failed > 0:
    sys.exit(1)
