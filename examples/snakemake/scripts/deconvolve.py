#!/usr/bin/env python3
"""
Script to deconvolve deskewed data using dual-view deconvolution with GPU acceleration.
Called by Snakemake workflow.
"""

import os
import numpy as np
import tifffile

# Try to use GPU acceleration with CuPy
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration available with CuPy")
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False
    print("CuPy not available, using CPU (NumPy)")

from pyspim.decon.rl.dualview_fft import deconvolve

# Get Snakemake variables
snakemake = globals()["snakemake"]

def main():
    # Create output directory
    os.makedirs(os.path.dirname(snakemake.output.deconvolved), exist_ok=True)
    
    # Load deskewed data from TIFF files
    a_deskewed = tifffile.imread(snakemake.input.a_deskewed)
    b_deskewed = tifffile.imread(snakemake.input.b_deskewed)
    
    # Load PSF files
    psf_a = np.load(snakemake.input.psf_a)
    psf_b = np.load(snakemake.input.psf_b)
    
    print(f"Deconvolving dual-view data:")
    print(f"  Channel A shape: {a_deskewed.shape}")
    print(f"  Channel B shape: {b_deskewed.shape}")
    print(f"  PSF A shape: {psf_a.shape}")
    print(f"  PSF B shape: {psf_b.shape}")
    print(f"  Iterations: {snakemake.params.iterations}")
    print(f"  GPU acceleration: {'Yes' if GPU_AVAILABLE else 'No'}")
    
    # Convert to GPU arrays if available
    if GPU_AVAILABLE:
        a_deskewed = cp.asarray(a_deskewed, dtype=cp.float32)
        b_deskewed = cp.asarray(b_deskewed, dtype=cp.float32)
        psf_a = cp.asarray(psf_a, dtype=cp.float32)
        psf_b = cp.asarray(psf_b, dtype=cp.float32)
    
    # Perform dual-view deconvolution
    # Note: The function signature is:
    # deconvolve(view_a, view_b, est_i, psf_a, psf_b, backproj_a, backproj_b, 
    #           decon_function, num_iter, epsilon, req_both, boundary_correction, 
    #           zero_padding, boundary_sigma_a, boundary_sigma_b, verbose)
    deconvolved = deconvolve(
        a_deskewed, 
        b_deskewed,
        None,  # Initial estimate (will use (A+B)/2)
        psf_a, 
        psf_b,
        None,  # Backprojector A (will use mirrored PSF)
        None,  # Backprojector B (will use mirrored PSF)
        "additive",  # Deconvolution function
        snakemake.params.iterations,
        1e-5,  # epsilon
        True,  # req_both
        False,  # boundary_correction (disabled to avoid bug)
        None,  # zero_padding
        1e-2,  # boundary_sigma_a
        1e-2,  # boundary_sigma_b
        True   # verbose
    )
    
    # Convert back to CPU if using GPU
    try:
        deconvolved = deconvolved.get()
    except AttributeError:
        deconvolved = np.asarray(deconvolved)
    
    # Save deconvolved result as TIFF file
    tifffile.imwrite(snakemake.output.deconvolved, deconvolved)
    
    print(f"Deconvolution complete:")
    print(f"  Output shape: {deconvolved.shape}")
    print(f"  Saved to: {snakemake.output.deconvolved}")

if __name__ == "__main__":
    main() 