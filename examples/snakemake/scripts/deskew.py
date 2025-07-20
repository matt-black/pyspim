#!/usr/bin/env python3
"""
Script to deskew diSPIM data using GPU acceleration.
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

from pyspim import deskew as dsk

# Get Snakemake variables
snakemake = globals()["snakemake"]

def main():
    # Create output directory
    os.makedirs(os.path.dirname(snakemake.output.deskewed), exist_ok=True)
    
    # Load cropped data
    data = tifffile.imread(snakemake.input.cropped)
    
    print(f"Deskewing data:")
    print(f"  Input shape: {data.shape}")
    print(f"  Pixel size: {snakemake.params.pixel_size}")
    print(f"  Step size (lateral): {snakemake.params.step_size_lat}")
    print(f"  Direction: {snakemake.params.direction}")
    print(f"  GPU acceleration: {'Yes' if GPU_AVAILABLE else 'No'}")
    
    # Convert to float32 for processing
    data = data.astype(np.float32)
    
    # Perform deskewing (using CPU for now due to pyspim CuPy compatibility issues)
    deskewed = dsk.deskew_stage_scan(
        data,
        snakemake.params.pixel_size,
        snakemake.params.step_size_lat,
        snakemake.params.direction,
        method='orthogonal'
    )
    
    # Ensure output is numpy array
    deskewed = np.asarray(deskewed)
    
    # Save deskewed result
    tifffile.imwrite(snakemake.output.deskewed, deskewed)
    
    print(f"Deskewing complete:")
    print(f"  Output shape: {deskewed.shape}")
    print(f"  Saved to: {snakemake.output.deskewed}")

if __name__ == "__main__":
    main() 