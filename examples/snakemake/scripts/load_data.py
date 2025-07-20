#!/usr/bin/env python3
"""
Script to load and preprocess raw data from the fruiting body subset example.
Called by Snakemake workflow.
"""

import os
import numpy as np
import tifffile
from pathlib import Path

# Get Snakemake variables
snakemake = globals()["snakemake"]

def subtract_constant_uint16arr(arr, constant):
    """Safely subtract constant from uint16 array."""
    result = arr.astype(np.int32) - constant
    return np.clip(result, 0, 65535).astype(np.uint16)

def main():
    # Create output directory
    os.makedirs(os.path.dirname(snakemake.output.a_raw), exist_ok=True)
    
    # Load data from TIFF file (fruiting body subset)
    data_path = snakemake.params.data_path
    tiff_file = os.path.join(data_path, "fruiting_body_subset.ome.tif")
    
    print(f"Loading data from: {tiff_file}")
    
    # Load the TIFF file
    data = tifffile.imread(tiff_file)
    
    # The fruiting body subset has shape (2, Z, Y, X) where first dimension is channels
    # Channel 0 = A, Channel 1 = B
    a_raw = data[0, :, :, :]  # Channel A
    b_raw = data[1, :, :, :]  # Channel B
    
    # Subtract camera offset (from acquisition_params.txt)
    camera_offset = 100
    a_raw = subtract_constant_uint16arr(a_raw, camera_offset)
    b_raw = subtract_constant_uint16arr(b_raw, camera_offset)
    
    # Save raw data as TIFF files
    tifffile.imwrite(snakemake.output.a_raw, a_raw)
    tifffile.imwrite(snakemake.output.b_raw, b_raw)
    
    print(f"Saved raw data:")
    print(f"  Channel A: {a_raw.shape} -> {snakemake.output.a_raw}")
    print(f"  Channel B: {b_raw.shape} -> {snakemake.output.b_raw}")

if __name__ == "__main__":
    main() 