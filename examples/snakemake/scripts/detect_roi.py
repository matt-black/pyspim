#!/usr/bin/env python3
"""
Script to detect ROI and crop data.
Called by Snakemake workflow.
"""

import os
import numpy as np
import json
import tifffile
from pyspim import roi

# Get Snakemake variables
snakemake = globals()["snakemake"]

def main():
    # Create output directory
    os.makedirs(os.path.dirname(snakemake.output.a_cropped), exist_ok=True)
    
    # Load raw data from TIFF files
    a_raw = tifffile.imread(snakemake.input.a_raw)
    b_raw = tifffile.imread(snakemake.input.b_raw)
    
    print(f"Detecting ROI with method: {snakemake.params.method}")
    print(f"Input shapes - A: {a_raw.shape}, B: {b_raw.shape}")
    
    # Detect ROI for both channels
    roia = roi.detect_roi_3d(a_raw, snakemake.params.method)
    roib = roi.detect_roi_3d(b_raw, snakemake.params.method)
    roic = roi.combine_rois(roia, roib)
    
    # Crop data using detected ROI
    a_cropped = a_raw[roic[0][0]:roic[0][1],
                      roic[1][0]:roic[1][1],
                      roic[2][0]:roic[2][1]]
    
    b_cropped = b_raw[roic[0][0]:roic[0][1],
                      roic[1][0]:roic[1][1],
                      roic[2][0]:roic[2][1]]
    
    # Save cropped data as TIFF files
    tifffile.imwrite(snakemake.output.a_cropped, a_cropped)
    tifffile.imwrite(snakemake.output.b_cropped, b_cropped)
    
    # Save ROI coordinates as JSON for reference
    roi_info = {
        "method": snakemake.params.method,
        "roi_a": roia,
        "roi_b": roib,
        "roi_combined": roic,
        "original_shape_a": a_raw.shape,
        "original_shape_b": b_raw.shape,
        "cropped_shape_a": a_cropped.shape,
        "cropped_shape_b": b_cropped.shape
    }
    
    with open(snakemake.output.roi_coords, 'w') as f:
        json.dump(roi_info, f, indent=2)
    
    print(f"ROI detection complete:")
    print(f"  ROI coordinates: {roic}")
    print(f"  Cropped A: {a_cropped.shape} -> {snakemake.output.a_cropped}")
    print(f"  Cropped B: {b_cropped.shape} -> {snakemake.output.b_cropped}")
    print(f"  ROI info: {snakemake.output.roi_coords}")

if __name__ == "__main__":
    main() 