#!/usr/bin/env python3
"""
Create a subset of the fruiting body data for documentation examples.
This script creates a smaller, manageable dataset that demonstrates the PySPIM workflow.
"""

import numpy as np
import tifffile
from pathlib import Path


def create_fruiting_body_subset() -> None:
    """Create a subset of the fruiting body data for documentation."""
    
    # Source data path
    source_path = Path(
        "examples/data/example_dispim_data/fruiting_body001/"
        "fruiting_body001_MMStack_Pos0.ome.tif"
    )
    
    # Output path for subset
    output_dir = Path("docs/examples/data/fruiting_body_subset")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not source_path.exists():
        print(f"Source file not found: {source_path}")
        print("Please ensure the fruiting body data is available.")
        return
    
    print(f"Loading source data from: {source_path}")
    
    # Load the full dataset
    with tifffile.TiffFile(source_path) as tif:
        # Get metadata
        metadata = tif.imagej_metadata
        print(f"Original data shape: {tif.series[0].shape}")
        print(f"Metadata: {metadata}")
        
        # Read a subset of the data (first 50 Z planes, 200x200 XY)
        data = tif.series[0].asarray()
        
        # Data shape is (channels, Z, Y, X) = (2, 320, 1138, 960)
        # Create subset from center (Z=80, Y=400, X=400) - under 50MB for GitHub
        z_start = data.shape[1] // 2 - 40  # Center 80 Z planes
        y_start = data.shape[2] // 2 - 200  # Center 400 Y planes  
        x_start = data.shape[3] // 2 - 200  # Center 400 X planes
        
        subset = data[:, z_start:z_start+80, y_start:y_start+400, x_start:x_start+400]
        
        print(f"Subset data shape: {subset.shape}")
        print(f"Subset size: {subset.nbytes / 1024 / 1024:.1f} MB")
        
        # Save the subset
        output_path = output_dir / "fruiting_body_subset.ome.tif"
        tifffile.imwrite(
            output_path,
            subset,
            imagej=True,
            resolution=(1/0.1625, 1/0.1625),
            metadata={
                'unit': 'um',
                'spacing': 0.1625,
                'description': 'Subset of fruiting body data for PySPIM documentation'
            }
        )
        
        print(f"Subset saved to: {output_path}")
        
        # Create a simple PSF for demonstration
        create_demo_psfs(output_dir)
        
        # Create acquisition parameters file
        create_acquisition_params(output_dir)
        
        print("Demo dataset creation complete!")


def create_demo_psfs(output_dir: Path) -> None:
    """Create simple PSFs for demonstration."""
    
    # Create simple 3D Gaussian PSFs
    z_size, y_size, x_size = 31, 31, 31
    center = np.array([z_size//2, y_size//2, x_size//2])
    
    # Create coordinate grids
    z, y, x = np.ogrid[:z_size, :y_size, :x_size]
    
    # PSF A (slightly different from PSF B)
    sigma_a = [2.0, 1.5, 1.5]  # Z, Y, X
    psf_a = np.exp(-((z - center[0])**2 / (2 * sigma_a[0]**2) + 
                     (y - center[1])**2 / (2 * sigma_a[1]**2) + 
                     (x - center[2])**2 / (2 * sigma_a[2]**2)))
    
    # PSF B
    sigma_b = [1.5, 2.0, 1.5]  # Z, Y, X  
    psf_b = np.exp(-((z - center[0])**2 / (2 * sigma_b[0]**2) + 
                     (y - center[1])**2 / (2 * sigma_b[1]**2) + 
                     (x - center[2])**2 / (2 * sigma_b[2]**2)))
    
    # Normalize
    psf_a /= psf_a.sum()
    psf_b /= psf_b.sum()
    
    # Save PSFs
    np.save(output_dir / "PSFA_demo.npy", psf_a)
    np.save(output_dir / "PSFB_demo.npy", psf_b)
    
    print(f"Demo PSFs created: {psf_a.shape}")


def create_acquisition_params(output_dir: Path) -> None:
    """Create acquisition parameters file."""
    
    params = {
        "step_size": 0.5,        # microns between image planes
        "pixel_size": 0.1625,    # microns per pixel
        "theta": 0.7853981633974483,  # π/4 radians (45 degrees)
        "camera_offset": 100,    # camera offset to subtract
        "description": "Acquisition parameters for fruiting body subset demo"
    }
    
    # Save as text file for easy reading
    with open(output_dir / "acquisition_params.txt", "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    
    print("Acquisition parameters saved")


if __name__ == "__main__":
    create_fruiting_body_subset() 