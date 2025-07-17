"""
Utility functions for the napari-pyspim plugin.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


def save_workflow_parameters(output_path: str, parameters: Dict[str, Any]) -> None:
    """Save workflow parameters to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(parameters, f, indent=2, default=str)


def load_workflow_parameters(input_path: str) -> Dict[str, Any]:
    """Load workflow parameters from a JSON file."""
    with open(input_path, 'r') as f:
        return json.load(f)


def get_layer_metadata(viewer, layer_name: str) -> Optional[Dict[str, Any]]:
    """Get metadata from a napari layer."""
    try:
        layer = viewer.layers[layer_name]
        return layer.metadata
    except (KeyError, AttributeError):
        return None


def format_memory_usage(array: np.ndarray) -> str:
    """Format memory usage of an array in human-readable format."""
    size_bytes = array.size * array.itemsize
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.1f} GB"


def validate_data_path(path: str) -> bool:
    """Validate that a data path exists and contains expected files."""
    if not os.path.exists(path):
        return False
    
    # Check for common μManager file patterns
    expected_files = ['metadata.txt', 'Pos0', 'Pos1']
    path_obj = Path(path)
    
    # Check if any expected files exist
    for file_pattern in expected_files:
        if list(path_obj.glob(f"*{file_pattern}*")):
            return True
    
    # If no expected files, check if it's a directory with subdirectories
    return any(path_obj.iterdir())


def create_output_directory(base_path: str, acquisition_name: str) -> str:
    """Create an output directory for processed data."""
    output_dir = os.path.join(base_path, f"{acquisition_name}_processed")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_default_psf_paths() -> tuple[str, str]:
    """Get default PSF paths based on common locations."""
    common_paths = [
        "/scratch/gpfs/SHAEVITZ/dispim/extract_spindles",
        "/projects/SHAEVITZ/dispim/psfs",
        os.path.expanduser("~/dispim/psfs")
    ]
    
    for path in common_paths:
        psf_a = os.path.join(path, "PSFA_500.npy")
        psf_b = os.path.join(path, "PSFB_500.npy")
        if os.path.exists(psf_a) and os.path.exists(psf_b):
            return psf_a, psf_b
    
    return "", ""


def estimate_processing_time(shape: tuple, iterations: int = 20) -> str:
    """Estimate processing time based on data size and parameters."""
    total_pixels = np.prod(shape)
    
    # Rough estimates based on typical processing speeds
    # These are very approximate and depend on hardware
    if total_pixels < 10**7:  # < 10M pixels
        time_minutes = iterations * 0.1
    elif total_pixels < 10**8:  # < 100M pixels
        time_minutes = iterations * 0.5
    elif total_pixels < 10**9:  # < 1B pixels
        time_minutes = iterations * 2
    else:  # > 1B pixels
        time_minutes = iterations * 10
    
    if time_minutes < 60:
        return f"~{time_minutes:.1f} minutes"
    else:
        hours = time_minutes / 60
        return f"~{hours:.1f} hours"


def format_shape_string(shape: tuple) -> str:
    """Format array shape as a readable string."""
    if len(shape) == 3:
        return f"{shape[0]}×{shape[1]}×{shape[2]} (Z×Y×X)"
    elif len(shape) == 2:
        return f"{shape[0]}×{shape[1]} (Y×X)"
    else:
        return str(shape) 