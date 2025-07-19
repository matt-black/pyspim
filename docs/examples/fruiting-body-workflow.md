# Fruiting Body Analysis Workflow

This example demonstrates the complete PySPIM workflow for analyzing dual-view SPIM (diSPIM) data using a subset of fruiting body microscopy data.

## Overview

The workflow processes dual-view SPIM data through the following steps:

1. **Data Loading**: Load and preprocess dual-view SPIM data
2. **ROI Detection**: Automatically detect regions of interest
3. **Deskewing**: Correct light sheet microscopy artifacts
4. **Registration**: Align the two views
5. **Deconvolution**: Improve resolution using dual-view deconvolution

## Dataset

We use a subset of fruiting body data (50 Z planes, 200×200 XY) extracted from the center of a larger acquisition. This manageable dataset demonstrates the complete workflow while being suitable for documentation and testing.

### Data Structure

```
docs/examples/data/fruiting_body_subset/
├── fruiting_body_subset.ome.tif    # Dual-view SPIM data (2 channels)
├── PSFA_demo.npy                   # Point spread function for view A
├── PSFB_demo.npy                   # Point spread function for view B
└── acquisition_params.txt          # Acquisition parameters
```

### Acquisition Parameters

- **Step size**: 0.5 μm (distance between image planes)
- **Pixel size**: 0.1625 μm
- **Theta**: π/4 radians (45° angle between objective and coverslip)
- **Camera offset**: 100 (to be subtracted from raw data)

## Workflow Steps

### 1. Data Loading and Preprocessing

```python
import numpy as np
import tifffile
from pathlib import Path

# Load dual-view data
with tifffile.TiffFile("fruiting_body_subset.ome.tif") as tif:
    data = tif.series[0].asarray()

# Extract the two views
a_raw = data[0]  # First channel (CameraRight)
b_raw = data[1]  # Second channel (CameraLeft)

# Subtract camera offset
def subtract_constant_uint16arr(arr, offset):
    result = arr.astype(np.int32) - offset
    return np.clip(result, 0, 65535).astype(np.uint16)

a_raw = subtract_constant_uint16arr(a_raw, camera_offset)
b_raw = subtract_constant_uint16arr(b_raw, camera_offset)
```

### 2. ROI Detection

Automatically detect regions of interest to crop out empty regions:

```python
def detect_roi_3d_simple(data, method='otsu'):
    # Use maximum projection for thresholding
    max_proj = np.max(data, axis=0)
    threshold = np.percentile(max_proj, 95)
    
    # Create binary mask and find bounding box
    mask = max_proj > threshold
    coords = np.where(mask)
    
    # Return bounding box with padding
    return [(z_min, z_max), (y_min, y_max), (x_min, x_max)]

# Detect and crop ROIs
roi_a = detect_roi_3d_simple(a_raw)
roi_b = detect_roi_3d_simple(b_raw)
roi_combined = combine_rois(roi_a, roi_b)

# Crop data
a_cropped = a_raw[roi_combined[0][0]:roi_combined[0][1],
                  roi_combined[1][0]:roi_combined[1][1],
                  roi_combined[2][0]:roi_combined[2][1]]
```

### 3. Deskewing

Correct light sheet microscopy artifacts:

```python
def deskew_stage_scan_simple(data, pixel_size, step_size_lat, direction=1):
    # Calculate skew parameters
    skew_factor = step_size_lat / pixel_size
    
    # Apply deskewing transformation
    # (Simplified implementation for demonstration)
    return deskewed_data

# Deskew both views (different directions)
a_deskewed = deskew_stage_scan_simple(a_cropped, pixel_size, step_size_lat, 1)
b_deskewed = deskew_stage_scan_simple(b_cropped, pixel_size, step_size_lat, -1)
```

### 4. Registration

Align the two views for dual-view processing:

```python
def simple_registration(a, b):
    # Use maximum projections for registration
    a_proj = np.max(a, axis=0)
    b_proj = np.max(b, axis=0)
    
    # Pad to same size and align
    a_padded, b_padded = pad_to_same_size(a, b)
    return a_padded, b_padded

# Register the views
a_reg, b_reg = simple_registration(a_deskewed, b_deskewed)
```

### 5. Deconvolution

Improve resolution using dual-view deconvolution:

```python
def simple_deconvolution(a, b, psf_a, psf_b, iterations=10):
    # Initialize result
    result = (a.astype(np.float32) + b.astype(np.float32)) / 2
    
    # Iterative Richardson-Lucy deconvolution
    for i in range(iterations):
        # Update using both views
        result = result * 0.5 * (a / (result + 1e-8) + b / (result + 1e-8))
    
    return result

# Perform deconvolution
deconvolved = simple_deconvolution(a_reg, b_reg, psf_a, psf_b, iterations=15)
```

## Results

The workflow produces:

- **Cropped data**: Focused on the sample region
- **Deskewed data**: Corrected for light sheet artifacts
- **Registered data**: Aligned dual views
- **Deconvolved data**: Improved resolution

### Visualization

The notebook includes comprehensive visualizations at each step:

- Raw data projections (XY, ZX, ZY)
- ROI detection results
- Deskewing before/after comparison
- Registration overlay
- Final deconvolved results

## Running the Example

1. **Install PySPIM**: Follow the [installation guide](../getting-started/installation.md)
2. **Download the data**: The subset data is included in the repository
3. **Run the notebook**: Execute `docs/examples/fruiting_body_workflow.ipynb`

```bash
# Navigate to the project directory
cd pyspim

# Activate the environment
conda activate pyspim

# Start Jupyter
jupyter lab docs/examples/fruiting_body_workflow.ipynb
```

## Performance Notes

This example uses simplified implementations for demonstration purposes. For production use:

- **GPU Acceleration**: Use CuPy for GPU-accelerated processing
- **Advanced Registration**: Use PySPIM's Powell optimization for precise registration
- **Chunked Processing**: Use Zarr for large datasets
- **Memory Management**: Process data in chunks for very large volumes

## Next Steps

- Explore the [API Reference](../packages/pyspim/api.md) for advanced features
- Try the [napari plugin](../packages/napari-pyspim/overview.md) for interactive analysis
- Check out [advanced features](advanced-features.md) for more complex workflows 