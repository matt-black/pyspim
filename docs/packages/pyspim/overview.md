# PySPIM Core Package

The PySPIM core package provides the fundamental functionality for processing selective plane illumination microscopy (SPIM) data.

## Overview

PySPIM is designed to handle the complete pipeline for SPIM data analysis, from raw data loading to final processed results. The package is built with performance and scalability in mind, using Dask for parallel processing and supporting both CPU and GPU acceleration.

## Key Components

### Data Loading
- Support for multiple file formats (TIFF, HDF5, Zarr)
- Efficient memory management with Dask arrays
- Metadata extraction and validation

### ROI Detection
- Automated region of interest detection
- Manual ROI selection tools
- ROI validation and refinement

### Deskewing
- Light sheet microscopy artifact correction
- GPU-accelerated processing with CuPy
- Configurable deskewing parameters

### Registration
- Multi-view image registration
- Rigid and non-rigid transformations
- Registration quality assessment

### Deconvolution
- Advanced deconvolution algorithms
- GPU acceleration support
- Configurable PSF models

## Architecture

The package is organized into modular components:

```
pyspim/
├── core/           # Core functionality
├── data/           # Data loading and I/O
├── processing/     # Processing algorithms
├── utils/          # Utility functions
└── config/         # Configuration management
```

## Performance Features

- **Parallel Processing**: Built on Dask for scalable computation
- **GPU Acceleration**: CuPy integration for CUDA-accelerated operations
- **Memory Efficiency**: Lazy evaluation and chunked processing
- **Caching**: Intelligent caching of intermediate results

## Usage Example

```python
import pyspim
import dask.array as da

# Load data
data = pyspim.load_data("path/to/data.tif")

# Detect ROIs
rois = pyspim.detect_rois(data)

# Process each ROI
for roi in rois:
    # Deskew
    deskewed = pyspim.deskew(roi)
    
    # Register (if multi-view)
    if roi.is_multiview:
        registered = pyspim.register(deskewed)
    else:
        registered = deskewed
    
    # Deconvolve
    deconvolved = pyspim.deconvolve(registered)
    
    # Save results
    pyspim.save_data(deconvolved, f"processed_{roi.id}.tif")
```

## Configuration

PySPIM uses a flexible configuration system:

```python
import pyspim

# Load configuration
config = pyspim.load_config("config.yaml")

# Or set parameters programmatically
config = pyspim.Config(
    deskew_angle=31.5,
    registration_method="phase_correlation",
    deconvolution_iterations=10
)
```

## Dependencies

- **Core**: numpy, dask, scikit-image
- **I/O**: tifffile, h5py, zarr
- **GPU**: cupy-cuda12x (optional)
- **Visualization**: matplotlib

## Next Steps

- Read the [API Reference](api.md) for detailed function documentation
- Check out [Examples](../../user-guide/basic-usage.md) for usage patterns
- Learn about [Advanced Features](../../user-guide/advanced-features.md) options 