# PySPIM Core Package

Core library for processing dual-view SPIM (diSPIM) microscopy data.

## Overview

PySPIM provides the fundamental functionality for SPIM data analysis with GPU acceleration support.

## Key Components

- **Data Loading** - μManager acquisitions, TIFF, HDF5, Zarr
- **ROI Detection** - Automated and manual region detection
- **Deskewing** - Light sheet artifact correction
- **Registration** - Multi-view alignment with GPU acceleration
- **Deconvolution** - Richardson-Lucy with PSF support

## Basic Usage

```python
import pyspim
from pyspim.data import dispim as data
from pyspim import roi, deskew as dsk

# Load data
with data.uManagerAcquisition(data_path, False, numpy) as acq:
    a_raw = acq.get('a', 0, 0)
    b_raw = acq.get('b', 0, 0)

# ROI detection
roia = roi.detect_roi_3d(a_raw, 'otsu')
roib = roi.detect_roi_3d(b_raw, 'otsu')

# Deskewing
a_dsk = dsk.deskew_stage_scan(a_raw, pixel_size, step_size, 1)
b_dsk = dsk.deskew_stage_scan(b_raw, pixel_size, step_size, -1)
```

## Performance Features

- **GPU Acceleration** - CuPy integration for CUDA operations
- **Memory Efficient** - Chunked processing for large datasets
- **Modular Design** - Separate components for each processing step

## Dependencies

- **Core**: numpy, scikit-image
- **GPU**: cupy (optional)
- **I/O**: tifffile, h5py

## Next Steps

- [API Reference](api.md) - Detailed function documentation
- [Basic Usage](../../user-guide/basic-usage.md) - Usage examples 