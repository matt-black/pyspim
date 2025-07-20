# Quick Start Guide

Get up and running with PySPIM in minutes.

## Prerequisites

- Python 3.8.1 or higher
- Napari (for the GUI plugin)
- CUDA-compatible GPU (optional, for GPU acceleration)

## Installation

```bash
# Clone and install
git clone https://github.com/matt-black/pyspim.git
cd pyspim
just install
```

## Basic Usage

### Napari Plugin Interface

1. **Launch Napari**
   ```bash
   napari
   ```

2. **Load the PySPIM Plugin**
   - Go to `Plugins` → `PySPIM` → `DiSPIM Pipeline`

3. **Follow the Workflow**
   - **Tab 1**: Load your data file
   - **Tab 2**: Detect or select regions of interest
   - **Tab 3**: Adjust deskewing parameters
   - **Tab 4**: Configure registration settings
   - **Tab 5**: Set deconvolution parameters

### Command Line Interface

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

## GPU Acceleration

For GPU acceleration:

```bash
# Install CuPy with CUDA support
pip install cupy-cuda12x
```

The plugin will automatically use GPU acceleration when available.

## Next Steps

- [Installation Guide](installation.md) - Detailed setup
- [Basic Usage](../user-guide/basic-usage.md) - More examples
- [API Reference](../packages/pyspim/api.md) - Detailed documentation 