# Basic Usage

Quick examples for processing SPIM data with PySPIM.

## Python API

### Basic Workflow

```python
import pyspim
from pyspim.data import dispim as data
from pyspim import roi, deskew as dsk

# 1. Load data
with data.uManagerAcquisition(data_path, False, numpy) as acq:
    a_raw = acq.get('a', 0, 0)
    b_raw = acq.get('b', 0, 0)

# 2. Camera offset correction
a_raw = data.subtract_constant_uint16arr(a_raw, 100)
b_raw = data.subtract_constant_uint16arr(b_raw, 100)

# 3. ROI detection and cropping
roia = roi.detect_roi_3d(a_raw, 'otsu')
roib = roi.detect_roi_3d(b_raw, 'otsu')
roic = roi.combine_rois(roia, roib)

# 4. Deskewing
step_size = 0.5      # microns
pixel_size = 0.1625  # microns
theta = math.pi / 4  # angle

a_dsk = dsk.deskew_stage_scan(a_raw, pixel_size, step_size, 1, method='orthogonal')
b_dsk = dsk.deskew_stage_scan(b_raw, pixel_size, step_size, -1, method='orthogonal')
```

## Command Line Tools

### Deskewing (CPU)
```bash
python deskew_cpu.py \
    --input-folder=/path/to/data \
    --output-folder=/path/to/output \
    --step-size=0.5 \
    --pixel-size=0.1625
```

### Registration (GPU)
```bash
python register.py \
    --input-folder=/path/to/deskewed/data \
    --output-folder=/path/to/registered/data \
    --transform='t+r+s' \
    --metric='cr'
```

### Deconvolution (GPU)
```bash
python deconvolve.py \
    --input-folder=/path/to/registered/data \
    --output-folder=/path/to/deconvolved/data \
    --psf-a=/path/to/psf_a.npy \
    --psf-b=/path/to/psf_b.npy \
    --iterations=10
```

## Performance Tips

| **Operation** | **Recommended** | **Reason** |
|---------------|-----------------|------------|
| Registration | GPU | 10-50x faster |
| Deconvolution | GPU | Significant speedup |
| Deskewing | CPU | Sufficient speed |
| ROI Detection | CPU | Fast enough |

## Example Data

- **Location**: `examples/data/sample_data/`
- **Contents**: Jupyter notebook with complete workflow
- **SLURM scripts**: Batch processing for HPC clusters 