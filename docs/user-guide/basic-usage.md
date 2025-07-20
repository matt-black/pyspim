# Basic Usage Examples

This page provides practical examples of how to use PySPIM for SPIM data processing, based on real workflows and the actual project structure.

## Project Structure

```
pyspim/
├── packages/
│   ├── pyspim/                    # Core SPIM processing library
│   └── napari-pyspim/             # Napari plugin for GUI
├── examples/
│   ├── nb/                        # Jupyter notebooks
│   │   └── basic_usage.ipynb      # Main usage example
│   └── data/
│       ├── sample_data/           # GitHub-friendly sample data
│       │   ├── basic_usage.ipynb  # Complete workflow notebook
│       │   ├── README.md          # Usage instructions
│       │   └── sh/                # SLURM job scripts
│       └── example_dispim_data/   # Full example datasets (excluded from git)
├── examples/
│   └── script/
│       └── dispim_pipeline/       # Command-line tools
└── docs/                          # Documentation
```

## Example Data

We provide sample data and scripts for testing PySPIM functionality:

### Sample Data Location
- **Path**: `examples/data/sample_data/`
- **Contents**: 
  - `basic_usage.ipynb` - Complete workflow demonstration
  - `README.md` - Detailed usage instructions
  - `sh/` - SLURM job scripts for batch processing

### What's Included
- **Jupyter Notebook**: Complete diSPIM processing pipeline
- **SLURM Scripts**: Batch processing for HPC clusters
- **Documentation**: Step-by-step usage instructions

## Notebook Example

The main usage example is in `examples/data/sample_data/basic_usage.ipynb`. This notebook demonstrates:

- Loading diSPIM data using `uManagerAcquisition`
- Automated ROI detection and cropping
- Deskewing for both heads (A and B)
- Registration and transformation
- Deconvolution with PSF support

### Key Workflow Steps

```python
import os
import math
import cupy
import numpy
import tifffile
from pyspim.data import dispim as data
from pyspim import roi, deskew as dsk

# 1. Data Setup
root_fldr = "/path/to/your/data"
acq = "acquisition_name"
data_path = os.path.join(root_fldr, acq)

# 2. Load Data
with data.uManagerAcquisition(data_path, False, numpy) as acq:
    a_raw = acq.get('a', 0, 0)
    b_raw = acq.get('b', 0, 0)

# 3. Camera Offset Correction
a_raw = data.subtract_constant_uint16arr(a_raw, 100)
b_raw = data.subtract_constant_uint16arr(b_raw, 100)

# 4. Automated ROI Detection
roia = roi.detect_roi_3d(a_raw, 'otsu')
roib = roi.detect_roi_3d(b_raw, 'otsu')
roic = roi.combine_rois(roia, roib)

# 5. Crop Data
a_raw = a_raw[roic[0][0]:roic[0][1],
              roic[1][0]:roic[1][1],
              roic[2][0]:roic[2][1]]
b_raw = b_raw[roic[0][0]:roic[0][1],
              roic[1][0]:roic[1][1],
              roic[2][0]:roic[2][1]]

# 6. Deskewing
step_size = 0.5      # microns
pixel_size = 0.1625  # microns
theta = math.pi / 4  # angle

a_dsk = dsk.deskew_stage_scan(a_raw, pixel_size, step_size_lat, 1,
                              method='orthogonal')
b_dsk = dsk.deskew_stage_scan(b_raw, pixel_size, step_size_lat, -1,
                              method='orthogonal')
```

## Command-Line Pipeline

PySPIM provides command-line tools for batch processing in `examples/scripts/dispim_pipeline/`:

### 1. Deskewing (CPU)

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=100G
#SBATCH --time=0:15:00
#SBATCH --job-name=deskew

module load cudatoolkit/12.6
module load anaconda3/2024.6
conda activate pyspim

python deskew_cpu.py \
    --input-folder=/path/to/data \
    --data-type="umasi" \
    --output-folder=/path/to/output \
    --deskew-method="ortho" \
    --step-size=0.5 \
    --pixel-size=0.1625 \
    --channels="0" \
    --interp-method='cubspl' \
    --verbose
```

### 2. Registration (GPU)

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=200G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=1:00:00
#SBATCH --job-name=dvreg

module load cudatoolkit/12.6
module load anaconda3/2024.6
conda activate pyspim

python register.py \
    --input-folder=/path/to/deskewed/data \
    --output-folder=/path/to/registered/data \
    --crop-box-a="115,275,167,437,500,850" \
    --crop-box-b="115,275,167,437,500,850" \
    --channel=0 \
    --metric='cr' \
    --transform='t+r+s' \
    --bounds='20,20,20,5,5,5,0.05,0.05,0.05' \
    --interp-method='cubspl' \
    --verbose
```

### 3. Deconvolution (GPU)

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=200G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=2:00:00
#SBATCH --job-name=deconv

module load cudatoolkit/12.6
module load anaconda3/2024.6
conda activate pyspim

python deconvolve.py \
    --input-folder=/path/to/registered/data \
    --output-folder=/path/to/deconvolved/data \
    --psf-a=/path/to/psf_a.npy \
    --psf-b=/path/to/psf_b.npy \
    --iterations=10 \
    --channel=0 \
    --verbose
```

## Performance Recommendations

### GPU vs CPU Processing

**Use GPU for:**
- Registration (significant speedup)
- Deconvolution (10-50x faster)
- Large dataset processing

**Use CPU for:**
- Deskewing (sufficient speed)
- Small datasets
- Development/testing

### Memory Management

```python
# For large datasets, use chunked processing
import dask.array as da

# Load data as Dask array
data = da.from_array(large_array, chunks=(64, 64, 64))

# Process in chunks
result = data.map_blocks(process_function)
```

### SLURM Job Configuration

**CPU Jobs:**
```bash
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=50G
```

**GPU Jobs:**
```bash
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem-per-cpu=200G
```

## Data Formats

### Input Data
- **uManager**: Micro-Manager acquisition format
- **TIFF**: Standard TIFF files
- **HDF5**: Hierarchical data format

### Output Data
- **TIFF**: Processed images
- **NPY**: NumPy arrays
- **Zarr**: Compressed arrays

## Configuration Examples

### Acquisition Parameters

```python
# Standard diSPIM parameters
params = {
    'step_size': 0.5,        # microns
    'pixel_size': 0.1625,    # microns
    'theta': math.pi / 4,    # radians
    'camera_offset': 100     # PCO.edge offset
}
```

### Processing Parameters

```python
# Deskewing
deskew_params = {
    'method': 'orthogonal',
    'interpolation': 'cubic_spline',
    'recrop': False
}

# Registration
reg_params = {
    'metric': 'correlation',
    'transform': 'translation+rotation+scale',
    'bounds': [20, 20, 20, 5, 5, 5, 0.05, 0.05, 0.05]
}

# Deconvolution
deconv_params = {
    'method': 'richardson_lucy',
    'iterations': 10,
    'psf_type': 'measured'
}
```

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce chunk size
   - Use CPU processing for large datasets
   - Increase SLURM memory allocation

2. **GPU Errors**
   - Check CUDA version compatibility
   - Ensure GPU memory is sufficient
   - Use CPU fallback if needed

3. **Registration Failures**
   - Check crop boxes are valid
   - Adjust registration bounds
   - Verify data quality

### Performance Tips

- Use GPU for registration and deconvolution
- Process in chunks for large datasets
- Monitor memory usage
- Use appropriate SLURM resources

## Next Steps

- Run the [basic usage notebook](../../../examples/nb/basic_usage.ipynb)
- Try the [command-line pipeline](examples/scripts/dispim_pipeline/)
- Use the [Snakemake workflow](snakemake-workflow.md) for automated processing
- Check the [API reference](packages/pyspim/api.md) for detailed function documentation
- Explore [advanced features](advanced-features.md) for complex workflows 