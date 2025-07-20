# Jupyter Notebooks

PySPIM provides Jupyter notebooks for interactive exploration and learning.

## Available Notebooks

### Basic Usage Notebook

**File**: `examples/data/sample_data/basic_usage.ipynb`

This is the main example notebook that demonstrates the complete PySPIM workflow:

- **Data Loading**: Loading diSPIM data using uManager acquisition
- **ROI Detection**: Automated region of interest detection and cropping
- **Deskewing**: Correcting light sheet microscopy artifacts
- **Registration**: Multi-view image alignment with GPU acceleration
- **Deconvolution**: Image restoration with PSF support

### Running the Notebook

1. **Install Jupyter** (if not already installed):
   ```bash
   uv pip install jupyter notebook matplotlib
   ```

2. **Start Jupyter**:
   ```bash
   cd examples/data/sample_data
   jupyter notebook
   ```

3. **Open the notebook**:
   - Navigate to `basic_usage.ipynb`
   - Run cells sequentially to follow the workflow

### Notebook Requirements

The notebook requires:
- PySPIM core package: `uv pip install -e packages/pyspim`
- CuPy for GPU acceleration
- NumPy for numerical operations
- Matplotlib for visualization
- Micro-Manager data files

### Example Data

The notebook is located in `examples/data/sample_data/` along with:
- **README.md**: Detailed usage instructions
- **SLURM scripts**: Batch processing examples
- **Sample data**: Test datasets for demonstration

For testing with your own data, modify the paths in the notebook to point to your data files.

## Key Sections

### 1. Data Setup

```python
# Configure data paths
root_fldr = "/path/to/your/data"
acq = "acquisition_name"
data_path = os.path.join(root_fldr, acq)

# Load PSFs
psf_dir = "/path/to/psfs"
psf_a = numpy.load(os.path.join(psf_dir, "PSFA_500.npy"))
psf_b = numpy.load(os.path.join(psf_dir, "PSFB_500.npy"))
```

### 2. Acquisition Parameters

```python
step_size = 0.5      # microns
pixel_size = 0.1625  # microns
theta = math.pi / 4  # radians
```

### 3. Data Loading

```python
from pyspim.data import dispim as data

with data.uManagerAcquisition(data_path, False, numpy) as acq:
    a_raw = acq.get('a', 0, 0)
    b_raw = acq.get('b', 0, 0)
```

### 4. ROI Detection

```python
from pyspim import roi

roia = roi.detect_roi_3d(a_raw, 'otsu')
roib = roi.detect_roi_3d(b_raw, 'otsu')
roic = roi.combine_rois(roia, roib)
```

### 5. Deskewing

```python
from pyspim import deskew as dsk

a_dsk = dsk.deskew_stage_scan(a_raw, pixel_size, step_size_lat, 1,
                              method='orthogonal')
b_dsk = dsk.deskew_stage_scan(b_raw, pixel_size, step_size_lat, -1,
                              method='orthogonal')
```

## Interactive Features

The notebook includes:
- **Visualization**: Matplotlib plots showing intermediate results
- **Parameter Tuning**: Easy modification of processing parameters
- **Progress Tracking**: Cell-by-cell execution for understanding
- **Error Handling**: Examples of common issues and solutions

## Customization

### Using Your Own Data

1. **Modify data paths**:
   ```python
   root_fldr = "/your/data/path"
   acq = "your_acquisition_name"
   ```

2. **Adjust parameters**:
   ```python
   step_size = 0.3      # Your step size
   pixel_size = 0.1     # Your pixel size
   theta = math.pi / 6  # Your angle
   ```

3. **Load your PSFs**:
   ```python
   psf_a = numpy.load("/path/to/your/psf_a.npy")
   psf_b = numpy.load("/path/to/your/psf_b.npy")
   ```

### GPU vs CPU

The notebook can run on both GPU and CPU:

**GPU (recommended)**:
```python
import cupy
# Use cupy arrays for GPU processing
```

**CPU (fallback)**:
```python
import numpy
# Use numpy arrays for CPU processing
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure PySPIM is installed: `uv pip install -e packages/pyspim`
   - Check Python environment: `conda activate pyspim`

2. **Memory Issues**
   - Reduce data size for testing
   - Use CPU processing for large datasets
   - Process in smaller chunks

3. **GPU Errors**
   - Check CUDA installation
   - Verify CuPy compatibility
   - Use CPU fallback if needed

### Performance Tips

- Use GPU for registration and deconvolution
- Process small datasets for development
- Monitor memory usage
- Save intermediate results

## Next Steps

- Run the [basic usage notebook](../../../examples/nb/basic_usage.ipynb)
- Try the [command-line pipeline](examples/scripts/dispim_pipeline/)
- Check the [API reference](packages/pyspim/api.md)
- Explore [advanced features](advanced-features.md) 