# Quick Start Guide

This guide will get you up and running with PySPIM in minutes.

## Prerequisites

- Python 3.8.1 or higher
- Napari (for the GUI plugin)
- CUDA-compatible GPU (optional, for GPU acceleration)

## Installation

```bash
# Install both packages
pip install pyspim napari-pyspim
```

## Basic Usage

### Command Line Interface

```python
import pyspim

# Load your SPIM data
data = pyspim.load_data("path/to/your/data.tif")

# Process the data
processed = pyspim.process_pipeline(data)

# Save results
pyspim.save_data(processed, "processed_data.tif")
```

### Napari Plugin Interface

1. **Launch Napari**
   ```python
   import napari
   viewer = napari.Viewer()
   ```

2. **Load the PySPIM Plugin**
   - Go to `Plugins` → `PySPIM` → `DiSPIM Pipeline`
   - This opens the main processing widget

3. **Follow the Workflow**
   - **Tab 1**: Load your data file
   - **Tab 2**: Detect or select regions of interest
   - **Tab 3**: Adjust deskewing parameters
   - **Tab 4**: Configure registration settings
   - **Tab 5**: Set deconvolution parameters

## Example Workflow

### 1. Load Data

```python
import pyspim
import napari

# Load data
data = pyspim.load_data("example_data.tif")

# View in napari
viewer = napari.Viewer()
viewer.add_image(data)
```

### 2. Process with Plugin

1. Open the PySPIM plugin from the plugins menu
2. In the "Data Loading" tab, select your data file
3. Switch to "ROI Detection" and adjust detection parameters
4. Move to "Deskewing" and set the deskew angle
5. Configure registration in the "Registration" tab
6. Set deconvolution parameters in the final tab
7. Click "Process" to run the complete pipeline

### 3. View Results

The processed results will appear as new layers in the Napari viewer, allowing you to compare original and processed data.

## Configuration

Create a configuration file for consistent processing:

```yaml
# config.yaml
deskew:
  angle: 31.5
  method: "gpu"

registration:
  method: "phase_correlation"
  max_shift: 50

deconvolution:
  method: "richardson_lucy"
  iterations: 10
  psf_type: "gaussian"
```

Load the configuration:

```python
import pyspim

# Load configuration
config = pyspim.load_config("config.yaml")

# Process with configuration
processed = pyspim.process_pipeline(data, config=config)
```

## GPU Acceleration

For GPU acceleration, ensure you have CUDA installed:

```bash
# Install CuPy with CUDA support
pip install cupy-cuda12x
```

The plugin will automatically use GPU acceleration when available.

## Next Steps

- Read the [Installation Guide](installation.md) for detailed setup
- Explore [Examples](../user-guide/basic-usage.md) for more use cases
- Check the [API Reference](../packages/pyspim/api.md) for detailed documentation
- Learn about [Advanced Features](../user-guide/advanced-features.md) options 