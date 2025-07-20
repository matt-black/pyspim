# Napari PySPIM Plugin

Interactive graphical interface for processing SPIM data within Napari.

## Overview

The plugin integrates seamlessly with Napari, providing a tabbed interface for all SPIM processing steps. It leverages the core PySPIM package for processing while offering an intuitive GUI for parameter adjustment and result visualization.

## Features

### Interactive Processing Pipeline
- **Tabbed Interface**: Organized workflow with dedicated tabs for each processing step
- **Real-time Visualization**: See results immediately in the Napari viewer
- **Parameter Adjustment**: Interactive sliders and controls for all processing parameters
- **Progress Tracking**: Real-time progress bars and status updates

### Processing Steps

1. **Data Loading**
   - Load various file formats (TIFF, HDF5, Zarr)
   - Preview data in Napari layers
   - Metadata display and validation

2. **ROI Detection**
   - Automated ROI detection with preview
   - Manual ROI selection tools
   - ROI refinement and validation

3. **Deskewing**
   - Interactive deskewing parameter adjustment
   - Real-time preview of deskewed results
   - GPU acceleration support

4. **Registration**
   - Multi-view registration with quality metrics
   - Registration parameter optimization
   - Result validation and refinement

5. **Deconvolution**
   - Advanced deconvolution algorithms
   - PSF model selection and configuration
   - Iteration control and convergence monitoring

## Installation

```bash
# Install from source
git clone https://github.com/matt-black/pyspim.git
cd pyspim
just install
```

## Usage

### Basic Workflow

1. **Launch Napari**
   ```bash
   napari
   ```

2. **Load the Plugin**
   - Go to `Plugins` → `PySPIM` → `DiSPIM Pipeline`

3. **Process Your Data**
   - Follow the tabbed workflow from left to right
   - Adjust parameters as needed
   - View results in the Napari viewer

### Advanced Usage

```python
import napari
from napari_pyspim import DispimPipelineWidget

# Create viewer
viewer = napari.Viewer()

# Load data
viewer.open("path/to/your/data.tif")

# Create and add the widget
widget = DispimPipelineWidget(viewer)
viewer.window.add_dock_widget(widget, name="PySPIM Pipeline")
```

## Widget Structure

The plugin provides a main widget (`DispimPipelineWidget`) with the following components:

- **DataLoaderWidget**: File loading and data preview
- **RoiDetectionWidget**: ROI detection and selection
- **DeskewingWidget**: Deskewing parameter adjustment
- **RegistrationWidget**: Registration and alignment
- **DeconvolutionWidget**: Deconvolution processing

## Integration with Napari

The plugin integrates deeply with Napari:

- **Layer Management**: Automatically creates and manages Napari layers
- **Event Handling**: Responds to Napari events and updates
- **Viewer Integration**: Seamless integration with Napari's viewer controls
- **Plugin Discovery**: Automatically discovered by Napari's plugin system

## Performance

- **Lazy Loading**: Data is loaded on-demand to minimize memory usage
- **Background Processing**: Long-running operations run in background threads
- **Progress Updates**: Real-time progress reporting for all operations
- **Memory Management**: Efficient memory usage with chunked processing

## Troubleshooting

| **Issue** | **Solution** |
|-----------|--------------|
| Plugin not appearing | Restart Napari after installation |
| CUDA errors | Ensure CuPy is properly installed |
| Memory issues | Use smaller data chunks |

## Next Steps

- [Installation Guide](installation.md) - Setup instructions
- [Usage Guide](usage.md) - Detailed workflow examples
- [API Reference](api.md) - Widget documentation 