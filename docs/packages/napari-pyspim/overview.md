# Napari PySPIM Plugin

The Napari PySPIM plugin provides an interactive graphical user interface for processing SPIM data within the Napari viewer.

## Overview

The plugin integrates seamlessly with Napari, providing a tabbed interface for all SPIM processing steps. It leverages the core PySPIM package for processing while offering an intuitive GUI for parameter adjustment and result visualization.


*Overview of the PySPIM napari plugin showing the complete workflow interface*

## Features

### Interactive Processing Pipeline
- **Tabbed Interface**: Organized workflow with dedicated tabs for each processing step
- **Real-time Visualization**: See results immediately in the Napari viewer
- **Parameter Adjustment**: Interactive sliders and controls for all processing parameters
- **Progress Tracking**: Real-time progress bars and status updates

### Processing Steps


*The 5 processing steps in the PySPIM workflow*

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
# Install the plugin
pip install napari-pyspim

# Or install in development mode
pip install -e packages/napari-pyspim
```

## Usage

### Basic Workflow


*Basic workflow showing the step-by-step process*

1. **Launch Napari**
   ```python
   import napari
   viewer = napari.Viewer()
   ```

2. **Load the Plugin**
   - Go to `Plugins` → `PySPIM` → `DiSPIM Pipeline`
   - Or use the command: `Plugins` → `PySPIM` → `Open DiSPIM Pipeline`

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

## Configuration

The plugin uses the same configuration system as the core package:

```python
# Load configuration
config = napari_pyspim.load_config("config.yaml")

# Or configure programmatically
config = napari_pyspim.Config(
    deskew_angle=31.5,
    registration_method="phase_correlation",
    deconvolution_iterations=10
)
```

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
- **Memory Management**: Efficient memory usage with Dask arrays

## Troubleshooting

### Common Issues

1. **Plugin not appearing**: Restart Napari after installation
2. **CUDA errors**: Ensure CuPy is properly installed for your CUDA version
3. **Memory issues**: Use smaller data chunks or reduce batch sizes

### Getting Help

- Check the [API Reference](api.md) for detailed widget documentation
- See [Examples](../user-guide/basic-usage.md) for usage patterns
- Report issues on [GitHub](https://github.com/matt-black/pyspim/issues)

## Next Steps

- Read the [Installation Guide](installation.md) for setup instructions
- Check the [Usage Guide](usage.md) for detailed workflow examples
- Explore the [API Reference](api.md) for widget documentation

## Image Placeholders

The images in this documentation are placeholders. To complete the documentation:

1. **Take Screenshots**: Capture the actual napari plugin interface
2. **Replace Placeholders**: Upload images to `docs/images/` directory
3. **Follow Guidelines**: See `docs/images/README.md` for detailed instructions

Key images needed:
- Plugin overview and main interface
- Processing steps diagram
- Basic workflow progression 