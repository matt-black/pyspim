# Napari Plugin Usage Guide

This guide provides detailed instructions for using the PySPIM napari plugin to process dual-view SPIM (diSPIM) data through an interactive graphical interface.

## Getting Started

### Launching the Plugin

1. **Start Napari**:
   ```bash
   napari
   ```

2. **Open the Plugin**:
   - Go to `Plugins` → `PySPIM` → `DiSPIM Pipeline`
   - The plugin will open as a docked widget with 5 tabs

3. **Alternative Method**:
   ```python
   import napari
   from napari_pyspim import DispimPipelineWidget
   
   viewer = napari.Viewer()
   widget = DispimPipelineWidget(viewer)
   viewer.window.add_dock_widget(widget, name="DiSPIM Pipeline")
   ```

## Plugin Interface Overview

The plugin provides a tabbed interface with 5 processing steps:

1. **Data Loading** - Load μManager acquisitions and set parameters
2. **ROI Detection** - Detect and crop regions of interest
3. **Deskewing** - Correct for oblique illumination geometry
4. **Registration** - Align dual views
5. **Deconvolution** - Richardson-Lucy deconvolution

Each tab contains controls specific to that processing step and automatically passes data to the next step.

![Plugin Main Interface](../images/napari-plugin-main-interface.png)
*Main plugin interface showing the tabbed workflow with all 5 processing steps*

## Step 1: Data Loading

### Loading μManager Data

![Data Loading Tab](../images/napari-plugin-data-loading.png)
*Data loading tab with file browser and acquisition parameter controls*

1. **Select Data Path**:
   - Click "Browse" to select your μManager acquisition folder
   - The folder should contain `a.tif` and `b.tif` files

2. **Set Acquisition Parameters**:
   - **Step Size**: Distance between z-slices (typically 0.5 μm)
   - **Pixel Size**: Physical pixel size (typically 0.1625 μm)
   - **Theta**: Angle between light sheet and detection objective (typically 45°)
   - **Camera Offset**: Camera dark level (typically 100 for PCO.edge)

3. **Load Data**:
   - Click "Load Data" to load the raw A and B volumes
   - Data will appear as new layers in the napari viewer
   - Progress is shown in the status bar

### Supported Data Formats

- **μManager acquisitions**: Standard μManager folder structure
- **TIFF files**: Individual TIFF files for each channel
- **HDF5 files**: Hierarchical data format files
- **Zarr arrays**: Compressed array format

### Data Validation

The plugin automatically validates:
- File existence and accessibility
- Data dimensions and format
- Parameter ranges and consistency

## Step 2: ROI Detection

### Channel Selection


*ROI detection tab showing channel selection and threshold method options*

Choose which channels to process:

- **Process both channels (A & B)**: Full dual-view processing
- **Process only channel A**: Single-channel processing
- **Process only channel B**: Single-channel processing

### ROI Detection Methods

#### Automated Methods

1. **Otsu Thresholding**:
   - Automatically determines optimal threshold
   - Good for data with clear background/foreground separation
   - Fast and reliable for most datasets

2. **Triangle Thresholding**:
   - Uses triangle algorithm for threshold determination
   - Better for data with gradual intensity transitions
   - More robust for noisy data

#### Manual ROI Selection

1. **Select "Manual" method**
2. **Use napari's rectangle selection tool**:
   - Draw a rectangle around your region of interest
   - The plugin will use this as the ROI
3. **Click "Apply Manual ROI"**

### ROI Parameters

- **Threshold Method**: Choose between "otsu", "triangle", or "manual"
- **Acquisition Parameters**: Automatically loaded from data metadata
  - Step Size, Pixel Size, and Theta angle

### ROI Results


*ROI detection results showing coordinates, volume size, and processing method*

After detection, the plugin displays:
- **ROI Coordinates**: Bounding box coordinates
- **Cropped Volume Size**: Dimensions of the cropped data
- **Processing Method**: Which method was used
- **New Layers**: Cropped data added to napari viewer

## Step 3: Deskewing

### Deskewing Parameters

1. **Method Selection**:
   - **Orthogonal**: Standard orthogonal deskewing
   - Additional methods may be added in future versions

2. **Acquisition Parameters**:
   - **Step Size**: Distance between z-slices
   - **Pixel Size**: Physical pixel size
   - **Theta**: Light sheet angle (in degrees)

3. **Advanced Options**:
   - **Re-crop after deskewing**: Automatically crop deskewed data
   - **GPU Acceleration**: Use GPU for faster processing

### Deskewing Process


*Deskewing tab with layer selection and parameter controls*

1. **Select Input Layers**:
   - Choose the cropped A and B layers from previous step
   - Parameters are automatically loaded from layer metadata

2. **Adjust Parameters**:
   - Fine-tune step size, pixel size, and theta if needed
   - Preview the effect of parameter changes

3. **Run Deskewing**:
   - Click "Deskew Data" to process
   - Progress is shown in real-time
   - Results appear as new layers

### Deskewing Results

- **Deskewed A**: Corrected channel A data
- **Deskewed B**: Corrected channel B data
- **Metadata**: Processing parameters stored in layer metadata

## Step 4: Registration

### Registration Parameters

1. **Transform Type**:
   - **t**: Translation only
   - **t+r**: Translation + rotation
   - **t+r+s**: Translation + rotation + scaling

2. **Phase Correlation**:
   - **Yes**: Use phase correlation for initial guess
   - **No**: Start from zero transformation

3. **Upsample Factor**:
   - Subpixel accuracy for phase correlation
   - Higher values = more accurate but slower

### Registration Process


*Registration tab showing transform options and phase correlation settings*

1. **Select Input Layers**:
   - Choose deskewed A and B layers
   - Parameters loaded from layer metadata

2. **Configure Registration**:
   - Select transform type based on expected misalignment
   - Enable/disable phase correlation
   - Set upsample factor

3. **Run Registration**:
   - Click "Register Data" to align the views
   - Progress updates in real-time
   - Quality metrics displayed

### Registration Results

- **Registered A**: Reference channel (unchanged)
- **Registered B**: Aligned channel B
- **Transform Matrix**: Applied transformation
- **Correlation Ratio**: Quality metric (higher = better alignment)

## Step 5: Deconvolution

### PSF Loading


*Deconvolution tab with PSF loading and algorithm selection*

1. **Load PSF Files**:
   - **PSF A**: Point spread function for channel A (.npy format)
   - **PSF B**: Point spread function for channel B (.npy format)
   - Click "Browse" to select files

2. **PSF Validation**:
   - Plugin validates PSF dimensions and format
   - Displays PSF information when loaded

### Deconvolution Parameters

1. **Algorithm Selection**:
   - **Additive**: Additive noise model
   - **DiSPIM**: DiSPIM-specific algorithm
   - **Efficient**: Memory-efficient Bayesian approach

2. **Processing Parameters**:
   - **Iterations**: Number of Richardson-Lucy iterations (10-50)
   - **Regularization**: Regularization parameter (typically 1e-6)
   - **Noise Model**: Additive or Poisson noise

3. **Chunked Processing**:
   - **Chunk Size**: Size of processing chunks (64-256)
   - **Overlap**: Overlap between chunks (16-64)
   - **GPU Memory**: Optimize for available GPU memory

### Deconvolution Process

1. **Select Input Layers**:
   - Choose registered A and B layers
   - Load PSF files for both channels

2. **Configure Parameters**:
   - Select algorithm based on your data characteristics
   - Set iteration count and regularization
   - Adjust chunking for memory constraints

3. **Run Deconvolution**:
   - Click "Deconvolve Data" to start processing
   - Progress bar shows completion status
   - Background processing allows continued interaction

### Deconvolution Results

- **Deconvolved**: Final deconvolved result
- **Processing Info**: Algorithm, iterations, and parameters used
- **Quality Metrics**: Convergence information

## Advanced Features

### GPU Acceleration

The plugin automatically uses GPU acceleration when available:

- **Registration**: GPU-accelerated optimization
- **Deconvolution**: GPU-accelerated Richardson-Lucy
- **Memory Management**: Automatic GPU memory management

### Background Processing

All long-running operations run in background threads:

- **Non-blocking**: Continue using napari while processing
- **Progress Updates**: Real-time progress reporting
- **Error Handling**: Graceful error reporting and recovery

### Data Flow

The plugin automatically manages data flow between steps:

- **Parameter Passing**: Parameters automatically passed between steps
- **Layer Management**: Automatic layer creation and naming
- **Metadata Preservation**: Processing parameters stored in layer metadata

### Memory Management

- **Lazy Loading**: Data loaded on-demand
- **Chunked Processing**: Large datasets processed in chunks
- **Automatic Cleanup**: Temporary data automatically cleaned up

## Troubleshooting

### Common Issues

1. **Plugin Not Appearing**:
   - Restart napari after installation
   - Check plugin installation: `pip list | grep napari-pyspim`

2. **CUDA Errors**:
   - Ensure CuPy is installed for your CUDA version
   - Check GPU availability: `nvidia-smi`
   - Use CPU fallback if GPU unavailable

3. **Memory Errors**:
   - Reduce chunk size in deconvolution
   - Process smaller datasets
   - Close other applications to free memory

4. **Registration Failures**:
   - Try different transform types
   - Disable phase correlation
   - Check data quality and alignment

5. **PSF Loading Errors**:
   - Ensure PSF files are .npy format
   - Check PSF dimensions match data
   - Verify PSF file integrity

### Performance Optimization

1. **GPU Memory**:
   - Monitor GPU memory usage
   - Adjust chunk sizes for your GPU
   - Use smaller datasets for testing

2. **Processing Speed**:
   - Use GPU acceleration when available
   - Reduce iteration count for faster results
   - Use lower upsample factors in registration

3. **Data Size**:
   - Use ROI detection to reduce data size
   - Process in smaller chunks
   - Consider downsampling for preview

### Getting Help

- **Error Messages**: Check the status bar for detailed error information
- **Logs**: Enable verbose logging for debugging
- **Documentation**: Refer to the API reference for technical details
- **Community**: Report issues on GitHub

## Example Workflow

### Complete Processing Pipeline

1. **Load Data**:
   - Browse to μManager acquisition folder
   - Set parameters: step_size=0.5, pixel_size=0.1625, theta=45
   - Click "Load Data"

2. **Detect ROI**:
   - Select "Process both channels"
   - Choose "Otsu" thresholding
   - Click "Detect and Apply ROI"

3. **Deskew Data**:
   - Select cropped A and B layers
   - Verify parameters from metadata
   - Click "Deskew Data"

4. **Register Views**:
   - Select deskewed A and B layers
   - Choose "t+r+s" transform
   - Enable phase correlation
   - Click "Register Data"

5. **Deconvolve**:
   - Load PSF files for both channels
   - Select "Efficient" algorithm
   - Set iterations=20, regularization=1e-6
   - Click "Deconvolve Data"

### Result Analysis

![Processing Results](../images/napari-plugin-processing-results.png)
*Final processing results showing all layers in napari viewer*

After processing, you'll have:
- **Original Data**: Raw A and B channels
- **Cropped Data**: ROI-detected and cropped volumes
- **Deskewed Data**: Corrected for light sheet geometry
- **Registered Data**: Aligned dual views
- **Deconvolved Data**: Final high-resolution result

Compare the layers in napari to see the improvement at each step.

## Next Steps

- Explore the [API Reference](api.md) for technical details
- Check [Examples](../user-guide/basic-usage.md) for usage patterns
- Review [Advanced Features](../user-guide/advanced-features.md) for complex workflows

## Image Placeholders

The images in this documentation are placeholders. To complete the documentation:

1. **Take Screenshots**: Capture the actual napari plugin interface
2. **Replace Placeholders**: Upload images to `docs/images/` directory
3. **Follow Guidelines**: See `docs/images/README.md` for detailed instructions

The required images include:
- Main interface overview
- Each processing tab (Data Loading, ROI Detection, Deskewing, Registration, Deconvolution)
- Installation and setup screenshots
- Processing results and workflow examples 