# napari-pyspim: diSPIM Processing Pipeline

A comprehensive napari plugin for processing dual-view SPIM (diSPIM) microscopy data. This plugin provides a user-friendly interface for the complete diSPIM processing pipeline, from raw data loading to final deconvolution.

## Features

- **Data Loading**: Load μManager acquisitions with automatic parameter detection
- **ROI Detection**: Automated region of interest detection with multiple threshold methods
- **Deskewing**: Correct for the oblique illumination geometry of SPIM
- **Registration**: Align dual views using phase correlation and optimization
- **Deconvolution**: Richardson-Lucy dual-view deconvolution with GPU acceleration

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (for deconvolution)
- napari 0.4.0 or higher

### Install with uv

```bash
# Clone the repository
git clone https://github.com/matt-black/pyspim.git
cd pyspim

# Install with uv
uv pip install -e .
```

### Install with pip

```bash
pip install napari-pyspim
```

## Usage

### Starting the Plugin

1. Launch napari
2. Go to `Plugins` → `diSPIM Processing Pipeline`
3. The plugin will open as a docked widget with 5 tabs

### Processing Pipeline

#### 1. Data Loading
- Select your μManager acquisition folder
- Set acquisition parameters (step size, pixel size, theta angle)
- Set camera offset (typically 100 for PCO.edge cameras)
- Click "Load Data" to load the raw A and B volumes

#### 2. ROI Detection
- Choose threshold method (Otsu, Triangle, or Manual)
- Click "Detect and Apply ROI" to automatically crop the data
- View ROI coordinates and cropped volume sizes

#### 3. Deskewing
- Select deskewing method (currently supports 'orthogonal')
- Optionally enable re-cropping after deskewing
- Click "Deskew Data" to correct for oblique illumination

#### 4. Registration
- Choose transform type (translation, rotation, scaling)
- Enable/disable phase correlation for initial guess
- Set upsample factor for phase correlation
- Click "Register Data" to align the dual views

#### 5. Deconvolution
- Load PSF files for both views (A and B)
- Set deconvolution parameters:
  - Iterations (typically 10-30)
  - Regularization (typically 1e-6)
  - Noise model (additive or poisson)
  - Chunk size and overlap for large datasets
- Click "Deconvolve Data" to perform Richardson-Lucy deconvolution

### Data Flow

The plugin automatically passes data between steps:
- Data Loading → ROI Detection
- ROI Detection → Deskewing  
- Deskewing → Registration
- Registration → Deconvolution

Each step adds new layers to the napari viewer, so you can compare results at each stage.

### Output

- All intermediate results are displayed as napari layers
- Final deconvolved result is available as the "Deconvolved" layer
- Processing parameters are stored in layer metadata
- Results can be exported using napari's built-in export functions

## File Formats

### Input
- **Raw Data**: μManager acquisition folders
- **PSFs**: NumPy (.npy) files containing 3D point spread functions

### Output
- **Intermediate**: napari layers with metadata
- **Final**: Deconvolved volume as napari layer
- **Export**: Can be saved as OME-TIFF, Zarr, or other formats

## Performance

- **GPU Acceleration**: Registration and deconvolution use CUDA acceleration
- **Memory Management**: Large datasets are processed in chunks
- **Background Processing**: All operations run in background threads
- **Progress Tracking**: Real-time progress updates for long operations

## Troubleshooting

### Common Issues

1. **CUDA Errors**: Ensure you have a compatible GPU and CUDA installation
2. **Memory Errors**: Reduce chunk size for large datasets
3. **PSF Loading**: Ensure PSF files are 3D NumPy arrays
4. **Registration Failures**: Try different transform types or disable phase correlation

### Getting Help

- Check the napari console for error messages
- Verify all input parameters are correct
- Ensure sufficient disk space for temporary files
- Check GPU memory usage during deconvolution

## Development

### Building from Source

```bash
git clone https://github.com/matt-black/pyspim.git
cd pyspim
uv pip install -e ".[testing]"
```

### Running Tests

```bash
pytest src/napari_pyspim/tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This plugin is licensed under the GPL-3.0 license. See the LICENSE file for details.

## Citation

If you use this plugin in your research, please cite:

```bibtex
@software{napari_pyspim,
  title={napari-pyspim: diSPIM Processing Pipeline},
  author={Black, Matthew},
  year={2024},
  url={https://github.com/matt-black/pyspim}
}
``` 