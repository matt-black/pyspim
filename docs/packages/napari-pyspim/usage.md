# Napari Plugin Usage

Process dual-view SPIM data through an interactive napari interface.

## Quick Start

1. **Launch napari**: `napari`
2. **Open plugin**: `Plugins` → `PySPIM` → `DiSPIM Pipeline`
3. **Follow the 5-step workflow** in the tabs

## Video Tutorial

!!! video "Complete Workflow Demonstration"
    <video width="100%" controls>
      <source src="../../../media/example_usage.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>

## Workflow Steps

### 1. Data Loading
- **Browse** to μManager folder containing `a.tif` and `b.tif`
- **Set parameters**: Step size (0.5 μm), Pixel size (0.1625 μm), Theta (45°)
- **Click "Load Data"**

### 2. ROI Detection
- **Choose channels**: Both A & B, or single channel
- **Select method**: Otsu (automatic) or Manual (draw rectangle)
- **Click "Detect and Apply ROI"**

### 3. Deskewing
- **Select** cropped A and B layers
- **Verify** parameters from metadata
- **Click "Deskew Data"**

### 4. Registration
- **Select** deskewed A and B layers
- **Choose transform**: `t+r+s` (translation + rotation + scaling)
- **Enable** phase correlation
- **Click "Register Data"**

### 5. Deconvolution
- **Load PSF files** for both channels (.npy format)
- **Select algorithm**: "Efficient" (recommended)
- **Set iterations**: 20, regularization: 1e-6
- **Click "Deconvolve Data"**

## Supported Formats

- **μManager acquisitions** (standard folder structure)
- **TIFF files** (individual files per channel)
- **HDF5 files** (hierarchical data format)
- **Zarr arrays** (compressed format)

## Troubleshooting

| **Issue** | **Solution** |
|-----------|--------------|
| Plugin not appearing | Restart napari after installation |
| CUDA errors | Check GPU availability with `nvidia-smi` |
| Memory errors | Reduce chunk size in deconvolution |
| Registration failures | Try different transform types |

## Advanced Features

- **GPU Acceleration**: Automatic when available
- **Background Processing**: Non-blocking operations
- **Memory Management**: Automatic cleanup and chunking
- **Data Flow**: Parameters automatically passed between steps 