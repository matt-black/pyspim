# napari-pyspim

A napari plugin for diSPIM processing pipeline.

## Description

This plugin provides a comprehensive interface for processing dual-view SPIM data, including data loading, ROI detection, deskewing, registration, and deconvolution.

## Installation

```bash
pip install napari-pyspim
```

## Development Setup

For development installation:
```bash
cd napari-pyspim
pip install --editable .
```

## Usage

1. Install the plugin: `pip install napari-pyspim`
2. Launch napari: `napari`
3. Go to Plugins > diSPIM Processing Pipeline
4. Use the diSPIM Pipeline widget to process your data

## Features

- Data loading for SPIM datasets
- ROI detection and management
- Deskewing algorithms
- Registration tools
- Deconvolution processing
- Complete diSPIM pipeline integration

## Dependencies

- pyspim (core library)
- napari>=0.4.0
- napari-plugin-engine>=0.2.0
- qtpy>=1.9.0
- PyQt5>=5.15.0
- cupy-cuda12x
- matplotlib>=3.5.0

## License

GPL-3.0-only 