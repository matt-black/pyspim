# PySPIM

A comprehensive Python library for analyzing and visualizing Selective plane illumination microscopy (SPIM) data, with special emphasis on dual-view SPIM (diSPIM) microscopy.

## 🚀 Features

- **Complete SPIM Pipeline**: Data loading, ROI detection, deskewing, registration, and deconvolution
- **High Performance**: Built on Dask for scalable, parallel processing
- **GPU Acceleration**: CUDA support with CuPy for fast computation
- **Interactive GUI**: Full napari integration with tabbed workflow
- **Modular Design**: Separate core library and napari plugin
- **Modern Tooling**: UV-based dependency management and build system

## 📦 Installation

### Quick Install

```bash
# Install both packages
pip install pyspim napari-pyspim
```

### Development Install

```bash
# Clone the repository
git clone https://github.com/matt-black/pyspim.git
cd pyspim

# Install with UV (recommended)
uv sync --extra dev
uv pip install -e packages/pyspim
uv pip install -e packages/napari-pyspim
```

## 🏗️ Project Structure

This is a modern monorepo using UV for dependency management:

```
pyspim/
├── packages/
│   ├── pyspim/              # Core SPIM processing library
│   └── napari-pyspim/       # Napari plugin for GUI
├── docs/                    # Documentation (MkDocs + Material)
├── examples/                # Example data and notebooks
├── tools/                   # Development tools
└── pyproject.toml          # UV workspace configuration
```

## 🚀 Quick Start

### Command Line Usage

```python
import pyspim

# Load and process SPIM data
data = pyspim.load_data("path/to/data.tif")
processed = pyspim.process_pipeline(data)
pyspim.save_data(processed, "output.tif")
```

### Napari Plugin Usage

```python
import napari

# Launch napari and load the PySPIM plugin
viewer = napari.Viewer()
# Go to Plugins → PySPIM → DiSPIM Pipeline
```

## 📚 Documentation

- **📖 [Full Documentation](https://pyspim.readthedocs.io/)** - Comprehensive guides and API reference
- **🚀 [Quick Start Guide](docs/getting-started/quickstart.md)** - Get up and running in minutes
- **📦 [Installation Guide](docs/getting-started/installation.md)** - Detailed setup instructions
- **🔧 [API Reference](docs/packages/pyspim/api.md)** - Complete function documentation

### Building Documentation Locally

```bash
# Build documentation
make docs

# Serve documentation locally
make docs-serve
```

## 🛠️ Development

### Prerequisites

- Python 3.8.1+
- UV (recommended) or pip
- CUDA-compatible GPU (optional)

### Development Commands

```bash
# Install development environment
make install-dev

# Run tests
make test

# Format code
make format

# Run linting
make lint

# Build packages
make build

# Clean build artifacts
make clean
```

### Contributing

We welcome contributions! Please see our [Contributing Guide](docs/development/contributing.md) for details.

## 📋 Requirements

### Core Dependencies

- **numpy**: Numerical computing
- **dask**: Parallel processing
- **scikit-image**: Image processing
- **tifffile**: TIFF file I/O
- **zarr**: Array storage
- **cupy-cuda12x**: GPU acceleration (optional)

### GUI Dependencies

- **napari**: Scientific image viewer
- **npe2**: Napari plugin engine
- **qtpy**: Qt bindings
- **PyQt5**: GUI framework

## 🎯 Use Cases

- **Light Sheet Microscopy**: Process SPIM/diSPIM data
- **Multi-view Registration**: Align multiple imaging views
- **Deconvolution**: Improve image resolution
- **ROI Analysis**: Focus on specific regions of interest
- **Batch Processing**: Process multiple datasets efficiently

## 📄 License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0) - see the [License](docs/about/license.md) file for details.

## 🤝 Citation

If you use PySPIM in your research, please cite:

```bibtex
@software{pyspim2024,
  title={PySPIM: Selective Plane Illumination Microscopy Analysis},
  author={PySPIM Team},
  year={2024},
  url={https://github.com/matt-black/pyspim}
}
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/matt-black/pyspim/issues)
- **Discussions**: [GitHub Discussions](https://github.com/matt-black/pyspim/discussions)
- **Documentation**: [Full Documentation](https://pyspim.readthedocs.io/)

---

**PySPIM** - Making SPIM data analysis accessible and efficient! 🔬✨
