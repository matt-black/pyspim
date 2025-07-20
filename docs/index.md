# Welcome to PySPIM

PySPIM is a comprehensive Python library for analyzing and visualizing selective plane illumination microscopy (SPIM) data, with special emphasis on dual-view SPIM (diSPIM) microscopy.

## 🧭 Quick Navigation

- **[🚀 Install Now](getting-started/installation.md)** - Get started in minutes
- **[📖 Quick Start](getting-started/quickstart.md)** - Basic workflow tutorial
- **[🖥️ Napari Plugin](packages/napari-pyspim/overview.md)** - Interactive GUI
- **[📦 Core Package](packages/pyspim/overview.md)** - Library documentation
- **[🧪 Examples](user-guide/basic-usage.md)** - Tutorials and use cases

## 🚀 Quick Installation

```bash
# Clone and install
git clone https://github.com/matt-black/pyspim.git
cd pyspim
just install-dev
```

## 🎯 What is PySPIM?

PySPIM provides a complete pipeline for processing SPIM microscopy data:

- **Data Loading**: Efficient loading of various microscopy data formats
- **ROI Detection**: Automated region of interest detection and cropping
- **Deskewing**: Correction of light sheet microscopy artifacts
- **Registration**: Multi-view image registration with GPU acceleration
- **Deconvolution**: Advanced deconvolution algorithms with PSF support

## 🔧 Key Features

- 🚀 **High Performance**: GPU-accelerated processing with CuPy
- 🔧 **Modular Design**: Separate core library and napari plugin
- 📊 **Interactive Visualization**: Full napari integration
- 🧪 **Research Ready**: Designed for scientific workflows with SLURM support
- 📚 **Well Documented**: Comprehensive API documentation and examples

## 🚀 Quick Start

### Using the Napari Plugin

```python
import napari

# Launch napari and load the PySPIM plugin
viewer = napari.Viewer()
# Go to Plugins → PySPIM → DiSPIM Pipeline
```

### Command Line Processing

```python
import pyspim

# Load and process SPIM data
data = pyspim.load_data("path/to/data.tif")
processed = pyspim.process_pipeline(data)
pyspim.save_data(processed, "output.tif")
```

### Automated Workflow

```bash
# Use Snakemake for high-throughput processing
cd examples/snakemake
snakemake --slurm --jobs 10
```

## 📚 Documentation

### 🚀 Getting Started
- **[Installation Guide](getting-started/installation.md)** - Detailed setup instructions
- **[Quick Start](getting-started/quickstart.md)** - Get up and running in minutes

### 📦 Core Package
- **[Overview](packages/pyspim/overview.md)** - Main PySPIM library documentation
- **[API Reference](packages/pyspim/api.md)** - Complete function documentation

### 🖥️ Napari Plugin
- **[Overview](packages/napari-pyspim/overview.md)** - Interactive GUI documentation
- **[Installation](packages/napari-pyspim/installation.md)** - Plugin setup guide
- **[Usage](packages/napari-pyspim/usage.md)** - Detailed usage instructions

### 📖 User Guide
- **[Basic Usage](user-guide/basic-usage.md)** - Tutorials and use cases
- **[Examples Overview](user-guide/examples-overview.md)** - Available examples
- **[Advanced Features](user-guide/advanced-features.md)** - Complex workflows

## 🧪 Examples

- **[Basic Usage](user-guide/basic-usage.md)** - Step-by-step tutorials
- **[Fruiting Body Workflow](user-guide/fruiting-body-workflow.md)** - Complete workflow demonstration
- **[Snakemake Workflow](user-guide/snakemake-workflow.md)** - Automated processing pipeline
- **[Jupyter Notebooks](user-guide/notebooks.md)** - Interactive examples

## 🛠️ Development

```bash
# Run tests
just test

# Format code
just format

# Build documentation
just docs-serve
```

See [Contributing Guide](development/contributing.md) for development details.

## 📄 License

This project is licensed under the GPL-3.0 License - see the [License](about/license.md) file for details.

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