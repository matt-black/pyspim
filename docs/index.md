# Welcome to PySPIM

PySPIM is a comprehensive Python library for analyzing and visualizing single plane illumination microscopy (SPIM) data, with special emphasis on dual-view SPIM (diSPIM) microscopy.

## What is PySPIM?

PySPIM provides a complete pipeline for processing SPIM microscopy data, including:

- **Data Loading**: Efficient loading of various microscopy data formats (uManager, TIFF, HDF5)
- **ROI Detection**: Automated region of interest detection and cropping
- **Deskewing**: Correction of light sheet microscopy artifacts
- **Registration**: Multi-view image registration with GPU acceleration
- **Deconvolution**: Advanced deconvolution algorithms with PSF support

## Key Features

- 🚀 **High Performance**: GPU-accelerated processing with CuPy
- 🔧 **Modular Design**: Separate core library and napari plugin
- 📊 **Interactive Visualization**: Full napari integration with parameter passing
- 🧪 **Research Ready**: Designed for scientific workflows with SLURM support
- 📚 **Well Documented**: Comprehensive API documentation and examples
- 🛠️ **Modern Development**: UV package management, ruff linting, pre-commit hooks

## Quick Start

### Using the Napari Plugin

```python
import napari
from napari_pyspim import DispimPipelineWidget

# Create viewer and add your data
viewer = napari.Viewer()
viewer.add_image(your_data, metadata={
    'step_size': 0.5,
    'pixel_size': 0.1625,
    'theta': 0.785,  # π/4 radians
    'camera_offset': 100
})

# Add the PySPIM plugin
widget = DispimPipelineWidget(viewer)
viewer.window.add_dock_widget(widget, name="diSPIM Pipeline")
```

### Command Line Processing

```bash
# Install in development mode
just install-dev

# Run the complete pipeline
python examples/data/sample_data/basic_usage.ipynb

# Or use SLURM batch processing
sbatch examples/data/sample_data/sh/1_deskew.sh
sbatch examples/data/sample_data/sh/2_register.sh
sbatch examples/data/sample_data/sh/3_transform.sh
sbatch examples/data/sample_data/sh/4_deconvolve.sh
```

## Installation

**Note**: PySPIM is currently in development and not yet published on PyPI or conda-forge.

### Installation

```bash
# Clone the repository
git clone https://github.com/matt-black/pyspim.git
cd pyspim

# Install with just (recommended)
just install

# Or manually
uv pip install -e packages/pyspim
uv pip install -e packages/napari-pyspim
```

### Future Production Installation

Once PySPIM is published on PyPI:

```bash
# Install the core package
pip install pyspim

# Install the napari plugin
pip install napari-pyspim
```

## Development Workflow

### Using Just (Task Runner)

```bash
# List all available commands
just --list

# Run tests
just test-fast

# Format and lint code
just format
just lint

# Build documentation
just docs-serve

# Run all checks
just check
```

### Code Quality

- **Ruff**: Fast Python linter and formatter
- **Pre-commit**: Automated code quality checks
- **Type hints**: Full mypy support
- **Testing**: Comprehensive test suite with GPU and integration tests

## Documentation Structure

- **[Getting Started](getting-started/installation.md)**: Installation and basic setup
- **[Core Package](packages/pyspim/overview.md)**: Main PySPIM library documentation
- **[Napari Plugin](packages/napari-pyspim/overview.md)**: Interactive GUI documentation
- **[Examples](examples/basic-usage.md)**: Tutorials and use cases
- **[API Reference](packages/pyspim/api.md)**: Complete API documentation

## Example Data

We provide sample data and scripts for testing:

- **Jupyter Notebook**: `examples/data/sample_data/basic_usage.ipynb`
- **SLURM Scripts**: `examples/data/sample_data/sh/*.sh`
- **Documentation**: See [examples/basic-usage.md](examples/basic-usage.md)

## Contributing

We welcome contributions! Please see our [Contributing Guide](development/contributing.md) for details.

## License

This project is licensed under the GPL-3.0 License - see the [License](about/license.md) file for details.

## Citation

If you use PySPIM in your research, please cite:

```bibtex
@software{pyspim2024,
  title={PySPIM: Single Plane Illumination Microscopy Analysis},
  author={PySPIM Team},
  year={2024},
  url={https://github.com/matt-black/pyspim}
}
``` 