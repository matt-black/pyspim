# Installation

This guide will help you install PySPIM and its dependencies.

## Prerequisites

- Python 3.8.1 or higher
- CUDA-compatible GPU (optional, for GPU acceleration)
- Git

## Installation Methods

**Note**: PySPIM is currently in development and not yet published on PyPI or conda-forge. All installation methods require cloning the repository.

### Method 1: Using UV (Recommended)

UV is a fast Python package manager that we recommend:

```bash
# Install UV if you haven't already
pip install uv

# Clone the repository
git clone https://github.com/matt-black/pyspim.git
cd pyspim

# Install packages (basic usage)
just install

# Or manually
uv pip install -e packages/pyspim
uv pip install -e packages/napari-pyspim
```

### Method 2: Using pip (Development Only)

**Note**: PySPIM is not yet published on PyPI. This method is for future use.

```bash
# Install the core package (when available)
pip install pyspim

# Install the napari plugin (when available)
pip install napari-pyspim
```

### Method 3: Using conda (Development Only)

**Note**: PySPIM is not yet published on conda-forge. This method is for future use.

```bash
# Create a new conda environment
conda create -n pyspim python=3.10
conda activate pyspim

# Install the packages (when available)
pip install pyspim napari-pyspim
```

## Development Installation

For development and contributing (includes ruff, pre-commit, testing tools):

```bash
# Clone the repository
git clone https://github.com/matt-black/pyspim.git
cd pyspim

# Install packages + development tools
just install-dev

# Or manually with UV
uv sync --extra dev
uv pip install -e packages/pyspim
uv pip install -e packages/napari-pyspim

# Install pre-commit hooks
just pre-commit
```

## CUDA Support

For GPU acceleration, you'll need CUDA-compatible hardware and drivers:

```bash
# Install CUDA toolkit (version 12.x recommended)
# Follow NVIDIA's installation guide for your system

# Install CuPy with CUDA support
pip install cupy-cuda12x
```

## Development Workflow

PySPIM uses modern development tools for code quality and automation:

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

### Code Quality Tools

- **Ruff**: Fast Python linter and formatter (replaces black, isort, flake8)
- **Pre-commit**: Automated code quality checks
- **Type hints**: Full mypy support
- **Testing**: Comprehensive test suite with GPU and integration tests

### Verification

After installation, verify that everything works:

```python
# Test core package
import pyspim
print(f"PySPIM version: {pyspim.__version__}")

# Test napari plugin
import napari
import napari_pyspim
viewer = napari.Viewer()
# Check that the plugin appears in the plugins menu
```

## Troubleshooting

### Common Issues

1. **CUDA not found**: Make sure you have CUDA drivers installed and the correct CuPy version
2. **Import errors**: Ensure you're in the correct Python environment
3. **Plugin not showing in napari**: Try restarting napari after installation
4. **Just command not found**: Install just: `conda install -c conda-forge just`

### Getting Help

- Check the [GitHub issues](https://github.com/matt-black/pyspim/issues)
- Join our [Discussions](https://github.com/matt-black/pyspim/discussions)
- Read the [API documentation](packages/pyspim/api.md)

## Next Steps

- Read the [Quick Start Guide](quickstart.md)
- Explore the [Examples](../user-guide/basic-usage.md)
- Check out the [API Reference](packages/pyspim/api.md) 