# Installation

Install PySPIM for development and research use.

## Prerequisites

- Python 3.8.1 or higher
- CUDA-compatible GPU (optional, for GPU acceleration)
- Git

## Installation

PySPIM is currently in development. Install from source:

```bash
# Clone the repository
git clone https://github.com/matt-black/pyspim.git
cd pyspim

# Install packages (basic usage)
just install

# Or manually with UV
uv pip install -e packages/pyspim
uv pip install -e packages/napari-pyspim
```

## Development Installation

For development and contributing:

```bash
# Install packages + development tools
just install-dev

# Or manually
uv sync --extra dev
uv pip install -e packages/pyspim
uv pip install -e packages/napari-pyspim

# Install pre-commit hooks
just pre-commit
```

## CUDA Support

For GPU acceleration:

```bash
# Install CuPy with CUDA support
pip install cupy-cuda12x
```

## Development Workflow

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
```

## Verification

After installation:

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

| **Issue** | **Solution** |
|-----------|--------------|
| CUDA not found | Install CUDA drivers and correct CuPy version |
| Import errors | Ensure you're in the correct Python environment |
| Plugin not showing | Restart napari after installation |
| Just command not found | Install just: `conda install -c conda-forge just` |

## Next Steps

- [Quick Start Guide](quickstart.md)
- [Basic Usage](../user-guide/basic-usage.md)
- [API Reference](../packages/pyspim/api.md) 