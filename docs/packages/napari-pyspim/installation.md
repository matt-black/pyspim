# Napari Plugin Installation

Install the PySPIM napari plugin for interactive SPIM data processing.

## Prerequisites

- **Python**: 3.8.1 or higher
- **napari**: 0.4.0 or higher
- **CUDA-compatible GPU** (optional, for GPU acceleration)

## Installation

PySPIM is currently in development. Install from source:

```bash
# Clone the repository
git clone https://github.com/matt-black/pyspim.git
cd pyspim

# Install with UV (recommended)
uv pip install -e packages/pyspim
uv pip install -e packages/napari-pyspim

# Or with just
just install
```

## GPU Support

For GPU acceleration:

```bash
# Install CuPy with CUDA support
pip install cupy-cuda12x
```

## Verification

1. **Launch napari**:
   ```bash
   napari
   ```

2. **Check Plugin**:
   - Go to `Plugins` menu
   - Look for `PySPIM` → `DiSPIM Pipeline`

3. **Test Loading**:
   ```python
   import napari
   from napari_pyspim import DispimPipelineWidget
   
   viewer = napari.Viewer()
   widget = DispimPipelineWidget(viewer)
   print("Plugin loaded successfully!")
   ```

## Troubleshooting

| **Issue** | **Solution** |
|-----------|--------------|
| Plugin not appearing | Restart napari after installation |
| CUDA errors | Check GPU availability with `nvidia-smi` |
| Import errors | Ensure you're in the correct Python environment |

## Next Steps

- [Usage Guide](usage.md) - How to use the plugin
- [Basic Usage](../../user-guide/basic-usage.md) - Examples 