# Napari Plugin Installation

This guide covers the installation of the PySPIM napari plugin for interactive SPIM data processing.

## Prerequisites

### System Requirements

- **Python**: 3.8.1 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 8GB RAM minimum, 16GB+ recommended
- **Storage**: 10GB free space for installation and data

### GPU Requirements (Optional but Recommended)

- **CUDA-compatible GPU**: NVIDIA GPU with CUDA support
- **CUDA Version**: 11.0 or higher (12.x recommended)
- **GPU Memory**: 4GB minimum, 8GB+ recommended for large datasets

### Software Dependencies

- **napari**: 0.4.0 or higher
- **PyQt5**: 5.15.0 or higher
- **CuPy**: For GPU acceleration (optional)

## Installation Methods

### Method 1: Quick Install (Recommended)

Install both the core PySPIM library and the napari plugin:

```bash
# Install both packages
pip install pyspim napari-pyspim
```

### Method 2: Development Install

For development or latest features:

```bash
# Clone the repository
git clone https://github.com/matt-black/pyspim.git
cd pyspim

# Install with UV (recommended)
uv sync --extra dev
uv pip install -e packages/pyspim
uv pip install -e packages/napari-pyspim

# Or install with pip
pip install -e packages/pyspim
pip install -e packages/napari-pyspim
```

### Method 3: Conda Install

Using conda-forge (when available):

```bash
# Create new environment
conda create -n pyspim python=3.10
conda activate pyspim

# Install napari
conda install -c conda-forge napari

# Install PySPIM packages
pip install pyspim napari-pyspim
```

## GPU Support Installation

### CUDA Installation

1. **Check GPU Compatibility**:
   ```bash
   nvidia-smi
   ```

2. **Install CUDA Toolkit**:
   - Download from [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
   - Follow platform-specific installation instructions

3. **Install CuPy**:
   ```bash
   # For CUDA 12.x
   pip install cupy-cuda12x
   
   # For CUDA 11.x
   pip install cupy-cuda11x
   
   # For specific CUDA version
   pip install cupy-cuda11x==11.8.0
   ```

### Verify GPU Installation

```python
import cupy as cp
print(f"CuPy version: {cp.__version__}")
print(f"CUDA available: {cp.cuda.is_available()}")
print(f"GPU count: {cp.cuda.runtime.getDeviceCount()}")
```

## Platform-Specific Instructions

### Windows

1. **Install Python**:
   - Download from [python.org](https://www.python.org/downloads/)
   - Ensure "Add Python to PATH" is checked

2. **Install Visual Studio Build Tools** (if needed):
   ```bash
   # Install build tools for C++ extensions
   pip install --upgrade setuptools wheel
   ```

3. **Install PySPIM**:
   ```bash
   pip install pyspim napari-pyspim
   ```

### macOS

1. **Install Homebrew** (if not installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python**:
   ```bash
   brew install python@3.10
   ```

3. **Install PySPIM**:
   ```bash
   pip install pyspim napari-pyspim
   ```

### Linux (Ubuntu/Debian)

1. **Install System Dependencies**:
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-dev
   sudo apt install libgl1-mesa-glx libglib2.0-0
   ```

2. **Install CUDA** (for GPU support):
   ```bash
   # Add NVIDIA repository
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
   sudo dpkg -i cuda-keyring_1.0-1_all.deb
   sudo apt update
   sudo apt install cuda
   ```

3. **Install PySPIM**:
   ```bash
   pip install pyspim napari-pyspim
   ```

## Verification

### Test Installation

1. **Launch napari**:
   ```bash
   napari
   ```

2. **Check Plugin Availability**:
   - Go to `Plugins` menu
   - Look for `PySPIM` → `DiSPIM Pipeline`
   - The plugin should appear in the list


*Plugin menu showing PySPIM DiSPIM Pipeline option*

3. **Test Plugin Loading**:
   ```python
   import napari
   from napari_pyspim import DispimPipelineWidget
   
   viewer = napari.Viewer()
   widget = DispimPipelineWidget(viewer)
   print("Plugin loaded successfully!")
   ```


*Plugin widget successfully loaded and docked in napari viewer*

### Test GPU Support

```python
import napari
import numpy as np
from napari_pyspim import DispimPipelineWidget

# Create test data
test_data = np.random.randint(0, 1000, (10, 50, 50), dtype=np.uint16)

# Test plugin with GPU
viewer = napari.Viewer()
viewer.add_image(test_data, name="Test")
widget = DispimPipelineWidget(viewer)

# Check if GPU is available
try:
    import cupy as cp
    print(f"GPU available: {cp.cuda.is_available()}")
except ImportError:
    print("CuPy not installed - using CPU only")
```

## Troubleshooting

### Common Installation Issues

1. **Plugin Not Appearing**:
   ```bash
   # Check installation
   pip list | grep napari-pyspim
   
   # Reinstall plugin
   pip uninstall napari-pyspim
   pip install napari-pyspim
   
   # Restart napari
   ```

2. **Import Errors**:
   ```bash
   # Check Python environment
   python -c "import napari_pyspim; print('OK')"
   
   # Install missing dependencies
   pip install --upgrade napari qtpy PyQt5
   ```

3. **CUDA Errors**:
   ```bash
   # Check CUDA installation
   nvidia-smi
   
   # Reinstall CuPy
   pip uninstall cupy-cuda12x
   pip install cupy-cuda12x
   
   # Test CUDA
   python -c "import cupy as cp; print(cp.cuda.is_available())"
   ```

4. **Memory Issues**:
   ```bash
   # Check available memory
   free -h  # Linux
   # or
   system_profiler SPHardwareDataType  # macOS
   ```

### Dependency Conflicts

1. **PyQt5 Conflicts**:
   ```bash
   # Uninstall conflicting packages
   pip uninstall PySide2 PySide6
   
   # Install specific PyQt5 version
   pip install PyQt5==5.15.9
   ```

2. **napari Version Issues**:
   ```bash
   # Check napari version
   napari --version
   
   # Upgrade napari
   pip install --upgrade napari
   ```

3. **Python Version Issues**:
   ```bash
   # Check Python version
   python --version
   
   # Use Python 3.8+ for compatibility
   ```

### Performance Issues

1. **Slow Startup**:
   - Disable unnecessary napari plugins
   - Use SSD storage for faster loading
   - Increase system memory

2. **GPU Memory Issues**:
   ```python
   # Monitor GPU memory
   import cupy as cp
   print(f"GPU Memory: {cp.get_default_memory_pool().used_bytes() / 1e9:.2f}GB")
   ```

3. **Processing Speed**:
   - Ensure GPU acceleration is enabled
   - Use appropriate chunk sizes
   - Close other GPU applications

## Environment Management

### Virtual Environments

```bash
# Create virtual environment
python -m venv pyspim_env

# Activate environment
# Windows:
pyspim_env\Scripts\activate
# macOS/Linux:
source pyspim_env/bin/activate

# Install packages
pip install pyspim napari-pyspim
```

### Conda Environments

```bash
# Create conda environment
conda create -n pyspim python=3.10
conda activate pyspim

# Install packages
conda install -c conda-forge napari
pip install pyspim napari-pyspim
```

### Docker (Advanced)

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install napari pyspim napari-pyspim

# Set working directory
WORKDIR /data

# Run napari
CMD ["napari"]
```

## Next Steps

After successful installation:

1. **Read the [Usage Guide](usage.md)** for detailed instructions
2. **Try the [Quick Start](../getting-started/quickstart.md)** for basic workflow
3. **Explore [Examples](../user-guide/basic-usage.md)** for sample data processing
4. **Check [Troubleshooting](usage.md#troubleshooting)** if you encounter issues

## Support

- **Documentation**: This installation guide and related documentation
- **GitHub Issues**: Report bugs and request features
- **Community**: Join discussions and get help from other users

## Image Placeholders

The images in this documentation are placeholders. To complete the documentation:

1. **Take Screenshots**: Capture the actual installation and setup process
2. **Replace Placeholders**: Upload images to `docs/images/` directory
3. **Follow Guidelines**: See `docs/images/README.md` for detailed instructions

Key images needed:
- Plugin menu in napari
- Plugin widget loaded in viewer
- Installation verification screenshots 