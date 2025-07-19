# pyspim

Python library for analyzing SPIM microscopy data.

This is a monorepo containing:
- **pyspim**: Core library for SPIM data analysis
- **napari-pyspim**: Napari plugin for diSPIM processing pipeline

## Repository Structure

```
pyspim/
├── pyspim/           # Core library package
├── napari-pyspim/    # Napari plugin package (npe2 compatible)
├── pyproject.toml    # Root pyproject.toml for monorepo
└── README.md         # This file
```

## Development Environment Setup

### Prerequisites

Dependencies:
- `cupy`
- `numpy`
- `scipy`
- `scikit-image`
- `tqdm`

Compilation of CUB kernels requires CUDA >=12.6. To ensure this works, `cupy` has to be linked to the version already installed, as is done below (*note*: done on `della@princeton`).

```bash
module load anaconda3/2024.6
module load cudatoolkit/12.6
conda create -p /tigress/[user name]/conda/pyspim python=3.12
conda activate /tigress/[user name]/conda/pyspim
CUDA_PATH=/usr/local/cuda-12.6 pip install cupy
pip install tqdm
```

Other useful things:
```bash
pip install jupyterlab
pip install ipympl
```

### Installing for Development

#### Core Library
```bash
cd pyspim
pip install --editable .
```

#### Napari Plugin
```bash
cd napari-pyspim
pip install --editable .
```

#### Both Packages
```bash
# Install core library first
cd pyspim
pip install --editable .

# Then install plugin
cd ../napari-pyspim
pip install --editable .
```

## Usage

### Core Library
```python
import pyspim

# Use the core functionality
```

### Napari Plugin
1. Install the plugin: `pip install napari-pyspim`
2. Launch napari: `napari`
3. Go to Plugins > diSPIM Processing Pipeline
4. Use the diSPIM Pipeline widget to process your data

## License

GPL-3.0-only
