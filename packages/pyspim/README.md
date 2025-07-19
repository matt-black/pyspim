# pyspim

Core Python library for analyzing SPIM microscopy data.

## Description

Core functionality for analyzing and visualizing selective plane illumination microscopy (spim) data with special emphasis on dual-view SPIM (diSPIM) microscopy.

## Installation

```bash
pip install pyspim
```

## Development Setup

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

For development installation:
```bash
cd pyspim
pip install --editable .
```

## Usage

```python
import pyspim

# Use the core functionality
```

## License

GPL-3.0-only 