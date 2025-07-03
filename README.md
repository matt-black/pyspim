# pyspim
Python library for analyzing SPIM microscopy data.

## development environment setup

Dependencies:
    - `cupy`
    - `numpy`
    - `scipy`
    - `scikit-image`
    - `tqdm`

Compilation of CUB kernels requires CUDA >=12.6. To ensure this works, `cupy` has to be linked to the version already installed, as is done below (*note*: done on `della@princeton`).
```
module load anaconda3/2024.6
module load cudatoolkit/12.6
conda create -p /tigress/[user name]/conda/pyspim python=3.12
conda activate /tigress/[user name]/conda/pyspim
CUDA_PATH=/usr/local/cuda-12.6 pip install cupy
pip install tqdm
```

Other useful things:
```
pip install jupyterlab
pip install ipympl
```

Then to install this repository into your environment for development:
```
cd [root of this repository]
pip install --editable .
```
