# spimlib
Python library & napari plugin for analyzing SPIM microscopy data.

## development environment setup

To get all of the dependencies (*note*: done on `della@princeton`):
```
module load anaconda3/2023.9
module load cudatoolkit/12.2
conda create -p /tigress/[user name]/conda/napari-env python=3.10
conda activate /tigress/[user name]/conda/napari-env
conda install -c conda-forge mamba
mamba install -c conda-forge napari
mamba install -c rapidsai -c conda-forge -c nvidia cuml=24.04 cucim=24.04 cuda-version=12.2
```
*note*: the rapids installation is quite general and probably brings along a bunch of shit we don't need. We'll trim later. I also don't know if the `module load cudatoolkit/12.2` is necessary, but for now it stays because this worked. 

I like to have a JupyterLab instance in my development environment:
```
mamba install -c conda-forge jupyterlab ipympl
```

Then to install this repository into your environment for development:
```
cd [root of this repository]
pip install --editable .
```
