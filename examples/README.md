# PySPIM Examples

This directory contains actual example code and data for PySPIM.

## Quick Start

```bash
# Run the basic usage notebook
cd data/sample_data
jupyter notebook basic_usage.ipynb

# Or use the Snakemake workflow
cd snakemake
snakemake --local --cores 4

# Or run individual processing scripts
cd scripts/dispim_pipeline
python deconvolve.py --help
```

## Directory Structure

- `snakemake/` - Snakemake workflow for automated processing
- `data/` - Example data and scripts
- `nb/` - Jupyter notebooks
- `scripts/` - Example processing scripts and utilities
  - `dispim_pipeline/` - Individual processing scripts (deconvolve, transform, deskew, register)
  - `profiling/` - Performance profiling scripts and tests 