# PySPIM Sample Data

This directory contains sample data and scripts for testing PySPIM functionality.

## Documentation

For detailed documentation, see:
- [Sample Data Documentation](../../../../examples/sample-data.md)
- [Basic Usage Documentation](../../../../examples/basic-usage.md)

## Quick Start

```bash
# Run the notebook
jupyter notebook basic_usage.ipynb

# Or run SLURM jobs
sbatch sh/1_deskew.sh
sbatch sh/2_register.sh
sbatch sh/3_transform.sh
sbatch sh/4_deconvolve.sh
```

## Files

- `basic_usage.ipynb` - Complete workflow demonstration
- `sh/` - SLURM job scripts for batch processing 