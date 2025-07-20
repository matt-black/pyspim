# Examples Overview

This page provides an overview of the available examples and how to use them for diSPIM data processing with PySPIM.

## Examples Directory Structure

The `examples/` directory contains actual example code and data:

```
examples/
├── snakemake/                    # Snakemake workflow
│   ├── Snakefile                 # Main workflow definition
│   ├── config.yaml              # Configuration file
│   ├── slurm.yaml               # SLURM cluster configuration
│   └── scripts/                 # Workflow scripts
├── data/                        # Example data
│   ├── sample_data/             # GitHub-friendly sample data
│   │   ├── basic_usage.ipynb    # Complete workflow notebook
│   │   ├── sh/                  # SLURM job scripts
│   │   └── README.md            # Usage instructions
│   └── example_dispim_data/     # Full example datasets
└── nb/                          # Jupyter notebooks
    └── basic_usage.ipynb        # Basic usage example
```

## Available Examples

### 1. Snakemake Workflow

**Location**: `examples/snakemake/`

A complete Snakemake workflow for automated diSPIM processing pipeline.

**Features**:
- Complete processing pipeline (load → ROI → deskew → deconvolve → save)
- SLURM cluster integration
- Uses example data from `examples/data/`
- Modular and extensible design

**Quick Start**:
```bash
cd examples/snakemake/
./run_workflow.sh --local --cores 4
```

**SLURM Execution**:
```bash
cd examples/snakemake/
./run_workflow.sh --slurm --jobs 10
```

**Documentation**: See [Snakemake Workflow](snakemake-workflow.md) for detailed instructions.

### 2. Sample Data and Scripts

**Location**: `examples/data/sample_data/`

Contains sample data and SLURM job scripts for testing PySPIM functionality.

**Contents**:
- `basic_usage.ipynb` - Complete workflow demonstration
- `sh/` - SLURM job scripts for batch processing
- `README.md` - Detailed usage instructions

**Usage**:
```bash
cd examples/data/sample_data
jupyter notebook basic_usage.ipynb
```

### 3. Jupyter Notebooks

**Location**: `examples/nb/`

Contains standalone Jupyter notebooks demonstrating PySPIM usage.

**Contents**:
- `basic_usage.ipynb` - Basic usage example

## Data Sources

### Example Data
The examples use data from `examples/data/fruiting_body_subset/`:
- `fruiting_body_subset.ome.tif` - Multi-channel TIFF data
- `PSFA_demo.npy` - PSF for channel A
- `PSFB_demo.npy` - PSF for channel B
- `acquisition_params.txt` - Acquisition parameters

### Using Your Own Data
1. **For Snakemake**: Update `config.yaml` with your data paths and parameters
2. **For Notebooks**: Modify the data paths in the notebook cells
3. **For SLURM Scripts**: Update the paths in the shell scripts

## Integration Examples

### Snakemake Integration
```python
# Include in larger workflow
include: "path/to/pyspim/examples/snakemake/Snakefile"
```

### Nextflow Integration
Convert the Snakemake workflow to Nextflow processes.

### Galaxy Integration
Use individual scripts as Galaxy tools.

## Workflow Comparison

| Feature | Snakemake Workflow | Jupyter Notebooks | SLURM Scripts |
|---------|-------------------|-------------------|---------------|
| **Use Case** | Automated processing | Interactive exploration | Batch processing |
| **ROI Selection** | Automatic only | Manual or automatic | Manual or automatic |
| **Processing Steps** | Complete pipeline | Step-by-step | Individual steps |
| **Scalability** | Multiple acquisitions | Single acquisition | Multiple acquisitions |
| **Cluster Support** | SLURM integration | Manual submission | Direct SLURM |
| **Reproducibility** | Automated | Manual | Manual |
| **Learning Curve** | Moderate | Low | Low |

## Recommendations

### For Production Workflows
- Use the **Snakemake workflow** for automated processing
- Ideal for multiple acquisitions
- Excellent for cluster environments
- Ensures reproducibility

### For Learning and Development
- Use **Jupyter notebooks** for interactive exploration
- Good for understanding individual steps
- Easy to modify and experiment

### For Integration
- Snakemake workflow is particularly suitable for HPC environments
- Can be integrated into larger bioinformatics pipelines
- Individual scripts can be used as building blocks

## Troubleshooting

### Common Issues

1. **napari not found**: Install with `pip install napari`
2. **Memory issues**: Reduce data size or use chunked processing
3. **SLURM errors**: Check `slurm.yaml` configuration
4. **GPU errors**: Ensure CUDA is available and GPU memory is sufficient

### Getting Help

- Check the individual README files in each example directory
- Review the PySPIM documentation
- Examine the example data structure for reference
- Open an issue on GitHub for specific problems

## Contributing Examples

To add new examples:

1. Create a new directory or file in `examples/`
2. Include a README with usage instructions
3. Use the example data when possible
4. Document integration possibilities
5. Test with the provided example data
6. Update this overview page

## Next Steps

- Try the [Basic Usage](basic-usage.md) tutorial
- Explore the [Snakemake Workflow](snakemake-workflow.md)
- Check out [Advanced Features](advanced-features.md)
- Review the [API Reference](../packages/pyspim/api.md) 