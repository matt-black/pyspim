# Sample Data and Scripts

This page describes the sample data and scripts available in `examples/data/sample_data/` for testing and demonstrating PySPIM functionality.

## Contents

### Jupyter Notebooks

- **`basic_usage.ipynb`**: Main example notebook demonstrating the complete PySPIM workflow
  - Data loading with uManager acquisition
  - ROI detection and cropping
  - Deskewing for both heads
  - Registration and transformation
  - Deconvolution

### SLURM Job Scripts

The `sh/` directory contains SLURM job scripts for batch processing:

- **`1_deskew.sh`**: CPU-based deskewing
- **`2_register.sh`**: GPU-based registration
- **`3_transform.sh`**: Data transformation
- **`4_deconvolve.sh`**: GPU-based deconvolution
- **`5a_rotate.sh`**: Data rotation

## Usage

### Running the Notebook

1. **Install dependencies**:
   ```bash
   uv pip install jupyter notebook matplotlib
   ```

2. **Start Jupyter**:
   ```bash
   cd examples/data/sample_data
   jupyter notebook
   ```

3. **Open the notebook**:
   - Navigate to `basic_usage.ipynb`
   - Run cells sequentially

### Running SLURM Jobs

1. **Modify paths** in the shell scripts to point to your data
2. **Submit jobs**:
   ```bash
   sbatch sh/1_deskew.sh
   sbatch sh/2_register.sh
   sbatch sh/3_transform.sh
   sbatch sh/4_deconvolve.sh
   sbatch sh/5a_rotate.sh
   ```

## Data Requirements

### Input Data Format

The scripts expect:
- **uManager acquisition format**: Micro-Manager data files
- **TIFF files**: Standard TIFF format
- **HDF5 files**: Hierarchical data format

### PSF Files

For deconvolution, you need:
- **PSF-A**: Point spread function for head A (`.npy` format)
- **PSF-B**: Point spread function for head B (`.npy` format)

### Acquisition Parameters

Standard diSPIM parameters:
- **Step size**: 0.5 microns
- **Pixel size**: 0.1625 microns
- **Angle**: π/4 radians (45 degrees)
- **Camera offset**: 100 (PCO.edge)

## Customization

### Using Your Own Data

1. **Update paths** in the notebook:
   ```python
   root_fldr = "/path/to/your/data"
   acq = "your_acquisition_name"
   ```

2. **Modify parameters**:
   ```python
   step_size = 0.3      # Your step size
   pixel_size = 0.1     # Your pixel size
   theta = math.pi / 6  # Your angle
   ```

3. **Load your PSFs**:
   ```python
   psf_a = numpy.load("/path/to/your/psf_a.npy")
   psf_b = numpy.load("/path/to/your/psf_b.npy")
   ```

### SLURM Configuration

Adjust SLURM parameters in the shell scripts:

**CPU Jobs**:
```bash
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=50G
```

**GPU Jobs**:
```bash
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem-per-cpu=200G
```

## Performance Tips

### GPU vs CPU

- **Use GPU for**: Registration, deconvolution, large datasets
- **Use CPU for**: Deskewing, small datasets, development

### Memory Management

- Monitor memory usage during processing
- Use chunked processing for large datasets
- Adjust SLURM memory allocation as needed

### Processing Pipeline

1. **Deskew** (CPU): Correct light sheet artifacts
2. **Register** (GPU): Align multi-view data
3. **Transform** (CPU): Apply transformations
4. **Deconvolve** (GPU): Restore image quality
5. **Rotate** (CPU): Final orientation

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce data size for testing
   - Increase SLURM memory allocation
   - Use CPU processing for large datasets

2. **GPU Errors**
   - Check CUDA installation
   - Verify GPU memory availability
   - Use CPU fallback if needed

3. **Import Errors**
   - Ensure PySPIM is installed: `uv pip install -e packages/pyspim`
   - Check Python environment

### Getting Help

- Check the [main documentation](../index.md)
- Review the [API reference](../packages/pyspim/api.md)
- Open an issue on GitHub

## File Structure

```
examples/data/sample_data/
├── basic_usage.ipynb          # Main workflow notebook
├── sh/                        # SLURM job scripts
│   ├── 1_deskew.sh           # CPU deskewing
│   ├── 2_register.sh         # GPU registration
│   ├── 3_transform.sh        # Data transformation
│   ├── 4_deconvolve.sh       # GPU deconvolution
│   └── 5a_rotate.sh          # Data rotation
└── README.md                  # Usage instructions
```

## Next Steps

- Try the [Basic Usage](basic-usage.md) tutorial
- Explore the [Snakemake Workflow](snakemake-workflow.md)
- Check out [Advanced Features](advanced-features.md)
- Review the [API Reference](../packages/pyspim/api.md) 