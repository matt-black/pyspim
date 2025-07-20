# Snakemake Workflow

A streamlined Snakemake workflow for automated diSPIM data processing with GPU acceleration on SLURM clusters.

## Overview

The Snakemake workflow provides a complete, automated pipeline for processing diSPIM data from raw TIFF files to deconvolved results. It's designed for high-throughput processing on SLURM clusters with GPU acceleration.

## Quick Start

```bash
# Navigate to the workflow directory
cd ../../../examples/snakemake

# Activate the pyspim environment
conda activate pyspim

# Run locally
snakemake

# Run on SLURM cluster
snakemake --slurm --jobs 10
```

## Workflow Steps

The pipeline processes data through these automated steps:

1. **Load Data** - Load raw TIFF files with camera correction
2. **ROI Detection** - Automatically crop to regions of interest  
3. **Deskew** - Correct stage scanning angle (GPU-accelerated)
4. **Deconvolve** - Richardson-Lucy deconvolution (GPU-accelerated)

## Configuration

Edit `config.yaml` to configure your data:

```yaml
# Data paths
data_dir: "path/to/your/data"
psf_dir: "path/to/psf/files"

# Acquisitions to process
acquisitions:
  - "your_acquisition_name"

# Acquisition parameters
step_size: 0.5      # microns
pixel_size: 0.1625  # microns  
theta: 45           # degrees

# Processing parameters
roi_method: "otsu"
decon_iterations: 10

# PSF files
psf_a: "PSFA_demo.npy"
psf_b: "PSFB_demo.npy"
```

## Output Structure

```
results/{acquisition}/
├── raw/           # Loaded raw data
├── cropped/       # ROI-detected and cropped data
├── deskewed/      # Deskewed data (GPU-accelerated)
└── deconvolved/   # Final deconvolved result (GPU-accelerated)
```

## SLURM Integration

The workflow automatically handles SLURM job submission with GPU resource requests:

### Resource Management
- **GPU Jobs**: Automatically request `--gres=gpu:1` for deskew and deconvolution
- **Memory**: Configurable per rule (4GB for loading, 8GB for ROI, 16GB for deskew, 32GB for deconvolution)
- **Runtime**: Configurable time limits per processing step
- **Parallel Execution**: Use `--jobs N` for concurrent processing

### Example SLURM Execution
```bash
# Submit workflow to SLURM
snakemake --slurm --jobs 10

# Monitor jobs
squeue -u $USER

# Check logs
ls .snakemake/slurm_logs/
```

## Adding New Acquisitions

1. **Add to config.yaml**:
   ```yaml
   acquisitions:
     - "existing_acquisition"
     - "new_acquisition"
   ```

2. **Ensure data files exist**:
   ```
   {data_dir}/new_acquisition/
   ├── a.tif
   ├── b.tif
   ├── PSFA.npy
   └── PSFB.npy
   ```

3. **Run the workflow**:
   ```bash
   snakemake --slurm --jobs 10
   ```

## Customization

### Modifying Resource Requirements
Edit the `resources` section in `Snakefile`:

```python
resources:
    runtime = 240,        # minutes
    mem_mb = 32000,       # memory in MB
    slurm_extra = "--gres=gpu:1"  # GPU request
```

### Adding Processing Steps
The workflow is modular and easily extensible. Add new rules to `Snakefile` for additional processing steps.

### Different ROI Methods
Change the ROI detection method in `config.yaml`:
```yaml
roi_method: "triangle"  # Options: "otsu", "triangle"
```

## Troubleshooting

### Common Issues

**GPU Issues**:
- Ensure you're on a GPU node
- Check CUDA availability: `nvidia-smi`
- Verify GPU resource requests in Snakefile

**Memory Issues**:
- Increase `mem_mb` in Snakefile rules
- Monitor memory usage: `squeue -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"`

**Time Limits**:
- Increase `runtime` in Snakefile rules
- Monitor job duration: `sacct -j <jobid>`

### Debug Mode
```bash
# Verbose output
snakemake --verbose

# Dry run
snakemake -n

# Clean restart
rm -rf results && snakemake
```

## Performance Tips

1. **Parallel Processing**: Use `--jobs N` to process multiple acquisitions concurrently
2. **GPU Utilization**: Ensure GPU jobs run on appropriate nodes
3. **Storage**: Use fast storage for intermediate files
4. **Monitoring**: Check SLURM logs for performance insights

## Example Output

After successful execution, you'll find:
- Raw, cropped, and deskewed data as TIFF files
- Final deconvolved result as TIFF file
- ROI coordinates and processing metadata
- SLURM job logs for monitoring

## Integration with Other Workflows

The Snakemake workflow can be integrated into larger bioinformatics pipelines:

### Snakemake Integration
```python
# In your main Snakefile
include: "path/to/pyspim/examples/snakemake/Snakefile"

rule process_all:
    input:
        expand("results/{acquisition}/deconvolved/deconvolved.tif", 
               acquisition=config["acquisitions"])
    # ... additional processing
```

### Nextflow Integration
Convert the workflow to Nextflow or integrate with existing Nextflow pipelines.

## Comparison with Other Methods

| Method | Pros | Cons |
|--------|------|------|
| **Snakemake** | Automated, reproducible, SLURM integration | Learning curve, setup required |
| **Notebook** | Interactive, educational | Manual, not scalable |
| **Command-line** | Flexible, direct control | Manual job management |

The Snakemake workflow is ideal for production processing of multiple acquisitions with minimal manual intervention. 