# Advanced Features

This page covers advanced features and capabilities of PySPIM for sophisticated SPIM data processing workflows.

## Advanced Deconvolution Algorithms

PySPIM provides multiple deconvolution algorithms optimized for different scenarios:

### 1. Richardson-Lucy Variants

#### Additive Joint RL
```python
from pyspim.decon.rl.dualview_fft import additive_joint_rl

# Additive noise model
result = additive_joint_rl(
    view_a, view_b, initial_estimate,
    psf_a, psf_b, bp_a, bp_b,
    num_iter=20, epsilon=1e-6,
    boundary_correction=True
)
```

#### DiSPIM Joint RL
```python
from pyspim.decon.rl.dualview_fft import joint_rl_dispim

# DiSPIM-specific algorithm
result = joint_rl_dispim(
    view_a, view_b, initial_estimate,
    psf_a, psf_b, bp_a, bp_b,
    num_iter=20, epsilon=1e-6,
    boundary_correction=True
)
```

#### Efficient Bayesian RL
```python
from pyspim.decon.rl.dualview_fft import efficient_bayesian

# Memory-efficient Bayesian approach
result = efficient_bayesian(
    view_a, view_b, initial_estimate,
    psf_a, psf_b,
    num_iter=20, epsilon=1e-6,
    boundary_correction=True
)
```

### 2. Chunked Processing for Large Datasets

For datasets that don't fit in GPU memory:

```python
from pyspim.decon.rl.dualview_fft import deconvolve_chunkwise
import zarr

# Create output zarr array
output_zarr = zarr.create(
    shape=view_a.shape, 
    dtype=np.float32,
    chunks=(64, 64, 64)
)

# Process in chunks
deconvolve_chunkwise(
    view_a_zarr, view_b_zarr, output_zarr,
    chunk_size=(128, 128, 128),
    overlap=(32, 32, 32),
    psf_a, psf_b, bp_a, bp_b,
    decon_function="additive",
    num_iter=20,
    verbose=True
)
```

## Advanced Registration

### 1. Multi-Scale Registration

```python
from pyspim.reg import powell
from pyspim.util import downscale_2x

# Downscale for initial registration
a_down = downscale_2x(a_deskewed, (8, 8, 8))
b_down = downscale_2x(b_deskewed, (8, 8, 8))

# Initial registration on downscaled data
T_down, res_down = powell.optimize_affine_piecewise(
    a_down, b_down,
    metric="cr",
    transform="t+r+s",
    interp_method="cubspl",
    par0=[0, 0, 0, 0, 0, 0, 1, 1, 1],
    bounds=[(t-10, t+10) for t in [0,0,0]] + [(-2, 2)]*3 + [(0.9, 1.1)]*3
)

# Use result as initial guess for full resolution
par0 = np.concatenate([res_down.x[:3] * 2, res_down.x[3:]])
```

### 2. Phase Cross Correlation Pre-registration

```python
from pyspim.reg import pcc

# Get initial translation estimate
t0 = pcc.translation_for_volumes(
    a_deskewed, b_deskewed, 
    upsample_factor=10
)

# Use as initial parameters
par0 = np.concatenate([t0, [0, 0, 0, 1, 1, 1]])
```

### 3. Custom Transform Types

```python
# Translation only
transform = "t"
bounds = [(t-20, t+20) for t in [0, 0, 0]]

# Translation + Rotation
transform = "t+r"
bounds = [(t-20, t+20) for t in [0, 0, 0]] + [(-5, 5)] * 3

# Translation + Rotation + Scale
transform = "t+r+s"
bounds = [(t-20, t+20) for t in [0, 0, 0]] + [(-5, 5)] * 3 + [(0.9, 1.1)] * 3
```

## Advanced Interpolation Methods

### 1. Cubic Spline Interpolation

```python
from pyspim.interp import affine

# High-quality cubic spline interpolation
result = affine.transform(
    volume, transform_matrix,
    interp_method="cubspl",
    preserve_dtype=True,
    block_size_z=8, block_size_y=8, block_size_x=8
)
```

### 2. Boundary Correction

```python
# Apply boundary correction during deconvolution
result = deconvolve(
    view_a, view_b, initial_estimate,
    psf_a, psf_b, bp_a, bp_b,
    boundary_correction=True,
    zero_padding=(16, 16, 16),
    boundary_sigma_a=1e-2,
    boundary_sigma_b=1e-2
)
```

## GPU Memory Management

### 1. Multi-GPU Processing

```python
import cupy as cp

# Check available GPUs
n_gpu = cp.cuda.runtime.getDeviceCount()
print(f"Available GPUs: {n_gpu}")

# Use specific GPU
with cp.cuda.Device(0):
    # Process on GPU 0
    result = process_on_gpu(data)

# Distribute across multiple GPUs
with concurrent.futures.ProcessPoolExecutor(max_workers=n_gpu) as executor:
    futures = []
    for gpu_id in range(n_gpu):
        future = executor.submit(process_chunk, chunk, gpu_id)
        futures.append(future)
```

### 2. Memory Pool Management

```python
# Free GPU memory
cp.get_default_memory_pool().free_all_blocks()

# Monitor memory usage
used_mb = cp.get_default_memory_pool().used_bytes() / 1e6
total_mb = cp.get_default_memory_pool().total_bytes() / 1e6
print(f"GPU Memory: {used_mb:.1f}MB / {total_mb:.1f}MB")
```

## Advanced Data Loading

### 1. Zarr Integration

```python
import zarr

# Load from zarr
data = zarr.open("path/to/data.zarr")

# Create zarr array with specific chunks
output = zarr.create(
    shape=data.shape,
    dtype=np.float32,
    chunks=(64, 64, 64),
    compressor=zarr.Blosc(cname='zstd', clevel=3)
)
```

### 2. OME-Zarr Support

```python
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image

# Load OME-Zarr data
store = parse_url("path/to/data.ome.zarr")
data = store.load()

# Save as OME-Zarr
write_image(data, "output.ome.zarr")
```

## Performance Optimization

### 1. Kernel Launch Parameters

```python
from pyspim.util import launch_params_for_volume

# Optimize GPU kernel parameters
launch_params = launch_params_for_volume(
    volume.shape, 
    block_size_z=8, 
    block_size_y=8, 
    block_size_x=8
)

# Use in registration
T, res = powell.optimize_affine_piecewise(
    a, b,
    kernel_launch_params=launch_params,
    verbose=True
)
```

### 2. Chunked Processing Strategies

```python
def calculate_optimal_chunks(volume_shape, gpu_memory_gb=8):
    """Calculate optimal chunk size based on GPU memory."""
    # Estimate memory per voxel (float32)
    bytes_per_voxel = 4
    
    # Available memory (leave 20% for overhead)
    available_memory = gpu_memory_gb * 0.8 * 1e9
    
    # Calculate chunk size
    total_voxels = available_memory / bytes_per_voxel
    chunk_size = int(total_voxels ** (1/3))
    
    return (chunk_size, chunk_size, chunk_size)
```

## Advanced Workflow Integration

### 1. Snakemake Integration

```python
# Include in larger workflow
include: "path/to/pyspim/examples/snakemake/Snakefile"

# Custom rule using PySPIM
rule custom_deconvolution:
    input:
        a="data/{sample}/a.zarr",
        b="data/{sample}/b.zarr",
        psf_a="psfs/a.npy",
        psf_b="psfs/b.npy"
    output:
        "results/{sample}/deconvolved.ome.tif"
    params:
        iterations=20,
        algorithm="efficient"
    shell:
        "python deconvolve.py "
        "--a-zarr-path {input.a} "
        "--b-zarr-path {input.b} "
        "--psf-a-path {input.psf_a} "
        "--psf-b-path {input.psf_b} "
        "--output-path {output} "
        "--function {params.algorithm} "
        "--num-iter {params.iterations}"
```

### 2. Nextflow Integration

```groovy
process Deconvolution {
    publishDir "results", mode: 'copy'
    
    input:
    tuple val(sample), path(a), path(b), path(psf_a), path(psf_b)
    
    output:
    tuple val(sample), path("*.ome.tif")
    
    script:
    """
    python deconvolve.py \\
        --a-zarr-path ${a} \\
        --b-zarr-path ${b} \\
        --psf-a-path ${psf_a} \\
        --psf-b-path ${psf_b} \\
        --output-path ${sample}_deconvolved.ome.tif \\
        --function efficient \\
        --num-iter 20
    """
}
```

## Troubleshooting Advanced Features

### Common Issues

1. **GPU Memory Errors**: Reduce chunk size or use multi-GPU processing
2. **Registration Convergence**: Try different initial parameters or transform types
3. **Deconvolution Artifacts**: Adjust regularization parameters or use different algorithms
4. **Performance Issues**: Optimize kernel launch parameters and chunk sizes

### Debugging Tips

```python
# Enable verbose output
verbose = True

# Check intermediate results
print(f"Registration correlation ratio: {correlation_ratio:.4f}")
print(f"Deconvolution iterations: {iterations}")

# Monitor GPU memory
print(f"GPU Memory: {cp.get_default_memory_pool().used_bytes() / 1e9:.2f}GB")
```

## Next Steps

- Explore the [API Reference](../packages/pyspim/api.md) for complete function documentation
- Check out [basic usage examples](basic-usage.md) for fundamental workflows
- Try the [napari plugin](../packages/napari-pyspim/overview.md) for interactive analysis
- Review [performance benchmarks](../development/testing.md) for optimization tips 