# PySPIM

**Selective Plane Illumination Microscopy Analysis**

!!! abstract "Documentation that simply works"

    Process SPIM data with GPU acceleration – powerful, interactive, and research-ready.

    [:octicons-arrow-right-24: Get started](getting-started/installation.md){ .md-button .md-button--primary }
    [:octicons-book-24: Learn more](user-guide/examples-overview.md){ .md-button .md-button--secondary }

<div class="mdx-video-showcase">
  <h2>See PySPIM in Action</h2>
  <div class="mdx-video-container">
    <video controls preload="metadata" poster="media/example_view.png">
      <source src="media/example_usage.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>
  <p>PySPIM makes advanced SPIM data analysis easy and interactive. Watch this short video to see how you can go from raw data to beautiful results in just a few clicks!</p>
</div>

## Quick Install

!!! example "Install from git repository"

    ```bash
    git clone https://github.com/matt-black/pyspim.git
    cd pyspim
    uv sync --extra dev
    uv pip install -e packages/pyspim
    uv pip install -e packages/napari-pyspim
    ```

!!! note "Note"
    PySPIM is currently in development and not yet available on PyPI. Installation requires cloning the repository.

## Quick Start

!!! example "Launch PySPIM in napari"

    ```bash
    # Launch napari
    napari
    
    # Navigate to: Plugins → pyspim → pyspim
    # This opens the DiSPIM Pipeline interface
    ```

## Key Features

- **GPU Acceleration** - Fast processing with CuPy-powered operations
- **Interactive Visualization** - Seamless napari integration for instant results
- **Workflow Automation** - Batch processing with Snakemake integration
- **HPC Ready** - Built-in SLURM support for cluster computing
- **Open Source** - GPL-3.0 licensed and actively maintained

## Documentation

<div class="grid" markdown>

<div markdown>

### Getting Started
- [Installation](getting-started/installation.md) - Set up PySPIM
- [Quick Start](getting-started/quickstart.md) - Your first workflow
- [Basic Usage](user-guide/basic-usage.md) - Step-by-step tutorial

### User Guide
- [Examples Overview](user-guide/examples-overview.md) - Available examples
- [Fruiting Body Workflow](user-guide/fruiting-body-workflow.md) - Complete pipeline
- [Advanced Features](user-guide/advanced-features.md) - Advanced techniques

</div>

<div markdown>

### Packages
- [Napari Plugin](packages/napari-pyspim/overview.md) - Interactive GUI
- [Core Package](packages/pyspim/overview.md) - Library reference
- [API Documentation](packages/pyspim/api.md) - Complete API

### Development
- [Contributing](development/contributing.md) - How to contribute
- [License](about/license.md) - GPL-3.0 License

</div>

</div> 