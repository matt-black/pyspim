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

## Everything you would expect

<div class="grid cards" markdown>

-   :fontawesome-solid-rocket:{ .lg .middle } __It's just Python__
    
    Focus on your SPIM data analysis and create professional results in minutes. No need to know complex GPU programming – let PySPIM do the heavy lifting for you.

-   :fontawesome-solid-desktop:{ .lg .middle } __Works on all platforms__
    
    Process your SPIM data with confidence – PySPIM automatically adapts to your computing environment, from laptops to HPC clusters. Desktop. Server. Cloud. All great.

-   :fontawesome-solid-cogs:{ .lg .middle } __Made to measure__
    
    Make it yours – customize processing parameters, visualization options, and workflow automation with a few lines of configuration. PySPIM can be easily extended and provides many options to alter behavior.

-   :fontawesome-solid-bolt:{ .lg .middle } __Fast and lightweight__
    
    Don't let your users wait – get incredible performance with GPU acceleration by using CuPy-powered processing, yielding optimal results and happy researchers that return.

-   :fontawesome-solid-shield:{ .lg .middle } __Maintain ownership__
    
    Own your data processing pipeline completely, guaranteeing both integrity and security – no need to entrust your microscopy analysis to third-party platforms. Retain full control.

-   :fontawesome-solid-heart:{ .lg .middle } __Open Source__
    
    You're in good company – choose a mature and actively maintained solution built with state-of-the-art Open Source technologies, trusted by research labs worldwide. Licensed under GPL-3.0.

</div>

## More than just data processing

<div class="grid cards" markdown>

-   :fontawesome-solid-chart-line:{ .lg .middle } __Interactive visualization__
    
    PySPIM makes your SPIM data **instantly visualizable** with zero effort: say goodbye to complex visualization setups that can take hours to configure. Process your data with a **highly customizable** and **blazing fast** napari integration running entirely **in your browser** at no extra cost.

-   :fontawesome-solid-robot:{ .lg .middle } __Workflow automation__
    
    Some datasets need more processing than others, which is why PySPIM offers a **unique and elegant** way to create **automated workflows** for **batch processing**.

-   :fontawesome-solid-server:{ .lg .middle } __HPC integration__
    
    **Scale your processing** and **increase throughput** when working with large datasets by leveraging the built-in SLURM integration. PySPIM makes it effortless to submit **batch jobs** to your cluster, which will drive more efficient research workflows.

</div>

## Quick Start

!!! example "Install and run PySPIM"

    ```bash
    # Install PySPIM
    git clone https://github.com/matt-black/pyspim.git
    cd pyspim
    just install-dev

    # Launch napari with PySPIM plugin
    napari
    # Navigate to: Plugins → PySPIM → DiSPIM Pipeline
    ```

## Documentation

<div class="grid" markdown>

<div markdown>

### Getting Started
- [Installation Guide](getting-started/installation.md) - Set up PySPIM on your system
- [Quick Start Tutorial](getting-started/quickstart.md) - Your first PySPIM workflow
- [Examples Overview](user-guide/examples-overview.md) - Explore available examples

### User Guide
- [Basic Usage](user-guide/basic-usage.md) - Step-by-step tutorial
- [Fruiting Body Workflow](user-guide/fruiting-body-workflow.md) - Complete pipeline example
- [Snakemake Workflow](user-guide/snakemake-workflow.md) - Automated processing
- [Advanced Features](user-guide/advanced-features.md) - Advanced techniques

</div>

<div markdown>

### Napari Plugin
- [Plugin Overview](packages/napari-pyspim/overview.md) - Interactive GUI features
- [Plugin Usage](packages/napari-pyspim/usage.md) - How to use the GUI
- [Plugin API](packages/napari-pyspim/api.md) - Technical reference

### Core Package
- [Core Overview](packages/pyspim/overview.md) - Library architecture
- [API Reference](packages/pyspim/api.md) - Complete API documentation

</div>

</div>

## Examples

!!! tip "Ready to explore?"

    - [Basic Usage](user-guide/basic-usage.md) - Step-by-step tutorial
    - [Fruiting Body Workflow](user-guide/fruiting-body-workflow.md) - Complete pipeline
    - [Snakemake Workflow](user-guide/snakemake-workflow.md) - Automated processing

## License & Citation

<div class="grid" markdown>

<div markdown>

!!! info "License"

    PySPIM is released under the GPL-3.0 License. See [License Details](about/license.md) for more information.

</div>

<div markdown>

!!! quote "Citation"

    If you use PySPIM in your research, please cite:

    ```bibtex
    @software{pyspim2024,
      title={PySPIM: Selective Plane Illumination Microscopy Analysis},
      author={PySPIM Team},
      year={2024},
      url={https://github.com/matt-black/pyspim}
    }
    ```

</div>

</div> 