# PySPIM Snakemake Workflow

This directory contains a Snakemake workflow for automated diSPIM data processing.

## Documentation

For detailed documentation, see:
- [Snakemake Workflow Documentation](../../../examples/snakemake-workflow.md)
- [Examples Overview](../../../examples/examples-overview.md)

## Quick Start

```bash
# Run locally
snakemake

# Run on SLURM cluster
snakemake --slurm --jobs 10
```

## Files

- `Snakefile` - Main workflow definition
- `config.yaml` - Configuration file
- `slurm.yaml` - SLURM cluster configuration
- `scripts/` - Workflow scripts 