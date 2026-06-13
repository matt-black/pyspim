"""Utilities for generating SLURM batch scripts for pyspim computations.

Generates standalone bash scripts that activate the remote virtual environment
and invoke ``_batch_runner.py`` with the appropriate parameters.
"""

from __future__ import annotations

import json
import os
import uuid


def generate_batch_script(
    command_type: str,
    params_json_path: str,
    result_path: str,
    remote_venv: str,
    batch_runner_path: str,
    time_string: str,
    memory_gb: int,
    gpus: int,
    ntasks: int,
) -> str:
    """Generate a SLURM batch script for a pyspim computation.

    Parameters
    ----------
    command_type : str
        Computation label for the job name (e.g. ``"deconvolution"``).
    params_json_path : str
        Remote path to the JSON file containing computation parameters.
    result_path : str
        Remote path where ``_batch_runner.py`` should write results.
    remote_venv : str
        Path to the Python virtual environment on the remote server.
    batch_runner_path : str
        Path to ``_batch_runner.py`` on the remote server.
    time_string : str
        SLURM time format ``HH:MM:SS``.
    memory_gb : int
        Memory per node in GB.
    gpus : int
        Number of GPUs to request (0 = CPU only).
    ntasks : int
        Number of tasks.

    Returns
    -------
    str
        The complete bash script content.
    """
    python_bin = os.path.join(remote_venv, "bin", "python")
    job_name = f"pyspim_{command_type}"

    # Build GPU directive (omit if 0)
    gpu_directive = f"#SBATCH --gres=gpu:{gpus}\n" if gpus > 0 else ""

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time={time_string}
#SBATCH --cpus-per-task{ntasks}
#SBATCH --mem={memory_gb}G
{gpu_directive}#SBATCH --output=/tmp/pyspim_batch_%j.out
#SBATCH --error=/tmp/pyspim_batch_%j.err

# Activate virtual environment
source {remote_venv}/bin/activate

# Run computation
{python_bin} {batch_runner_path} \\
    --command {command_type} \\
    --params-json {params_json_path} \\
    --output {result_path}
"""
    return script


def generate_params_json(
    command_type: str,
    computation_params: dict,
) -> str:
    """Serialize computation parameters to a JSON string.

    Parameters
    ----------
    command_type : str
        Computation type (``"deconvolution"`` or ``"registration"``).
    computation_params : dict
        Parameters specific to the computation (paths, settings, etc.).

    Returns
    -------
    str
        JSON string ready to be written to a file.
    """
    payload = {
        "command_type": command_type,
        "params": computation_params,
    }
    return json.dumps(payload, indent=2)


def get_unique_paths(base_dir: str, command_type: str) -> tuple[str, str, str]:
    """Generate unique file paths for a batch job submission.

    Parameters
    ----------
    base_dir : str
        Base directory (typically ``/tmp``).
    command_type : str
        Computation label for filename prefix.

    Returns
    -------
    tuple[str, str, str]
        (script_path, params_json_path, result_path)
    """
    job_uuid = str(uuid.uuid4())[:8]
    prefix = f"pyspim_batch_{command_type}_{job_uuid}"
    return (
        os.path.join(base_dir, f"{prefix}.sh"),
        os.path.join(base_dir, f"{prefix}_params.json"),
        os.path.join(base_dir, f"{prefix}_results.json"),
    )


def get_batch_runner_path(remote_venv: str) -> str:
    """Derive the path to ``_batch_runner.py`` from the remote venv path.

    The remote venv is typically at ``<pyspim_root>/venv``, so the
    batch runner is at ``<pyspim_root>/packages/napari-pyspim/src/napari_pyspim/_batch_runner.py``.

    Parameters
    ----------
    remote_venv : str
        Path to the remote Python virtual environment.

    Returns
    -------
    str
        Absolute path to ``_batch_runner.py``.
    """
    pyspim_root = os.path.dirname(remote_venv)
    return os.path.join(
        pyspim_root,
        "packages",
        "napari-pyspim",
        "src",
        "napari_pyspim",
        "_batch_runner.py",
    )
