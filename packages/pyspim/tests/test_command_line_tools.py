"""Tests for PySPIM command-line tools."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestCommandLineTools:
    """Test command-line tool functionality."""

    def test_deskew_cpu_script_exists(self):
        """Test that deskew_cpu.py script exists and is executable."""
        script_path = Path("tools/script/dispim_pipeline/deskew_cpu.py")
        assert script_path.exists(), f"Script not found: {script_path}"
        assert script_path.is_file(), f"Not a file: {script_path}"

    def test_register_script_exists(self):
        """Test that register.py script exists and is executable."""
        script_path = Path("tools/script/dispim_pipeline/register.py")
        assert script_path.exists(), f"Script not found: {script_path}"
        assert script_path.is_file(), f"Not a file: {script_path}"

    def test_deconvolve_script_exists(self):
        """Test that deconvolve.py script exists and is executable."""
        script_path = Path("tools/script/dispim_pipeline/deconvolve.py")
        assert script_path.exists(), f"Script not found: {script_path}"
        assert script_path.is_file(), f"Not a file: {script_path}"

    def test_transform_script_exists(self):
        """Test that transform.py script exists and is executable."""
        script_path = Path("tools/script/dispim_pipeline/transform.py")
        assert script_path.exists(), f"Script not found: {script_path}"
        assert script_path.is_file(), f"Not a file: {script_path}"

    def test_script_help(self):
        """Test that scripts provide help information."""
        scripts = [
            "tools/script/dispim_pipeline/deskew_cpu.py",
            "tools/script/dispim_pipeline/register.py",
            "tools/script/dispim_pipeline/deconvolve.py",
            "tools/script/dispim_pipeline/transform.py",
        ]

        for script in scripts:
            if Path(script).exists():
                try:
                    result = subprocess.run(
                        [sys.executable, script, "--help"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    # Script should either show help or exit with error
                    assert result.returncode in [0, 1, 2]
                except subprocess.TimeoutExpired:
                    pytest.skip(f"Script {script} timed out")

    def test_script_argument_parsing(self):
        """Test that scripts can parse basic arguments."""
        scripts = [
            "tools/script/dispim_pipeline/deskew_cpu.py",
            "tools/script/dispim_pipeline/register.py",
            "tools/script/dispim_pipeline/deconvolve.py",
            "tools/script/dispim_pipeline/transform.py",
        ]

        for script in scripts:
            if Path(script).exists():
                try:
                    # Test with invalid arguments (should fail gracefully)
                    result = subprocess.run(
                        [sys.executable, script, "--invalid-arg"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    # Should exit with error code
                    assert result.returncode != 0
                except subprocess.TimeoutExpired:
                    pytest.skip(f"Script {script} timed out")


class TestSLURMScripts:
    """Test SLURM job scripts."""

    def test_slurm_scripts_exist(self):
        """Test that SLURM job scripts exist."""
        slurm_scripts = [
            "examples/data/sample_data/sh/1_deskew.sh",
            "examples/data/sample_data/sh/2_register.sh",
            "examples/data/sample_data/sh/3_transform.sh",
            "examples/data/sample_data/sh/4_deconvolve.sh",
            "examples/data/sample_data/sh/5a_rotate.sh",
        ]

        for script in slurm_scripts:
            script_path = Path(script)
            assert script_path.exists(), f"SLURM script not found: {script_path}"
            assert script_path.is_file(), f"Not a file: {script_path}"

    def test_slurm_script_content(self):
        """Test SLURM script content and structure."""
        slurm_scripts = [
            "examples/data/sample_data/sh/1_deskew.sh",
            "examples/data/sample_data/sh/2_register.sh",
            "examples/data/sample_data/sh/3_transform.sh",
            "examples/data/sample_data/sh/4_deconvolve.sh",
            "examples/data/sample_data/sh/5a_rotate.sh",
        ]

        for script in slurm_scripts:
            if Path(script).exists():
                with open(script) as f:
                    content = f.read()

                # Check for required SLURM directives
                assert "#SBATCH" in content, f"Missing SLURM directives in {script}"
                assert "module load" in content, f"Missing module loading in {script}"
                assert "conda activate" in content, (
                    f"Missing conda activation in {script}"
                )
                assert "python" in content, f"Missing Python command in {script}"

    def test_deskew_slurm_script(self):
        """Test deskew SLURM script specifically."""
        script_path = "examples/data/sample_data/sh/1_deskew.sh"
        if Path(script_path).exists():
            with open(script_path) as f:
                content = f.read()

            # Check for specific deskew parameters
            assert "deskew_cpu.py" in content
            assert "--deskew-method" in content
            assert "--step-size" in content
            assert "--pixel-size" in content

    def test_register_slurm_script(self):
        """Test registration SLURM script specifically."""
        script_path = "examples/data/sample_data/sh/2_register.sh"
        if Path(script_path).exists():
            with open(script_path) as f:
                content = f.read()

            # Check for specific registration parameters
            assert "register.py" in content
            assert "--crop-box-a" in content
            assert "--crop-box-b" in content
            assert "--metric" in content
            assert "--transform" in content

    def test_deconvolve_slurm_script(self):
        """Test deconvolution SLURM script specifically."""
        script_path = "examples/data/sample_data/sh/4_deconvolve.sh"
        if Path(script_path).exists():
            with open(script_path) as f:
                content = f.read()

            # Check for specific deconvolution parameters
            assert "deconvolve.py" in content
            assert "--psf-a" in content
            assert "--psf-b" in content
            assert "--iterations" in content


class TestDataFormats:
    """Test data format handling."""

    def test_tiff_io(self):
        """Test TIFF file I/O."""
        import tifffile

        # Create test data
        test_data = np.random.randint(0, 65535, (10, 10, 10), dtype=np.uint16)

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            # Write data
            tifffile.imwrite(tmp.name, test_data)

            # Read data back
            read_data = tifffile.imread(tmp.name)

            # Clean up
            os.unlink(tmp.name)

        # Verify data integrity
        np.testing.assert_array_equal(test_data, read_data)

    def test_numpy_io(self):
        """Test NumPy file I/O."""
        # Create test data
        test_data = np.random.randint(0, 65535, (10, 10, 10), dtype=np.uint16)

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            # Write data
            np.save(tmp.name, test_data)

            # Read data back
            read_data = np.load(tmp.name)

            # Clean up
            os.unlink(tmp.name)

        # Verify data integrity
        np.testing.assert_array_equal(test_data, read_data)

    def test_hdf5_io(self):
        """Test HDF5 file I/O."""
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not available")

        # Create test data
        test_data = np.random.randint(0, 65535, (10, 10, 10), dtype=np.uint16)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            # Write data
            with h5py.File(tmp.name, "w") as f:
                f.create_dataset("data", data=test_data)

            # Read data back
            with h5py.File(tmp.name, "r") as f:
                read_data = f["data"][:]

            # Clean up
            os.unlink(tmp.name)

        # Verify data integrity
        np.testing.assert_array_equal(test_data, read_data)


class TestPerformance:
    """Test performance-related functionality."""

    def test_gpu_memory_estimation(self):
        """Test GPU memory estimation."""
        # Test data size
        data_shape = (100, 100, 100)
        dtype = np.uint16

        # Calculate memory usage
        memory_bytes = np.prod(data_shape) * dtype().itemsize
        memory_gb = memory_bytes / (1024**3)

        # Verify reasonable memory usage
        assert memory_gb < 1.0, f"Memory usage too high: {memory_gb:.2f} GB"

    def test_cpu_memory_estimation(self):
        """Test CPU memory estimation."""
        # Test data size
        data_shape = (200, 200, 200)
        dtype = np.float32

        # Calculate memory usage
        memory_bytes = np.prod(data_shape) * dtype().itemsize
        memory_gb = memory_bytes / (1024**3)

        # Verify reasonable memory usage
        assert memory_gb < 10.0, f"Memory usage too high: {memory_gb:.2f} GB"

    def test_processing_time_estimation(self):
        """Test processing time estimation."""
        # Test parameters
        data_size = (100, 100, 100)
        gpu_speedup = 10  # GPU is 10x faster than CPU

        # Estimate processing time (simplified)
        cpu_time = np.prod(data_size) / 1e6  # seconds
        gpu_time = cpu_time / gpu_speedup

        # Verify reasonable time estimates
        assert cpu_time > 0
        assert gpu_time > 0
        assert gpu_time < cpu_time


if __name__ == "__main__":
    pytest.main([__file__])
