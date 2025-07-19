#!/usr/bin/env python3
"""
Build script for the pyspim monorepo.
This script helps install both the core library and napari plugin for
development.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    if cwd:
        print(f"Working directory: {cwd}")
    
    result = subprocess.run(
        cmd, 
        shell=True, 
        cwd=cwd,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        return False
    
    print(f"Success: {result.stdout}")
    return True


def main():
    """Main build function."""
    root_dir = Path(__file__).parent
    
    # Install core library first
    print("Installing core pyspim library...")
    if not run_command("pip install --editable .", cwd=root_dir / "pyspim"):
        print("Failed to install core library")
        sys.exit(1)
    
    # Install napari plugin
    print("Installing napari-pyspim plugin...")
    if not run_command("pip install --editable .",
                      cwd=root_dir / "napari-pyspim"):
        print("Failed to install napari plugin")
        sys.exit(1)
    
    print("Successfully installed both packages!")


if __name__ == "__main__":
    main() 