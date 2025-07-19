"""Generate API reference pages for PySPIM packages."""

from pathlib import Path
from mkdocs_gen_files import open as mkdocs_open
from mkdocs_gen_files import nav

# Define the packages to document
packages = {
    "pyspim": {
        "path": "packages/pyspim/src/pyspim",
        "output": "packages/pyspim/api.md",
        "title": "PySPIM Core API Reference",
        "module": "pyspim"
    },
    "napari-pyspim": {
        "path": "packages/napari-pyspim/src/napari_pyspim", 
        "output": "packages/napari-pyspim/api.md",
        "title": "Napari PySPIM Plugin API Reference",
        "module": "napari_pyspim"
    }
}

def generate_api_docs():
    """Generate API documentation for all packages."""
    
    for package_name, config in packages.items():
        package_path = Path(config["path"])
        output_path = config["output"]
        module_name = config["module"]
        
        if not package_path.exists():
            print(f"Warning: Package path {package_path} does not exist")
            continue
            
        # Create the API reference page
        with mkdocs_open(output_path, "w") as f:
            f.write(f"# {config['title']}\n\n")
            f.write(f"::: {module_name}\n")
            f.write("    options:\n")
            f.write("      show_root_heading: true\n")
            f.write("      show_source: true\n")
            f.write("      show_category_heading: true\n")
            f.write("      show_signature_annotations: true\n")
            f.write("      show_bases: true\n")
            f.write("      show_submodules: true\n")
            f.write("      heading_level: 2\n")
            f.write("      members_order: source\n")
            f.write("      docstring_style: google\n")
            f.write("      preload_modules: []\n")
            f.write("      filters: ['!^_']\n")
            f.write("      merge_init_into_class: true\n")
            f.write("\n")
            
            # Add all Python files in the package
            for py_file in package_path.rglob("*.py"):
                # Skip most underscore files, but include specific ones we want to document
                if py_file.name.startswith("_"):
                    # Allow specific underscore files to be documented
                    if py_file.name == "_util.py" and "deskew" in str(py_file):
                        pass  # Include this file
                    else:
                        continue
                    
                # Convert file path to module path
                module_path = py_file.relative_to(package_path.parent)
                full_module_name = str(module_path).replace("/", ".").replace("\\", ".").replace(".py", "")
                
                f.write(f"::: {full_module_name}\n")
                f.write("    options:\n")
                f.write("      show_root_heading: true\n")
                f.write("      show_source: true\n")
                f.write("      show_category_heading: true\n")
                f.write("      show_signature_annotations: true\n")
                f.write("      show_bases: true\n")
                f.write("      show_submodules: true\n")
                f.write("      heading_level: 3\n")
                f.write("      members_order: source\n")
                f.write("      docstring_style: google\n")
                f.write("      preload_modules: []\n")
                f.write("      filters: ['!^_']\n")
                f.write("      merge_init_into_class: true\n")
                f.write("\n")

if __name__ == "__main__":
    generate_api_docs() 