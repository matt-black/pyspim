# pyspim Monorepo Guide

This repository has been restructured as a monorepo containing two separate packages:

## Structure

```
pyspim/
├── pyspim/                    # Core library package
│   ├── pyproject.toml        # Core package configuration
│   ├── README.md             # Core package documentation
│   └── src/pyspim/           # Core library source code
├── napari-pyspim/            # Napari plugin package (npe2 compatible)
│   ├── pyproject.toml        # Plugin package configuration
│   ├── README.md             # Plugin package documentation
│   ├── napari.yaml           # npe2 manifest
│   └── src/napari_pyspim/    # Plugin source code
├── pyproject.toml            # Root monorepo configuration
├── README.md                 # Main documentation
├── Makefile                  # Development commands
├── build.py                  # Build script
└── MONOREPO.md              # This file
```

## Key Changes

### 1. Separation of Concerns
- **pyspim**: Core library with all SPIM analysis functionality
- **napari-pyspim**: Napari plugin that depends on the core library

### 2. npe2 Compatibility
The napari plugin has been updated to be compatible with npe2:
- Removed old `napari_plugin_engine` imports and hooks
- Added `napari.yaml` manifest file
- Updated `pyproject.toml` with npe2 configuration
- Plugin discovery is now handled automatically by npe2

### 3. Independent Package Management
Each package has its own:
- `pyproject.toml` with specific dependencies
- `README.md` with package-specific documentation
- Version management
- Dependency isolation

## Development Workflow

### Installing for Development

#### Option 1: Using Makefile (Recommended)
```bash
# Install both packages
make install

# Install only core library
make install-core

# Install only plugin
make install-plugin
```

#### Option 2: Using build script
```bash
python build.py
```

#### Option 3: Manual installation
```bash
# Install core library first
cd pyspim
pip install --editable .

# Then install plugin
cd ../napari-pyspim
pip install --editable .
```

### Development Commands

```bash
# Clean build artifacts
make clean

# Run tests
make test

# Show available commands
make help
```

## Package Dependencies

### pyspim (Core Library)
- dask, dask[distributed]
- fsspec>=2022.8.2
- npy2bdv
- numpy
- pandas
- pyyaml
- read-roi
- resource-backed-dask-array>=0.1.0
- scikit-image
- tifffile
- tqdm
- typing_extensions
- zarr
- cupy-cuda12x
- matplotlib>=3.5.0

### napari-pyspim (Plugin)
- pyspim (core library)
- napari>=0.4.0
- napari-plugin-engine>=0.2.0
- qtpy>=1.9.0
- PyQt5>=5.15.0
- cupy-cuda12x
- matplotlib>=3.5.0

## Benefits of This Structure

1. **Clear Separation**: Core functionality is separate from UI code
2. **Independent Versioning**: Each package can be versioned independently
3. **Focused Dependencies**: Each package only includes what it needs
4. **npe2 Ready**: Plugin is compatible with modern napari plugin system
5. **Easier Testing**: Can test core library without napari dependencies
6. **Better Distribution**: Can distribute core library separately from plugin

## Migration Notes

- The old `src/` directory structure is preserved for reference
- All existing functionality has been moved to the new structure
- The plugin now properly depends on the core library
- npe2 compatibility ensures future napari compatibility

## Next Steps

1. Test the installation process
2. Verify that the plugin works in napari
3. Update any import statements if needed
4. Consider adding CI/CD for both packages
5. Update documentation as needed 