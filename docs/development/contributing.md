# Contributing to PySPIM

Thank you for your interest in contributing to PySPIM! This guide will help you get started.

## Getting Started

### Prerequisites

- Python 3.8.1 or higher
- Git
- UV (recommended) or pip
- CUDA-compatible GPU (optional, for GPU development)

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/pyspim.git
   cd pyspim
   ```

2. **Install Development Environment**
   ```bash
   # Using UV (recommended)
   uv sync --extra dev
   uv pip install -e packages/pyspim
   uv pip install -e packages/napari-pyspim
   
   # Or using pip
   pip install -e packages/pyspim
   pip install -e packages/napari-pyspim
   pip install -r requirements-dev.txt
   ```

3. **Verify Installation**
   ```bash
   python -c "import pyspim; import napari_pyspim; print('Setup complete!')"
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Follow the existing code style (see [Code Style](#code-style))
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests

```bash
# Run all tests
make test

# Run specific package tests
make test-pyspim
make test-plugin
```

### 4. Code Quality Checks

```bash
# Format code
make format

# Run linting
make lint

# Run all checks
make check
```

### 5. Commit Your Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Code Style

### Python Code

We use:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

### Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance tasks

### Documentation

- Use Google-style docstrings
- Include type hints
- Add examples for public APIs
- Update relevant documentation pages

## Project Structure

```
pyspim/
├── packages/
│   ├── pyspim/              # Core package
│   │   ├── src/pyspim/      # Source code
│   │   ├── tests/           # Tests
│   │   └── pyproject.toml   # Package configuration
│   └── napari-pyspim/       # Napari plugin
│       ├── src/napari_pyspim/
│       ├── tests/
│       └── pyproject.toml
├── docs/                    # Documentation
├── examples/                # Example data and notebooks
├── tools/                   # Development tools
└── pyproject.toml          # Workspace configuration
```

## Testing

### Writing Tests

- Use pytest for testing
- Place tests in `packages/*/tests/` directories
- Follow the naming convention: `test_*.py`
- Include both unit and integration tests

### Test Structure

```python
def test_function_name():
    """Test description."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_to_test(input_data)
    
    # Assert
    assert result is not None
    assert result.shape == expected_shape
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pyspim --cov=napari_pyspim

# Run specific test file
pytest packages/pyspim/tests/test_specific.py

# Run with verbose output
pytest -v
```

## Documentation

### Building Documentation

```bash
# Build documentation
make docs

# Serve documentation locally
make docs-serve
```

### Documentation Guidelines

- Write clear, concise documentation
- Include code examples
- Use proper Markdown formatting
- Keep documentation up to date with code changes

## GPU Development

### CUDA Development

If working on GPU-accelerated features:

1. **Install CUDA Toolkit**
   ```bash
   # Follow NVIDIA's installation guide for your system
   ```

2. **Install CuPy**
   ```bash
   pip install cupy-cuda12x
   ```

3. **Test GPU Functionality**
   ```python
   import cupy as cp
   print(f"CUDA available: {cp.cuda.is_available()}")
   ```

## Release Process

### Version Management

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Creating a Release

1. **Update Version Numbers**
   - Update `pyproject.toml` files
   - Update `__version__` in `__init__.py` files

2. **Update Changelog**
   - Add release notes to `docs/about/changelog.md`

3. **Build and Test**
   ```bash
   make clean
   make build
   make test
   ```

4. **Create Release**
   - Create a new tag
   - Push to GitHub
   - Create GitHub release

## Getting Help

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for questions and general discussion
- **Code Review**: All contributions require code review

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) for details.

## License

By contributing to PySPIM, you agree that your contributions will be licensed under the GPL-3.0 license.

Thank you for contributing to PySPIM! 🎉 