# PySPIM Development Tasks
# Run `just --list` to see all available commands

# Default recipe - show help
default:
    @just --list

# Install packages in development mode
install:
    uv pip install -e packages/pyspim
    uv pip install -e packages/napari-pyspim

# Install development dependencies and packages
install-dev:
    uv sync --extra dev
    uv pip install -e packages/pyspim
    uv pip install -e packages/napari-pyspim

# Install development dependencies & packages with GPU support
install-dev-gpu:
    uv sync --extra dev
    uv pip install -e packages/pyspim[gpu]
    uv pip install -e packages/napari-pyspim

# Install only the napari plugin
install-plugin:
    uv pip install -e packages/napari-pyspim

# Clean build artifacts and cache
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name "*.egg-info" -exec rm -rf {} +
    find . -type d -name ".pytest_cache" -exec rm -rf {} +
    find . -type d -name ".mypy_cache" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type f -name ".DS_Store" -delete
    rm -rf build/
    rm -rf dist/
    rm -rf .venv/

# Run tests for all packages
test:
    uv run pytest packages/pyspim/tests/ -v --cov=packages/pyspim/src/pyspim --cov-report=term-missing
    uv run pytest packages/napari-pyspim/tests/ -v --cov=packages/napari-pyspim/src/napari_pyspim --cov-report=term-missing

# Run tests for core pyspim package
test-pyspim:
    uv run pytest packages/pyspim/tests/ -v --cov=packages/pyspim/src/pyspim --cov-report=term-missing

# Run tests for napari plugin
test-plugin:
    uv run pytest packages/napari-pyspim/tests/ -v --cov=packages/napari-pyspim/src/napari_pyspim --cov-report=term-missing

# Run fast tests (exclude slow and GPU tests)
test-fast:
    uv run pytest packages/pyspim/tests/ -v -m "not slow and not gpu" --tb=short
    uv run pytest packages/napari-pyspim/tests/ -v -m "not slow and not gpu" --tb=short

# Run GPU tests only
test-gpu:
    uv run pytest packages/pyspim/tests/ -v -m "gpu" --tb=short
    uv run pytest packages/napari-pyspim/tests/ -v -m "gpu" --tb=short

# Run integration tests only
test-integration:
    uv run pytest packages/pyspim/tests/ -v -m "integration" --tb=short
    uv run pytest packages/napari-pyspim/tests/ -v -m "integration" --tb=short

# Run tests with detailed coverage reports
test-coverage:
    uv run pytest packages/pyspim/tests/ --cov=packages/pyspim/src/pyspim --cov-report=html --cov-report=xml
    uv run pytest packages/napari-pyspim/tests/ --cov=packages/napari-pyspim/src/napari_pyspim --cov-report=html --cov-report=xml

# Build documentation
docs:
    uv run mkdocs build

# Serve documentation locally
docs-serve:
    uv run mkdocs serve

# Build and serve documentation
docs-build:
    uv run mkdocs build
    uv run mkdocs serve

# Format code with ruff
format:
    uv run ruff format packages/
    uv run ruff check --fix packages/

# Run linting with ruff and mypy
lint:
    uv run ruff check packages/
    uv run mypy packages/

# Run ruff checks and formatting
ruff:
    uv run ruff check packages/
    uv run ruff format packages/

# Install pre-commit hooks
pre-commit:
    uv run pre-commit install

# Run pre-commit on all files
pre-commit-run:
    uv run pre-commit run --all-files

# Run all checks (format, lint, test)
check:
    just ruff
    just lint
    just test-fast

# Build both packages
build:
    uv build packages/pyspim
    uv build packages/napari-pyspim

# Publish packages to PyPI (requires authentication)
publish:
    uv publish packages/pyspim
    uv publish packages/napari-pyspim

# Show help
help:
    @echo "PySPIM Development Commands:"
    @echo ""
    @echo "Installation:"
    @echo "  just install          - Install packages (basic usage)"
    @echo "  just install-dev      - Install packages + dev tools (contributing)"
    @echo "  just install-plugin   - Install only the napari plugin"
    @echo ""
    @echo "Testing:"
    @echo "  just test             - Run all tests with coverage"
    @echo "  just test-fast        - Run fast tests (no slow/GPU)"
    @echo "  just test-gpu         - Run GPU tests only"
    @echo "  just test-pyspim      - Test core package only"
    @echo "  just test-plugin      - Test napari plugin only"
    @echo "  just test-coverage    - Generate coverage reports"
    @echo ""
    @echo "Code Quality:"
    @echo "  just format           - Format code with ruff"
    @echo "  just lint             - Run linting checks"
    @echo "  just ruff             - Run ruff checks and formatting"
    @echo "  just check            - Run all checks (format, lint, test)"
    @echo ""
    @echo "Documentation:"
    @echo "  just docs             - Build documentation"
    @echo "  just docs-serve       - Serve documentation locally"
    @echo ""
    @echo "Development:"
    @echo "  just pre-commit       - Install pre-commit hooks"
    @echo "  just pre-commit-run   - Run pre-commit on all files"
    @echo "  just clean            - Clean build artifacts"
    @echo "  just build            - Build packages"
    @echo "  just publish          - Publish to PyPI"
    @echo ""
    @echo "Run 'just --list' for all available commands" 