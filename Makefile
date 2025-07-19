.PHONY: install install-core install-plugin clean test

# Install both packages
install: install-core install-plugin

# Install core library only
install-core:
	cd pyspim && pip install --editable .

# Install napari plugin only
install-plugin:
	cd napari-pyspim && pip install --editable .

# Clean build artifacts
clean:
	rm -rf pyspim/build pyspim/dist pyspim/*.egg-info
	rm -rf napari-pyspim/build napari-pyspim/dist napari-pyspim/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run tests (if available)
test:
	cd pyspim && python -m pytest tests/ || echo "No tests in core package"
	cd napari-pyspim && python -m pytest tests/ || echo "No tests in plugin package"

# Show help
help:
	@echo "Available commands:"
	@echo "  install      - Install both packages"
	@echo "  install-core - Install core library only"
	@echo "  install-plugin - Install napari plugin only"
	@echo "  clean        - Clean build artifacts"
	@echo "  test         - Run tests"
	@echo "  help         - Show this help" 