# Makefile for M.I.A - Personal Virtual Assistant
# Provides convenient commands for development and deployment

.PHONY: help install install-dev run test lint format clean docs security uninstall

# Default target
help:
	@echo "M.I.A - Personal Virtual Assistant"
	@echo "Available commands:"
	@echo ""
	@echo "  make install      - Install M.I.A and dependencies"
	@echo "  make install-dev  - Install with development dependencies"
	@echo "  make run          - Run M.I.A with default settings"
	@echo "  make test         - Run tests with coverage"
	@echo "  make lint         - Run code linting"
	@echo "  make format       - Format code with black and isort"
	@echo "  make docs         - Build documentation"
	@echo "  make security     - Run security checks"
	@echo "  make clean        - Clean cache and build files"
	@echo "  make uninstall    - Uninstall M.I.A"
	@echo ""
	@echo "Development workflow:"
	@echo "  1. make install-dev"
	@echo "  2. Edit .env with your API keys"
	@echo "  3. make test"
	@echo "  4. make run"

# Installation
install:
	@echo "Installing M.I.A..."
	@chmod +x install.sh
	@./install.sh

install-dev: install
	@echo "Installing development dependencies..."
	@chmod +x dev.sh
	@./dev.sh install-dev

# Running
run:
	@echo "Starting M.I.A..."
	@chmod +x run.sh
	@./run.sh

run-debug:
	@echo "Starting M.I.A in debug mode..."
	@chmod +x run.sh
	@./run.sh --debug

# Development
test:
	@echo "Running tests..."
	@chmod +x dev.sh
	@./dev.sh test

lint:
	@echo "Running linter..."
	@chmod +x dev.sh
	@./dev.sh lint

format:
	@echo "Formatting code..."
	@chmod +x dev.sh
	@./dev.sh format

docs:
	@echo "Building documentation..."
	@chmod +x dev.sh
	@./dev.sh docs

security:
	@echo "Running security checks..."
	@chmod +x dev.sh
	@./dev.sh security

# Cleanup
clean:
	@echo "Cleaning up..."
	@chmod +x dev.sh
	@./dev.sh clean

uninstall:
	@echo "Uninstalling M.I.A..."
	@chmod +x uninstall.sh
	@./uninstall.sh

# Docker targets (if needed)
docker-build:
	@echo "Building Docker image..."
	@docker build -t mia-assistant .

docker-run:
	@echo "Running M.I.A in Docker..."
	@docker run -it --rm -v $(PWD)/.env:/app/.env mia-assistant

# Quick development setup
dev-setup: install-dev
	@echo "Setting up development environment..."
	@./dev.sh hooks
	@echo ""
	@echo "Development environment ready!"
	@echo "Don't forget to:"
	@echo "1. Edit .env with your API keys"
	@echo "2. Run 'make test' to verify installation"
	@echo "3. Run 'make run' to start M.I.A"

# CI/CD targets
ci-test: install-dev
	@./dev.sh test
	@./dev.sh lint
	@./dev.sh security

# Package for distribution
package:
	@echo "Creating distribution package..."
	@python setup.py sdist bdist_wheel

# Check system requirements
check-system:
	@echo "Checking system requirements..."
	@./dev.sh info
