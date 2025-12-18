# M.I.A - Multimodal Intelligent Assistant
# Makefile for Linux/Unix systems

.PHONY: all install install-core install-dev install-extras setup venv deps config \
        run run-debug run-text run-audio run-api run-ui \
        test lint format clean clean-all uninstall \
        docker docker-build docker-run docker-stop \
        help check-deps check-python check-system

# Default target
all: help

# Variables
PYTHON := python3
PIP := pip3
VENV_DIR := venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip
CONFIG_DIR := config
SRC_DIR := src

# Colors for terminal output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m

# ============================================================================
# INSTALLATION
# ============================================================================

## install: Full installation (venv + all dependencies + config)
install: check-python venv deps config
	@echo -e "$(GREEN)[SUCCESS]$(NC) M.I.A installed successfully!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Edit .env file: nano .env"
	@echo "  2. Run M.I.A: make run"
	@echo ""

## install-core: Install only core dependencies (minimal)
install-core: check-python venv
	@echo -e "$(BLUE)[INFO]$(NC) Installing core dependencies..."
	@$(VENV_PIP) install --upgrade pip
	@$(VENV_PIP) install -r requirements-core.txt
	@$(VENV_PIP) install -e .
	@echo -e "$(GREEN)[SUCCESS]$(NC) Core dependencies installed"

## install-dev: Install development dependencies
install-dev: venv
	@echo -e "$(BLUE)[INFO]$(NC) Installing development dependencies..."
	@$(VENV_PIP) install -r requirements-dev.txt
	@echo -e "$(GREEN)[SUCCESS]$(NC) Development dependencies installed"

## install-extras: Install extra/optional dependencies
install-extras: venv
	@echo -e "$(BLUE)[INFO]$(NC) Installing extra dependencies..."
	@$(VENV_PIP) install -r requirements-extras.txt
	@echo -e "$(GREEN)[SUCCESS]$(NC) Extra dependencies installed"

## setup: Quick setup for first-time installation
setup: check-system install config
	@echo -e "$(GREEN)[SUCCESS]$(NC) Setup complete!"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

## venv: Create Python virtual environment
venv:
	@if [ -d "$(VENV_DIR)" ]; then \
		echo -e "$(YELLOW)[WARNING]$(NC) Virtual environment already exists"; \
	else \
		echo -e "$(BLUE)[INFO]$(NC) Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV_DIR); \
		echo -e "$(GREEN)[SUCCESS]$(NC) Virtual environment created"; \
	fi

## deps: Install all Python dependencies
deps: venv
	@echo -e "$(BLUE)[INFO]$(NC) Installing Python dependencies..."
	@$(VENV_PIP) install --upgrade pip setuptools wheel
	@$(VENV_PIP) install -r requirements.txt
	@$(VENV_PIP) install -e .
	@echo -e "$(GREEN)[SUCCESS]$(NC) Dependencies installed"

## config: Setup configuration files
config:
	@echo -e "$(BLUE)[INFO]$(NC) Setting up configuration..."
	@if [ ! -f ".env" ]; then \
		if [ -f "$(CONFIG_DIR)/.env.example" ]; then \
			cp $(CONFIG_DIR)/.env.example .env; \
			echo -e "$(GREEN)[SUCCESS]$(NC) Created .env from template"; \
			echo -e "$(YELLOW)[WARNING]$(NC) Please edit .env with your API keys"; \
		else \
			echo -e "$(YELLOW)[WARNING]$(NC) No .env.example found"; \
		fi \
	else \
		echo -e "$(BLUE)[INFO]$(NC) .env already exists"; \
	fi
	@mkdir -p logs memory cache

# ============================================================================
# RUNNING M.I.A
# ============================================================================

## run: Start M.I.A in default mode
run: check-venv
	@echo -e "$(BLUE)[M.I.A]$(NC) Starting M.I.A..."
	@$(VENV_PYTHON) -m mia.main

## run-debug: Start M.I.A in debug mode
run-debug: check-venv
	@echo -e "$(BLUE)[M.I.A]$(NC) Starting M.I.A in debug mode..."
	@$(VENV_PYTHON) -m mia.main --debug

## run-text: Start M.I.A in text-only mode
run-text: check-venv
	@echo -e "$(BLUE)[M.I.A]$(NC) Starting M.I.A in text-only mode..."
	@$(VENV_PYTHON) -m mia.main --text-only

## run-audio: Start M.I.A with audio enabled
run-audio: check-venv
	@echo -e "$(BLUE)[M.I.A]$(NC) Starting M.I.A with audio..."
	@$(VENV_PYTHON) -m mia.main --audio-mode

## run-api: Start M.I.A API server
run-api: check-venv
	@echo -e "$(BLUE)[M.I.A]$(NC) Starting API server..."
	@$(VENV_PYTHON) -m uvicorn mia.api.server:app --host 0.0.0.0 --port 8080 --reload

## run-ui: Start M.I.A Streamlit UI
run-ui: check-venv
	@echo -e "$(BLUE)[M.I.A]$(NC) Starting Streamlit UI..."
	@$(VENV_PYTHON) -m streamlit run $(SRC_DIR)/mia/ui/app.py

## info: Show M.I.A version info
info: check-venv
	@$(VENV_PYTHON) -m mia.main --info

# ============================================================================
# DEVELOPMENT
# ============================================================================

## test: Run all tests
test: check-venv
	@echo -e "$(BLUE)[DEV]$(NC) Running tests..."
	@$(VENV_PYTHON) -m pytest tests/ -v

## test-unit: Run unit tests only
test-unit: check-venv
	@$(VENV_PYTHON) -m pytest tests/unit/ -v

## test-integration: Run integration tests only
test-integration: check-venv
	@$(VENV_PYTHON) -m pytest tests/integration/ -v

## test-cov: Run tests with coverage report
test-cov: check-venv
	@$(VENV_PYTHON) -m pytest tests/ -v --cov=$(SRC_DIR)/mia --cov-report=html

## lint: Run code linter
lint: check-venv
	@echo -e "$(BLUE)[DEV]$(NC) Running linter..."
	@$(VENV_PYTHON) -m flake8 $(SRC_DIR)/ --max-line-length=100 --ignore=E203,W503

## format: Format code with black
format: check-venv
	@echo -e "$(BLUE)[DEV]$(NC) Formatting code..."
	@$(VENV_PYTHON) -m black $(SRC_DIR)/ --line-length=100

## type-check: Run type checking with mypy
type-check: check-venv
	@echo -e "$(BLUE)[DEV]$(NC) Running type check..."
	@$(VENV_PYTHON) -m mypy $(SRC_DIR)/mia

# ============================================================================
# DOCKER
# ============================================================================

## docker-build: Build Docker image
docker-build:
	@echo -e "$(BLUE)[DOCKER]$(NC) Building Docker image..."
	@docker build -t mia-assistant .

## docker-run: Run M.I.A in Docker container
docker-run:
	@echo -e "$(BLUE)[DOCKER]$(NC) Starting M.I.A container..."
	@docker-compose up -d

## docker-stop: Stop M.I.A Docker container
docker-stop:
	@echo -e "$(BLUE)[DOCKER]$(NC) Stopping M.I.A container..."
	@docker-compose down

## docker-logs: View Docker container logs
docker-logs:
	@docker-compose logs -f mia

## docker-shell: Open shell in Docker container
docker-shell:
	@docker exec -it mia-assistant /bin/bash

# ============================================================================
# CLEANUP
# ============================================================================

## clean: Remove cache and temporary files
clean:
	@echo -e "$(BLUE)[INFO]$(NC) Cleaning cache files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@rm -rf .pytest_cache/ .mypy_cache/ .coverage htmlcov/
	@echo -e "$(GREEN)[SUCCESS]$(NC) Cache cleaned"

## clean-all: Remove all generated files including venv
clean-all: clean
	@echo -e "$(YELLOW)[WARNING]$(NC) Removing virtual environment..."
	@rm -rf $(VENV_DIR)/
	@rm -rf build/ dist/ *.egg-info/
	@rm -rf logs/*.log
	@echo -e "$(GREEN)[SUCCESS]$(NC) All cleaned"

## uninstall: Full uninstall (removes venv, caches, but preserves config)
uninstall: clean-all
	@echo -e "$(YELLOW)[WARNING]$(NC) M.I.A uninstalled"
	@echo -e "$(BLUE)[INFO]$(NC) Configuration files (.env, config/) preserved"

# ============================================================================
# SYSTEM CHECKS
# ============================================================================

## check-python: Verify Python installation
check-python:
	@echo -e "$(BLUE)[INFO]$(NC) Checking Python installation..."
	@which $(PYTHON) > /dev/null 2>&1 || (echo -e "$(RED)[ERROR]$(NC) Python3 not found" && exit 1)
	@$(PYTHON) -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" || \
		(echo -e "$(RED)[ERROR]$(NC) Python 3.8+ required" && exit 1)
	@echo -e "$(GREEN)[OK]$(NC) Python $$($(PYTHON) --version | cut -d' ' -f2) found"

## check-venv: Verify virtual environment exists
check-venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo -e "$(RED)[ERROR]$(NC) Virtual environment not found. Run: make install"; \
		exit 1; \
	fi

## check-system: Check system dependencies
check-system:
	@echo -e "$(BLUE)[INFO]$(NC) Checking system dependencies..."
	@echo ""
	@echo "Python:"
	@$(PYTHON) --version || echo "  $(RED)Not found$(NC)"
	@echo ""
	@echo "Audio libraries:"
	@which arecord > /dev/null 2>&1 && echo "  arecord: $(GREEN)OK$(NC)" || echo "  arecord: $(YELLOW)Not found$(NC)"
	@which pactl > /dev/null 2>&1 && echo "  pulseaudio: $(GREEN)OK$(NC)" || echo "  pulseaudio: $(YELLOW)Not found$(NC)"
	@echo ""
	@echo "Media tools:"
	@which ffmpeg > /dev/null 2>&1 && echo "  ffmpeg: $(GREEN)OK$(NC)" || echo "  ffmpeg: $(YELLOW)Not found$(NC)"
	@echo ""
	@echo "GPU support:"
	@which nvidia-smi > /dev/null 2>&1 && echo "  NVIDIA GPU: $(GREEN)Detected$(NC)" || echo "  NVIDIA GPU: $(BLUE)Not detected (CPU mode)$(NC)"
	@echo ""
	@echo "Container tools:"
	@which docker > /dev/null 2>&1 && echo "  Docker: $(GREEN)OK$(NC)" || echo "  Docker: $(YELLOW)Not found$(NC)"
	@echo ""

## check-deps: Check all dependencies status
check-deps: check-python check-system
	@echo -e "$(GREEN)[INFO]$(NC) Dependency check complete"

# ============================================================================
# HELP
# ============================================================================

## help: Show this help message
help:
	@echo ""
	@echo "M.I.A - Multimodal Intelligent Assistant"
	@echo "========================================="
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Installation:"
	@echo "  install          Full installation (recommended)"
	@echo "  install-core     Install core dependencies only"
	@echo "  install-dev      Install development dependencies"
	@echo "  install-extras   Install optional dependencies"
	@echo "  setup            Quick setup for first-time users"
	@echo ""
	@echo "Running:"
	@echo "  run              Start M.I.A (default mode)"
	@echo "  run-debug        Start in debug mode"
	@echo "  run-text         Start in text-only mode"
	@echo "  run-audio        Start with audio enabled"
	@echo "  run-api          Start API server"
	@echo "  run-ui           Start Streamlit UI"
	@echo "  info             Show version info"
	@echo ""
	@echo "Development:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests"
	@echo "  test-integration Run integration tests"
	@echo "  lint             Run code linter"
	@echo "  format           Format code with black"
	@echo "  type-check       Run type checking"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run in Docker"
	@echo "  docker-stop      Stop Docker container"
	@echo "  docker-logs      View container logs"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean            Remove cache files"
	@echo "  clean-all        Remove all generated files"
	@echo "  uninstall        Full uninstall"
	@echo "  check-system     Check system dependencies"
	@echo ""
	@echo "For more information, see README.md"
	@echo ""
