#!/bin/bash

# M.I.A Development Script
# Provides development utilities for Unix/Linux

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[DEV]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment exists
check_venv() {
    if [ ! -d "venv" ]; then
        print_error "Virtual environment not found. Please run ./install.sh first"
        exit 1
    fi
}

# Activate virtual environment
activate_venv() {
    source venv/bin/activate
}

# Run tests
run_tests() {
    print_status "Running tests..."
    check_venv
    activate_venv
    
    if [ ! -d "tests" ]; then
        print_error "Tests directory not found"
        exit 1
    fi
    
    # Run pytest with coverage
    python -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term
    
    if [ $? -eq 0 ]; then
        print_success "All tests passed!"
        print_status "Coverage report generated in htmlcov/"
    else
        print_error "Some tests failed"
        exit 1
    fi
}

# Run linter
run_lint() {
    print_status "Running code linter..."
    check_venv
    activate_venv
    
    # Run flake8
    print_status "Running flake8..."
    flake8 --max-line-length=100 --ignore=E203,W503 --exclude=venv,__pycache__,.git .
    
    # Run mypy
    print_status "Running mypy..."
    mypy --ignore-missing-imports --disallow-untyped-defs main_modules/ core/ llm/ security/ utils/
    
    print_success "Linting completed"
}

# Format code
format_code() {
    print_status "Formatting code..."
    check_venv
    activate_venv
    
    # Run black
    print_status "Running black..."
    black --line-length=100 --exclude=venv .
    
    # Run isort
    print_status "Running isort..."
    isort --profile black --line-length=100 --skip venv .
    
    print_success "Code formatted"
}

# Install development dependencies
install_dev() {
    print_status "Installing development dependencies..."
    check_venv
    activate_venv
    
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
        print_success "Development dependencies installed"
    else
        print_error "requirements-dev.txt not found"
        exit 1
    fi
}

# Clean cache and build files
clean() {
    print_status "Cleaning cache and build files..."
    
    # Remove Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    
    # Remove build artifacts
    rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/
    
    # Remove logs (with confirmation)
    if [ -d "logs" ]; then
        read -p "Remove log files? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf logs/
            print_status "Log files removed"
        fi
    fi
    
    print_success "Cleanup completed"
}

# Setup pre-commit hooks
setup_hooks() {
    print_status "Setting up pre-commit hooks..."
    check_venv
    activate_venv
    
    # Create pre-commit config if it doesn't exist
    if [ ! -f ".pre-commit-config.yaml" ]; then
        cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
  
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3
        args: [--line-length=100]
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile=black, --line-length=100]
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100, --ignore=E203,W503]
EOF
    fi
    
    # Install pre-commit hooks
    pre-commit install
    print_success "Pre-commit hooks installed"
}

# Generate documentation
build_docs() {
    print_status "Building documentation..."
    check_venv
    activate_venv
    
    if [ ! -d "docs" ]; then
        print_status "Creating docs directory..."
        sphinx-quickstart docs --quiet --project="M.I.A" --author="M.I.A Team" --release="0.1.0" --language="en" --suffix=".rst" --master="index" --epub
    fi
    
    cd docs
    make html
    cd ..
    
    print_success "Documentation built in docs/_build/html/"
}

# Run security check
security_check() {
    print_status "Running security checks..."
    check_venv
    activate_venv
    
    # Install bandit if not present
    pip install bandit
    
    # Run bandit security linter
    bandit -r . -x venv,tests
    
    print_success "Security check completed"
}

# Show system information
show_info() {
    print_status "System Information:"
    echo "  OS: $(uname -s) $(uname -r)"
    echo "  Architecture: $(uname -m)"
    
    if [ -d "venv" ]; then
        source venv/bin/activate
        echo "  Python: $(python --version)"
        echo "  Virtual Environment: Active"
        
        # Check key dependencies
        if python -c "import torch" 2>/dev/null; then
            echo "  PyTorch: $(python -c "import torch; print(torch.__version__)")"
            if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
                echo "  CUDA: Available"
            else
                echo "  CUDA: Not available"
            fi
        else
            echo "  PyTorch: Not installed"
        fi
        
        if python -c "import transformers" 2>/dev/null; then
            echo "  Transformers: $(python -c "import transformers; print(transformers.__version__)")"
        else
            echo "  Transformers: Not installed"
        fi
        
        deactivate
    else
        echo "  Virtual Environment: Not found"
    fi
}

# Show usage
show_usage() {
    echo "M.I.A Development Script"
    echo "Usage: ./dev.sh {command} [options]"
    echo ""
    echo "Commands:"
    echo "  test          - Run tests with coverage"
    echo "  lint          - Run code linter (flake8, mypy)"
    echo "  format        - Format code with black and isort"
    echo "  install-dev   - Install development dependencies"
    echo "  clean         - Clean cache and build files"
    echo "  hooks         - Setup pre-commit hooks"
    echo "  docs          - Build documentation"
    echo "  security      - Run security checks"
    echo "  info          - Show system information"
    echo "  help          - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./dev.sh test          # Run all tests"
    echo "  ./dev.sh lint          # Check code quality"
    echo "  ./dev.sh format        # Format all code"
    echo "  ./dev.sh clean         # Clean build files"
}

# Main function
main() {
    case "$1" in
        "test")
            run_tests
            ;;
        "lint")
            run_lint
            ;;
        "format")
            format_code
            ;;
        "install-dev")
            install_dev
            ;;
        "clean")
            clean
            ;;
        "hooks")
            setup_hooks
            ;;
        "docs")
            build_docs
            ;;
        "security")
            security_check
            ;;
        "info")
            show_info
            ;;
        "help"|"")
            show_usage
            ;;
        *)
            print_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
