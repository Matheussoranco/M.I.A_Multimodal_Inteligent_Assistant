#!/bin/bash

# M.I.A Installation Script for Unix/Linux
# This script sets up the M.I.A virtual assistant environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
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

# Check if Python 3.8+ is installed
check_python() {
    print_status "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python3"
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            exit 1
        fi
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python"
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python not found. Please install Python 3.8+"
        exit 1
    fi
}

# Check for system dependencies
check_system_deps() {
    print_status "Checking system dependencies..."
    
    # Check for audio libraries
    if ! pkg-config --exists alsa 2>/dev/null && ! pkg-config --exists portaudio-2.0 2>/dev/null; then
        print_warning "Audio libraries not found. Installing recommended packages..."
        
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y python3-dev python3-venv portaudio19-dev alsa-utils
        elif command -v yum &> /dev/null; then
            sudo yum install -y python3-devel python3-venv portaudio-devel alsa-lib-devel
        elif command -v pacman &> /dev/null; then
            sudo pacman -S --noconfirm python python-virtualenv portaudio alsa-utils
        elif command -v brew &> /dev/null; then
            brew install portaudio
        else
            print_warning "Could not install audio dependencies automatically. Please install manually."
        fi
    fi
    
    # Check for FFmpeg
    if ! command -v ffmpeg &> /dev/null; then
        print_warning "FFmpeg not found. Some audio features may not work."
        print_status "To install FFmpeg:"
        echo "  Ubuntu/Debian: sudo apt-get install ffmpeg"
        echo "  RHEL/CentOS:   sudo yum install ffmpeg"
        echo "  Arch:          sudo pacman -S ffmpeg"
        echo "  macOS:         brew install ffmpeg"
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Removing old one..."
        rm -rf venv
    fi
    
    $PYTHON_CMD -m venv venv
    print_success "Virtual environment created"
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install Python dependencies
install_deps() {
    print_status "Installing Python dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        print_error "requirements.txt not found"
        exit 1
    fi
    
    # Install package in development mode
    pip install -e .
    
    print_success "Dependencies installed"
}

# Setup configuration
setup_config() {
    print_status "Setting up configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_success "Configuration template copied to .env"
            print_warning "Please edit .env file with your API keys before running M.I.A"
        else
            print_warning ".env.example not found. You'll need to create .env manually"
        fi
    else
        print_status ".env file already exists"
    fi
}

# Create run script
create_run_script() {
    print_status "Creating run script..."
    
    cat > run.sh << 'EOF'
#!/bin/bash

# M.I.A Run Script for Unix/Linux
# Activates virtual environment and starts M.I.A

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[M.I.A]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_error "Virtual environment not found. Please run ./install.sh first"
    exit 1
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    print_error ".env file not found. Please copy .env.example to .env and configure your API keys"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Start M.I.A
print_status "Starting M.I.A - Your Personal Virtual Assistant"
print_status "Press Ctrl+C to stop"

python -m mia.main "$@"
EOF

    chmod +x run.sh
    print_success "Run script created (run.sh)"
}

# Create development script
create_dev_script() {
    print_status "Creating development script..."
    
    cat > dev.sh << 'EOF'
#!/bin/bash

# M.I.A Development Script
# Provides development utilities

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[DEV]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

case "$1" in
    "test")
        print_status "Running tests..."
        source venv/bin/activate
        python -m pytest tests/ -v
        ;;
    "lint")
        print_status "Running linter..."
        source venv/bin/activate
        flake8 --max-line-length=100 --ignore=E203,W503 .
        ;;
    "format")
        print_status "Formatting code..."
        source venv/bin/activate
        black --line-length=100 .
        ;;
    "install-dev")
        print_status "Installing development dependencies..."
        source venv/bin/activate
        pip install -r requirements-dev.txt
        ;;
    "clean")
        print_status "Cleaning cache and build files..."
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find . -type f -name "*.pyc" -delete 2>/dev/null || true
        rm -rf build/ dist/ *.egg-info/ .pytest_cache/
        print_success "Cleaned"
        ;;
    *)
        echo "M.I.A Development Script"
        echo "Usage: ./dev.sh {test|lint|format|install-dev|clean}"
        echo ""
        echo "Commands:"
        echo "  test        - Run tests"
        echo "  lint        - Run code linter"
        echo "  format      - Format code with black"
        echo "  install-dev - Install development dependencies"
        echo "  clean       - Clean cache and build files"
        ;;
esac
EOF

    chmod +x dev.sh
    print_success "Development script created (dev.sh)"
}

# Create uninstall script
create_uninstall_script() {
    print_status "Creating uninstall script..."
    
    cat > uninstall.sh << 'EOF'
#!/bin/bash

# M.I.A Uninstall Script
# Removes virtual environment and generated files

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "M.I.A Uninstall Script"
echo "This will remove:"
echo "  - Virtual environment (venv/)"
echo "  - Cache files (__pycache__/)"
echo "  - Build files (build/, dist/, *.egg-info/)"
echo "  - Log files (logs/)"
echo ""

read -p "Are you sure you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_error "Uninstall cancelled"
    exit 1
fi

print_warning "Removing virtual environment..."
rm -rf venv/

print_warning "Removing cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

print_warning "Removing build files..."
rm -rf build/ dist/ *.egg-info/ .pytest_cache/

print_warning "Removing log files..."
rm -rf logs/

print_success "M.I.A uninstalled successfully"
print_warning "Note: Configuration files (.env) and memory files were preserved"
EOF

    chmod +x uninstall.sh
    print_success "Uninstall script created (uninstall.sh)"
}

# Main installation process
main() {
    echo "=============================================="
    echo "M.I.A - Personal Virtual Assistant Installer"
    echo "=============================================="
    echo ""
    
    check_python
    check_system_deps
    create_venv
    activate_venv
    install_deps
    setup_config
    create_run_script
    create_dev_script
    create_uninstall_script
    
    echo ""
    echo "=============================================="
    print_success "Installation completed successfully!"
    echo "=============================================="
    echo ""
    echo "Next steps:"
    echo "1. Edit .env file with your API keys:"
    echo "   nano .env"
    echo ""
    echo "2. Run M.I.A:"
    echo "   ./run.sh"
    echo ""
    echo "3. For development:"
    echo "   ./dev.sh test     # Run tests"
    echo "   ./dev.sh lint     # Check code quality"
    echo "   ./dev.sh format   # Format code"
    echo ""
    echo "4. To uninstall:"
    echo "   ./uninstall.sh"
    echo ""
}

# Run main function
main "$@"
