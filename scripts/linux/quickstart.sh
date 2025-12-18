#!/bin/bash
# M.I.A - Quick Start Script for Linux
# One-line installation and setup

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
cat << 'EOF'
 __  __   ___    _    
|  \/  | |_ _|  / \   
| |\/| |  | |  / _ \  
| |  | | _| |_/ ___ \ 
|_|  |_||___/_/   \_\
                      
Multimodal Intelligent Assistant
EOF
echo -e "${NC}"

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_DIR"

# Check for Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    print_status "Install with your package manager:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    echo "  Fedora:        sudo dnf install python3 python3-pip"
    echo "  Arch:          sudo pacman -S python python-pip"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_status "Found Python $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
print_status "Installing dependencies..."
if [ -f "requirements-core.txt" ]; then
    pip install -r requirements-core.txt
else
    pip install -r requirements.txt
fi

# Install package
pip install -e .

# Setup configuration
if [ ! -f ".env" ]; then
    if [ -f "config/.env.example" ]; then
        cp config/.env.example .env
        print_status "Created .env from template"
    fi
fi

# Create necessary directories
mkdir -p logs memory cache

echo ""
print_success "M.I.A is ready!"
echo ""
echo "To start M.I.A:"
echo "  source venv/bin/activate"
echo "  python -m mia.main"
echo ""
echo "Or use make commands:"
echo "  make run          # Start M.I.A"
echo "  make run-text     # Text-only mode"
echo "  make run-debug    # Debug mode"
echo "  make help         # Show all commands"
echo ""
echo "Don't forget to edit .env with your API keys!"
echo ""

# Ask if user wants to start M.I.A
read -p "Start M.I.A now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python -m mia.main --info
fi
