#!/bin/bash
# M.I.A Startup Script for Linux/macOS
# This script activates the virtual environment and starts M.I.A

set -euo pipefail  # Exit on any error, undefined variables, and pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo ""
echo "================================="
echo "     M.I.A Startup Script"
echo "================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to the M.I.A directory
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    print_error "Virtual environment not found!"
    print_error "Please run the installation script first:"
    print_error "  scripts/install/install.sh"
    echo ""
    exit 1
fi

print_info "Activating virtual environment..."
source venv/bin/activate

# Check if main.py exists
if [ ! -f "main.py" ]; then
    print_error "main.py not found!"
    print_error "Please ensure you are in the correct directory."
    echo ""
    exit 1
fi

# Check for config file
if [ ! -f "config/.env" ]; then
    if [ -f ".env" ]; then
        print_info "Migrating .env to config/.env..."
        mv ".env" "config/.env"
    else
        print_warning "config/.env file not found!"
        print_warning "Using default configuration..."
        print_warning "Please copy config/.env.example to config/.env and configure your API keys"
    fi
fi

print_info "Starting M.I.A..."
echo ""
echo "🚀 M.I.A - The successor of pseudoJarvis"
echo "=========================================="
echo ""

# Start M.I.A with any command line arguments
python main.py "$@"

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    print_error "M.I.A exited with error code $EXIT_CODE"
    exit $EXIT_CODE
else
    print_success "M.I.A session ended successfully"
fi
