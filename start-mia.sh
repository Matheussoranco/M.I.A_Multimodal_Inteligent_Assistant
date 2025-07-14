#!/bin/bash
# M.I.A Startup Script for Linux/macOS
# This script activates the virtual environment and starts M.I.A

set -e  # Exit on any error

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

print_info "Starting M.I.A..."
echo ""
echo "🚀 M.I.A - The successor of pseudoJarvis"
echo "=========================================="
echo ""

# Start M.I.A with any command line arguments
python main.py "$@"

# Check exit status
if [ $? -ne 0 ]; then
    print_error "M.I.A exited with an error"
    exit 1
fi
