#!/bin/bash

# M.I.A Quick Start Script for Unix/Linux
# One-command setup and run

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_banner() {
    echo -e "${BLUE}"
    echo "=============================================="
    echo "   M.I.A - Personal Virtual Assistant        "
    echo "         Quick Start Script                   "
    echo "=============================================="
    echo -e "${NC}"
}

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

# Main function
main() {
    print_banner
    
    print_status "This script will:"
    echo "  1. Install M.I.A and all dependencies"
    echo "  2. Set up configuration"
    echo "  3. Run M.I.A for the first time"
    echo ""
    
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Setup cancelled"
        exit 0
    fi
    
    # Step 1: Install
    print_status "Step 1: Installing M.I.A..."
    chmod +x install.sh
    ./install.sh
    
    # Step 2: Configure
    print_status "Step 2: Setting up configuration..."
    if [ ! -f ".env" ] && [ -f ".env.example" ]; then
        cp .env.example .env
        print_warning "Please edit .env with your API keys"
        read -p "Open .env in editor now? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            ${EDITOR:-nano} .env
        fi
    fi
    
    # Step 3: Test installation
    print_status "Step 3: Testing installation..."
    chmod +x dev.sh
    ./dev.sh info
    
    # Step 4: Run
    print_status "Step 4: Starting M.I.A..."
    print_success "Setup complete! Starting M.I.A..."
    echo ""
    
    chmod +x run.sh
    ./run.sh "$@"
}

main "$@"
