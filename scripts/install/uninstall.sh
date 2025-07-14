#!/bin/bash

# M.I.A Uninstall Script for Unix/Linux
# Removes virtual environment and generated files

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[UNINSTALL]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to get user confirmation
confirm() {
    local prompt="$1"
    local default="${2:-N}"
    
    if [ "$default" = "Y" ]; then
        prompt="$prompt (Y/n): "
    else
        prompt="$prompt (y/N): "
    fi
    
    read -p "$prompt" -n 1 -r
    echo
    
    if [ "$default" = "Y" ]; then
        [[ $REPLY =~ ^[Nn]$ ]] && return 1 || return 0
    else
        [[ $REPLY =~ ^[Yy]$ ]] && return 0 || return 1
    fi
}

# Show what will be removed
show_removal_plan() {
    echo "M.I.A Uninstall Script"
    echo "======================"
    echo ""
    echo "This script will remove the following:"
    echo ""
    
    # Check what exists and will be removed
    if [ -d "venv" ]; then
        echo "✓ Virtual environment (venv/)"
    fi
    
    if find . -name "__pycache__" -type d 2>/dev/null | grep -q .; then
        echo "✓ Python cache files (__pycache__/)"
    fi
    
    if find . -name "*.pyc" -type f 2>/dev/null | grep -q .; then
        echo "✓ Compiled Python files (*.pyc)"
    fi
    
    if [ -d "build" ] || [ -d "dist" ] || ls *.egg-info 2>/dev/null | grep -q .; then
        echo "✓ Build files (build/, dist/, *.egg-info/)"
    fi
    
    if [ -d ".pytest_cache" ]; then
        echo "✓ Test cache (.pytest_cache/)"
    fi
    
    if [ -f ".coverage" ] || [ -d "htmlcov" ]; then
        echo "✓ Coverage files (.coverage, htmlcov/)"
    fi
    
    if [ -d "logs" ]; then
        echo "✓ Log files (logs/)"
    fi
    
    if [ -d "memory" ]; then
        echo "✓ Memory files (memory/)"
    fi
    
    echo ""
    echo "The following will be PRESERVED:"
    echo "✓ Source code"
    echo "✓ Configuration files (.env)"
    echo "✓ Documentation"
    echo "✓ README and LICENSE files"
    echo ""
}

# Remove virtual environment
remove_venv() {
    if [ -d "venv" ]; then
        print_status "Removing virtual environment..."
        rm -rf venv/
        print_success "Virtual environment removed"
    else
        print_status "Virtual environment not found (already removed)"
    fi
}

# Remove cache files
remove_cache() {
    print_status "Removing cache files..."
    
    # Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    
    # Test cache
    rm -rf .pytest_cache/
    
    # Coverage files
    rm -f .coverage
    rm -rf htmlcov/
    
    print_success "Cache files removed"
}

# Remove build files
remove_build() {
    print_status "Removing build files..."
    
    rm -rf build/ dist/ *.egg-info/
    
    print_success "Build files removed"
}

# Remove log files
remove_logs() {
    if [ -d "logs" ]; then
        if confirm "Remove log files?"; then
            print_status "Removing log files..."
            rm -rf logs/
            print_success "Log files removed"
        else
            print_status "Log files preserved"
        fi
    fi
}

# Remove memory files
remove_memory() {
    if [ -d "memory" ]; then
        if confirm "Remove memory/learning data? (This will reset M.I.A's learned preferences)"; then
            print_status "Removing memory files..."
            rm -rf memory/
            print_success "Memory files removed"
        else
            print_status "Memory files preserved"
        fi
    fi
}

# Remove configuration files
remove_config() {
    if [ -f ".env" ]; then
        if confirm "Remove configuration file (.env)? (You'll need to reconfigure API keys)"; then
            print_status "Removing configuration file..."
            rm -f .env
            print_success "Configuration file removed"
        else
            print_status "Configuration file preserved"
        fi
    fi
}

# Complete uninstall
complete_uninstall() {
    print_warning "This will perform a COMPLETE uninstall, removing ALL M.I.A files including:"
    print_warning "- Configuration (.env)"
    print_warning "- Memory/learning data"
    print_warning "- Log files"
    print_warning "- Virtual environment"
    print_warning "- All cache and build files"
    echo ""
    
    if confirm "Are you ABSOLUTELY sure you want to completely remove M.I.A?"; then
        remove_venv
        remove_cache
        remove_build
        rm -rf logs/ memory/
        rm -f .env
        
        # Remove generated scripts
        rm -f run.sh dev.sh uninstall.sh
        
        print_success "Complete uninstall finished"
        print_status "M.I.A has been completely removed from your system"
    else
        print_status "Complete uninstall cancelled"
    fi
}

# Show usage
show_usage() {
    echo "M.I.A Uninstall Script"
    echo "Usage: ./uninstall.sh [option]"
    echo ""
    echo "Options:"
    echo "  --complete    - Complete uninstall (removes everything)"
    echo "  --help, -h    - Show this help message"
    echo ""
    echo "Interactive mode (default):"
    echo "  Prompts for each type of file to remove"
    echo ""
}

# Main uninstall process
main() {
    case "$1" in
        "--complete")
            complete_uninstall
            ;;
        "--help"|"-h")
            show_usage
            ;;
        "")
            # Interactive mode
            show_removal_plan
            
            if confirm "Proceed with uninstall?"; then
                echo ""
                remove_venv
                remove_cache
                remove_build
                remove_logs
                remove_memory
                remove_config
                
                echo ""
                echo "=============================================="
                print_success "M.I.A uninstall completed!"
                echo "=============================================="
                echo ""
                print_status "To reinstall M.I.A, run: ./install.sh"
            else
                print_status "Uninstall cancelled"
                exit 0
            fi
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
