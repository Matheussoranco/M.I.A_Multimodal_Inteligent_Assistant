#!/bin/bash

# M.I.A System Compatibility Checker
# Verifies system requirements and dependencies

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}"
    echo "=============================================="
    echo "   M.I.A System Compatibility Checker        "
    echo "=============================================="
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check Python version
check_python() {
    echo ""
    echo "Checking Python..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python $PYTHON_VERSION (python3)"
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            return 1
        fi
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python $PYTHON_VERSION (python)"
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            return 1
        fi
    else
        print_error "Python not found"
        print_info "Install Python 3.8+ from https://python.org"
        return 1
    fi
}

# Check system packages
check_system_packages() {
    echo ""
    echo "Checking system packages..."
    
    # Check package manager
    if command -v apt-get &> /dev/null; then
        PKG_MGR="apt-get"
        print_success "Package manager: apt-get (Debian/Ubuntu)"
    elif command -v yum &> /dev/null; then
        PKG_MGR="yum"
        print_success "Package manager: yum (RHEL/CentOS)"
    elif command -v pacman &> /dev/null; then
        PKG_MGR="pacman"
        print_success "Package manager: pacman (Arch Linux)"
    elif command -v brew &> /dev/null; then
        PKG_MGR="brew"
        print_success "Package manager: Homebrew (macOS)"
    else
        print_warning "No recognized package manager found"
        PKG_MGR="none"
    fi
    
    # Check audio libraries
    if pkg-config --exists alsa 2>/dev/null; then
        print_success "ALSA audio library found"
    elif pkg-config --exists portaudio-2.0 2>/dev/null; then
        print_success "PortAudio library found"
    else
        print_warning "Audio libraries not found"
        case $PKG_MGR in
            "apt-get")
                print_info "Install with: sudo apt-get install portaudio19-dev alsa-utils"
                ;;
            "yum")
                print_info "Install with: sudo yum install portaudio-devel alsa-lib-devel"
                ;;
            "pacman")
                print_info "Install with: sudo pacman -S portaudio alsa-utils"
                ;;
            "brew")
                print_info "Install with: brew install portaudio"
                ;;
        esac
    fi
    
    # Check FFmpeg
    if command -v ffmpeg &> /dev/null; then
        FFMPEG_VERSION=$(ffmpeg -version 2>/dev/null | head -n1 | cut -d' ' -f3)
        print_success "FFmpeg $FFMPEG_VERSION found"
    else
        print_warning "FFmpeg not found (optional but recommended)"
        case $PKG_MGR in
            "apt-get")
                print_info "Install with: sudo apt-get install ffmpeg"
                ;;
            "yum")
                print_info "Install with: sudo yum install ffmpeg"
                ;;
            "pacman")
                print_info "Install with: sudo pacman -S ffmpeg"
                ;;
            "brew")
                print_info "Install with: brew install ffmpeg"
                ;;
        esac
    fi
    
    # Check Git
    if command -v git &> /dev/null; then
        GIT_VERSION=$(git --version | cut -d' ' -f3)
        print_success "Git $GIT_VERSION found"
    else
        print_error "Git not found (required for installation)"
        return 1
    fi
}

# Check hardware
check_hardware() {
    echo ""
    echo "Checking hardware..."
    
    # Memory
    if command -v free &> /dev/null; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$MEMORY_GB" -ge 4 ]; then
            print_success "Memory: ${MEMORY_GB}GB (sufficient)"
        else
            print_warning "Memory: ${MEMORY_GB}GB (4GB+ recommended)"
        fi
    elif command -v system_profiler &> /dev/null; then
        MEMORY_GB=$(system_profiler SPHardwareDataType | grep "Memory:" | awk '{print $2}' | cut -d' ' -f1)
        print_success "Memory: ${MEMORY_GB} (macOS)"
    else
        print_info "Memory check not available on this system"
    fi
    
    # GPU (NVIDIA)
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        print_success "NVIDIA GPU: $GPU_INFO"
        print_info "CUDA acceleration available"
    else
        print_info "No NVIDIA GPU detected (CPU mode will be used)"
    fi
    
    # Audio devices
    if command -v arecord &> /dev/null; then
        AUDIO_DEVICES=$(arecord -l 2>/dev/null | grep "card" | wc -l)
        if [ "$AUDIO_DEVICES" -gt 0 ]; then
            print_success "Audio input devices: $AUDIO_DEVICES found"
        else
            print_warning "No audio input devices found"
        fi
    elif command -v system_profiler &> /dev/null; then
        print_info "Audio devices check not implemented for macOS"
    fi
}

# Check network connectivity
check_network() {
    echo ""
    echo "Checking network connectivity..."
    
    # Check internet connectivity
    if ping -c 1 8.8.8.8 &> /dev/null; then
        print_success "Internet connectivity available"
    else
        print_error "No internet connectivity"
        print_info "Internet required for downloading models and API access"
        return 1
    fi
    
    # Check API endpoints
    if curl -s --connect-timeout 5 https://api.openai.com &> /dev/null; then
        print_success "OpenAI API endpoint reachable"
    else
        print_warning "OpenAI API endpoint not reachable"
    fi
    
    if curl -s --connect-timeout 5 http://localhost:11434 &> /dev/null; then
        print_success "Ollama local server detected"
    else
        print_info "Ollama local server not running (optional)"
    fi
}

# Check disk space
check_disk_space() {
    echo ""
    echo "Checking disk space..."
    
    AVAILABLE_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    
    if [ "$AVAILABLE_GB" -ge 5 ]; then
        print_success "Available disk space: ${AVAILABLE_GB}GB (sufficient)"
    elif [ "$AVAILABLE_GB" -ge 2 ]; then
        print_warning "Available disk space: ${AVAILABLE_GB}GB (minimum met, 5GB+ recommended)"
    else
        print_error "Available disk space: ${AVAILABLE_GB}GB (insufficient, 2GB+ required)"
        return 1
    fi
}

# Generate compatibility report
generate_report() {
    echo ""
    echo "=============================================="
    echo "           Compatibility Report               "
    echo "=============================================="
    
    if [ $PYTHON_OK -eq 1 ] && [ $SYSTEM_OK -eq 1 ] && [ $NETWORK_OK -eq 1 ] && [ $DISK_OK -eq 1 ]; then
        print_success "System is compatible with M.I.A!"
        echo ""
        echo "Next steps:"
        echo "  1. Run: ./install.sh"
        echo "  2. Configure: nano .env"
        echo "  3. Start: ./run.sh"
        return 0
    else
        print_error "System compatibility issues found"
        echo ""
        echo "Please resolve the issues above before installing M.I.A"
        return 1
    fi
}

# Main function
main() {
    print_header
    
    # Initialize status variables
    PYTHON_OK=0
    SYSTEM_OK=0
    NETWORK_OK=0
    DISK_OK=0
    
    # Run checks
    if check_python; then PYTHON_OK=1; fi
    if check_system_packages; then SYSTEM_OK=1; fi
    check_hardware
    if check_network; then NETWORK_OK=1; fi
    if check_disk_space; then DISK_OK=1; fi
    
    # Generate report
    generate_report
}

# Run main function
main "$@"
