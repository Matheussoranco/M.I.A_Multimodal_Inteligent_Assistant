#!/bin/bash
# M.I.A - Linux System Installation Script
# This script installs M.I.A system-wide dependencies and sets up the service

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running as root for system operations
check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "This script requires root privileges for system package installation"
        print_status "Run with: sudo $0"
        exit 1
    fi
}

# Detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        DISTRO_VERSION=$VERSION_ID
        print_status "Detected: $NAME $VERSION_ID"
    elif [ -f /etc/debian_version ]; then
        DISTRO="debian"
    elif [ -f /etc/redhat-release ]; then
        DISTRO="rhel"
    else
        DISTRO="unknown"
    fi
}

# Install system dependencies based on distribution
install_system_deps() {
    print_status "Installing system dependencies..."
    
    case $DISTRO in
        ubuntu|debian|linuxmint|pop)
            apt-get update
            apt-get install -y \
                python3 \
                python3-pip \
                python3-venv \
                python3-dev \
                build-essential \
                portaudio19-dev \
                alsa-utils \
                pulseaudio \
                ffmpeg \
                git \
                curl \
                wget \
                libsndfile1 \
                libasound2-dev \
                libportaudio2 \
                libportaudiocpp0
            ;;
        fedora)
            dnf install -y \
                python3 \
                python3-pip \
                python3-devel \
                portaudio-devel \
                alsa-utils \
                pulseaudio \
                ffmpeg \
                git \
                curl \
                wget \
                libsndfile-devel \
                gcc \
                gcc-c++
            ;;
        centos|rhel|rocky|almalinux)
            yum install -y epel-release
            yum install -y \
                python3 \
                python3-pip \
                python3-devel \
                portaudio-devel \
                alsa-utils \
                pulseaudio \
                ffmpeg \
                git \
                curl \
                wget \
                libsndfile-devel \
                gcc \
                gcc-c++
            ;;
        arch|manjaro|endeavouros)
            pacman -Syu --noconfirm
            pacman -S --noconfirm --needed \
                python \
                python-pip \
                python-virtualenv \
                portaudio \
                alsa-utils \
                pulseaudio \
                ffmpeg \
                git \
                curl \
                wget \
                libsndfile \
                base-devel
            ;;
        opensuse*|suse*)
            zypper install -y \
                python3 \
                python3-pip \
                python3-devel \
                portaudio-devel \
                alsa-utils \
                pulseaudio \
                ffmpeg \
                git \
                curl \
                wget \
                libsndfile-devel \
                gcc \
                gcc-c++
            ;;
        *)
            print_warning "Unknown distribution: $DISTRO"
            print_status "Please install the following packages manually:"
            echo "  - python3, python3-pip, python3-venv, python3-dev"
            echo "  - portaudio (development files)"
            echo "  - alsa-utils, pulseaudio"
            echo "  - ffmpeg"
            echo "  - git, curl, wget"
            echo "  - build tools (gcc, make)"
            return 1
            ;;
    esac
    
    print_success "System dependencies installed"
}

# Install optional GPU support (NVIDIA CUDA)
install_gpu_support() {
    print_status "Checking GPU support..."
    
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        
        read -p "Install CUDA support for PyTorch? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "CUDA support will be installed via pip when setting up M.I.A"
        fi
    else
        print_status "No NVIDIA GPU detected, using CPU mode"
    fi
}

# Install Ollama (local LLM)
install_ollama() {
    print_status "Would you like to install Ollama for local LLM support?"
    
    read -p "Install Ollama? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        if command -v ollama &> /dev/null; then
            print_status "Ollama is already installed"
        else
            print_status "Installing Ollama..."
            curl -fsSL https://ollama.ai/install.sh | sh
            print_success "Ollama installed"
            print_status "You can pull models with: ollama pull mistral:instruct"
        fi
    fi
}

# Setup systemd service
setup_systemd_service() {
    print_status "Would you like to install M.I.A as a systemd service?"
    
    read -p "Install systemd service? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
        
        if [ -f "$SCRIPT_DIR/mia@.service" ]; then
            cp "$SCRIPT_DIR/mia@.service" /etc/systemd/system/
            systemctl daemon-reload
            print_success "Systemd service installed"
            print_status "Enable with: sudo systemctl enable mia@\$USER"
            print_status "Start with:  sudo systemctl start mia@\$USER"
        else
            print_warning "Service file not found"
        fi
    fi
}

# Add user to audio group
setup_audio_permissions() {
    print_status "Setting up audio permissions..."
    
    REAL_USER="${SUDO_USER:-$USER}"
    
    if groups "$REAL_USER" | grep -q '\baudio\b'; then
        print_status "User $REAL_USER already in audio group"
    else
        usermod -aG audio "$REAL_USER"
        print_success "Added $REAL_USER to audio group"
        print_warning "You may need to log out and back in for audio permissions to take effect"
    fi
    
    # Check PulseAudio
    if command -v pulseaudio &> /dev/null; then
        if groups "$REAL_USER" | grep -q '\bpulse\b'; then
            print_status "User $REAL_USER already in pulse group"
        else
            if getent group pulse > /dev/null 2>&1; then
                usermod -aG pulse "$REAL_USER"
                print_success "Added $REAL_USER to pulse group"
            fi
        fi
    fi
}

# Main installation
main() {
    echo ""
    echo "=============================================="
    echo "  M.I.A Linux System Setup"
    echo "=============================================="
    echo ""
    
    check_root
    detect_distro
    install_system_deps
    setup_audio_permissions
    install_gpu_support
    install_ollama
    setup_systemd_service
    
    echo ""
    echo "=============================================="
    print_success "System setup complete!"
    echo "=============================================="
    echo ""
    echo "Next steps (as regular user):"
    echo "  1. cd /path/to/M.I.A_Multimodal_Inteligent_Assistant"
    echo "  2. make install    # or ./scripts/install/install.sh"
    echo "  3. Edit .env with your API keys"
    echo "  4. make run        # or ./scripts/run/run.sh"
    echo ""
}

main "$@"
