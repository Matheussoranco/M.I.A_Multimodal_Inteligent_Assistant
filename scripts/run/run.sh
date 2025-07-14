#!/bin/bash

# M.I.A Run Script for Unix/Linux
# Activates virtual environment and starts M.I.A

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[M.I.A]${NC} $1"
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

# Function to show usage
show_usage() {
    echo "M.I.A - Personal Virtual Assistant"
    echo "Usage: ./run.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --url URL              LLM API URL (default: http://localhost:11434/v1)"
    echo "  --model-id MODEL       Model ID to use (default: mistral:instruct)"
    echo "  --api-key KEY          API key for the LLM service (default: ollama)"
    echo "  --stt-model MODEL      Speech-to-text model (default: openai/whisper-base.en)"
    echo "  --image-input PATH     Path to image file for multimodal processing"
    echo "  --enable-reasoning     Enable advanced reasoning"
    echo "  --debug               Enable debug mode"
    echo "  --help, -h            Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh                                    # Start with default settings"
    echo "  ./run.sh --debug                          # Start in debug mode"
    echo "  ./run.sh --model-id gpt-3.5-turbo         # Use OpenAI GPT-3.5"
    echo "  ./run.sh --enable-reasoning --debug       # Advanced mode with debugging"
}

# Check if virtual environment exists
check_environment() {
    if [ ! -d "venv" ]; then
        print_error "Virtual environment not found."
        print_status "Please run: ./install.sh"
        exit 1
    fi
    
    if [ ! -f ".env" ]; then
        print_warning ".env file not found."
        if [ -f ".env.example" ]; then
            print_status "Copying .env.example to .env..."
            cp .env.example .env
            print_warning "Please edit .env file with your API keys before running M.I.A"
            print_status "Opening .env in default editor..."
            ${EDITOR:-nano} .env
        else
            print_error "No configuration template found. Please create .env manually."
            exit 1
        fi
    fi
}

# Check system requirements
check_system() {
    print_status "Checking system requirements..."
    
    # Check for audio devices
    if ! command -v arecord &> /dev/null && ! command -v pactl &> /dev/null; then
        print_warning "Audio recording tools not found. Audio input may not work."
    fi
    
    # Check for GPU support
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
    else
        print_status "No NVIDIA GPU detected, using CPU mode"
    fi
}

# Handle command line arguments
parse_arguments() {
    ARGS=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --url)
                ARGS="$ARGS --url $2"
                shift 2
                ;;
            --model-id)
                ARGS="$ARGS --model-id $2"
                shift 2
                ;;
            --api-key)
                ARGS="$ARGS --api-key $2"
                shift 2
                ;;
            --stt-model)
                ARGS="$ARGS --stt-model $2"
                shift 2
                ;;
            --image-input)
                if [ ! -f "$2" ]; then
                    print_error "Image file not found: $2"
                    exit 1
                fi
                ARGS="$ARGS --image-input $2"
                shift 2
                ;;
            --enable-reasoning)
                ARGS="$ARGS --enable-reasoning"
                shift
                ;;
            --debug)
                ARGS="$ARGS --debug"
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Main execution
main() {
    # Parse command line arguments
    parse_arguments "$@"
    
    # Check environment
    check_environment
    check_system
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    # Create logs directory if it doesn't exist
    mkdir -p logs
    
    # Start M.I.A
    print_success "Starting M.I.A - Your Personal Virtual Assistant"
    print_status "Press Ctrl+C to stop"
    echo ""
    
    # Set up signal handling for graceful shutdown
    trap 'print_status "Shutting down M.I.A..."; exit 0' INT TERM
    
    # Run the application
    python -m main_modules.main $ARGS
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
