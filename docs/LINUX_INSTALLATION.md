# Linux Installation Guide for M.I.A

This guide provides detailed instructions for installing and running M.I.A on Linux systems.

## Supported Distributions

- **Ubuntu/Debian** (20.04+, 22.04+, Debian 11+)
- **Fedora** (36+)
- **CentOS/RHEL/Rocky Linux** (8+)
- **Arch Linux/Manjaro**
- **openSUSE**

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/M.I.A_Multimodal_Inteligent_Assistant.git
cd M.I.A_Multimodal_Inteligent_Assistant

# Set permissions (first time only)
chmod +x setup-permissions.sh
./setup-permissions.sh

# Install and run
make install
make run
```

## Detailed Installation

### Step 1: System Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y \
    python3 python3-pip python3-venv python3-dev \
    build-essential \
    portaudio19-dev libportaudio2 libportaudiocpp0 \
    alsa-utils libasound2-dev \
    pulseaudio \
    ffmpeg \
    git curl wget \
    libsndfile1
```

#### Fedora
```bash
sudo dnf install -y \
    python3 python3-pip python3-devel \
    portaudio-devel \
    alsa-utils \
    pulseaudio \
    ffmpeg \
    git curl wget \
    libsndfile-devel \
    gcc gcc-c++
```

#### Arch Linux
```bash
sudo pacman -Syu
sudo pacman -S --needed \
    python python-pip python-virtualenv \
    portaudio \
    alsa-utils \
    pulseaudio \
    ffmpeg \
    git curl wget \
    libsndfile \
    base-devel
```

#### openSUSE
```bash
sudo zypper install -y \
    python3 python3-pip python3-devel \
    portaudio-devel \
    alsa-utils \
    pulseaudio \
    ffmpeg \
    git curl wget \
    libsndfile-devel \
    gcc gcc-c++
```

### Step 2: Audio Configuration

Ensure your user is in the `audio` group:

```bash
sudo usermod -aG audio $USER
# Log out and back in for changes to take effect
```

Test audio:
```bash
# Test recording
arecord -d 3 test.wav
# Test playback
aplay test.wav
```

### Step 3: Python Environment

```bash
cd M.I.A_Multimodal_Inteligent_Assistant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Install M.I.A in development mode
pip install -e .
```

### Step 4: Configuration

```bash
# Copy configuration template
cp config/.env.example .env

# Edit with your preferred editor
nano .env
# or
vim .env
```

**Important settings in `.env`:**
- `OPENAI_API_KEY` - If using OpenAI
- `OLLAMA_API_KEY` - If using local Ollama (usually not needed)

### Step 5: Create Required Directories

```bash
mkdir -p logs memory cache
```

## Running M.I.A

### Using Make (Recommended)

```bash
make run           # Default mode
make run-text      # Text-only mode (no audio)
make run-audio     # With audio input/output
make run-debug     # Debug mode with verbose output
make run-api       # Start API server
make run-ui        # Start web UI (Streamlit)
```

### Using Scripts

```bash
./scripts/run/run.sh                    # Default
./scripts/run/run.sh --text-only        # Text mode
./scripts/run/run.sh --debug            # Debug mode
./scripts/run/run.sh --model-id gpt-4   # Specific model
```

### Direct Python

```bash
source venv/bin/activate
python -m mia.main
python -m mia.main --info
python -m mia.main --text-only
```

## Local LLM with Ollama

### Install Ollama
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Pull Models
```bash
# Recommended models
ollama pull mistral:instruct       # General conversation
ollama pull qwen2.5:3b-instruct-q4_K_M  # Fast responses
ollama pull deepseek-r1:1.5b       # Reasoning tasks
ollama pull qwen2.5-coder:3b-q4_0  # Code assistance
```

### Start Ollama
```bash
# As service (recommended)
sudo systemctl enable ollama
sudo systemctl start ollama

# Or manually
ollama serve
```

### Configure M.I.A for Ollama
Edit `config/config.yaml`:
```yaml
llm:
  provider: ollama
  model_id: mistral:instruct
  url: http://localhost:11434
```

## Running as a Service

### Systemd Service

```bash
# Install service file
sudo cp scripts/linux/mia@.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable service (starts on boot)
sudo systemctl enable mia@$USER

# Start service
sudo systemctl start mia@$USER

# Check status
sudo systemctl status mia@$USER

# View logs
journalctl -u mia@$USER -f

# Stop service
sudo systemctl stop mia@$USER
```

## Docker

### Build and Run
```bash
# Build image
docker build -t mia-assistant .

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f mia

# Stop
docker-compose down
```

### GPU Support (NVIDIA)
Edit `docker-compose.yml` and uncomment the GPU sections, then:
```bash
docker-compose up -d
```

## Troubleshooting

### Audio Issues

**No audio input:**
```bash
# Check audio devices
arecord -l

# Test PulseAudio
pactl info

# Restart PulseAudio
pulseaudio -k
pulseaudio --start
```

**Permission denied:**
```bash
# Add user to audio group
sudo usermod -aG audio $USER
# Log out and back in
```

### Python Issues

**Module not found:**
```bash
# Ensure venv is activated
source venv/bin/activate

# Reinstall
pip install -e .
```

**Wrong Python version:**
```bash
# Check version (need 3.8+)
python3 --version

# Use specific Python
python3.10 -m venv venv
```

### Ollama Issues

**Connection refused:**
```bash
# Check if Ollama is running
sudo systemctl status ollama

# Start Ollama
sudo systemctl start ollama
# Or
ollama serve
```

### Memory Issues

If running out of memory with large models:
```bash
# Use smaller models
ollama pull qwen2.5:0.5b

# Or reduce context in config.yaml
llm:
  max_tokens: 512
```

## Development

```bash
# Install dev dependencies
make install-dev

# Run tests
make test

# Run linter
make lint

# Format code
make format

# Clean cache
make clean
```

## Uninstalling

```bash
# Remove installation (preserves config)
make uninstall

# Or complete removal
make clean-all
rm -rf M.I.A_Multimodal_Inteligent_Assistant
```

## Getting Help

- Check `make help` for available commands
- Review logs in `logs/` directory
- Open an issue on GitHub for bugs
