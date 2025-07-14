# Friday/M.I.A - Your Personal Virtual Assistant

This is Friday/M.I.A, your all-in-one personal virtual assistant for PC, phone, and smartwatch. M.I.A leverages advanced LLMs and modular automation to help you with just about anything on your computer and beyond.

## 🚀 Quick Start

### 🐧 Unix/Linux/macOS (Recommended)

#### Super Quick Start (One Command)
```bash
# Clone and run in one go
git clone https://github.com/yourusername/friday-mia.git
cd friday-mia
chmod +x quickstart.sh
./quickstart.sh
```

#### Step-by-Step Installation
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/friday-mia.git
cd friday-mia

# 2. Run installation script
chmod +x install.sh
./install.sh

# 3. Configure your API keys
nano .env  # Edit with your API keys

# 4. Run M.I.A
./run.sh
```

#### Using Make (Alternative)
```bash
# Install and setup development environment
make dev-setup

# Edit configuration
nano .env

# Run M.I.A
make run
```

### 🪟 Windows

#### Option 1: Automated Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/friday-mia.git
cd friday-mia

# Run the installation script
install.bat

# Configure your API keys in .env file
# Then run:
run.bat
```

#### Option 2: Manual Installation
```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies  
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 3. Configure environment
copy .env.example .env
# Edit .env with your API keys

# 4. Run M.I.A
python -m main_modules.main
```

### 🐳 Docker (Cross-Platform)

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t mia-assistant .
docker run -it --rm -v $(pwd)/.env:/app/.env mia-assistant
```

## 🔧 Configuration

Copy `.env.example` to `.env` and configure your API keys:

```bash
# Required for basic functionality
OPENAI_API_KEY=your-openai-api-key    # For GPT models
OLLAMA_API_KEY=ollama                 # For local models

# Optional API keys
ANTHROPIC_API_KEY=your-anthropic-key  # For Claude
GOOGLE_CALENDAR_API_KEY=your-key      # For calendar integration
GEMINI_API_KEY=your-gemini-key       # For Google AI
```

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **Operating System**: 
  - ✅ Linux (Ubuntu 18.04+, Debian 10+, CentOS 7+, Arch Linux)
  - ✅ macOS (10.14+)
  - ✅ Windows 10/11
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space for models and cache

### Linux Dependencies
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev python3-venv portaudio19-dev alsa-utils ffmpeg

# RHEL/CentOS/Fedora
sudo yum install python3-devel python3-venv portaudio-devel alsa-lib-devel ffmpeg

# Arch Linux
sudo pacman -S python python-virtualenv portaudio alsa-utils ffmpeg
```

### macOS Dependencies
```bash
# Using Homebrew
brew install portaudio ffmpeg
```

### Windows Dependencies
- Most dependencies are handled automatically
- For audio: Windows Audio Session API (WASAPI) support
- Optional: Microsoft C++ Build Tools for some packages

### Optional Dependencies

- **PyAudio**: For advanced audio processing (automatically handled)
- **CUDA**: For GPU acceleration with PyTorch models
- **FFmpeg**: For advanced audio/video processing (recommended)
- **Docker**: For containerized deployment

## 🛠️ Development

### Development Setup
```bash
# Install with development dependencies
./install.sh
./dev.sh install-dev

# Or using Make
make dev-setup
```

### Available Development Commands
```bash
# Using dev.sh script
./dev.sh test          # Run tests with coverage
./dev.sh lint          # Run code linting
./dev.sh format        # Format code with black
./dev.sh docs          # Build documentation
./dev.sh security      # Run security checks
./dev.sh clean         # Clean cache files

# Using Make
make test              # Run tests
make lint              # Run linting
make format            # Format code
make docs              # Build documentation
make clean             # Clean up
```

### Running Options
```bash
# Basic run
./run.sh

# Debug mode
./run.sh --debug

# With custom model
./run.sh --model-id gpt-4 --api-key your-key

# With image input
./run.sh --image-input path/to/image.jpg

# Enable advanced reasoning
./run.sh --enable-reasoning --debug
```

# Key Capabilities

- **Conversational AI**: Natural, context-aware chat powered by LLMs (OpenAI, Ollama, HuggingFace, etc.)
- **Speech Recognition & Synthesis**: Voice input (ASR) and human-like speech output (TTS)
- **File Management**: Open, move, delete, and search files/folders
- **Application Control**: Launch and close programs, interact with running apps
- **Web Automation**: Automated browsing, form filling, and web actions (Selenium)
- **Clipboard & Notifications**: Copy/paste and send system notifications
- **System Settings**: Change system settings (stub, extensible)
- **Email & Calendar**: Send emails (SMTP), create calendar events (Google Calendar API stub)
- **Messaging**: Send messages via WhatsApp, Telegram, and more (stub, extensible)
- **Smart Home**: Control smart devices via Home Assistant (stub, extensible)
- **Note-Taking & Reminders**: Create notes, schedule events, and set reminders
- **Multimodal Processing**: Understand and process audio, images, and (stub) video
- **Memory & Learning**: Long-term memory, user feedback, and personalization
- **Plugin System**: Dynamically load new skills and tools
- **Security & Privacy**: Permission checks, user consent, and logging
- **Cross-Platform**: Works on PC, phone, and smartwatch (Bluetooth, extensible)
- **Extensible & Open Source**: Easy to add new modules, plugins, and integrations

# How It Works

1. **Input**: Speak, type, or send a command
2. **LLM Processing**: Your request is interpreted by a Large Language Model
3. **Action Execution**: M.I.A securely executes the relevant module or plugin
4. **Output**: Get a response via speech, text, or system action

## 🔧 Troubleshooting

### Common Issues

#### Installation Issues
```bash
# Permission denied
chmod +x *.sh

# Python version issues
python3 --version  # Must be 3.8+

# Audio dependencies (Linux)
sudo apt-get install portaudio19-dev alsa-utils

# Missing compiler (for some packages)
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev
# macOS
xcode-select --install
```

#### Runtime Issues
```bash
# Check system info
./dev.sh info

# Test installation
./dev.sh test

# Debug mode
./run.sh --debug

# Check logs
tail -f logs/mia.log
```

#### API Key Issues
```bash
# Verify .env file exists and has correct keys
cat .env | grep API_KEY

# Test specific provider
./run.sh --model-id gpt-3.5-turbo --debug
```

### Getting Help
- Check logs in `logs/` directory
- Run `./dev.sh info` for system information
- Run tests with `./dev.sh test`
- Enable debug mode with `./run.sh --debug`

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Configuration Guide](docs/configuration.md)
- [API Reference](docs/api.md)
- [Plugin Development](docs/plugins.md)
- [Troubleshooting](docs/troubleshooting.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Install development dependencies: `./dev.sh install-dev`
4. Make your changes and add tests
5. Run tests and linting: `./dev.sh test && ./dev.sh lint`
6. Format code: `./dev.sh format`
7. Commit your changes: `git commit -am 'Add new feature'`
8. Push to the branch: `git push origin feature/new-feature`
9. Create a Pull Request

## 📄 License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models and Whisper
- HuggingFace for Transformers library
- The open-source community for various dependencies
- Contributors and beta testers

## 📞 Support

- 📧 Email: matheussoranco@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/Matheussoranco/M.I.A-The-successor-of-pseudoJarvis/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/Matheussoranco/M.I.A-The-successor-of-pseudoJarvis/discussions)

---

Made with ❤️ by the M.I.A development team

# File Organization

- `main.py` — Main entry point
- `main_modules/` — Core logic and chat interface
- `llm/` — LLM management and inference
- `langchain/` — Output verification/workflows
- `system/` — System actions (file, process, clipboard, etc.)
- `audio/` — Audio and speech modules
- `utils/` — Automation, notes, messaging, etc.
- `multimodal/` — Multimodal (audio/image/video) processing
- `memory/` — Agent memory/context
- `tools/` — Action execution and integrations
- `plugins/` — Dynamic plugin system
- `learning/` — User learning and personalization
- `security/` — Security and permissions
- `deployment/` — Deployment and platform support

# Requirements

The following Python libraries are required (see `requirements.txt` for full list):

- transformers
- torch
- openai
- pydub
- sounddevice
- soundfile
- PyAudio
- numpy
- matplotlib
- Pillow
- requests
- selenium
- pyperclip
- psutil
- argparse
- smtplib
- email
- (Optional) pyqt5, tkinter, plyer, win10toast, google-api-python-client, homeassistant, etc.

# To Do

- [ ] Add a GUI interface (Tkinter, PyQt, or web-based)
- [ ] Add mobile companion app and sync
- [ ] Expand plugin marketplace and auto-update system
- [ ] Enhance context retention and session memory
- [ ] Add advanced error recovery and user prompts
- [ ] Harden security (sandboxing, 2FA, encrypted storage)
- [ ] Integrate more messaging and cloud storage services
- [ ] Enable learning from user demonstration/scripts
- [ ] Improve multimodal (video, sensor) capabilities
- [ ] Add comprehensive logging and audit trail
- [ ] Expand smart home integrations
- [ ] Add more LLM providers and plug-ins
- [ ] Enhance user authentication and profiles

# Scientific Articles and Sources

This project draws inspiration and implementation details from the following research papers and documentation:

- Radford, A. et al. (2023). Robust Speech Recognition via OpenAI Whisper.
- Touvron, H. et al. (2023). LLaMA: Open and Efficient Foundation Language Models.
- Shankar, V. et al. (2022). SpeechT5: Unified-Text-to-Speech.
- Selenium WebDriver Documentation: Automating Browser Tasks
- Webbrowser Module: Python Docs
- Hugging Face Documentation: Transformers Library

# Contributing

Feel free to fork this repository and make pull requests. Suggestions and improvements are always welcome!

# License

This project is licensed under the AGPL-3.0. See the LICENSE file for more details.
