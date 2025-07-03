# Friday/M.I.A - Your Personal Virtual Assistant

This is Friday/M.I.A, your all-in-one personal virtual assistant for PC, phone, and smartwatch. M.I.A leverages advanced LLMs and modular automation to help you with just about anything on your computer and beyond.

## 🚀 Quick Start (Windows)

### Option 1: Automated Installation
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

### Option 2: Manual Installation
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

- **Python**: 3.8 or higher
- **Operating System**: Windows 10/11 (macOS/Linux support planned)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space for models and cache

### Optional Dependencies

- **PyAudio**: For advanced audio processing (may require system libraries)
- **CUDA**: For GPU acceleration with PyTorch models
- **FFmpeg**: For advanced audio/video processing

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

# Installation

## Windows Installation

### Quick Setup
1. Run `install.bat` as administrator
2. Configure `.env` with your API keys  
3. Run `run.bat` to start M.I.A

### Manual Installation
```bash
# Clone and setup
git clone https://github.com/yourusername/friday-mia.git
cd friday-mia
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# Configure and run
copy .env.example .env
# Edit .env with your credentials
python -m main_modules.main
```

## Linux/macOS Installation

```bash
git clone https://github.com/yourusername/friday-mia.git
cd friday-mia
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

cp .env.example .env
# Configure .env with your API keys
python -m main_modules.main
```

## Docker Installation (Coming Soon)

```bash
docker build -t mia-assistant .
docker run -it --env-file .env mia-assistant
```

## Usage

See [USAGE.md](USAGE.md) for command-line examples and options.

Configure API keys and device settings in `.env` and relevant modules.

    python main.py

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
