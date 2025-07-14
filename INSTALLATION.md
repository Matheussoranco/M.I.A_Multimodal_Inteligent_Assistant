# M.I.A Installation Guide

M.I.A (The successor of pseudoJarvis) is a multimodal intelligent assistant that supports both text and audio interactions.

## 🚀 Quick Start

### Windows
1. Run the installation script:
   ```cmd
   scripts\install\install.bat
   ```
2. Start M.I.A:
   ```cmd
   start-mia.bat
   ```

### Linux/macOS
1. Run the installation script:
   ```bash
   chmod +x scripts/install/install.sh
   ./scripts/install/install.sh
   ```
2. Start M.I.A:
   ```bash
   ./start-mia.sh
   ```

## 📋 Requirements

- **Python 3.8+** (3.10 recommended)
- **pip** (Python package manager)
- **Git** (for cloning the repository)
- **Virtual environment support** (venv)

### System-specific Requirements

#### Windows
- Windows 10 or later
- Visual C++ Build Tools (for some audio dependencies)

#### Linux
- Audio system packages (installed automatically):
  - `portaudio19-dev`
  - `python3-pyaudio`
  - `alsa-utils`

#### macOS
- macOS 10.14 or later
- Homebrew (recommended for dependencies)

## 🔧 Manual Installation

If the automatic installation scripts don't work, you can install manually:

### 1. Clone the Repository
```bash
git clone <repository-url>
cd M.I.A-The-successor-of-pseudoJarvis
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate.bat

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install M.I.A (Development Mode)
```bash
pip install -e .
```

### 5. Run M.I.A
```bash
python main.py
```

## 🎵 Audio Dependencies

M.I.A supports audio input/output. The installation scripts attempt to install these automatically, but you may need to install them manually:

### Windows
```cmd
pip install pyaudio sounddevice
```

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio alsa-utils
pip install pyaudio sounddevice
```

### macOS
```bash
brew install portaudio
pip install pyaudio sounddevice
```

## 🤖 LLM Setup

M.I.A uses Ollama for local LLM inference. You need to:

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Pull a model**: 
   ```bash
   ollama pull gemma2:9b
   # or
   ollama pull llama3:8b
   ```
3. **Start Ollama server**:
   ```bash
   ollama serve
   ```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root (optional):
```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma2:9b
```

### Audio Settings
Audio is optional. If you experience issues:
- On Windows: Install Visual C++ Build Tools
- On Linux: Ensure ALSA is properly configured
- On macOS: Check microphone permissions

## 🚨 Troubleshooting

### Common Issues

#### Python Not Found
- **Windows**: Install Python from [python.org](https://python.org) and add to PATH
- **Linux**: `sudo apt-get install python3 python3-pip`
- **macOS**: Install via Homebrew: `brew install python3`

#### PyAudio Installation Failed
- **Windows**: Install Visual C++ Build Tools or download PyAudio wheel
- **Linux**: Install development headers: `sudo apt-get install portaudio19-dev`
- **macOS**: Install portaudio: `brew install portaudio`

#### Virtual Environment Issues
- Remove existing venv: `rm -rf venv` (Linux/macOS) or `rmdir /s venv` (Windows)
- Recreate: `python -m venv venv`

#### LLM Connection Issues
- Ensure Ollama is running: `ollama serve`
- Check if model is available: `ollama list`
- Verify connection: `curl http://localhost:11434/api/version`

### Getting Help

1. Check the [documentation](docs/)
2. Review the [troubleshooting guide](docs/TROUBLESHOOTING.md)
3. Open an issue on the repository

## 📁 Project Structure

```
M.I.A-The-successor-of-pseudoJarvis/
├── main.py                 # Main entry point
├── start-mia.bat          # Windows startup script
├── start-mia.sh           # Linux/macOS startup script
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
├── src/mia/              # Main application code
├── scripts/              # Installation and utility scripts
├── docs/                 # Documentation
├── config/               # Configuration files
└── tests/                # Test files
```

## 🎯 Usage

After installation, you can start M.I.A with:

### Interactive Mode Selection
```bash
python main.py
```

### Direct Mode Selection
```bash
python main.py --mode text    # Text-only mode
python main.py --mode audio   # Audio-only mode  
python main.py --mode mixed   # Mixed mode
```

### Help
```bash
python main.py --help
```

## 📝 Next Steps

1. **Configure LLM**: Set up Ollama and pull your preferred model
2. **Test Audio**: Run a quick audio test if you plan to use voice features
3. **Explore Features**: Check the documentation for advanced features
4. **Customize**: Modify configuration files to suit your needs

Enjoy using M.I.A! 🚀
