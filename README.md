# M.I.A - The Successor of pseudoJarvis

<div align="center">

![M.I.A Logo](https://img.shields.io/badge/M.I.A-Multimodal_Intelligent_Assistant-blue?style=for-the-badge&logo=robot)

**A powerful, multimodal intelligent assistant that combines text, audio, and vision capabilities with advanced AI reasoning.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Ollama](https://img.shields.io/badge/Powered_by-Ollama-orange?style=flat-square)](https://ollama.ai)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow?style=flat-square)](https://huggingface.co/transformers)

</div>

## 🚀 Quick Start

### Windows
```cmd
# Install M.I.A
scripts\install\install.bat

# Run M.I.A
start-mia.bat
```

### Linux/macOS
```bash
# Install M.I.A
./scripts/install/install.sh

# Run M.I.A
./start-mia.sh
```

## 🎯 What is M.I.A?

**M.I.A (Multimodal Intelligent Assistant)** is an advanced AI assistant that seamlessly integrates multiple interaction modalities:

- **🔤 Text Mode** - Natural language conversations
- **🎤 Audio Mode** - Voice interactions with speech-to-text and text-to-speech
- **🖼️ Vision Mode** - Image understanding and analysis
- **🔄 Mixed Mode** - Combine text, audio, and vision in one conversation

Built on cutting-edge technologies including local LLMs (via Ollama), transformers, and advanced cognitive architectures.

## ✨ Features

### Core Capabilities
- **🧠 Advanced Reasoning** - Powered by DeepSeek-R1 and other state-of-the-art models
- **🎙️ Speech Processing** - Real-time speech-to-text and text-to-speech
- **👁️ Computer Vision** - Image analysis using CLIP and other vision models
- **🧠 Memory System** - Persistent memory with ChromaDB vector database
- **🔌 Plugin System** - Extensible architecture for custom tools and skills
- **🛡️ Security** - Built-in security management and validation

### User Experience
- **🎨 Clean Interface** - Professional command-line interface with colored output
- **⚡ Fast Setup** - One-click installation scripts for all platforms
- **🔧 Flexible Configuration** - Extensive command-line options and configuration files
- **📱 Cross-Platform** - Works on Windows, Linux, and macOS
- **🎯 Mode Selection** - Interactive mode selection or direct CLI arguments

### Technical Features
- **🤖 Local LLM Integration** - Runs entirely on your machine with Ollama
- **🔄 Multimodal Processing** - Seamlessly handle text, audio, and images
- **📊 Real-time Analytics** - Performance monitoring and usage statistics
- **🔌 Extensible Architecture** - Plugin system for custom functionality
- **💾 Persistent Memory** - Long-term conversation memory and learning

## 📋 Requirements

### System Requirements
- **Python 3.8+** (Python 3.10+ recommended)
- **Operating System**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.14+
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space for models and dependencies

### Dependencies
- **Ollama** - Local LLM runtime
- **FFmpeg** - Audio processing (optional)
- **CUDA** - GPU acceleration (optional)

## 🛠️ Installation

### Automatic Installation

The easiest way to install M.I.A is using our installation scripts:

#### Windows
```cmd
# Download or clone the repository
git clone https://github.com/yourusername/M.I.A-The-successor-of-pseudoJarvis.git
cd M.I.A-The-successor-of-pseudoJarvis

# Run installation script
scripts\install\install.bat
```

#### Linux/macOS
```bash
# Download or clone the repository
git clone https://github.com/yourusername/M.I.A-The-successor-of-pseudoJarvis.git
cd M.I.A-The-successor-of-pseudoJarvis

# Make installation script executable and run
chmod +x scripts/install/install.sh
./scripts/install/install.sh
```

### Manual Installation

If you prefer manual installation or need customization:

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/M.I.A-The-successor-of-pseudoJarvis.git
   cd M.I.A-The-successor-of-pseudoJarvis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama**
   - Download from [ollama.ai](https://ollama.ai)
   - Pull a model: `ollama pull deepseek-r1:1.5b`

5. **Run M.I.A**
   ```bash
   python main.py
   ```

## 🎮 Usage

### Basic Usage

```bash
# Interactive mode selection
python main.py

# Direct mode selection
python main.py --mode text    # Text-only mode
python main.py --mode audio   # Audio-only mode
python main.py --mode mixed   # Mixed mode (text + audio)
python main.py --mode auto    # Auto-detect best mode

# Custom model
python main.py --model-id gemma2:9b

# Debug mode
python main.py --mode text --debug
```

### Startup Scripts

For convenience, use the provided startup scripts:

```bash
# Windows
start-mia.bat

# Linux/macOS  
./start-mia.sh
```

### Command-Line Options

```bash
python main.py --help
```

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Interaction mode (text/audio/mixed/auto) | Interactive selection |
| `--model-id` | LLM model to use | `deepseek-r1:1.5b` |
| `--url` | Ollama API URL | `http://localhost:11434/api/generate` |
| `--debug` | Enable debug logging | `False` |
| `--image-input` | Path to image for analysis | `None` |
| `--enable-reasoning` | Enable advanced reasoning | `False` |

### Interactive Commands

Once M.I.A is running, you can use these commands:

- `help` - Show available commands
- `status` - Display system status
- `mode` - Switch interaction mode
- `clear` - Clear conversation history
- `save` - Save conversation
- `quit` - Exit M.I.A

## 🧠 Supported Models

M.I.A works with various LLM models via Ollama:

### Recommended Models
- **DeepSeek-R1 1.5B** - Fast, efficient reasoning (default)
- **Gemma2 9B** - Balanced performance and quality
- **Llama3 8B** - High-quality conversations
- **Phi3 3.8B** - Compact and efficient

### Installation
```bash
# Install models via Ollama
ollama pull deepseek-r1:1.5b
ollama pull gemma2:9b
ollama pull llama3:8b
ollama pull phi3:3.8b
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:1.5b
OLLAMA_API_KEY=ollama

# Audio Configuration
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1

# Vision Configuration
VISION_MODEL=openai/clip-vit-base-patch32

# Debug Configuration
DEBUG_MODE=false
LOG_LEVEL=INFO
```

### Configuration Files

Advanced configuration can be done via `config/` directory:

- `config/llm_config.yaml` - LLM settings
- `config/audio_config.yaml` - Audio processing settings
- `config/vision_config.yaml` - Vision model settings

## 🏗️ Architecture

M.I.A follows a modular architecture:

```
src/mia/
├── audio/           # Audio processing modules
├── core/            # Core cognitive architecture
├── llm/             # LLM integration and management
├── memory/          # Memory and knowledge graph
├── multimodal/      # Multimodal processing
├── plugins/         # Plugin system
├── security/        # Security and validation
├── tools/           # Action execution tools
└── utils/           # Utility functions
```

### Key Components

- **Cognitive Architecture** - Central reasoning engine
- **LLM Manager** - Handles communication with language models
- **Memory System** - Persistent conversation memory
- **Multimodal Processor** - Integrates text, audio, and vision
- **Plugin Manager** - Extensible functionality system
- **Security Manager** - Input validation and security

## 🔌 Plugins

M.I.A supports a plugin system for extending functionality:

### Available Plugins
- **Web Search** - Search the internet for information
- **File Operations** - Read, write, and manipulate files
- **System Control** - System commands and automation
- **Calendar Integration** - Schedule and calendar management

### Creating Plugins

Create a new plugin by adding a Python file to `src/mia/plugins/`:

```python
# src/mia/plugins/my_plugin.py
class MyPlugin:
    def __init__(self):
        self.name = "My Custom Plugin"
        
    def execute(self, command, args):
        # Plugin logic here
        return f"Executed: {command}"
```

## 📊 Performance

### System Performance
- **Startup Time**: ~5-10 seconds
- **Response Time**: 1-3 seconds (depends on model)
- **Memory Usage**: 2-6GB (depends on model)
- **CPU Usage**: Low to moderate

### Optimization Tips
- Use smaller models for faster responses
- Enable GPU acceleration for better performance
- Disable unnecessary features in text-only mode
- Use SSD storage for better model loading

## 🔍 Troubleshooting

### Common Issues

#### M.I.A won't start
```bash
# Check Python version
python --version

# Check dependencies
pip list

# Check Ollama status
ollama list
```

#### Audio not working
```bash
# Install audio dependencies
pip install sounddevice soundfile

# On Linux, install system packages
sudo apt-get install portaudio19-dev
```

#### Model loading errors
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Pull the model
ollama pull deepseek-r1:1.5b
```

### Debug Mode

Enable debug mode for detailed logging:
```bash
python main.py --debug
```

### Getting Help

1. Check the [documentation](docs/)
2. Review the [installation guide](INSTALLATION.md)
3. Search existing issues on GitHub
4. Create a new issue with detailed information

## 🤝 Contributing

We welcome contributions to M.I.A! Here's how to get started:

### Development Setup

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/yourusername/M.I.A-The-successor-of-pseudoJarvis.git
   cd M.I.A-The-successor-of-pseudoJarvis
   ```

3. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements-dev.txt
   ```

4. **Run tests**
   ```bash
   pytest tests/
   ```

### Contribution Guidelines

- Follow PEP 8 style guide
- Write comprehensive tests
- Update documentation
- Create detailed pull requests

### Development Commands

```bash
# Run tests
pytest tests/

# Code formatting
black src/

# Type checking
mypy src/

# Linting
flake8 src/
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama** - For providing excellent local LLM runtime
- **HuggingFace** - For transformers and model ecosystem
- **OpenAI** - For CLIP and other foundational models
- **ChromaDB** - For vector database capabilities
- **Community** - For feedback, contributions, and support

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/M.I.A-The-successor-of-pseudoJarvis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/M.I.A-The-successor-of-pseudoJarvis/discussions)

## 🔮 Roadmap

### Near Term (v2.1)
- [ ] Enhanced plugin system
- [ ] Web interface
- [ ] Better error handling
- [ ] Performance optimizations

### Medium Term (v2.5)
- [ ] Advanced reasoning capabilities
- [ ] Multi-language support
- [ ] Cloud deployment options
- [ ] Mobile app companion

### Long Term (v3.0)
- [ ] Advanced multimodal understanding
- [ ] Autonomous task execution
- [ ] Federated learning capabilities
- [ ] AR/VR integration

---
