# M.I.A - Multimodal Intelligent Assistant

**Enterprise-grade multimodal AI assistant with advanced cognitive architectures.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-4.35%2B-yellow?style=flat-square&logo=huggingface)](https://huggingface.co/transformers)
[![Ollama](https://img.shields.io/badge/Ollama-Runtime-orange?style=flat-square)](https://ollama.ai)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-green?style=flat-square)](https://chromadb.com)

## Overview

M.I.A is a sophisticated AI system implementing advanced cognitive architectures for multimodal understanding and generation. It provides capabilities for natural language processing, computer vision, and audio processing through a unified interface.

### Core Features
- **Multimodal Processing**: Text, audio, and vision input processing
- **LLM Integration**: Support for OpenAI, Ollama (optional), and other providers
- **Cognitive Architecture**: Chain-of-thought reasoning with visual grounding
- **Memory Systems**: Vector memory, knowledge graphs, and conversation history
- **Audio Processing**: Real-time speech recognition and synthesis
- **Vision Processing**: Image analysis and object detection

## Installation

### Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/M.I.A-The-successor-of-pseudoJarvis.git
cd M.I.A-The-successor-of-pseudoJarvis

# Install dependencies
pip install -r requirements.txt

# Optional: Install Ollama for local LLM support
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull deepseek-r1:1.5b

# Run M.I.A
python main.py --mode mixed
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- OpenAI API key (or Ollama for local models)
- ChromaDB

## Usage

### Basic Commands
```bash
# Text mode
python main.py --mode text

# Audio mode
python main.py --mode audio

# Mixed mode (default)
python main.py --mode mixed

# Custom model
python main.py --model-id gemma2:9b
```

### Interactive Commands
- `help` - Show available commands
- `status` - Display system status
- `clear` - Clear conversation history
- `quit` - Exit M.I.A

### Agent Commands
- `create file [name]` - Create a new file
- `analyze code [file]` - Analyze code file

## Configuration

Environment variables in `.env`:
```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:1.5b
DEBUG_MODE=false
```

## Architecture

```
Text Module → Cognitive Engine → LLM Layer → Vector Memory
Audio Module →              →            →
Vision Module →             →            →
```

## Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=src tests/
```

## License

GNU AFFERO GENERAL PUBLIC LICENSE