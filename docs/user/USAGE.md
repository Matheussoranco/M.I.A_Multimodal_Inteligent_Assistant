# M.I.A Usage Guide

## Quick Start

### Windows
```bash
# Using batch files (recommended)
run.bat

# Manual execution
venv\Scripts\activate
python -m main_modules.main
```

### Linux/macOS
```bash
source venv/bin/activate
python -m main_modules.main
```

## Command Line Options

```bash
python -m main_modules.main [OPTIONS]

Options:
  --url TEXT              LLM API URL (default: http://localhost:11434/v1)
  --model-id TEXT         Model ID to use (default: mistral:instruct)
  --api-key TEXT          API key for LLM service (default: ollama)
  --stt-model TEXT        Speech-to-text model (default: openai/whisper-base.en)
  --image-input PATH      Path to image file for processing
  --enable-reasoning      Enable advanced reasoning mode
  --debug                 Enable debug logging
  --help                  Show this message and exit
```

## Examples

### Basic Usage
```bash
# Start with default settings
python -m main_modules.main

# Use OpenAI GPT-4
python -m main_modules.main --model-id gpt-4 --api-key your-openai-key

# Use local Ollama model
python -m main_modules.main --url http://localhost:11434/v1 --model-id llama2

# Enable advanced reasoning
python -m main_modules.main --enable-reasoning

# Process an image
python -m main_modules.main --image-input photo.jpg
```

### Debug Mode
```bash
# Enable detailed logging
python -m main_modules.main --debug
```

## Configuration

### Environment Variables

Set these in your `.env` file:

```bash
# API Keys
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
GEMINI_API_KEY=your-key-here

# Debugging
DEBUG=true
LOG_LEVEL=DEBUG

# Audio settings
AUDIO_DEVICE=default
SPEECH_RATE=1.0
```

### Supported LLM Providers

1. **OpenAI** (GPT-3.5, GPT-4)
2. **Anthropic** (Claude)
3. **Google** (Gemini)
4. **Ollama** (Local models)
5. **Groq** (Fast inference)
6. **HuggingFace** (Open source models)

## Troubleshooting

### Common Issues

1. **Import Errors**: Install missing dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. **Audio Issues**: Install PyAudio
   ```bash
   pip install PyAudio
   ```

3. **GPU Issues**: Install CUDA-enabled PyTorch
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

4. **API Errors**: Check your API keys in `.env`

### Getting Help

- Check the logs for detailed error messages
- Use `--debug` flag for verbose output
- Ensure all dependencies are installed
- Verify API keys are correctly configured
