# M.I.A Usage Guide

## Starting M.I.A

### Interactive Mode Selection (Recommended)
```batch
.\run.bat
```
- Shows a menu to choose your preferred mode
- Options: Text-only, Audio-only, Mixed, or Auto-detect
- Detects available hardware automatically

### Quick Start Options
```batch
.\run-text-only.bat     # Text-only mode
.\run-audio.bat         # Audio-only mode  
.\run-mixed.bat         # Mixed mode (text + audio)
```

### Custom Options
```batch
.\run.bat --model-id gemma3:4b-it-qat --debug
```

## Mode Selection Menu

When you start M.I.A, you'll see:
```
Choose your interaction mode:
1. Text Mode (Recommended) - Type your messages
2. Audio Mode - Speak your messages  
3. Mixed Mode - Switch between text and audio
4. Auto-detect - Let M.I.A choose based on available hardware
```

## Available Commands

### Universal Commands
- `quit` - Exit the application
- `help` - Show available commands
- `status` - Show system status
- `models` - List available models
- `clear` - Clear conversation context

### Mode Switching (Mixed Mode)
- `audio` - Switch to audio input
- `text` - Switch to text input

### In Audio Mode
- Speak your message after "Listening..." prompt
- Press Ctrl+C to switch to text mode
- Say "text mode" to switch to text input

## Command Line Options

- `--skip-mode-selection`: Skip interactive mode selection
- `--text-only`: Force text-only mode
- `--audio-mode`: Force audio mode
- `--model-id MODEL`: Use specific Ollama model (default: deepseek-r1:1.5b)
- `--url URL`: Custom Ollama API URL
- `--debug`: Enable debug logging
- `--enable-reasoning`: Enable advanced reasoning features

## Available Models

Check your available models:
```batch
ollama list
```

Current available models:
- `deepseek-r1:1.5b` (default)
- `gemma3:4b-it-qat`

## Mode Descriptions

### 🔤 Text Mode
- Type your messages in the console
- No audio dependencies required
- Fastest and most reliable
- Best for development and debugging

### 🎤 Audio Mode
- Speak your messages aloud
- Requires microphone and audio libraries
- Responses can be spoken back
- Natural conversation experience

### 🔀 Mixed Mode
- Start with text input
- Switch to audio anytime with `audio` command
- Switch back to text with `text` command
- Best of both worlds

### 🔍 Auto-detect Mode
- Automatically detects available hardware
- Falls back to text-only if audio unavailable
- Recommended for first-time users

## Troubleshooting

### LLM Not Working
1. Make sure Ollama is running: `ollama serve`
2. Check available models: `ollama list`
3. Test connection: `python test_ollama.py`

### Audio Issues
1. Use text-only mode: `.\run-text-only.bat`
2. Install audio dependencies: `pip install PyAudio sounddevice`
3. Check microphone permissions
4. Try auto-detect mode to verify hardware

### Dependencies
- Core: `transformers`, `openai`, `chromadb`
- Audio: `PyAudio`, `sounddevice`, `pydub`
- Optional: `torch` for better performance

## Examples

### Basic Usage
```
🤖 M.I.A: Hello! How can I help you today?
💬 You: What's the weather like?
🤖 M.I.A: I don't have access to current weather data...
```

### Mode Switching
```
💬 You: audio
🎤 Switched to audio input mode. Say something...
🎤 Listening... (speak now or press Ctrl+C to switch to text)
🎙️  You said: Hello there
🤖 M.I.A: Hello! How can I assist you?
```

### Using Help
```
💬 You: help
📚 M.I.A Commands:
  quit         - Exit M.I.A
  help         - Show this help message
  audio        - Switch to audio input mode
  status       - Show current system status
  models       - List available models
  clear        - Clear conversation context
```
