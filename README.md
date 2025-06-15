# Friday/M.I.A - Your Personal Virtual Assistant

This is Friday/M.I.A, your all-in-one personal virtual assistant for PC, phone, and smartwatch. M.I.A leverages advanced LLMs and modular automation to help you with just about anything on your computer and beyond.

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

To get started with Friday/M.I.A:

    git clone https://github.com/yourusername/friday-mia.git
    cd friday-mia
    pip install -r requirements.txt

Configure API keys and device settings in `main.py` and relevant modules.

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
