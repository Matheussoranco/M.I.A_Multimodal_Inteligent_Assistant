# Friday/M.I.A - Your Personal Virtual Assistant

This is Friday/M.I.A, my personal virtual assistant that runs on my computer, phone, and smartwatch. After many requests from colleagues, I’m posting the modules. M.I.A works with an LLM (Large Language Model) to chat and interact with you, but she also has modules that automate many tasks, making her a versatile and powerful assistant.
# Key Features
Seamless Interaction with LLMs.

Powered by advanced Large Language Models like LLama 70B or APIs like OpenAI and Ollama. Provides accurate, natural, and engaging responses to queries.
Speech Recognition and Generation.

Utilizes state-of-the-art ASR (Automatic Speech Recognition) models, such as OpenAI Whisper. Generates human-like speech using a female voice, powered by SpeechT5.
Automated Task Modules.

    Email Writing and Sending: Compose and send emails effortlessly.
    Messaging Across Platforms: Send messages on WhatsApp, Slack, Telegram, and more.
    Note-Taking: Create and organize notes in apps like Notion and Evernote.
    Program and Website Launcher: Open files, programs, and websites with ease.
    Autofill Login: Automatically fills in username and password fields on websites.

Cross-Platform Compatibility

Works seamlessly across devices, ensuring a consistent user experience on PC, phone, and smartwatch via bluetooth.
# How It Works
Speech Input.

The assistant listens to your voice using the ASR module.
LLM Query.

Your request is processed and interpreted by a Large Language Model.
Action Execution.

Based on the request, M.I.A executes the relevant module—be it responding via speech, sending a message, or automating tasks.
Speech Output.

M.I.A responds with synthesized speech, creating a natural and interactive experience.
# Installation

To get started with Friday/M.I.A, follow these steps:
Clone the Repository

    git clone https://github.com/yourusername/friday-mia.git
    cd friday-mia

Install Dependencies

Ensure Python 3.9+ is installed, then run:

    pip install -r requirements.txt

Configure API Keys and Models

Update main.py and relevant modules with your API keys and model paths. Configure any device-specific settings as needed.
Run the Assistant

    python main.py

# File Organization

- `main.py` — Main entry point for the assistant
- `chat_interface.py` — Command-line chat interface (with optional voice input)
- `llm/llm_manager.py` — Unified interface for OpenAI, HuggingFace, Grok, and local LLMs
- `langchain/langchain_verifier.py` — LangChain-based output verification/workflows
- `system/system_control.py` — System actions (file, process, clipboard, etc.)
- `audio_utils.py`, `speech_processor.py`, `speech_generator.py` — Audio and speech modules
- `automation_util.py` — Automation utilities (open programs, websites, etc.)
- `note_taking_utils.py`, `message_util.py` — Note and message utilities
- `multimodal/processor.py` — Multimodal (audio/image) processing
- `memory/knowledge_graph.py` — Agent memory/context

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

# To Do

- [ ] Add a GUI interface (Tkinter, PyQt, or web-based)
- [ ] Add speech output (TTS) to chat interface
- [ ] Expand LangChain workflows and tools
- [ ] Add more system automation actions (scheduling, notifications, etc.)
- [ ] Improve error handling and logging
- [ ] Add more LLM providers and plug-ins
- [ ] Enhance multimodal (image/video) capabilities
- [ ] Add user authentication and profiles

Scientific Articles and Sources

This project draws inspiration and implementation details from the following research papers and documentation:
Speech Recognition and ASR Models

    Radford, A. et al. (2023). Robust Speech Recognition via OpenAI Whisper.
    Keras Documentation: Transformer ASR

LLM Implementation and Fine-Tuning

    Touvron, H. et al. (2023). LLaMA: Open and Efficient Foundation Language Models.
    Hugging Face Documentation: Transformers Library

Speech Synthesis

    Shankar, V. et al. (2022). SpeechT5: Unified-Text-to-Speech.

Task Automation with Python

    Selenium WebDriver Documentation: Automating Browser Tasks
    Webbrowser Module: Python Docs

Contributing

Feel free to fork this repository and make pull requests. Suggestions and improvements are always welcome!
# License

This project is licensed under the AGPL-3.0. See the LICENSE file for more details.
