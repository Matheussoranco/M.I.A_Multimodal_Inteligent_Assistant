# M.I.A — Intelligent Multimodal Personal Assistant (on‑device)

M.I.A is a multimodal, local‑first, extensible, and auditable personal assistant designed to run 100% on your device (with the option to use external APIs when desired). It combines natural language understanding, speech, computer vision, long‑term memory, and a toolchain to perform real tasks: send messages and emails, create documents (spreadsheets, presentations, text, and PDFs), automate the OS, search and browse the web (with Selenium), analyze data, program in multiple languages, send local notifications, open apps, and interact with IoT devices.

Design principles:

- Local by default: LLMs, vectors, memory, logs, and inference run locally; switching to API providers is optional.
- Security and control: action execution with confirmation, isolation of sensitive tools, and audit trails.
- Extensible: layered modular architecture with pluggable “tools” and swappable providers.
- Observable: latency/usage metrics, logs, and an optional admin console.


## Core capabilities

- Natural conversation via text and voice (in/out), with hotword activation and VAD.
- Multimodal vision: images and documents (OCR, description, visual Q&A).
- Short‑ and long‑term contextual memory (episodic, semantic, and local knowledge graph).
- Planning, reasoning, and multi‑step task execution (advanced cognition with tool calls).
- Document generation:
	- Word/Docx (python‑docx),
	- PowerPoint (python‑pptx),
	- Spreadsheets (openpyxl/pandas),
	- PDFs (ReportLab or optional headless LibreOffice conversion).
- Messaging: email sending (SMTP/Graph API) and WhatsApp Web automation (Selenium) on desktop.
- System automation: open apps, interact with windows, keyboard shortcuts, and local notifications.
- Web search and browsing with Selenium (search, pagination, scroll, clicks, result harvesting).
- Programming and code execution (Python and other languages) in an isolated environment.
- IoT control via MQTT (Home Assistant/Zigbee2MQTT) and local REST integrations.


## Architecture

M.I.A uses a layered architecture with LLM orchestration, memory, and tools. The main components already exist under `src/mia` and can be extended.

```
┌───────────────────────────────────────────────────────────────────────┐
│                           Interaction Layers                          │
│  • CLI / TUI • API (FastAPI) • Voice (VAD/Hotword/TTS/STT) • Messages │
└───────────────▲───────────────────────────────────────────────▲───────┘
								│                                               │
					 I/O streams                                     Webhooks/Apps
								│                                               │
┌───────────────┴───────────────────────────────────────────────┴───────┐
│                           LLM Orchestrator                             │
│  • Planning & Tools • Reasoning Chains • Guardrails                    │
│  • Model selection (local/API) • Prompting • Streaming                 │
└───────────────▲───────────────────────────────▲────────────────────────┘
								│                               │
				 Multimodal perception            Memory & Context
								│                               │
┌───────────────┴──────────────┐      ┌─────────┴─────────────────────┐
│  Audio (STT, TTS, VAD)       │      │  Short‑term (buffer)          │
│  Image/Docs (OCR/VQA)        │      │  Long‑term (Chroma/SQLite)    │
│  Multimodal Fusion           │      │  Knowledge Graph               │
└───────────────▲──────────────┘      └─────────▲─────────────────────┘
								│                               │
								└─────────┬───────────┬─────────┘
													│           │
									 ┌──────┴──────┐ ┌──┴──────────────────────┐
									 │  System/OS  │ │  Tools (Plugins)        │
									 │  Automation │ │  • Email • WhatsApp     │
									 │  Notifications││  • Office • Selenium    │
									 │  Apps/Process│ │  • IoT (MQTT) • Web     │
									 └──────────────┘ └─────────────────────────┘
```

Mapping to existing modules (high‑level):

- LLM orchestration: `src/mia/llm/llm_manager.py`, `src/mia/llm/llm_inference.py`, `src/mia/adaptive_intelligence/hybrid_llm_orchestration.py`
- Cognition and planning: `src/mia/core/cognitive_architecture.py`, `src/mia/adaptive_intelligence/workflow_automation_composer.py`
- Memory (local): `src/mia/memory/long_term_memory.py`, `src/mia/memory/knowledge_graph.py`, vector DB `memory/chroma.sqlite3`
- Multimodal perception: audio in `src/mia/audio/*` (hotword, VAD, TTS, STT); vision in `src/mia/multimodal/vision_processor.py`
- API/Service: `src/mia/api/server.py`
- Observability: `src/mia/performance_monitor.py`, `src/mia/adaptive_intelligence/observability_admin_console.py`
- Messaging: `src/mia/messaging/telegram_client.py` (example already implemented)
- Tools/Plugins: `src/mia/tools/`, `src/mia/plugins/` (extension points)
- Security: `src/mia/security/` (policies and utilities)


### Model providers (local and API)

- Local (preferred):
	- Text LLM: Ollama (llama, phi, mistral), llama.cpp (GGUF), LM Studio.
	- Multimodal vision: LLaVA, BakLLaVA, MiniCPM‑V (via Ollama/gguf), or local bridges.
	- STT: Whisper.cpp, Vosk.
	- TTS: Piper, Coqui TTS.
- API (optional): OpenAI, Azure OpenAI, Anthropic, Google, etc.

Backend and model selection are controlled by configuration (see `config/config.yaml`).


### Action execution and security

- Permissions: each tool can require confirmation (ask‑before‑act mode).
- Isolation: code execution in an isolated subprocess; Docker/WSL2 optional for stronger isolation.
- Secrets: `.env` for keys (never version control); accessed via the config manager.
- Auditing: structured logs for tool calls, errors, and action timelines.


## Installation (Windows)

Prerequisites:

- Windows 10/11 64‑bit
- Python 3.10+ and Git
- FFmpeg (audio), Tesseract OCR (vision), Google Chrome/Edge and matching WebDriver
- Optional: CUDA/cuDNN (GPU), Docker Desktop (sandbox), Ollama (local LLM)

Quick steps:

1) Clone the repository and create a virtual environment

2) Install core or development dependencies

3) Copy `config/.env.example` to `.env` and set credentials (email, optional APIs, etc.)

4) Review `config/config.yaml` and select local backends (LLM/STT/TTS/Vision)

Note: for WhatsApp Web automation, ensure Chrome/Edge is installed and the WebDriver version matches the browser.


## Configuration

`.env` file (typical examples):

- SMTP email: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`
- Optional providers: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.
- Local services toggles: ports, paths, and feature switches for tools

`config/config.yaml` (example keys):

- `llm.provider`: `ollama` | `llamacpp` | `openai` | `azure` ...
- `llm.model`: local model name (e.g., `llama3:8b`, `mistral:7b`)
- `audio.stt`: `whisper.cpp` | `vosk`; `audio.tts`: `piper` | `coqui`
- `vision.provider`: local `llava` (or OCR + captioning);
- `memory.vector_store`: `chroma`; database paths and context limits
- `tools.enabled`: list of enabled tools (email, whatsapp, office, selenium, iot, os, notifications)
- `security.confirm_before_action`: true/false


## How to run

- CLI (interactive): `python -m mia.main` or `python .` (root `main.py` triggers initialization)
- API (FastAPI): module `src/mia/api/server.py` (serve with Uvicorn/Gunicorn)
- Voice mode: enable hotword/VAD in `config.yaml` and connect mic/speakers

Outputs and logs are stored locally under `cache/` and per observability settings.


## Example flows (natural commands)

- “Send a WhatsApp message to Maria saying ‘arriving at 7pm’. If the chat doesn’t exist, open and search for the contact.”
- “Write an email to team@company.com with subject ‘Meeting’ and a summary of the topics, in a professional tone.”
- “Create an xlsx spreadsheet with three tabs: goals, budget, and expenses; use the attached data and generate charts.”
- “Generate a .docx document with a project brief and export a final PDF.”
- “Read this PDF, extract tables, and give me key insights with statistics.”
- “Search for the latest AI news, open the first 5 pages, and produce a summary with links.”
- “Open the calendar app, create an event tomorrow at 9 AM, and remind me 30 minutes earlier.”
- “Take a photo of the whiteboard, recognize the content, and attach it to the report.”
- “Write a Python script to rename files by pattern and run it in the sandbox.”
- “Turn on the living room light to 50% via MQTT.”


## Automation and tools

- Email: traditional SMTP (with app passwords) or Microsoft Graph API (OAuth2). Respect provider policies.
- WhatsApp: automate WhatsApp Web via Selenium (Chrome/Edge) using the user’s session; respect Terms of Use.
- Office/Docs: create with `python-docx`, `python-pptx`, `openpyxl`; PDFs with `reportlab` or headless conversion.
- Web/Selenium: search, pagination, clicks, scrolling, and data extraction (BeautifulSoup optional for post‑processing).
- OS/Apps: launch processes, focus windows, Windows Toast notifications, shortcuts, and clipboard.
- IoT: MQTT (paho‑mqtt), Home Assistant (REST/WebSocket APIs), Zigbee2MQTT.


## Local model execution

- LLM (text): Ollama with quantized models (4‑8 bit) for a good latency/quality balance on CPU/GPU.
- Vision: multimodal models via Ollama/gguf or a local OCR + captioning + VQA pipeline.
- STT/TTS: Whisper.cpp and Piper are lightweight, offline, and high‑quality.
- Caching: reuse embeddings/outputs to reduce latency; tune context length according to RAM.


## Security and privacy

- Local‑first: data, documents, history, and memory remain on your machine.
- Confirmation: sensitive actions (send messages/emails, modify files, IoT) require confirmation.
- Isolation: code execution and automations can run inside Docker/WSL2 to reduce host impact.
- Secrets: `.env` for minimal variables and periodic rotation; never commit.


## Testing and quality

- Tests: `tests/` folder with unit and integration tests; run with `pytest` (prepared environment).
- Lint: `.flake8` configured; recommended to integrate in editor/CI.
- Benchmarks: `src/mia/benchmark_runner.py` and `benchmark_config.py` for performance scenarios.


## Project structure (partial)

```
M.I.A_Multimodal_Inteligent_Assistant/
├─ main.py                 # local execution bootstrap
├─ config/
│  ├─ .env.example         # environment variables
│  └─ config.yaml          # main configuration
├─ src/mia/
│  ├─ main.py              # core entrypoint
│  ├─ llm/                 # model management and inference
│  ├─ audio/               # STT/TTS/VAD/Hotword
│  ├─ multimodal/          # vision and multimodal fusion
│  ├─ memory/              # long‑term memory and knowledge graph
│  ├─ adaptive_intelligence/ # orchestration, planning, feedback
│  ├─ api/                 # FastAPI server
│  ├─ tools/               # pluggable tools (OS, web, docs, etc.)
│  ├─ security/            # security policies/guardrails
│  └─ system/              # OS integration and local resources
└─ tests/                  # unit and integration tests
```


## Extensibility

- Plugins/Tools: add modules under `src/mia/tools` with a tool contract (name, input/output schema, permissions).
- Providers: implement adapters under `src/mia/llm` for new backends (e.g., another local model server).
- Channels: new inputs/outputs (e.g., inbound email, Slack, SMS) can be added as “connectors”.


## License

See the `LICENSE` file at the project root.


## Intentions not yet implemented (not a roadmap)

- Execution in an isolated microVM on Windows (via Windows Sandbox/WSL2/Docker with automated restrictive policies).
- Direct integration with official WhatsApp Business APIs (on‑prem model) while keeping local privacy.
- Navigation agent with fine‑tuning for long goals (live replanning and web‑specific memory).
- Embedded local IDE with structured code editing, refactors, and agent‑guided tests.
- Low‑latency streaming speech recognition with full on‑device multi‑speaker diarization.
- Deep integration with local calendar/contacts (private indexing and personal insights), without cloud.
- Native support for more languages/toolchains (Rust, Go, C/C++/CUDA) with isolated execution and build caching.
- Full graphical observability dashboard with distributed tracing and session replay.
- Local continual learning with human feedback (RLHF) preserving privacy.

## Licence

AGPL-3.0-or-later — consulte `LICENSE`.


## Autor

Matheus Pullig Soranço de Carvalho — matheussoranco@gmail.com
