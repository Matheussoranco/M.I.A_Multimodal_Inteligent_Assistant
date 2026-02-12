<p align="center">
  <h1 align="center">M.I.A — Multimodal Intelligent Assistant</h1>
  <p align="center">
    A local-first, tool-calling AI agent with hybrid reasoning, multi-agent orchestration, multimodal perception, and full desktop/web automation.
  </p>
  <p align="center">
    <a href="#architecture">Architecture</a> · <a href="#quick-start">Quick Start</a> · <a href="#configuration">Configuration</a> · <a href="#api-reference">API</a> · <a href="#benchmarks">Benchmarks</a> · <a href="#license">License</a>
  </p>
</p>

---

## Overview

M.I.A is an **agentic AI system** designed to operate as a general-purpose digital assistant across desktop and mobile environments. It combines:

- **Hybrid Reasoning Engine** — algorithmic-first problem solving (CSP, SAT/DPLL, Gaussian elimination, graph algorithms, Nelder-Mead optimization) with automatic LLM fallback for tasks that resist formal methods
- **Multi-Agent Orchestration** — role-specialized sub-agents (Researcher, Coder, Analyst, Writer, Executor, Reviewer) with keyword-based task routing and quality review
- **Native Function Calling** — 40+ tools exposed as OpenAI-compatible JSON schemas, with ReAct text-parsing fallback for models without native tool support
- **Multimodal I/O** — text, voice (STT/TTS/VAD/hotword), vision (OCR, VQA, image description), and document processing
- **Desktop Automation** — UI element interaction via `pywinauto` (UIA backend), keyboard/mouse control, window management
- **Persistent Memory** — episodic conversation history, semantic knowledge graph, and skill library with reliability scoring
- **Security-First Execution** — action confirmation gates, scoped permissions, sandboxed code execution, circuit breaker error handling

**Design Principle:** Everything runs locally by default — LLMs, vector stores, embeddings, memory, and inference. External APIs (OpenAI, Anthropic, Google, etc.) are optional, pluggable backends.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INTERACTION LAYER                                │
│  CLI (argparse + REPL)  │  Web UI (Flask)  │  FastAPI Server  │  Voice     │
└────────────┬────────────┴────────┬─────────┴────────┬─────────┴──────┬─────┘
             │                     │                   │                │
             ▼                     ▼                   ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TOOL-CALLING AGENT (core/agent.py)                  │
│                                                                             │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────────────┐    │
│  │  TaskPlanner  │  │  AgentOrchestrator│  │  CognitiveKernel           │    │
│  │  DAG decomp.  │  │  6 specialist     │  │  WorkingMemory (LRU+TTL)  │    │
│  │  should_plan()│  │  sub-agents       │  │  SkillLibrary (MD5 cache) │    │
│  │  replan_step()│  │  classify+review  │  │  IntrospectionState       │    │
│  └──────────────┘  └──────────────────┘  └────────────────────────────┘    │
│                                                                             │
│  Pipeline: user_msg → memory_ctx → plan? → delegate? → tool_loop → guard   │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                     ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────────────┐
│  REASONING LAYER │ │  INTELLIGENCE    │ │  LLM PROVIDERS           │
│                  │ │                  │ │                          │
│  HybridEngine    │ │  ArcSolver       │ │  Ollama (local)          │
│  11 task domains │ │  7 strategies    │ │  OpenAI  / Azure OpenAI  │
│  AlgorithmicReas.│ │  ProgramSynth.   │ │  Anthropic               │
│  LogicEngine     │ │  PatternMatcher  │ │  Google (Gemini)         │
│  MetaCognition   │ │  HypothesisTester│ │  HuggingFace (local)     │
│  Solvers (Z3/Sym)│ │  GridDSL         │ │  llama.cpp (GGUF)        │
└──────────────────┘ └──────────────────┘ └──────────────────────────┘
          │                    │                     │
          └────────────────────┼─────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TOOL EXECUTION LAYER                              │
│                                                                             │
│  ActionExecutor (3000+ LOC)  ←─  ToolRegistry (40+ OpenAI-schema tools)    │
│                                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌────────────────┐  │
│  │ Web      │ │ File I/O │ │ Docs     │ │ Desktop   │ │ Communication  │  │
│  │ search   │ │ CRUD     │ │ docx     │ │ pywinauto │ │ email (SMTP)   │  │
│  │ scrape   │ │ dir ops  │ │ pptx     │ │ keyboard  │ │ WhatsApp (Sel.)│  │
│  │ research │ │ search   │ │ xlsx     │ │ mouse     │ │ Telegram       │  │
│  │ wikipedia│ │ archive  │ │ pdf      │ │ apps      │ │ notifications  │  │
│  └──────────┘ └──────────┘ └──────────┘ └───────────┘ └────────────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌────────────────┐  │
│  │ Code     │ │ Memory   │ │ System   │ │ IoT/Smart │ │ Sandbox        │  │
│  │ create   │ │ store    │ │ commands │ │ MQTT      │ │ WASI runner    │  │
│  │ analyze  │ │ search   │ │ process  │ │ HomeAsst. │ │ subprocess     │  │
│  │ execute  │ │ retrieve │ │ clipboard│ │ Zigbee    │ │ Docker (opt.)  │  │
│  └──────────┘ └──────────┘ └──────────┘ └───────────┘ └────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
          │                                              │
          ▼                                              ▼
┌──────────────────────────────┐   ┌──────────────────────────────────────────┐
│  MULTIMODAL PERCEPTION       │   │  MEMORY & PERSISTENCE                    │
│                              │   │                                          │
│  Audio:                      │   │  PersistentMemory (JSON file-backed)     │
│    STT (Whisper/Vosk)        │   │  KnowledgeGraph (NetworkX)               │
│    TTS (Piper/Coqui/pyttsx3) │   │  LongTermMemory (ChromaDB vectors)      │
│    VAD (WebRTC)              │   │  ConversationBuffer (in-memory ring)     │
│    Hotword detection         │   │  RAG Pipeline (retrieval + generation)   │
│                              │   │  EmbeddingManager (sentence-transformers)│
│  Vision:                     │   │                                          │
│    OCR (Tesseract/EasyOCR)   │   │  Storage:                               │
│    Image description (VLM)   │   │    memory/persistent/*.json              │
│    Document intelligence     │   │    memory/chroma.sqlite3                 │
│    Visual Q&A                │   │    config/config.yaml                    │
└──────────────────────────────┘   └──────────────────────────────────────────┘
```

### Module Map

```
src/mia/                          # 97 Python source files
├── core/                         # Central agent architecture
│   ├── agent.py                  #   ToolCallingAgent — main loop (native FC + ReAct)
│   ├── orchestrator.py           #   Multi-agent: 6 roles, classify+delegate+review
│   ├── planner.py                #   DAG task decomposition with dependency tracking
│   ├── persistent_memory.py      #   JSON file-backed skills, sessions, interactions
│   ├── guardrails.py             #   Output filtering and safety checks
│   ├── tool_registry.py          #   40+ OpenAI-schema tool definitions
│   ├── cognitive_architecture.py #   Legacy ReAct loop (retained for compatibility)
│   ├── benchmarks.py             #   Framework execution harness
│   └── skills.py                 #   Skill definition & execution
│
├── reasoning/                    # Algorithmic + hybrid reasoning
│   ├── algorithmic.py            #   CSP (AC-3+backtrack+MRV), DPLL SAT, Gaussian
│   │                             #   elimination, Dijkstra, Nelder-Mead, LogicEngine
│   ├── hybrid_engine.py          #   11-domain classifier, algorithmic-first routing
│   ├── cognitive_kernel.py       #   WorkingMemory, SkillLibrary, IntrospectionState
│   ├── metacognition.py          #   Self-monitoring, strategy adjustment, retry logic
│   └── solvers.py                #   SymPy/Z3 integration for formal verification
│
├── intelligence/                 # ARC-AGI and abstract reasoning
│   ├── arc_solver.py             #   7-strategy solver (analogy, gravity, symmetry,
│   │                             #   tiling, crop, object transform, synthesis)
│   ├── program_synthesis.py      #   DSL program search for grid transformations
│   ├── patterns.py               #   Symmetry, periodicity, object detection
│   ├── grid_dsl.py               #   Domain-specific language for grid operations
│   ├── hypothesis.py             #   Hypothesis generation and testing
│   └── search.py                 #   Program search strategies
│
├── llm/                          # LLM provider abstraction
│   ├── llm_manager.py            #   Multi-provider manager (Ollama, OpenAI, etc.)
│   ├── llm_inference.py          #   Inference pipeline with streaming
│   └── embedding_manager.py      #   Sentence-transformer embeddings
│
├── tools/                        # Tool execution engine
│   ├── action_executor.py        #   3000+ LOC — all tool implementations
│   ├── document_generator.py     #   docx/pptx/xlsx/pdf generation
│   └── document_intelligence.py  #   Document analysis and extraction
│
├── web/                          # Web interfaces
│   ├── webui.py                  #   Ollama-style chat UI (Flask)
│   └── web_agent.py              #   Selenium-based browsing, scraping, search
│
├── api/                          # REST API
│   └── server.py                 #   FastAPI: /chat, /chat/stream (SSE), /actions,
│                                 #   /memory, /health, /ready, /status
│
├── audio/                        # Speech I/O pipeline
│   ├── speech_processor.py       #   STT (Whisper, Vosk, SpeechRecognition)
│   ├── speech_generator.py       #   TTS (Piper, Coqui, pyttsx3)
│   ├── vad_detector.py           #   Voice Activity Detection (WebRTC VAD)
│   ├── hotword_detector.py       #   Wake-word activation
│   ├── audio_utils.py            #   Recording, playback, format conversion
│   └── audio_resource_manager.py #   Device management and cleanup
│
├── multimodal/                   # Vision and multimodal fusion
│   ├── vision_processor.py       #   Image analysis, VLM integration
│   ├── ocr_processor.py          #   Tesseract, EasyOCR, PaddleOCR
│   ├── processor.py              #   Multimodal fusion pipeline
│   └── vision_resource_manager.py#   Model loading and GPU management
│
├── memory/                       # Knowledge persistence
│   ├── long_term_memory.py       #   ChromaDB vector store
│   ├── knowledge_graph.py        #   NetworkX entity-relation graph
│   └── chroma.sqlite3            #   Vector DB storage
│
├── system/                       # OS integration
│   ├── desktop_automation.py     #   pywinauto UIA backend
│   └── system_control.py         #   Process, clipboard, command execution
│
├── security/                     # Access control
│   └── security_manager.py       #   Permission policies, action scoping
│
├── messaging/                    # Communication channels
│   └── telegram_client.py        #   Telethon-based Telegram integration
│
├── mcp/                          # Model Context Protocol
│   ├── client.py                 #   MCP client implementation
│   └── manager.py                #   MCP server management
│
├── sandbox/                      # Isolated execution
│   └── wasi_runner.py            #   WebAssembly (WASI) sandbox via wasmtime
│
├── benchmarks/                   # AGI benchmark frameworks
│   ├── arc_agi.py                #   ARC-AGI evaluation
│   ├── swe_bench.py              #   SWE-bench (software engineering)
│   ├── gaia.py                   #   GAIA multi-step reasoning
│   ├── gpqa.py                   #   GPQA graduate-level science
│   ├── mmmu.py                   #   MMMU multimodal understanding
│   ├── osworld.py                #   OSWorld desktop automation
│   ├── webvoyager.py             #   WebVoyager web navigation
│   ├── runner.py                 #   Benchmark orchestration
│   └── base.py                   #   Abstract benchmark interface
│
├── providers/                    # Plugin registry
│   ├── registry.py               #   ProviderRegistry with lazy imports
│   └── defaults.py               #   Built-in provider registrations
│
├── config/                       # Configuration
│   └── config_manager.py         #   YAML/env config, LLM profiles
│
├── cli/                          # Command-line interface
│   ├── parser.py                 #   Argument parsing
│   ├── display.py                #   Terminal output formatting
│   └── utils.py                  #   CLI utilities
│
├── error_handler.py              # Circuit breaker (closed/open/half-open),
│                                 # exponential backoff, fallback providers
├── performance_monitor.py        # Latency/throughput metrics
├── resource_manager.py           # GPU/CPU resource allocation
├── cache_manager.py              # LRU caching for LLM/embedding responses
├── config_manager.py             # Global configuration singleton
├── localization.py               # i18n string management
├── exceptions.py                 # Custom exception hierarchy
└── main.py                       # Entrypoint: initialize_components() → REPL
```

---

## Key Technical Details

### Agent Execution Pipeline

The `ToolCallingAgent` processes every user message through a 5-stage pipeline:

```
1. CONTEXT ENRICHMENT
   └─ Retrieve relevant memories (PersistentMemory + RAG)
   └─ Inject into system prompt as context

2. PLANNING (optional)
   └─ TaskPlanner.should_plan() — keyword heuristics detect multi-step goals
   └─ LLM decomposes into DAG of sub-tasks with depends_on edges
   └─ Execution tiers group independent tasks for parallel dispatch

3. MULTI-AGENT DELEGATION (optional)
   └─ AgentOrchestrator.should_delegate() — keyword scoring across 6 roles
   └─ Best-fit SubAgent executes with role-specific system prompt
   └─ Reviewer agent evaluates output quality (if enabled)

4. TOOL-CALLING LOOP
   └─ Native path: LLM returns tool_calls → execute → feed results back
   └─ ReAct path: parse Thought/Action/Action Input from text
   └─ Retry on failure (configurable max_retries with linear backoff)
   └─ Max iterations guard prevents infinite loops

5. GUARDRAILS
   └─ GuardrailsManager.check_output() filters response
   └─ Memory: log interaction to PersistentMemory
   └─ CognitiveKernel: update working memory + skill library
```

### Hybrid Reasoning Engine

The `HybridReasoningEngine` classifies incoming tasks into 11 domains via regex patterns and routes them to pure algorithmic solvers before touching the LLM:

| Domain | Algorithm | LLM Fallback |
|--------|-----------|--------------|
| `ARITHMETIC` | Direct Python eval (safe subset) | Complex word problems |
| `LOGIC` | Forward/backward chaining (LogicEngine) | Natural language logic |
| `CONSTRAINT` | AC-3 + backtracking + MRV heuristic | Underspecified CSPs |
| `SAT` | DPLL with unit propagation + pure literal | Large/complex formulae |
| `LINEAR_ALGEBRA` | Gaussian elimination with pivoting | Symbolic/abstract |
| `GRAPH` | Dijkstra, shortest path, BFS/DFS | Graph construction from NL |
| `OPTIMIZATION` | Nelder-Mead simplex | High-dimensional/non-numeric |
| `SEQUENCE` | Pattern detection + extrapolation | Novel sequence types |
| `ANALOGY` | Structural mapping (numeric + symbolic) | Abstract metaphors |
| `PROBABILITY` | Analytical computation | Bayesian networks |
| `GENERAL` | – | Always LLM |

The engine also supports `reason_ensemble()` — multi-strategy voting where algorithmic and LLM solutions are compared and the highest-confidence answer wins.

### Provider System

All major subsystems are loaded through a `ProviderRegistry` with lazy imports:

```python
provider_registry.register_lazy(
    category="llm",           # "llm", "audio", "vision", "memory", "web", ...
    name="default",           # Named variant within category
    module="mia.llm.llm_manager",  # Dotted import path
    class_name="LLMManager",       # Class to instantiate
    default=True,                   # Is the default for this category?
)
```

This allows **zero-cost imports** — modules are only loaded when first requested via `provider_registry.create("llm")`. Adding a new LLM backend or memory store requires only registering a factory; no core code changes.

### Tool Registry

Tools are defined as OpenAI function-calling JSON schemas:

```python
_tool(
    name="web_search",
    description="Search the web and return structured results with titles, URLs, and snippets.",
    properties={
        "query": _prop("Search query"),
        "num_results": _prop("Number of results", "integer"),
    },
    required=["query"],
)
```

For models without native function calling, `get_tool_descriptions_text()` generates a ReAct-compatible plain-text summary that the agent parses with regex.

**Tool categories:** Web (search, scrape, research, Wikipedia, YouTube) · File I/O (CRUD, directory ops, search) · Documents (docx, pptx, xlsx, pdf) · Communication (email, WhatsApp, Telegram, notifications) · Desktop (type, click, keys, app launch, window text) · System (commands, processes, clipboard) · Code (create, analyze, execute) · Memory (store, search, retrieve) · IoT (MQTT, Home Assistant) · Calendar · OCR · Embeddings

### Security Model

```
ActionExecutor.ACTION_SCOPES = {
    "read_file":    {"file"},        # Scoped permissions per tool
    "send_email":   {"email"},
    "run_command":  {"system"},
    "web_search":   {"web"},
    ...
}

Execution flow:
  1. Scope check    → Is this tool allowed in current session?
  2. Consent gate   → consent_callback() prompts user for sensitive actions
  3. Execution      → Tool runs with error handling
  4. Audit log      → Structured log entry with tool name, params, result
```

Sensitive actions (file writes, email, messaging, shell commands, desktop automation) require explicit user confirmation via a configurable `consent_callback`. Code execution uses subprocess isolation with optional Docker/WASI sandboxing.

---

## Quick Start

### Prerequisites

| Component | Required | Optional |
|-----------|----------|----------|
| Python 3.10+ | ✅ | |
| Git | ✅ | |
| Ollama | | Local LLM inference |
| FFmpeg | | Audio processing |
| Tesseract OCR | | Vision/OCR |
| Chrome/Edge + WebDriver | | Web automation (Selenium) |
| CUDA/cuDNN | | GPU acceleration |
| Docker | | Sandbox isolation |

### Installation

```bash
# Clone
git clone https://github.com/Matheussoranco/M.I.A-The-successor-of-pseudoJarvis.git
cd M.I.A-The-successor-of-pseudoJarvis

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/macOS)
source venv/bin/activate

# Install — choose your tier:
pip install -r requirements-core.txt    # Text-only, minimal (~15 deps)
pip install -r requirements.txt         # Everything (core + extras + dev)

# Install as editable package
pip install -e .
```

### Running

```bash
# Interactive CLI (default)
python -m mia
# or
mia                                    # If installed via pip install -e .

# Web UI (Ollama-style chat)
python -m mia --web
python -m mia --web --port 8080

# API server (FastAPI + Uvicorn)
mia-api
# or
python -m mia.api.server

# Text-only mode (no audio hardware required)
python -m mia --mode text

# Debug mode (verbose logging)
python -m mia --debug
```

### First Run with Ollama

```bash
# Install Ollama (https://ollama.com)
# Pull a model
ollama pull llama3.1:8b

# M.I.A auto-detects Ollama on localhost:11434
python -m mia
```

### Linux System Dependencies

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install python3 python3-pip python3-venv \
    python3-dev portaudio19-dev ffmpeg git

# Fedora
sudo dnf install python3 python3-pip python3-devel portaudio-devel ffmpeg git

# Arch
sudo pacman -S python python-pip python-virtualenv portaudio ffmpeg git
```

### Docker

```bash
docker-compose up -d
```

---

## Configuration

### Environment Variables (`.env`)

```bash
# LLM Providers (set one or more)
OLLAMA_HOST=http://localhost:11434       # Local Ollama
OPENAI_API_KEY=sk-...                    # OpenAI
ANTHROPIC_API_KEY=sk-ant-...             # Anthropic
GOOGLE_API_KEY=...                       # Google Gemini
HUGGINGFACE_HUB_TOKEN=hf_...            # HuggingFace

# Web Search (optional — DuckDuckGo works with no keys)
GOOGLE_API_KEY=...                       # Google Custom Search
GOOGLE_CSE_ID=...                        # Custom Search Engine ID

# Email
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USERNAME=...
EMAIL_PASSWORD=...                       # App password

# Telegram
TELEGRAM_BOT_TOKEN=...
TELEGRAM_API_ID=...
TELEGRAM_API_HASH=...

# IoT
HOME_ASSISTANT_URL=http://homeassistant.local:8123
HOME_ASSISTANT_TOKEN=...
```

### Configuration File (`config/config.yaml`)

```yaml
llm:
  provider: ollama                  # ollama | openai | anthropic | azure | huggingface | llamacpp
  model: llama3.1:8b                # Model identifier
  temperature: 0.7
  max_tokens: 4096
  context_length: 8192

audio:
  stt: whisper                      # whisper | vosk | speechrecognition
  tts: piper                        # piper | coqui | pyttsx3
  vad: webrtc                       # WebRTC VAD
  hotword: enabled                  # Wake-word activation

vision:
  provider: llava                   # llava | tesseract | easyocr
  ocr_engine: tesseract

memory:
  vector_store: chroma              # ChromaDB
  embedding_model: all-MiniLM-L6-v2
  max_context_items: 10
  persist_dir: memory/

tools:
  enabled:
    - web_search
    - file_operations
    - email
    - desktop_automation
    - code_execution
    - documents

security:
  confirm_before_action: true       # Ask before sensitive ops
  allowed_scopes:                   # Restrict tool categories
    - web
    - file
    - system
```

---

## API Reference

The FastAPI server exposes the following endpoints. All chat endpoints route through the `ToolCallingAgent`, giving API clients full access to tools, memory, planning, and multi-agent orchestration.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns `{"status": "ok", "version": "..."}` |
| `GET` | `/ready` | Readiness — reports agent and LLM availability |
| `POST` | `/chat` | Non-streaming chat. Body: `{"prompt": "..."}` → `{"response": "...", "status": "success"}` |
| `GET` | `/chat/stream?prompt=...` | SSE streaming. Returns `data: <token>` events, `event: done` on completion |
| `GET` | `/actions` | List available tools |
| `POST` | `/actions/{name}` | Execute a specific tool with parameters |
| `GET` | `/memory?query=...` | Search persistent memory |
| `POST` | `/memory` | Store memory item. Body: `{"text": "...", "metadata": {...}}` |
| `GET` | `/status` | System status: LLM, memory, actions, speech, web agent availability |

### Example

```bash
# Chat (with full tool access)
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Search the web for the latest Python release and summarize"}'

# Streaming
curl -N "http://localhost:8080/chat/stream?prompt=Explain+quantum+computing"

# Execute a tool directly
curl -X POST http://localhost:8080/actions/web_search \
  -H "Content-Type: application/json" \
  -d '{"query": "latest AI news"}'
```

---

## Benchmarks

M.I.A includes benchmark harnesses for evaluating against standard AGI/agent benchmarks. Framework-level adapters are implemented under `src/mia/benchmarks/`:

| Benchmark | Module | Domain |
|-----------|--------|--------|
| [ARC-AGI](https://arcprize.org/) | `arc_agi.py` | Abstract reasoning, grid transformations |
| [SWE-bench](https://swebench.com/) | `swe_bench.py` | Autonomous software engineering |
| [GAIA](https://huggingface.co/gaia-benchmark) | `gaia.py` | Multi-step reasoning with tool use |
| [GPQA](https://arxiv.org/abs/2311.12022) | `gpqa.py` | Graduate-level science questions |
| [MMMU](https://mmmu-benchmark.github.io/) | `mmmu.py` | Multimodal understanding |
| [OSWorld](https://os-world.github.io/) | `osworld.py` | Desktop task automation |
| [WebVoyager](https://arxiv.org/abs/2401.13919) | `webvoyager.py` | Web navigation and interaction |

```bash
python -m mia.benchmarks.runner --benchmark arc_agi --data path/to/tasks.json
```

---

## Testing

```bash
# All tests (190 pass, 3 skipped)
python -m pytest tests/ -q

# By category
python -m pytest tests/unit/             # 134 unit tests
python -m pytest tests/integration/      # 49 integration tests
python -m pytest tests/e2e/              # 10 end-to-end tests

# With coverage
python -m pytest tests/ --cov=src/mia --cov-report=term-missing

# Specific module
python -m pytest tests/unit/test_tool_calling_agent.py -v
```

### Test Structure

```
tests/
├── conftest.py                     # Shared fixtures, mocks
├── unit/                           # Isolated component tests
│   ├── test_action_executor.py
│   ├── test_agent_memory.py
│   ├── test_cognitive_architecture.py
│   ├── test_error_handler.py
│   ├── test_llm_manager.py
│   ├── test_main.py
│   ├── test_multimodal_processor.py
│   ├── test_speech_generator.py
│   ├── test_tool_calling_agent.py
│   └── test_vision_processor.py
├── integration/                    # Cross-module integration
│   ├── test_comprehensive_integration.py
│   ├── test_end_to_end.py
│   ├── test_mia_integration.py
│   └── test_performance.py
└── e2e/                            # Full-stack scenarios
    └── test_e2e_scenarios.py
```

---

## Dependencies

M.I.A uses a tiered dependency model:

| Tier | File | Contents | Count |
|------|------|----------|-------|
| **Core** | `requirements-core.txt` | Runtime essentials (openai, anthropic, requests, pydantic, numpy, sentence-transformers) | ~15 |
| **Extras** | `requirements-extras.txt` | Audio (whisper, piper, sounddevice), vision (opencv, PIL, tesseract), documents (docx, pptx, openpyxl), web (selenium, beautifulsoup4), sandbox (wasmtime, docker), benchmarks (swe-bench, gymnasium) | ~60 |
| **Dev** | `requirements-dev.txt` | Testing (pytest), linting (flake8, black, isort, mypy) | ~10 |
| **All** | `requirements.txt` | Aggregates all three tiers | ~85 |

---

## Project Metadata

| | |
|---|---|
| **Language** | Python 3.10+ |
| **Package** | `mia-successor` |
| **Source** | `src/mia/` (97 files) |
| **License** | AGPL-3.0-or-later |
| **Build** | setuptools + pyproject.toml |
| **Linting** | black (79 cols), isort, flake8, mypy (strict) |
| **Tests** | pytest — 193 collected, 190 pass, 3 skip |
| **Entry points** | `mia` (CLI), `mia-api` (server), `mia-web` (Web UI) |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write tests for new functionality
4. Ensure all tests pass: `python -m pytest tests/ -q`
5. Ensure code quality: `black src/ tests/` and `mypy src/mia/`
6. Submit a pull request

---

## License

**AGPL-3.0-or-later** — see [LICENSE](LICENSE).

## Author

**Matheus Pullig Soranço de Carvalho** — [matheussoranco@gmail.com](mailto:matheussoranco@gmail.com)

GitHub: [Matheussoranco](https://github.com/Matheussoranco)
