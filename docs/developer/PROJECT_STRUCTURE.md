# M.I.A Project Structure - Technical Documentation

## 🏗️ Enterprise-Grade Architecture

M.I.A follows a sophisticated modular architecture designed for scalability, maintainability, and enterprise deployment. The project structure implements clean architecture principles with clear separation of concerns.

```
M.I.A-The-successor-of-pseudoJarvis/
├── main.py                          # 🚀 Application entry point and orchestrator
├── requirements.txt                 # 📦 Unified dependency management (all components)
├── setup.py                        # 📋 Package configuration and metadata
├── pyproject.toml                  # 🔧 Modern Python project configuration
├── Dockerfile                      # 🐳 Container deployment configuration
├── docker-compose.yml              # 🐙 Multi-service orchestration
├── Makefile                        # 🔨 Build automation and development tasks
├── .env.template                   # 🔐 Environment configuration template
├── .gitignore                      # 🚫 Version control exclusions
├── LICENSE                         # 📄 MIT License
├── README.md                       # 📖 Technical documentation and quickstart
│
├── src/mia/                        # 🧠 Core application source code
│   ├── __init__.py                 # 📚 Package initialization and version
│   ├── main.py                     # 🎯 Application bootstrap and configuration
│   │
│   ├── core/                       # 🏛️ Core cognitive architecture
│   │   ├── __init__.py            # Core module initialization
│   │   ├── cognitive_architecture.py  # 🧠 Main cognitive engine
│   │   ├── reasoning_engine.py     # 🔬 Advanced reasoning capabilities
│   │   ├── decision_tree.py        # 🌳 Decision-making algorithms
│   │   ├── context_manager.py      # 📝 Context state management
│   │   ├── attention_mechanism.py  # 👁️ Attention and focus systems
│   │   └── state_machine.py        # 🔄 State transition management
│   │
│   ├── llm/                        # 🤖 Large Language Model integration
│   │   ├── __init__.py            # LLM module initialization
│   │   ├── manager.py             # 🎛️ LLM provider management
│   │   ├── ollama_provider.py     # 🦙 Ollama integration
│   │   ├── openai_provider.py     # 🤖 OpenAI API integration
│   │   ├── anthropic_provider.py  # 🧬 Anthropic Claude integration
│   │   ├── huggingface_provider.py # 🤗 HuggingFace transformers
│   │   ├── model_loader.py        # 📥 Dynamic model loading
│   │   ├── prompt_engine.py       # 💬 Prompt engineering and optimization
│   │   ├── response_processor.py  # 📤 Response parsing and validation
│   │   └── quantization.py        # ⚡ Model quantization and optimization
│   │
│   ├── multimodal/                 # 🔄 Multimodal processing pipeline
│   │   ├── __init__.py            # Multimodal module initialization
│   │   ├── processor.py           # 🔄 Main multimodal processor
│   │   ├── text_processor.py      # 📝 Text processing and NLP
│   │   ├── audio_processor.py     # 🎵 Audio processing and ASR/TTS
│   │   ├── vision_processor.py    # 👁️ Computer vision and image analysis
│   │   ├── fusion_layer.py        # 🔗 Modal fusion and alignment
│   │   ├── attention_fusion.py    # 🎯 Cross-modal attention mechanisms
│   │   └── modality_detector.py   # 🔍 Automatic modality detection
│   │
│   ├── audio/                      # 🎤 Audio processing subsystem
│   │   ├── __init__.py            # Audio module initialization
│   │   ├── speech_recognition.py  # 🗣️ Speech-to-text conversion
│   │   ├── text_to_speech.py      # 🔊 Text-to-speech synthesis
│   │   ├── audio_enhancement.py   # 🎧 Audio preprocessing and enhancement
│   │   ├── vad.py                 # 🔇 Voice activity detection
│   │   ├── noise_reduction.py     # 🔕 Noise filtering and reduction
│   │   └── audio_formats.py       # 🎵 Audio format conversion
│   │
│   ├── vision/                     # 👁️ Computer vision subsystem
│   │   ├── __init__.py            # Vision module initialization
│   │   ├── image_processor.py     # 🖼️ Image processing and analysis
│   │   ├── clip_integration.py    # 📎 CLIP model integration
│   │   ├── object_detection.py    # 🎯 Object detection and recognition
│   │   ├── ocr.py                 # 📄 Optical character recognition
│   │   ├── face_detection.py      # 👤 Face detection and recognition
│   │   ├── scene_understanding.py # 🏞️ Scene analysis and understanding
│   │   └── image_generation.py    # 🎨 Image generation capabilities
│   │
│   ├── memory/                     # 💾 Memory and knowledge systems
│   │   ├── __init__.py            # Memory module initialization
│   │   ├── vector_memory.py       # 📊 Vector database operations
│   │   ├── conversation_memory.py # 💬 Conversation history management
│   │   ├── knowledge_graph.py     # 🕸️ Knowledge graph operations
│   │   ├── embedding_manager.py   # 🔢 Embedding generation and management
│   │   ├── semantic_search.py     # 🔍 Semantic similarity search
│   │   ├── memory_consolidation.py # 🧠 Memory consolidation algorithms
│   │   └── forgetting_curve.py    # 📉 Memory decay and retention
│   │
│   ├── security/                   # 🔒 Security and privacy systems
│   │   ├── __init__.py            # Security module initialization
│   │   ├── input_validator.py     # ✅ Input validation and sanitization
│   │   ├── content_filter.py      # 🚫 Content filtering and safety
│   │   ├── encryption_manager.py  # 🔐 Encryption and decryption
│   │   ├── authentication.py      # 🔑 User authentication systems
│   │   ├── authorization.py       # 🛡️ Access control and permissions
│   │   ├── audit_logger.py        # 📝 Security audit logging
│   │   ├── privacy_manager.py     # 🔒 Privacy protection and anonymization
│   │   └── rate_limiter.py        # ⏱️ Rate limiting and DDoS protection
│   │
│   ├── tools/                      # 🔧 Action execution and tool integration
│   │   ├── __init__.py            # Tools module initialization
│   │   ├── action_executor.py     # ⚡ Action execution engine
│   │   ├── web_search.py          # 🔍 Web search integration
│   │   ├── file_operations.py     # 📁 File system operations
│   │   ├── system_control.py      # 🖥️ System command execution
│   │   ├── calendar_integration.py # 📅 Calendar and scheduling
│   │   ├── email_client.py        # 📧 Email integration
│   │   └── api_client.py          # 🌐 External API integration
│   │
│   ├── plugins/                    # 🔌 Plugin system and extensions
│   │   ├── __init__.py            # Plugin module initialization
│   │   ├── plugin_manager.py      # 🎛️ Plugin lifecycle management
│   │   ├── base_plugin.py         # 📋 Base plugin interface
│   │   ├── plugin_loader.py       # 📥 Dynamic plugin loading
│   │   ├── plugin_registry.py     # 📚 Plugin registration system
│   │   └── example_plugin.py      # 📝 Example plugin implementation
│   │
│   ├── utils/                      # 🛠️ Utility functions and helpers
│   │   ├── __init__.py            # Utils module initialization
│   │   ├── logging.py             # 📊 Structured logging configuration
│   │   ├── config.py              # ⚙️ Configuration management
│   │   ├── helpers.py             # 🔧 Common utility functions
│   │   ├── decorators.py          # 🎭 Custom decorators
│   │   ├── exceptions.py          # ⚠️ Custom exception classes
│   │   ├── constants.py           # 📏 Application constants
│   │   └── validators.py          # ✅ Data validation utilities
│   │
│   ├── api/                        # 🌐 API and web interface
│   │   ├── __init__.py            # API module initialization
│   │   ├── server.py              # 🖥️ FastAPI server implementation
│   │   ├── routes.py              # 🛣️ API route definitions
│   │   ├── middleware.py          # 🔄 HTTP middleware
│   │   ├── websocket.py           # 🔌 WebSocket handlers
│   │   ├── models.py              # 📊 Pydantic data models
│   │   └── dependencies.py        # 🔗 FastAPI dependencies
│   │
│   ├── cache_manager.py           # 🗄️ Intelligent caching system
│   ├── performance_monitor.py     # 📈 Performance monitoring
│   ├── documentation_generator.py # 📖 Automated documentation
│   └── code_quality_manager.py    # 🔍 Code quality analysis
│
├── tests/                          # 🧪 Comprehensive testing suite
│   ├── __init__.py                # Test package initialization
│   ├── conftest.py                # 🔧 Pytest configuration and fixtures
│   ├── test_core/                 # 🧠 Core system tests
│   │   ├── test_cognitive_architecture.py
│   │   ├── test_reasoning_engine.py
│   │   ├── test_decision_tree.py
│   │   └── test_context_manager.py
│   ├── test_llm/                  # 🤖 LLM integration tests
│   │   ├── test_manager.py
│   │   ├── test_ollama_provider.py
│   │   ├── test_openai_provider.py
│   │   └── test_prompt_engine.py
│   ├── test_multimodal/           # 🔄 Multimodal processing tests
│   │   ├── test_processor.py
│   │   ├── test_text_processor.py
│   │   ├── test_audio_processor.py
│   │   └── test_vision_processor.py
│   ├── test_memory/               # 💾 Memory system tests
│   │   ├── test_vector_memory.py
│   │   ├── test_conversation_memory.py
│   │   └── test_knowledge_graph.py
│   ├── test_security/             # 🔒 Security tests
│   │   ├── test_input_validator.py
│   │   ├── test_content_filter.py
│   │   └── test_encryption_manager.py
│   ├── test_integration/          # 🔗 Integration tests
│   │   ├── test_end_to_end.py
│   │   ├── test_api_integration.py
│   │   └── test_plugin_integration.py
│   ├── test_performance/          # ⚡ Performance tests
│   │   ├── test_benchmarks.py
│   │   ├── test_load_testing.py
│   │   └── test_memory_usage.py
│   └── test_priority_4.py         # 🎯 Priority 4 comprehensive tests
│
├── docs/                          # 📚 Technical documentation
│   ├── README.md                  # 📖 Documentation overview
│   ├── TECHNICAL_ARCHITECTURE.md # 🏗️ Technical architecture guide
│   ├── API_REFERENCE.md           # 🔗 API documentation
│   ├── DEPLOYMENT_GUIDE.md        # 🚀 Deployment instructions
│   ├── DEVELOPMENT_GUIDE.md       # 👨‍💻 Development guidelines
│   ├── SECURITY_GUIDE.md          # 🔒 Security best practices
│   ├── PERFORMANCE_GUIDE.md       # ⚡ Performance optimization
│   ├── PLUGIN_DEVELOPMENT.md      # 🔌 Plugin development guide
│   ├── TROUBLESHOOTING.md         # 🔧 Common issues and solutions
│   ├── CHANGELOG.md               # 📝 Version history
│   ├── CONTRIBUTING.md            # 🤝 Contribution guidelines
│   ├── REQUIREMENTS_GUIDE.md      # 📦 Dependencies documentation
│   ├── PRIORITY_4_IMPLEMENTATION.md # 🎯 Priority 4 features
│   └── REQUIREMENTS_CONSOLIDATION_SUMMARY.md # 📋 Requirements summary
│
├── config/                        # ⚙️ Configuration management
│   ├── __init__.py                # Config package initialization
│   ├── default.yaml               # 📄 Default configuration
│   ├── development.yaml           # 👨‍💻 Development environment config
│   ├── production.yaml            # 🚀 Production environment config
│   ├── testing.yaml               # 🧪 Testing environment config
│   ├── models.yaml                # 🤖 Model configurations
│   ├── security.yaml              # 🔒 Security configurations
│   └── logging.yaml               # 📊 Logging configurations
│
├── scripts/                       # 🔨 Automation and utility scripts
│   ├── install/                   # 📦 Installation scripts
│   │   ├── install.sh            # 🐧 Linux/macOS installation
│   │   ├── install.bat           # 🪟 Windows installation
│   │   └── install.ps1           # 💻 PowerShell installation
│   ├── deployment/                # 🚀 Deployment scripts
│   │   ├── deploy.sh             # 🚀 Production deployment
│   │   ├── docker-build.sh       # 🐳 Docker build automation
│   │   └── k8s-deploy.sh         # ☸️ Kubernetes deployment
│   ├── development/               # 👨‍💻 Development utilities
│   │   ├── setup-dev.sh          # 🔧 Development environment setup
│   │   ├── run-tests.sh          # 🧪 Test execution
│   │   └── code-quality.sh       # 🔍 Code quality checks
│   ├── maintenance/               # 🔧 Maintenance scripts
│   │   ├── backup.sh             # 💾 Data backup
│   │   ├── cleanup.sh            # 🧹 Cleanup operations
│   │   └── health-check.sh       # 🏥 Health monitoring
│   └── dev_qa.py                 # 🎯 Development and QA automation
│
├── k8s/                          # ☸️ Kubernetes deployment manifests
│   ├── namespace.yaml            # 🏷️ Namespace configuration
│   ├── deployment.yaml           # 🚀 Application deployment
│   ├── service.yaml              # 🔗 Service configuration
│   ├── ingress.yaml              # 🌐 Ingress configuration
│   ├── configmap.yaml            # ⚙️ Configuration mapping
│   ├── secret.yaml               # 🔐 Secret management
│   ├── hpa.yaml                  # 📈 Horizontal Pod Autoscaler
│   └── rbac.yaml                 # 🛡️ Role-based access control
│
├── helm/                         # ⛵ Helm charts for deployment
│   ├── Chart.yaml                # 📋 Chart metadata
│   ├── values.yaml               # 📊 Default values
│   ├── values-prod.yaml          # 🚀 Production values
│   └── templates/                # 📝 Kubernetes templates
│       ├── deployment.yaml
│       ├── service.yaml
│       ├── ingress.yaml
│       └── configmap.yaml
│
├── .github/                      # 🐙 GitHub Actions and workflows
│   ├── workflows/                # 🔄 CI/CD workflows
│   │   ├── ci.yml                # 🔄 Continuous integration
│   │   ├── cd.yml                # 🚀 Continuous deployment
│   │   ├── security-scan.yml     # 🔒 Security scanning
│   │   └── performance-test.yml  # ⚡ Performance testing
│   ├── ISSUE_TEMPLATE/           # 📝 Issue templates
│   └── PULL_REQUEST_TEMPLATE.md  # 🔄 PR template
│
├── .vscode/                      # 💻 VS Code configuration
│   ├── settings.json             # ⚙️ Editor settings
│   ├── launch.json               # 🚀 Debug configuration
│   ├── tasks.json                # 🔨 Task automation
│   └── extensions.json           # 🧩 Recommended extensions
│
├── monitoring/                   # 📊 Monitoring and observability
│   ├── prometheus/               # 📈 Prometheus configuration
│   │   ├── prometheus.yml
│   │   └── alerts.yml
│   ├── grafana/                  # 📊 Grafana dashboards
│   │   ├── dashboards/
│   │   └── provisioning/
│   └── jaeger/                   # 🔍 Distributed tracing
│       └── jaeger.yml
│
├── data/                         # 📊 Data storage and models
│   ├── models/                   # 🤖 Model storage
│   ├── embeddings/               # 🔢 Embedding cache
│   ├── conversations/            # 💬 Conversation history
│   └── logs/                     # 📝 Application logs
│
├── start-mia.sh                  # 🚀 Linux/macOS startup script
├── start-mia.bat                 # 🪟 Windows startup script
├── start-mia.ps1                 # 💻 PowerShell startup script
└── .env                          # 🔐 Environment variables (local)
```

## 🧩 Module Architecture

### Core Modules

#### 1. **Cognitive Architecture** (`src/mia/core/`)
- **Primary Purpose**: Central reasoning and decision-making engine
- **Key Components**:
  - `cognitive_architecture.py`: Main cognitive engine with attention mechanisms
  - `reasoning_engine.py`: Logical, causal, and analogical reasoning
  - `decision_tree.py`: Hierarchical decision-making algorithms
  - `context_manager.py`: Context state management and preservation
  - `attention_mechanism.py`: Multi-head attention for cross-modal processing
  - `state_machine.py`: State transition management and workflow control

#### 2. **LLM Integration** (`src/mia/llm/`)
- **Primary Purpose**: Large language model provider abstraction and management
- **Key Components**:
  - `manager.py`: Provider-agnostic LLM management
  - `*_provider.py`: Provider-specific implementations (Ollama, OpenAI, etc.)
  - `model_loader.py`: Dynamic model loading and caching
  - `prompt_engine.py`: Prompt engineering and optimization
  - `quantization.py`: Model quantization for performance

#### 3. **Multimodal Processing** (`src/mia/multimodal/`)
- **Primary Purpose**: Cross-modal understanding and fusion
- **Key Components**:
  - `processor.py`: Main multimodal orchestration
  - `*_processor.py`: Modality-specific processors
  - `fusion_layer.py`: Modal fusion and alignment
  - `attention_fusion.py`: Cross-modal attention mechanisms

#### 4. **Memory Systems** (`src/mia/memory/`)
- **Primary Purpose**: Knowledge storage, retrieval, and management
- **Key Components**:
  - `vector_memory.py`: Vector database operations with ChromaDB
  - `conversation_memory.py`: Conversation history management
  - `knowledge_graph.py`: Graph-based knowledge representation
  - `semantic_search.py`: Semantic similarity search
  - `memory_consolidation.py`: Memory consolidation algorithms

#### 5. **Security Framework** (`src/mia/security/`)
- **Primary Purpose**: Security, privacy, and compliance
- **Key Components**:
  - `input_validator.py`: Input validation and sanitization
  - `content_filter.py`: Content filtering and safety
  - `encryption_manager.py`: Encryption and decryption
  - `authentication.py`: User authentication systems
  - `authorization.py`: Role-based access control
  - `audit_logger.py`: Security audit logging

## 🔧 Development Architecture

### Build System
- **Makefile**: Unified build automation and development tasks
- **setup.py**: Package configuration and distribution
- **pyproject.toml**: Modern Python project configuration (PEP 518)

### Testing Strategy
- **Unit Tests**: Individual component testing (90%+ coverage)
- **Integration Tests**: Cross-component interaction testing
- **End-to-End Tests**: Full workflow validation
- **Performance Tests**: Benchmark and load testing
- **Security Tests**: Vulnerability and compliance testing

### Code Quality
- **Black**: Code formatting and style consistency
- **isort**: Import organization and sorting
- **flake8**: Linting and style checking
- **mypy**: Static type checking
- **bandit**: Security vulnerability detection
- **pytest**: Testing framework with coverage

### Documentation
- **Sphinx**: API documentation generation
- **Markdown**: Technical documentation
- **Automated Generation**: Code-driven documentation updates

## 🚀 Deployment Architecture

### Container Strategy
- **Docker**: Multi-stage builds for optimized containers
- **Docker Compose**: Multi-service development environment
- **Kubernetes**: Production orchestration and scaling

### Infrastructure as Code
- **Helm Charts**: Kubernetes application packaging
- **Terraform**: Infrastructure provisioning (optional)
- **GitOps**: Deployment automation with ArgoCD

### Monitoring and Observability
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **Jaeger**: Distributed tracing
- **ELK Stack**: Centralized logging

## 📊 Data Flow Architecture

### Input Processing
1. **Raw Input** → **Modality Detection** → **Preprocessing** → **Encoding**
2. **Encoded Data** → **Attention Mechanisms** → **Fusion Layer** → **Cognitive Processing**
3. **Processed Data** → **Memory Storage** → **Context Management** → **Response Generation**

### Memory Flow
1. **Input** → **Embedding Generation** → **Vector Storage** → **Similarity Search**
2. **Retrieved Context** → **Relevance Scoring** → **Context Integration** → **Response Enhancement**

### Security Flow
1. **Input** → **Validation** → **Sanitization** → **Authentication** → **Authorization**
2. **Processed Data** → **Content Filtering** → **Encryption** → **Audit Logging**

This project structure implements enterprise-grade software architecture principles with clear separation of concerns, comprehensive testing, and production-ready deployment capabilities.
│   ├── main.py           # 🎮 Core application logic
│   ├── audio/            # 🎵 Audio processing modules
│   │   ├── __init__.py
│   │   ├── audio_utils.py
│   │   ├── speech_processor.py
│   │   └── speech_generator.py
│   ├── core/             # 🧬 Core cognitive architecture
│   │   ├── __init__.py
│   │   └── cognitive_architecture.py
│   ├── llm/              # 🤖 LLM integration
│   │   ├── __init__.py
│   │   └── llm_manager.py
│   ├── memory/           # 🧠 Memory management
│   │   ├── __init__.py
│   │   ├── knowledge_graph.py
│   │   └── long_term_memory.py
│   ├── multimodal/       # 🎨 Multimodal processing
│   │   ├── __init__.py
│   │   ├── processor.py
│   │   └── vision_processor.py
│   ├── plugins/          # 🔌 Plugin system
│   │   ├── __init__.py
│   │   └── plugin_manager.py
│   ├── security/         # 🛡️ Security management
│   │   ├── __init__.py
│   │   └── security_manager.py
│   ├── tools/            # 🔧 Tool execution
│   │   ├── __init__.py
│   │   └── action_executor.py
│   ├── utils/            # 🛠️ Utility functions
│   │   ├── __init__.py
│   │   └── automation_util.py
│   ├── learning/         # 📚 User learning
│   │   ├── __init__.py
│   │   └── user_learning.py
│   ├── planning/         # 📅 Planning and scheduling
│   │   ├── __init__.py
│   │   └── calendar_integration.py
│   ├── deployment/       # 🚀 Deployment management
│   │   ├── __init__.py
│   │   └── deployment_manager.py
│   ├── langchain/        # 🔗 LangChain integration
│   │   ├── __init__.py
│   │   └── langchain_verifier.py
│   └── system/           # 💻 System control
│       ├── __init__.py
│       └── system_control.py
│
├── scripts/              # 📜 Utility scripts
│   ├── install/          # 📦 Installation scripts
│   │   ├── install.bat
│   │   ├── install.sh
│   │   ├── install_ffmpeg.bat
│   │   └── uninstall.sh
│   ├── run/              # 🏃 Run scripts
│   │   ├── run.bat
│   │   ├── run-audio.bat
│   │   ├── run-mixed.bat
│   │   ├── run-text-only.bat
│   │   └── run.sh
│   └── development/      # 🔧 Development tools
│       ├── check-system.sh
│       ├── dev.sh
│       ├── quickstart.sh
│       └── test_ollama.py
│
├── docs/                 # 📖 Documentation
│   ├── README.md         # 📄 Main documentation
│   ├── CHANGELOG.md      # 📝 Change history
│   ├── PROJECT_STRUCTURE.md # 🏗️ This file
│   ├── user/             # 👤 User documentation
│   │   ├── USAGE.md
│   │   └── USAGE_GUIDE.md
│   └── developer/        # 👨‍💻 Developer documentation
│       └── API.md
│
├── config/               # ⚙️ Configuration files
│   └── .env.example      # 🔧 Environment template
│
├── tests/                # 🧪 Test files
│   ├── unit/             # 🔬 Unit tests
│   ├── integration/      # 🔗 Integration tests
│   └── __init__.py
│
└── venv/                 # 🐍 Virtual environment (local)
```

## 🎯 Key Improvements

### 1. **Clear Entry Points**
- **`main.py`**: Primary entry point at root level
- **`start.bat`**: Quick start script for users
- **`scripts/run/`**: Multiple run options organized

### 2. **Organized Source Code**
- **`src/mia/`**: All Python modules under single namespace
- **Proper package structure**: Each module has `__init__.py`
- **Relative imports**: Clean import statements

### 3. **Logical Script Organization**
- **`scripts/install/`**: Installation and setup scripts
- **`scripts/run/`**: Different run configurations
- **`scripts/development/`**: Development and testing tools

### 4. **Better Documentation**
- **`docs/user/`**: User-facing documentation
- **`docs/developer/`**: Technical documentation
- **`docs/`**: Main project documentation

### 5. **Configuration Management**
- **`config/`**: Centralized configuration files
- **Environment variables**: Organized in dedicated directory

## 🚀 Running the Application

### Quick Start
```bash
# Windows
start.bat

# Linux/Mac
python main.py
```

### Specific Modes
```bash
# Text-only mode
python main.py --text-only

# Audio mode
python main.py --audio-mode

# Mixed mode
python main.py

# With specific model
python main.py --model-id gemma3:4b-it-qat
```

### Using Scripts
```bash
# Windows
scripts\run\run.bat
scripts\run\run-text-only.bat
scripts\run\run-audio.bat

# Linux/Mac
scripts/run/run.sh
```

## 📦 Installation

### Windows
```bash
scripts\install\install.bat
```

### Linux/Mac
```bash
scripts/install/install.sh
```

## 🔧 Development

### Setup Development Environment
```bash
# Windows
scripts\development\dev.sh

# Linux/Mac
scripts/development/dev.sh
```

### Testing
```bash
# Test Ollama connection
python scripts/development/test_ollama.py

# Run tests
python -m pytest tests/
```

## 🗂️ Module Organization

### Core Modules
- **`main.py`**: Application entry point and main loop
- **`core/`**: Cognitive architecture and core AI logic
- **`llm/`**: Language model integration and management

### I/O Modules
- **`audio/`**: Audio input/output processing
- **`multimodal/`**: Vision and multimodal processing
- **`memory/`**: Knowledge graphs and long-term memory

### System Modules
- **`tools/`**: Action execution and system tools
- **`security/`**: Security and permission management
- **`plugins/`**: Plugin system and extensions

### Utility Modules
- **`utils/`**: General utility functions
- **`learning/`**: User learning and adaptation
- **`planning/`**: Calendar and scheduling integration
- **`deployment/`**: Deployment and configuration management

## 🎨 Benefits of New Structure

1. **🧹 Cleaner imports**: Relative imports reduce coupling
2. **📚 Better organization**: Related functionality grouped together
3. **🔧 Easier maintenance**: Clear separation of concerns
4. **🚀 Simpler deployment**: Single entry point and clear structure
5. **📖 Better documentation**: Organized docs for different audiences
6. **🧪 Improved testing**: Clear test organization
7. **🔌 Plugin support**: Well-organized plugin system
8. **⚙️ Configuration management**: Centralized config handling

## 🔄 Migration Guide

### For Users
1. Use `start.bat` instead of `run.bat`
2. Configuration moved to `config/` directory
3. Documentation moved to `docs/` directory

### For Developers
1. Import from `src.mia.module` instead of `module`
2. Scripts moved to `scripts/` subdirectories
3. Tests organized in `tests/` with proper structure

This new structure provides a much cleaner, more maintainable, and professional codebase! 🎉
