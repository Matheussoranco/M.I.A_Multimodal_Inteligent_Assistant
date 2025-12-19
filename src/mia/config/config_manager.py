"""
Unified Configuration Management and Validation System.

This module provides a centralized, type-safe configuration management
system with validation, environment variable support, and hot-reloading.

Features:
- Pydantic-based validation with type safety
- Environment variable override support
- Configuration inheritance and composition
- Hot-reload capability
- Configuration versioning
- Secrets management
"""

import os
import json
import yaml
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)
from datetime import datetime
import hashlib

from typing import TYPE_CHECKING

# Pydantic imports with proper type stubs for static analysis
try:
    from pydantic import (
        BaseModel,
        Field,
        validator,
        root_validator,
        ValidationError,
        SecretStr,
    )
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    
    # Type stubs for static analysis when pydantic is not available
    if TYPE_CHECKING:
        from pydantic import BaseModel, Field, validator, root_validator, ValidationError, SecretStr
    else:
        # Runtime fallbacks
        class BaseModel:  # type: ignore[no-redef]
            """Fallback BaseModel when pydantic is not available."""
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
            
            class Config:
                extra = "allow"
        
        def Field(*args, **kwargs):  # type: ignore[no-redef]
            """Fallback Field when pydantic is not available."""
            return kwargs.get('default', kwargs.get('default_factory', lambda: None)())
        
        def validator(*args, **kwargs):  # type: ignore[no-redef]
            """Fallback validator when pydantic is not available."""
            def decorator(func):
                return func
            return decorator
        
        def root_validator(*args, **kwargs):  # type: ignore[no-redef]
            """Fallback root_validator when pydantic is not available."""
            def decorator(func):
                return func
            return decorator
        
        class ValidationError(Exception):  # type: ignore[no-redef]
            """Fallback ValidationError when pydantic is not available."""
            pass
        
        class SecretStr(str):  # type: ignore[no-redef]
            """Fallback SecretStr when pydantic is not available."""
            pass

logger = logging.getLogger(__name__)


class ConfigEnvironment(Enum):
    """Configuration environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ============================================================================
# Base Configuration Models
# ============================================================================

class LLMProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    enabled: bool = True
    api_key: Optional[SecretStr] = None
    api_base: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)  # type: ignore[assignment]
    max_tokens: int = Field(default=4096, gt=0)  # type: ignore[assignment]
    timeout: float = Field(default=30.0, gt=0)  # type: ignore[assignment]
    max_retries: int = Field(default=3, ge=0)  # type: ignore[assignment]
    
    class Config:
        extra = "allow"


class OpenAIConfig(LLMProviderConfig):
    """OpenAI-specific configuration."""
    model: str = "gpt-4"
    organization: Optional[str] = None
    
    @validator('model')
    def validate_model(cls, v):
        valid_models = [
            'gpt-4', 'gpt-4-turbo', 'gpt-4-turbo-preview',
            'gpt-3.5-turbo', 'gpt-3.5-turbo-16k',
            'gpt-4o', 'gpt-4o-mini',
        ]
        if v not in valid_models and not v.startswith('gpt-'):
            logger.warning(f"Non-standard OpenAI model: {v}")
        return v


class AnthropicConfig(LLMProviderConfig):
    """Anthropic-specific configuration."""
    model: str = "claude-3-opus-20240229"
    
    @validator('model')
    def validate_model(cls, v):
        if not v.startswith('claude'):
            logger.warning(f"Non-standard Anthropic model: {v}")
        return v


class OllamaConfig(LLMProviderConfig):
    """Ollama-specific configuration."""
    api_base: str = "http://localhost:11434"
    model: str = "llama3"
    api_key: Optional[SecretStr] = None  # Ollama typically doesn't need API key


class GroqConfig(LLMProviderConfig):
    """Groq-specific configuration."""
    model: str = "mixtral-8x7b-32768"


class LLMConfig(BaseModel):
    """Complete LLM configuration."""
    default_provider: str = "openai"
    fallback_providers: List[str] = ["ollama"]
    
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)  # type: ignore[assignment]
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)  # type: ignore[assignment]
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)  # type: ignore[assignment]
    groq: GroqConfig = Field(default_factory=GroqConfig)  # type: ignore[assignment]
    
    # Routing configuration
    task_routing: Dict[str, str] = Field(default_factory=lambda: {  # type: ignore[assignment]
        "code": "codellama",
        "chat": "gpt-4",
        "reasoning": "claude-3-opus-20240229",
    })
    
    @root_validator  # type: ignore[misc]
    def validate_providers(cls, values):
        default = values.get('default_provider')
        valid_providers = ['openai', 'anthropic', 'ollama', 'groq']
        if default not in valid_providers:
            raise ValueError(f"Invalid default provider: {default}")
        return values


class MemoryConfig(BaseModel):
    """Memory system configuration."""
    enabled: bool = True
    
    # Vector store
    vector_store_type: str = "chromadb"
    vector_store_path: str = "./memory/vectors"
    collection_name: str = "mia_memories"
    
    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Long-term memory
    long_term_enabled: bool = True
    long_term_path: str = "./memory/long_term.json"
    max_memories: int = 10000
    
    # Knowledge graph
    knowledge_graph_enabled: bool = True
    knowledge_graph_path: str = "./memory/knowledge_graph"
    
    # RAG settings
    rag_chunk_size: int = Field(default=512, gt=0)  # type: ignore[assignment]
    rag_chunk_overlap: int = Field(default=50, ge=0)  # type: ignore[assignment]
    rag_top_k: int = Field(default=5, gt=0)  # type: ignore[assignment]
    
    @validator('rag_chunk_overlap')
    def validate_overlap(cls, v, values):
        if 'rag_chunk_size' in values and v >= values['rag_chunk_size']:
            raise ValueError("Chunk overlap must be less than chunk size")
        return v


class VisionConfig(BaseModel):
    """Vision processing configuration."""
    enabled: bool = True
    provider: str = "blip"
    
    # OpenAI vision
    openai_model: str = "gpt-4-vision-preview"
    openai_max_tokens: int = 300
    
    # Ollama vision
    ollama_model: str = "llava"
    
    # BLIP
    blip_model: str = "Salesforce/blip-image-captioning-base"
    
    # OCR
    ocr_enabled: bool = True
    ocr_language: str = "eng"
    
    # Processing
    max_image_size: int = Field(default=2048, gt=0)  # type: ignore[assignment]
    default_quality: str = "high"
    
    @validator('provider')
    def validate_provider(cls, v):
        valid_providers = ['openai', 'ollama', 'blip', 'clip']
        if v not in valid_providers:
            raise ValueError(f"Invalid vision provider: {v}")
        return v


class AudioConfig(BaseModel):
    """Audio processing configuration."""
    enabled: bool = True
    
    # Speech-to-text
    stt_engine: str = "whisper"
    whisper_model: str = "base"
    whisper_device: str = "auto"
    
    # Text-to-speech
    tts_engine: str = "piper"
    tts_voice: str = "en_US-lessac-medium"
    tts_rate: float = Field(default=1.0, gt=0)  # type: ignore[assignment]
    
    # VAD
    vad_enabled: bool = True
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)  # type: ignore[assignment]
    
    # Hotword
    hotword_enabled: bool = True
    hotword: str = "hey mia"


class SecurityConfig(BaseModel):
    """Security configuration."""
    enabled: bool = True
    
    # Sandbox
    sandbox_enabled: bool = True
    sandbox_type: str = "wasi"
    
    # Permissions
    filesystem_access: bool = True
    allowed_paths: List[str] = Field(default_factory=lambda: ["./", "~/Documents"])  # type: ignore[assignment]
    blocked_paths: List[str] = Field(default_factory=lambda: ["/etc", "/sys", "/proc"])  # type: ignore[assignment]
    
    network_access: bool = True
    allowed_domains: List[str] = Field(default_factory=list)  # type: ignore[assignment]
    blocked_domains: List[str] = Field(default_factory=list)  # type: ignore[assignment]
    
    code_execution: bool = True
    max_execution_time: float = Field(default=30.0, gt=0)  # type: ignore[assignment]
    
    # Audit
    audit_enabled: bool = True
    audit_log_path: str = "./logs/audit.log"
    
    @validator('sandbox_type')
    def validate_sandbox(cls, v):
        valid_types = ['wasi', 'docker', 'none']
        if v not in valid_types:
            raise ValueError(f"Invalid sandbox type: {v}")
        return v


class WebUIConfig(BaseModel):
    """Web UI configuration."""
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = Field(default=8000, gt=0, lt=65536)  # type: ignore[assignment]
    
    # CORS
    cors_enabled: bool = True
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])  # type: ignore[assignment]
    
    # WebSocket
    websocket_enabled: bool = True
    websocket_ping_interval: float = Field(default=30.0, gt=0)  # type: ignore[assignment]
    
    # Static files
    static_dir: str = "./static"
    template_dir: str = "./templates"
    
    # Session
    session_secret: Optional[SecretStr] = None
    session_expiry: int = Field(default=3600, gt=0)  # type: ignore[assignment]


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File logging
    file_enabled: bool = True
    file_path: str = "./logs/mia.log"
    file_max_size: int = Field(default=10_000_000, gt=0)  # type: ignore[assignment]  # 10MB
    file_backup_count: int = Field(default=5, ge=0)  # type: ignore[assignment]
    
    # Console logging
    console_enabled: bool = True
    console_colored: bool = True


class AgentConfig(BaseModel):
    """Multi-agent system configuration."""
    enabled: bool = True
    
    # Default crew
    create_default_crew: bool = True
    
    # Task execution
    max_iterations: int = Field(default=10, gt=0)  # type: ignore[assignment]
    task_timeout: float = Field(default=300.0, gt=0)  # type: ignore[assignment]
    
    # Consensus
    consensus_threshold: float = Field(default=0.5, ge=0.0, le=1.0)  # type: ignore[assignment]
    
    # Memory
    agent_memory_enabled: bool = True


class MIAConfig(BaseModel):
    """
    Main MIA configuration.
    
    This is the root configuration model that contains all subsystem configurations.
    """
    # Meta
    version: str = "1.0.0"
    environment: ConfigEnvironment = ConfigEnvironment.DEVELOPMENT
    debug: bool = False
    
    # Paths
    base_dir: str = "./"
    cache_dir: str = "./cache"
    config_dir: str = "./config"
    
    # Subsystems
    llm: LLMConfig = Field(default_factory=LLMConfig)  # type: ignore[assignment]
    memory: MemoryConfig = Field(default_factory=MemoryConfig)  # type: ignore[assignment]
    vision: VisionConfig = Field(default_factory=VisionConfig)  # type: ignore[assignment]
    audio: AudioConfig = Field(default_factory=AudioConfig)  # type: ignore[assignment]
    security: SecurityConfig = Field(default_factory=SecurityConfig)  # type: ignore[assignment]
    webui: WebUIConfig = Field(default_factory=WebUIConfig)  # type: ignore[assignment]
    logging: LoggingConfig = Field(default_factory=LoggingConfig)  # type: ignore[assignment]
    agents: AgentConfig = Field(default_factory=AgentConfig)  # type: ignore[assignment]
    
    # Extra settings
    extra: Dict[str, Any] = Field(default_factory=dict)  # type: ignore[assignment]
    
    class Config:
        extra = "allow"
        use_enum_values = True
    
    @root_validator  # type: ignore[misc]
    def validate_paths(cls, values):
        """Ensure paths exist or can be created."""
        base_dir = values.get('base_dir', './')
        
        # Resolve relative paths
        for key in ['cache_dir', 'config_dir']:
            if key in values and not os.path.isabs(values[key]):
                values[key] = os.path.join(base_dir, values[key])
        
        return values


# ============================================================================
# Configuration Manager
# ============================================================================

class ConfigSource(Enum):
    """Configuration sources."""
    FILE = "file"
    ENVIRONMENT = "environment"
    DEFAULT = "default"
    OVERRIDE = "override"


@dataclass
class ConfigEntry:
    """Tracks a configuration value and its source."""
    key: str
    value: Any
    source: ConfigSource
    timestamp: datetime = field(default_factory=datetime.now)


class ConfigManager:
    """
    Centralized configuration manager.
    
    Features:
    - Multi-source configuration loading
    - Environment variable overrides
    - Configuration validation
    - Hot-reload support
    - Configuration history
    """
    
    ENV_PREFIX = "MIA_"
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        environment: Optional[ConfigEnvironment] = None,
    ):
        self.config_path = config_path
        self.environment = environment or self._detect_environment()
        
        self._config: Optional[MIAConfig] = None
        self._config_hash: Optional[str] = None
        self._history: List[ConfigEntry] = []
        self._watchers: List[Callable] = []
        
        logger.info(f"ConfigManager initialized for {self.environment.value} environment")
    
    def _detect_environment(self) -> ConfigEnvironment:
        """Detect environment from environment variable."""
        env_str = os.getenv(f"{self.ENV_PREFIX}ENVIRONMENT", "development").lower()
        try:
            return ConfigEnvironment(env_str)
        except ValueError:
            logger.warning(f"Unknown environment: {env_str}, defaulting to development")
            return ConfigEnvironment.DEVELOPMENT
    
    def load(self, config_path: Optional[str] = None) -> MIAConfig:
        """Load configuration from file and environment."""
        path = config_path or self.config_path
        
        # Start with defaults
        config_dict = {}
        
        # Load from file
        if path and os.path.exists(path):
            config_dict = self._load_file(path)
            logger.info(f"Loaded configuration from {path}")
        
        # Apply environment-specific config
        env_config_path = self._get_env_config_path(path)
        if env_config_path and os.path.exists(env_config_path):
            env_config = self._load_file(env_config_path)
            config_dict = self._deep_merge(config_dict, env_config)
            logger.info(f"Applied {self.environment.value} configuration")
        
        # Apply environment variable overrides
        config_dict = self._apply_env_overrides(config_dict)
        
        # Validate and create config
        try:
            if PYDANTIC_AVAILABLE:
                self._config = MIAConfig(**config_dict)
            else:
                self._config = MIAConfig(**config_dict)
            
            # Calculate hash for change detection
            self._config_hash = self._calculate_hash(config_dict)
            
            logger.info("Configuration loaded and validated successfully")
            return self._config
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def _load_file(self, path: str) -> Dict:
        """Load configuration from file."""
        file_path = Path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        content = file_path.read_text()
        
        if file_path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(content) or {}
        elif file_path.suffix == '.json':
            return json.loads(content)
        else:
            raise ValueError(f"Unsupported configuration format: {file_path.suffix}")
    
    def _get_env_config_path(self, base_path: Optional[str]) -> Optional[str]:
        """Get environment-specific config file path."""
        if not base_path:
            return None
        
        base = Path(base_path)
        env_path = base.parent / f"{base.stem}.{self.environment.value}{base.suffix}"
        return str(env_path) if env_path.exists() else None
    
    def _apply_env_overrides(self, config: Dict) -> Dict:
        """Apply environment variable overrides."""
        for key, value in os.environ.items():
            if key.startswith(self.ENV_PREFIX):
                config_key = key[len(self.ENV_PREFIX):].lower()
                
                # Convert KEY_SUBKEY to nested dict
                parts = config_key.split('_')
                
                # Navigate/create nested structure
                current = config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Set value with type inference
                current[parts[-1]] = self._parse_env_value(value)
                
                self._history.append(ConfigEntry(
                    key=config_key,
                    value=value,
                    source=ConfigSource.ENVIRONMENT,
                ))
        
        return config
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # Number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # JSON
        if value.startswith('{') or value.startswith('['):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        return value
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _calculate_hash(self, config: Dict) -> str:
        """Calculate hash of configuration for change detection."""
        content = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()
    
    @property
    def config(self) -> MIAConfig:
        """Get current configuration."""
        if self._config is None:
            self.load()
        assert self._config is not None  # Type narrowing for type checker
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""
        config = self.config
        
        parts = key.split('.')
        current = config
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current
    
    def set(self, key: str, value: Any):
        """Set configuration value (runtime override)."""
        parts = key.split('.')
        
        if not self._config:
            self.load()
        
        # Navigate to parent
        current = self._config
        for part in parts[:-1]:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                raise KeyError(f"Configuration key not found: {key}")
        
        # Set value
        setattr(current, parts[-1], value)
        
        self._history.append(ConfigEntry(
            key=key,
            value=value,
            source=ConfigSource.OVERRIDE,
        ))
        
        # Notify watchers
        self._notify_watchers(key, value)
    
    def watch(self, callback: Callable[[str, Any], None]):
        """Register a callback for configuration changes."""
        self._watchers.append(callback)
    
    def _notify_watchers(self, key: str, value: Any):
        """Notify watchers of configuration change."""
        for callback in self._watchers:
            try:
                callback(key, value)
            except Exception as e:
                logger.error(f"Watcher callback failed: {e}")
    
    def reload(self) -> bool:
        """Reload configuration from file."""
        if not self.config_path:
            return False
        
        try:
            old_hash = self._config_hash
            self.load(self.config_path)
            
            if self._config_hash != old_hash:
                logger.info("Configuration reloaded with changes")
                self._notify_watchers('*', self._config)
                return True
            
            return False
        except Exception as e:
            logger.error(f"Configuration reload failed: {e}")
            return False
    
    def export(self, path: str, format: str = 'yaml'):
        """Export current configuration to file."""
        if not self._config:
            raise ValueError("No configuration loaded")
        
        if PYDANTIC_AVAILABLE:
            config_dict = self._config.dict()
        else:
            config_dict = vars(self._config)
        
        # Remove secrets
        config_dict = self._redact_secrets(config_dict)
        
        file_path = Path(path)
        
        if format == 'yaml':
            content = yaml.dump(config_dict, default_flow_style=False)
        elif format == 'json':
            content = json.dumps(config_dict, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        file_path.write_text(content)
        logger.info(f"Configuration exported to {file_path}")
    
    def _redact_secrets(self, config: Dict, keys_to_redact: Optional[Set[str]] = None) -> Dict:
        """Redact sensitive values from configuration."""
        keys_to_redact = keys_to_redact or {'api_key', 'secret', 'password', 'token'}
        
        result = {}
        for key, value in config.items():
            if isinstance(value, dict):
                result[key] = self._redact_secrets(value, keys_to_redact)
            elif any(k in key.lower() for k in keys_to_redact):
                result[key] = "***REDACTED***"
            elif hasattr(value, 'get_secret_value'):
                result[key] = "***REDACTED***"
            else:
                result[key] = value
        
        return result
    
    def validate(self) -> List[str]:
        """Validate configuration and return any warnings."""
        warnings = []
        config = self.config
        
        # Check LLM configuration
        if config.llm.default_provider == 'openai':
            api_key = config.llm.openai.api_key
            if not api_key or (hasattr(api_key, 'get_secret_value') and not api_key.get_secret_value()):
                warnings.append("OpenAI API key not set")
        
        # Check paths
        for path_attr in ['cache_dir', 'config_dir']:
            path = getattr(config, path_attr, None)
            if path and not os.path.exists(path):
                warnings.append(f"Directory does not exist: {path}")
        
        # Check security settings in production
        if config.environment == ConfigEnvironment.PRODUCTION:
            if config.debug:
                warnings.append("Debug mode enabled in production")
            if not config.security.sandbox_enabled:
                warnings.append("Sandbox disabled in production")
        
        return warnings
    
    def get_history(self) -> List[ConfigEntry]:
        """Get configuration change history."""
        return self._history.copy()


# ============================================================================
# Global Configuration Instance
# ============================================================================

_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> MIAConfig:
    """Get the current configuration."""
    return get_config_manager().config


def init_config(
    config_path: Optional[str] = None,
    environment: Optional[ConfigEnvironment] = None,
) -> MIAConfig:
    """Initialize configuration from file."""
    global _config_manager
    _config_manager = ConfigManager(config_path, environment)
    return _config_manager.load()


# ============================================================================
# Configuration Schema Export
# ============================================================================

def export_schema(path: str = "config_schema.json"):
    """Export configuration JSON schema."""
    if not PYDANTIC_AVAILABLE:
        raise ImportError("Pydantic required for schema export")
    
    schema = MIAConfig.schema_json(indent=2)
    Path(path).write_text(schema)
    logger.info(f"Configuration schema exported to {path}")


def generate_example_config(path: str = "config.example.yaml"):
    """Generate example configuration file."""
    if PYDANTIC_AVAILABLE:
        config = MIAConfig()
        config_dict = config.dict()
    else:
        config_dict = {}
    
    # Add helpful comments
    content = """# MIA Configuration File
# =====================
# This is an example configuration file for M.I.A (Multimodal Intelligent Assistant)
# Copy this file to config.yaml and customize as needed.

"""
    content += yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
    
    Path(path).write_text(content)
    logger.info(f"Example configuration exported to {path}")
