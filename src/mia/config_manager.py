"""
Configuration Management System for M.I.A
Provides centralized configuration loading, validation, and management.
"""
import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from .exceptions import ConfigurationError, ValidationError
from .error_handler import global_error_handler, with_error_handling

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: str = "ollama"
    model_id: str = "gemma3:4b-it-qat"
    api_key: Optional[str] = None
    url: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout: int = 30
    
    def validate(self) -> None:
        """Validate LLM configuration."""
        if not self.provider:
            raise ValidationError("Provider cannot be empty", "EMPTY_PROVIDER")
        
        if self.provider not in ['openai', 'ollama', 'huggingface', 'anthropic', 'gemini', 'groq', 'grok', 'local']:
            raise ValidationError(f"Unsupported provider: {self.provider}", "UNSUPPORTED_PROVIDER")
        
        if self.max_tokens <= 0:
            raise ValidationError("Max tokens must be positive", "INVALID_MAX_TOKENS")
        
        if not 0.0 <= self.temperature <= 2.0:
            raise ValidationError("Temperature must be between 0.0 and 2.0", "INVALID_TEMPERATURE")
        
        if self.timeout <= 0:
            raise ValidationError("Timeout must be positive", "INVALID_TIMEOUT")

@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    enabled: bool = True
    sample_rate: int = 16000
    chunk_size: int = 1024
    device_id: Optional[int] = None
    input_threshold: float = 0.01
    speech_model: str = "openai/whisper-base.en"
    tts_enabled: bool = True
    
    def validate(self) -> None:
        """Validate audio configuration."""
        if self.sample_rate <= 0:
            raise ValidationError("Sample rate must be positive", "INVALID_SAMPLE_RATE")
        
        if self.chunk_size <= 0:
            raise ValidationError("Chunk size must be positive", "INVALID_CHUNK_SIZE")
        
        if not 0.0 <= self.input_threshold <= 1.0:
            raise ValidationError("Input threshold must be between 0.0 and 1.0", "INVALID_THRESHOLD")

@dataclass
class VisionConfig:
    """Configuration for vision processing."""
    enabled: bool = True
    model: str = "openai/clip-vit-base-patch32"
    device: str = "auto"
    max_image_size: int = 1024
    supported_formats: List[str] = field(default_factory=lambda: ["jpg", "jpeg", "png", "bmp", "gif"])
    
    def validate(self) -> None:
        """Validate vision configuration."""
        if self.max_image_size <= 0:
            raise ValidationError("Max image size must be positive", "INVALID_IMAGE_SIZE")
        
        if not self.supported_formats:
            raise ValidationError("Supported formats cannot be empty", "EMPTY_FORMATS")
        
        if self.device not in ["auto", "cpu", "cuda", "mps"]:
            raise ValidationError(f"Unsupported device: {self.device}", "UNSUPPORTED_DEVICE")

@dataclass
class MemoryConfig:
    """Configuration for memory systems."""
    enabled: bool = True
    vector_db_path: str = "memory/"
    max_memory_size: int = 10000
    embedding_dimension: int = 768
    similarity_threshold: float = 0.7
    
    def validate(self) -> None:
        """Validate memory configuration."""
        if self.max_memory_size <= 0:
            raise ValidationError("Max memory size must be positive", "INVALID_MEMORY_SIZE")
        
        if self.embedding_dimension <= 0:
            raise ValidationError("Embedding dimension must be positive", "INVALID_EMBEDDING_DIM")
        
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValidationError("Similarity threshold must be between 0.0 and 1.0", "INVALID_THRESHOLD")

@dataclass
class SecurityConfig:
    """Configuration for security settings."""
    enabled: bool = True
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: List[str] = field(default_factory=lambda: ["txt", "md", "py", "json", "yaml"])
    blocked_commands: List[str] = field(default_factory=lambda: ["rm -rf", "del /f", "format", "fdisk"])
    max_command_length: int = 1000
    audit_logging: bool = True
    
    def validate(self) -> None:
        """Validate security configuration."""
        if self.max_file_size <= 0:
            raise ValidationError("Max file size must be positive", "INVALID_FILE_SIZE")
        
        if self.max_command_length <= 0:
            raise ValidationError("Max command length must be positive", "INVALID_COMMAND_LENGTH")

@dataclass
class SystemConfig:
    """Configuration for system settings."""
    debug: bool = False
    log_level: str = "INFO"
    max_workers: int = 4
    request_timeout: int = 30
    retry_attempts: int = 3
    cache_enabled: bool = True
    cache_ttl: int = 3600
    
    def validate(self) -> None:
        """Validate system configuration."""
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise ValidationError(f"Invalid log level: {self.log_level}", "INVALID_LOG_LEVEL")
        
        if self.max_workers <= 0:
            raise ValidationError("Max workers must be positive", "INVALID_MAX_WORKERS")
        
        if self.request_timeout <= 0:
            raise ValidationError("Request timeout must be positive", "INVALID_TIMEOUT")
        
        if self.retry_attempts < 0:
            raise ValidationError("Retry attempts cannot be negative", "INVALID_RETRY_ATTEMPTS")

@dataclass
class MIAConfig:
    """Main configuration class for M.I.A."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def validate(self) -> None:
        """Validate all configuration sections."""
        self.llm.validate()
        self.audio.validate()
        self.vision.validate()
        self.memory.validate()
        self.security.validate()
        self.system.validate()

class ConfigManager:
    """Configuration manager for M.I.A."""
    
    def __init__(self, config_dir: str = "config"):
        if config_dir and os.path.isfile(config_dir):
            # If config_dir is actually a file path, extract directory
            config_file = Path(config_dir)
            self.config_dir = config_file.parent
            self._explicit_config_file = config_file
        else:
            self.config_dir = Path(config_dir)
            self._explicit_config_file = None
        
        self.config: Optional[MIAConfig] = None
        self._config_file_paths = [
            self.config_dir / "config.json",
            self.config_dir / "config.yaml",
            self.config_dir / "config.yml",
            "config.json",
            "config.yaml",
            "config.yml"
        ]
        # Load .env early if present (non-fatal if missing)
        try:
            from dotenv import load_dotenv  # type: ignore
            env_path = self.config_dir / ".env"
            if env_path.exists():
                load_dotenv(dotenv_path=str(env_path))
            else:
                # Also load project root .env if exists
                if os.path.exists('.env'):
                    load_dotenv(dotenv_path='.env')
        except Exception:
            # dotenv is optional; ignore any errors
            pass
        
        # Auto-load config if explicit file was provided
        if self._explicit_config_file:
            try:
                self.load_config(str(self._explicit_config_file))
            except Exception as e:
                logger.warning(f"Failed to auto-load config: {e}")
        
    @with_error_handling(global_error_handler, fallback_value=None)
    def load_config(self, config_path: Optional[str] = None) -> MIAConfig:
        """Load configuration from file or use defaults."""
        if config_path:
            config_file = Path(config_path)
            if not config_file.exists():
                raise ConfigurationError(f"Configuration file not found: {config_path}", "CONFIG_NOT_FOUND")
        else:
            config_file = self._find_config_file()
        
        if config_file:
            try:
                config_data = self._load_config_file(config_file)
                self.config = self._create_config_from_dict(config_data)
            except Exception as e:
                logger.error(f"Failed to load configuration from {config_file}: {e}")
                raise ConfigurationError(f"Configuration load failed: {str(e)}", "CONFIG_LOAD_FAILED")
        else:
            logger.info("No configuration file found, using defaults")
            self.config = MIAConfig()
        
        # Load environment variables
        self._load_env_overrides()
        
        # Validate configuration
        try:
            self.config.validate()
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {str(e)}", "CONFIG_VALIDATION_FAILED")
        
        logger.info("Configuration loaded successfully")
        return self.config
    
    def _find_config_file(self) -> Optional[Path]:
        """Find the first available configuration file."""
        for config_path in self._config_file_paths:
            if config_path.exists():
                return config_path
        return None
    
    def _load_config_file(self, config_file: Path) -> Dict[str, Any]:
        """Load configuration from JSON or YAML file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() == '.json':
                    return json.load(f)
                elif config_file.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                else:
                    raise ConfigurationError(f"Unsupported config file format: {config_file.suffix}", 
                                           "UNSUPPORTED_FORMAT")
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in config file: {str(e)}", "INVALID_JSON")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in config file: {str(e)}", "INVALID_YAML")
        except Exception as e:
            raise ConfigurationError(f"Error reading config file: {str(e)}", "FILE_READ_ERROR")
    
    def _create_config_from_dict(self, config_data: Dict[str, Any]) -> MIAConfig:
        """Create MIAConfig from dictionary."""
        try:
            return MIAConfig(
                llm=LLMConfig(**config_data.get('llm', {})),
                audio=AudioConfig(**config_data.get('audio', {})),
                vision=VisionConfig(**config_data.get('vision', {})),
                memory=MemoryConfig(**config_data.get('memory', {})),
                security=SecurityConfig(**config_data.get('security', {})),
                system=SystemConfig(**config_data.get('system', {}))
            )
        except TypeError as e:
            raise ConfigurationError(f"Invalid configuration structure: {str(e)}", "INVALID_STRUCTURE")
    
    def _load_env_overrides(self):
        """Load environment variable overrides."""
        if not self.config:
            return
        
        env_mappings = {
            'MIA_LLM_PROVIDER': ('llm', 'provider'),
            'MIA_LLM_MODEL_ID': ('llm', 'model_id'),
            'MIA_LLM_API_KEY': ('llm', 'api_key'),
            'MIA_LLM_URL': ('llm', 'url'),
            'MIA_AUDIO_ENABLED': ('audio', 'enabled'),
            'MIA_VISION_ENABLED': ('vision', 'enabled'),
            'MIA_MEMORY_ENABLED': ('memory', 'enabled'),
            'MIA_SECURITY_ENABLED': ('security', 'enabled'),
            'MIA_SYSTEM_DEBUG': ('system', 'debug'),
            'MIA_SYSTEM_LOG_LEVEL': ('system', 'log_level'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                section_obj = getattr(self.config, section)
                
                # Type conversion
                if key in ['enabled', 'debug']:
                    value = value.lower() in ['true', '1', 'yes', 'on']
                elif key in ['max_tokens', 'timeout', 'sample_rate', 'chunk_size']:
                    value = int(value)
                elif key in ['temperature', 'input_threshold', 'similarity_threshold']:
                    value = float(value)
                
                setattr(section_obj, key, value)
                logger.debug(f"Environment override: {env_var} = {value}")
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        if not self.config:
            raise ConfigurationError("No configuration to save", "NO_CONFIG")
        
        if not config_path:
            config_path = str(self.config_dir / "config.json")
        
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dict = self._config_to_dict()
            
            with open(config_file, 'w', encoding='utf-8') as f:
                if config_file.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                elif config_file.suffix.lower() in ['.yaml', '.yml']:
                    yaml.safe_dump(config_dict, f, default_flow_style=False)
                else:
                    raise ConfigurationError(f"Unsupported save format: {config_file.suffix}", 
                                           "UNSUPPORTED_SAVE_FORMAT")
            
            logger.info(f"Configuration saved to {config_file}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {str(e)}", "CONFIG_SAVE_FAILED")
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert MIAConfig to dictionary."""
        if not self.config:
            return {}
        
        return {
            'llm': {
                'provider': self.config.llm.provider,
                'model_id': self.config.llm.model_id,
                'api_key': self.config.llm.api_key,
                'url': self.config.llm.url,
                'max_tokens': self.config.llm.max_tokens,
                'temperature': self.config.llm.temperature,
                'timeout': self.config.llm.timeout
            },
            'audio': {
                'enabled': self.config.audio.enabled,
                'sample_rate': self.config.audio.sample_rate,
                'chunk_size': self.config.audio.chunk_size,
                'device_id': self.config.audio.device_id,
                'input_threshold': self.config.audio.input_threshold,
                'speech_model': self.config.audio.speech_model,
                'tts_enabled': self.config.audio.tts_enabled
            },
            'vision': {
                'enabled': self.config.vision.enabled,
                'model': self.config.vision.model,
                'device': self.config.vision.device,
                'max_image_size': self.config.vision.max_image_size,
                'supported_formats': self.config.vision.supported_formats
            },
            'memory': {
                'enabled': self.config.memory.enabled,
                'vector_db_path': self.config.memory.vector_db_path,
                'max_memory_size': self.config.memory.max_memory_size,
                'embedding_dimension': self.config.memory.embedding_dimension,
                'similarity_threshold': self.config.memory.similarity_threshold
            },
            'security': {
                'enabled': self.config.security.enabled,
                'max_file_size': self.config.security.max_file_size,
                'allowed_file_types': self.config.security.allowed_file_types,
                'blocked_commands': self.config.security.blocked_commands,
                'max_command_length': self.config.security.max_command_length,
                'audit_logging': self.config.security.audit_logging
            },
            'system': {
                'debug': self.config.system.debug,
                'log_level': self.config.system.log_level,
                'max_workers': self.config.system.max_workers,
                'request_timeout': self.config.system.request_timeout,
                'retry_attempts': self.config.system.retry_attempts,
                'cache_enabled': self.config.system.cache_enabled,
                'cache_ttl': self.config.system.cache_ttl
            }
        }
    
    def get_config(self) -> Optional[MIAConfig]:
        """Get current configuration."""
        return self.config
    
    def update_config(self, section: str, key: str, value: Any) -> None:
        """Update a specific configuration value."""
        if not self.config:
            raise ConfigurationError("No configuration loaded", "NO_CONFIG")
        
        if not hasattr(self.config, section):
            raise ConfigurationError(f"Unknown configuration section: {section}", "UNKNOWN_SECTION")
        
        section_obj = getattr(self.config, section)
        if not hasattr(section_obj, key):
            raise ConfigurationError(f"Unknown configuration key: {key}", "UNKNOWN_KEY")
        
        setattr(section_obj, key, value)
        
        # Re-validate configuration
        try:
            self.config.validate()
        except ValidationError as e:
            raise ConfigurationError(f"Configuration update validation failed: {str(e)}", 
                                   "UPDATE_VALIDATION_FAILED")
        
        logger.info(f"Configuration updated: {section}.{key} = {value}")
    
    def validate_config(self) -> bool:
        """Validate current configuration."""
        if not self.config:
            raise ConfigurationError("No configuration loaded", "NO_CONFIG")
        
        try:
            self.config.validate()
            return True
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise ConfigurationError(f"Configuration validation failed: {str(e)}", "VALIDATION_FAILED")

# Global configuration manager instance
config_manager = ConfigManager()
