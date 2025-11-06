"""
Configuration Management System for M.I.A
Provides centralized configuration loading, validation, and management.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from .error_handler import global_error_handler, with_error_handling
from .exceptions import ConfigurationError, ValidationError

logger = logging.getLogger(__name__)

SUPPORTED_LLM_PROVIDERS = {
    "openai",
    "ollama",
    "huggingface",
    "anthropic",
    "gemini",
    "groq",
    "grok",
    "local",
    "nanochat",
    "minimax",
}


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""

    provider: str = "ollama"
    model_id: str = "gpt-oss:latest"
    api_key: Optional[str] = None
    url: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout: int = 30

    def validate(self) -> None:
        """Validate LLM configuration."""
        if not self.provider:
            raise ValidationError("Provider cannot be empty", "EMPTY_PROVIDER")

        if self.provider not in SUPPORTED_LLM_PROVIDERS:
            raise ValidationError(
                f"Unsupported provider: {self.provider}",
                "UNSUPPORTED_PROVIDER",
            )

        if self.max_tokens <= 0:
            raise ValidationError(
                "Max tokens must be positive", "INVALID_MAX_TOKENS"
            )

        if not 0.0 <= self.temperature <= 2.0:
            raise ValidationError(
                "Temperature must be between 0.0 and 2.0",
                "INVALID_TEMPERATURE",
            )

        if self.timeout <= 0:
            raise ValidationError(
                "Timeout must be positive", "INVALID_TIMEOUT"
            )


@dataclass
class LLMProfileConfig:
    """Named LLM profile that can override base configuration and provide metadata."""

    name: str = "default"
    label: Optional[str] = None
    description: Optional[str] = None
    provider: Optional[str] = None
    model_id: Optional[str] = None
    api_key: Optional[str] = None
    url: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    timeout: Optional[int] = None
    system_prompt: Optional[str] = None
    stream: Optional[bool] = None
    scopes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ValidationError(
                "LLM profile must have a name", "LLM_PROFILE_NO_NAME"
            )

        if self.provider and self.provider not in SUPPORTED_LLM_PROVIDERS:
            raise ValidationError(
                f"Unsupported provider in profile '{self.name}': {self.provider}",
                "LLM_PROFILE_PROVIDER_INVALID",
            )

        if self.temperature is not None and not 0.0 <= self.temperature <= 2.0:
            raise ValidationError(
                f"Temperature for profile '{self.name}' must be between 0.0 and 2.0",
                "LLM_PROFILE_TEMPERATURE_INVALID",
            )

        if self.timeout is not None and self.timeout <= 0:
            raise ValidationError(
                f"Timeout for profile '{self.name}' must be positive",
                "LLM_PROFILE_TIMEOUT_INVALID",
            )

    def apply_overrides(self, base: LLMConfig) -> LLMConfig:
        """Return a new LLMConfig using profile overrides on top of base."""
        return LLMConfig(
            provider=self.provider or base.provider,
            model_id=self.model_id or base.model_id,
            api_key=self.api_key or base.api_key,
            url=self.url or base.url,
            max_tokens=self.max_tokens or base.max_tokens,
            temperature=(
                self.temperature
                if self.temperature is not None
                else base.temperature
            ),
            timeout=self.timeout or base.timeout,
        )


SUPPORTED_TTS_PROVIDERS = {"local", "nanochat", "minimax", "openai", "custom"}


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
    tts_provider: str = "local"
    tts_model_id: Optional[str] = None
    tts_api_key: Optional[str] = None
    tts_url: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model_id: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_url: Optional[str] = None
    vad_enabled: bool = False
    vad_aggressiveness: int = 2
    vad_frame_duration_ms: int = 30
    vad_silence_duration_ms: int = 600
    playback_enabled: bool = True
    playback_device_id: Optional[int] = None
    push_to_talk: bool = False
    hotword_enabled: bool = False
    hotword: str = "mia"
    hotword_sensitivity: float = 0.6
    hotword_timeout: float = 15.0

    def validate(self) -> None:
        """Validate audio configuration."""
        if self.sample_rate <= 0:
            raise ValidationError(
                "Sample rate must be positive", "INVALID_SAMPLE_RATE"
            )

        if self.chunk_size <= 0:
            raise ValidationError(
                "Chunk size must be positive", "INVALID_CHUNK_SIZE"
            )

        if not 0.0 <= self.input_threshold <= 1.0:
            raise ValidationError(
                "Input threshold must be between 0.0 and 1.0",
                "INVALID_THRESHOLD",
            )

        if (
            self.tts_provider
            and self.tts_provider not in SUPPORTED_TTS_PROVIDERS
        ):
            raise ValidationError(
                f"Unsupported TTS provider: {self.tts_provider}",
                "UNSUPPORTED_TTS_PROVIDER",
            )

        if (
            self.llm_provider
            and self.llm_provider not in SUPPORTED_LLM_PROVIDERS
        ):
            raise ValidationError(
                f"Unsupported audio LLM provider: {self.llm_provider}",
                "UNSUPPORTED_AUDIO_LLM_PROVIDER",
            )

        if not 0 <= self.vad_aggressiveness <= 3:
            raise ValidationError(
                "VAD aggressiveness must be between 0 and 3",
                "INVALID_VAD_AGGRESSIVENESS",
            )

        if self.vad_frame_duration_ms not in {10, 20, 30}:
            raise ValidationError(
                "VAD frame duration must be 10, 20, or 30 ms",
                "INVALID_VAD_FRAME_DURATION",
            )

        if self.vad_silence_duration_ms <= 0:
            raise ValidationError(
                "VAD silence duration must be positive",
                "INVALID_VAD_SILENCE_DURATION",
            )

        if (
            self.hotword_sensitivity is not None
            and not 0.0 <= self.hotword_sensitivity <= 1.0
        ):
            raise ValidationError(
                "Hotword sensitivity must be between 0.0 and 1.0",
                "INVALID_HOTWORD_SENSITIVITY",
            )

        if self.hotword_timeout is not None and self.hotword_timeout <= 0:
            raise ValidationError(
                "Hotword timeout must be positive", "INVALID_HOTWORD_TIMEOUT"
            )


@dataclass
class VisionConfig:
    """Configuration for vision processing."""

    enabled: bool = True
    model: str = "openai/clip-vit-base-patch32"
    device: str = "auto"
    max_image_size: int = 1024
    supported_formats: List[str] = field(
        default_factory=lambda: ["jpg", "jpeg", "png", "bmp", "gif"]
    )

    def validate(self) -> None:
        """Validate vision configuration."""
        if self.max_image_size <= 0:
            raise ValidationError(
                "Max image size must be positive", "INVALID_IMAGE_SIZE"
            )

        if not self.supported_formats:
            raise ValidationError(
                "Supported formats cannot be empty", "EMPTY_FORMATS"
            )

        if self.device not in ["auto", "cpu", "cuda", "mps"]:
            raise ValidationError(
                f"Unsupported device: {self.device}", "UNSUPPORTED_DEVICE"
            )


@dataclass
class MemoryConfig:
    """Configuration for memory systems."""

    enabled: bool = True
    vector_enabled: bool = True
    graph_enabled: bool = True
    long_term_enabled: bool = True
    vector_db_path: str = "memory/"
    max_memory_size: int = 10000
    embedding_dimension: int = 768
    similarity_threshold: float = 0.7
    max_results: int = 5

    def validate(self) -> None:
        """Validate memory configuration."""
        if self.max_memory_size <= 0:
            raise ValidationError(
                "Max memory size must be positive", "INVALID_MEMORY_SIZE"
            )

        if self.embedding_dimension <= 0:
            raise ValidationError(
                "Embedding dimension must be positive", "INVALID_EMBEDDING_DIM"
            )

        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValidationError(
                "Similarity threshold must be between 0.0 and 1.0",
                "INVALID_THRESHOLD",
            )

        if not self.vector_db_path:
            raise ValidationError(
                "Vector DB path cannot be empty", "INVALID_MEMORY_PATH"
            )

        if self.max_results <= 0:
            raise ValidationError(
                "Max results must be positive", "INVALID_MAX_RESULTS"
            )


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""

    enabled: bool = True
    work_dir: str = "sandbox_runs"
    log_dir: str = "logs/sandbox"
    runtime: str = "auto"
    max_memory_mb: int = 256
    timeout_ms: int = 10_000
    fuel: Optional[int] = None

    def validate(self) -> None:
        if self.max_memory_mb <= 0:
            raise ValidationError(
                "Sandbox memory must be positive", "INVALID_SANDBOX_MEMORY"
            )
        if self.timeout_ms <= 0:
            raise ValidationError(
                "Sandbox timeout must be positive", "INVALID_SANDBOX_TIMEOUT"
            )
        if self.runtime not in {"auto", "wasmtime", "wasmer"}:
            raise ValidationError(
                "Sandbox runtime must be auto, wasmtime or wasmer",
                "INVALID_SANDBOX_RUNTIME",
            )


@dataclass
class DocumentConfig:
    """Configuration for document generation."""

    template_dir: str = "templates/documents"
    output_dir: str = "output/documents"
    default_template: str = "proposal"

    def validate(self) -> None:
        if not self.template_dir:
            raise ValidationError(
                "Template directory cannot be empty", "INVALID_TEMPLATE_DIR"
            )
        if not self.output_dir:
            raise ValidationError(
                "Output directory cannot be empty", "INVALID_OUTPUT_DIR"
            )


@dataclass
class TelegramConfig:
    """Configuration for Telegram messaging."""

    enabled: bool = False
    api_id: Optional[int] = None
    api_hash: Optional[str] = None
    bot_token: Optional[str] = None
    phone_number: Optional[str] = None
    session_dir: str = "sessions"
    session_name: str = "mia_telegram"
    default_peer: Optional[str] = None
    parse_mode: str = "markdown"
    request_timeout: int = 30

    def validate(self) -> None:
        if self.request_timeout <= 0:
            raise ValidationError(
                "Telegram request timeout must be positive",
                "INVALID_TELEGRAM_TIMEOUT",
            )
        if not self.enabled:
            return
        if not self.api_id or not self.api_hash:
            raise ValidationError(
                "Telegram API ID and API hash are required when Telegram is enabled",
                "MISSING_TELEGRAM_API",
            )
        if not (self.bot_token or self.phone_number):
            raise ValidationError(
                "Telegram requires either a bot token or a phone number when enabled",
                "MISSING_TELEGRAM_AUTH",
            )


@dataclass
class SecurityConfig:
    """Configuration for security settings."""

    enabled: bool = True
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: List[str] = field(
        default_factory=lambda: ["txt", "md", "py", "json", "yaml"]
    )
    blocked_commands: List[str] = field(
        default_factory=lambda: ["rm -rf", "del /f", "format", "fdisk"]
    )
    max_command_length: int = 1000
    audit_logging: bool = True
    require_consent: bool = True
    scopes: Dict[str, List[str]] = field(default_factory=dict)
    default_allow: List[str] = field(default_factory=list)
    api_enabled: bool = False
    api_keys: Dict[str, List[str]] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate security configuration."""
        if self.max_file_size <= 0:
            raise ValidationError(
                "Max file size must be positive", "INVALID_FILE_SIZE"
            )

        if self.max_command_length <= 0:
            raise ValidationError(
                "Max command length must be positive", "INVALID_COMMAND_LENGTH"
            )

        for key, scope_list in self.api_keys.items():
            if not key:
                raise ValidationError(
                    "API key identifier cannot be empty", "INVALID_API_KEY"
                )
            if scope_list and not all(
                isinstance(scope, str) for scope in scope_list
            ):
                raise ValidationError(
                    "API key scopes must be strings", "INVALID_API_SCOPE"
                )


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
            raise ValidationError(
                f"Invalid log level: {self.log_level}", "INVALID_LOG_LEVEL"
            )

        if self.max_workers <= 0:
            raise ValidationError(
                "Max workers must be positive", "INVALID_MAX_WORKERS"
            )

        if self.request_timeout <= 0:
            raise ValidationError(
                "Request timeout must be positive", "INVALID_TIMEOUT"
            )

        if self.retry_attempts < 0:
            raise ValidationError(
                "Retry attempts cannot be negative", "INVALID_RETRY_ATTEMPTS"
            )


@dataclass
class MIAConfig:
    """Main configuration class for M.I.A."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    llm_profiles: Dict[str, LLMProfileConfig] = field(default_factory=dict)
    default_llm_profile: Optional[str] = None
    audio: AudioConfig = field(default_factory=AudioConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    documents: DocumentConfig = field(default_factory=DocumentConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)

    def validate(self) -> None:
        """Validate all configuration sections."""
        self.llm.validate()
        for profile in self.llm_profiles.values():
            profile.validate()
        if (
            self.default_llm_profile
            and self.default_llm_profile not in self.llm_profiles
        ):
            raise ValidationError(
                f"Default LLM profile '{self.default_llm_profile}' not found",
                "DEFAULT_LLM_PROFILE_UNKNOWN",
            )
        self.audio.validate()
        self.vision.validate()
        self.memory.validate()
        self.security.validate()
        self.system.validate()
        self.sandbox.validate()
        self.documents.validate()
        self.telegram.validate()


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
        self.active_llm_profile: Optional[str] = None
        self._config_file_paths = [
            self.config_dir / "config.json",
            self.config_dir / "config.yaml",
            self.config_dir / "config.yml",
            Path("config.json"),
            Path("config.yaml"),
            Path("config.yml"),
        ]
        # Load .env early if present (non-fatal if missing)
        try:
            from dotenv import load_dotenv  # type: ignore

            env_path = self.config_dir / ".env"
            if env_path.exists():
                load_dotenv(dotenv_path=str(env_path))
            else:
                # Also load project root .env if exists
                if os.path.exists(".env"):
                    load_dotenv(dotenv_path=".env")
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
                raise ConfigurationError(
                    f"Configuration file not found: {config_path}",
                    "CONFIG_NOT_FOUND",
                )
        elif self._explicit_config_file:
            config_file = self._explicit_config_file
        else:
            config_file = self._find_config_file()

        if config_file:
            try:
                config_data = self._load_config_file(config_file)
                self.config = self._create_config_from_dict(config_data)
            except Exception as e:
                logger.error(
                    f"Failed to load configuration from {config_file}: {e}"
                )
                raise ConfigurationError(
                    f"Configuration load failed: {str(e)}",
                    "CONFIG_LOAD_FAILED",
                )
        else:
            logger.info("No configuration file found, using defaults")
            self.config = MIAConfig()

        # Load environment variables
        self._load_env_overrides()

        # Validate configuration
        try:
            self.config.validate()
        except ValidationError as e:
            raise ConfigurationError(
                f"Configuration validation failed: {str(e)}",
                "CONFIG_VALIDATION_FAILED",
            )

        self.active_llm_profile = self.config.default_llm_profile
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
            with open(config_file, "r", encoding="utf-8") as f:
                if config_file.suffix.lower() == ".json":
                    return json.load(f)
                elif config_file.suffix.lower() in [".yaml", ".yml"]:
                    return yaml.safe_load(f) or {}
                else:
                    raise ConfigurationError(
                        f"Unsupported config file format: {config_file.suffix}",
                        "UNSUPPORTED_FORMAT",
                    )
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Invalid JSON in config file: {str(e)}", "INVALID_JSON"
            )
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in config file: {str(e)}", "INVALID_YAML"
            )
        except Exception as e:
            raise ConfigurationError(
                f"Error reading config file: {str(e)}", "FILE_READ_ERROR"
            )

    def _create_config_from_dict(
        self, config_data: Dict[str, Any]
    ) -> MIAConfig:
        """Create MIAConfig from dictionary."""
        try:
            profiles: Dict[str, LLMProfileConfig] = {}
            raw_profiles = config_data.get("llm_profiles", {})

            def _iter_profiles(
                source: Any,
            ) -> Iterable[tuple[str, Dict[str, Any]]]:
                if isinstance(source, dict):
                    for name, payload in source.items():
                        if isinstance(payload, dict):
                            yield str(name), payload
                elif isinstance(source, list):
                    for entry in source:
                        if isinstance(entry, dict) and entry.get("name"):
                            yield str(entry["name"]), entry

            for profile_name, payload in _iter_profiles(raw_profiles):
                profiles[profile_name] = LLMProfileConfig(
                    name=profile_name,
                    **{k: v for k, v in payload.items() if k != "name"},
                )

            return MIAConfig(
                llm=LLMConfig(**config_data.get("llm", {})),
                llm_profiles=profiles,
                default_llm_profile=config_data.get("default_llm_profile"),
                audio=AudioConfig(**config_data.get("audio", {})),
                vision=VisionConfig(**config_data.get("vision", {})),
                memory=MemoryConfig(**config_data.get("memory", {})),
                security=SecurityConfig(**config_data.get("security", {})),
                system=SystemConfig(**config_data.get("system", {})),
                sandbox=SandboxConfig(**config_data.get("sandbox", {})),
                documents=DocumentConfig(**config_data.get("documents", {})),
                telegram=TelegramConfig(**config_data.get("telegram", {})),
            )
        except TypeError as e:
            raise ConfigurationError(
                f"Invalid configuration structure: {str(e)}",
                "INVALID_STRUCTURE",
            )

    def _load_env_overrides(self):
        """Load environment variable overrides."""
        if not self.config:
            return

        env_mappings = {
            "MIA_LLM_PROVIDER": ("llm", "provider"),
            "MIA_LLM_MODEL_ID": ("llm", "model_id"),
            "MIA_LLM_API_KEY": ("llm", "api_key"),
            "MIA_LLM_URL": ("llm", "url"),
            "MIA_AUDIO_ENABLED": ("audio", "enabled"),
            "MIA_AUDIO_TTS_PROVIDER": ("audio", "tts_provider"),
            "MIA_AUDIO_TTS_MODEL_ID": ("audio", "tts_model_id"),
            "MIA_AUDIO_TTS_API_KEY": ("audio", "tts_api_key"),
            "MIA_AUDIO_TTS_URL": ("audio", "tts_url"),
            "MIA_AUDIO_LLM_PROVIDER": ("audio", "llm_provider"),
            "MIA_AUDIO_LLM_MODEL_ID": ("audio", "llm_model_id"),
            "MIA_AUDIO_LLM_API_KEY": ("audio", "llm_api_key"),
            "MIA_AUDIO_LLM_URL": ("audio", "llm_url"),
            "MIA_AUDIO_VAD_ENABLED": ("audio", "vad_enabled"),
            "MIA_AUDIO_VAD_AGGRESSIVENESS": ("audio", "vad_aggressiveness"),
            "MIA_AUDIO_VAD_FRAME_MS": ("audio", "vad_frame_duration_ms"),
            "MIA_AUDIO_VAD_SILENCE_MS": ("audio", "vad_silence_duration_ms"),
            "MIA_AUDIO_PLAYBACK_ENABLED": ("audio", "playback_enabled"),
            "MIA_AUDIO_PLAYBACK_DEVICE_ID": ("audio", "playback_device_id"),
            "MIA_AUDIO_PUSH_TO_TALK": ("audio", "push_to_talk"),
            "MIA_AUDIO_HOTWORD_ENABLED": ("audio", "hotword_enabled"),
            "MIA_AUDIO_HOTWORD": ("audio", "hotword"),
            "MIA_AUDIO_HOTWORD_SENSITIVITY": ("audio", "hotword_sensitivity"),
            "MIA_AUDIO_HOTWORD_TIMEOUT": ("audio", "hotword_timeout"),
            "MIA_VISION_ENABLED": ("vision", "enabled"),
            "MIA_MEMORY_ENABLED": ("memory", "enabled"),
            "MIA_MEMORY_VECTOR_ENABLED": ("memory", "vector_enabled"),
            "MIA_MEMORY_GRAPH_ENABLED": ("memory", "graph_enabled"),
            "MIA_MEMORY_LONG_TERM_ENABLED": ("memory", "long_term_enabled"),
            "MIA_MEMORY_PATH": ("memory", "vector_db_path"),
            "MIA_MEMORY_MAX_RESULTS": ("memory", "max_results"),
            "MIA_SECURITY_ENABLED": ("security", "enabled"),
            "MIA_SECURITY_REQUIRE_CONSENT": ("security", "require_consent"),
            "MIA_SECURITY_API_ENABLED": ("security", "api_enabled"),
            "MIA_SYSTEM_DEBUG": ("system", "debug"),
            "MIA_SYSTEM_LOG_LEVEL": ("system", "log_level"),
            "MIA_SANDBOX_ENABLED": ("sandbox", "enabled"),
            "MIA_SANDBOX_WORKDIR": ("sandbox", "work_dir"),
            "MIA_SANDBOX_LOGDIR": ("sandbox", "log_dir"),
            "MIA_SANDBOX_RUNTIME": ("sandbox", "runtime"),
            "MIA_SANDBOX_MEMORY_MB": ("sandbox", "max_memory_mb"),
            "MIA_SANDBOX_TIMEOUT_MS": ("sandbox", "timeout_ms"),
            "MIA_SANDBOX_FUEL": ("sandbox", "fuel"),
            "MIA_DOC_TEMPLATE_DIR": ("documents", "template_dir"),
            "MIA_DOC_OUTPUT_DIR": ("documents", "output_dir"),
            "MIA_DOC_DEFAULT_TEMPLATE": ("documents", "default_template"),
            "MIA_TELEGRAM_ENABLED": ("telegram", "enabled"),
            "TELEGRAM_ENABLED": ("telegram", "enabled"),
            "MIA_TELEGRAM_API_ID": ("telegram", "api_id"),
            "TELEGRAM_API_ID": ("telegram", "api_id"),
            "MIA_TELEGRAM_API_HASH": ("telegram", "api_hash"),
            "TELEGRAM_API_HASH": ("telegram", "api_hash"),
            "MIA_TELEGRAM_BOT_TOKEN": ("telegram", "bot_token"),
            "TELEGRAM_BOT_TOKEN": ("telegram", "bot_token"),
            "MIA_TELEGRAM_PHONE": ("telegram", "phone_number"),
            "TELEGRAM_PHONE": ("telegram", "phone_number"),
            "MIA_TELEGRAM_SESSION_DIR": ("telegram", "session_dir"),
            "TELEGRAM_SESSION_DIR": ("telegram", "session_dir"),
            "MIA_TELEGRAM_SESSION_NAME": ("telegram", "session_name"),
            "TELEGRAM_SESSION_NAME": ("telegram", "session_name"),
            "MIA_TELEGRAM_DEFAULT_PEER": ("telegram", "default_peer"),
            "TELEGRAM_DEFAULT_PEER": ("telegram", "default_peer"),
            "MIA_TELEGRAM_PARSE_MODE": ("telegram", "parse_mode"),
            "TELEGRAM_PARSE_MODE": ("telegram", "parse_mode"),
            "MIA_TELEGRAM_REQUEST_TIMEOUT": ("telegram", "request_timeout"),
            "TELEGRAM_REQUEST_TIMEOUT": ("telegram", "request_timeout"),
        }

        default_profile_env = os.getenv("MIA_LLM_PROFILE") or os.getenv(
            "MIA_DEFAULT_LLM_PROFILE"
        )
        if default_profile_env:
            self.config.default_llm_profile = default_profile_env

        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                section_obj = getattr(self.config, section)

                # Type conversion
                if (
                    key in ["enabled", "debug"]
                    or key.endswith("_enabled")
                    or key
                    in [
                        "push_to_talk",
                        "playback_enabled",
                        "require_consent",
                        "api_enabled",
                    ]
                ):
                    value = value.lower() in ["true", "1", "yes", "on"]
                elif key in [
                    "max_tokens",
                    "timeout",
                    "sample_rate",
                    "chunk_size",
                    "max_memory_mb",
                    "timeout_ms",
                    "request_timeout",
                    "api_id",
                    "vad_aggressiveness",
                    "vad_frame_duration_ms",
                    "vad_silence_duration_ms",
                    "max_results",
                    "playback_device_id",
                ]:
                    try:
                        value = int(value)
                    except ValueError:
                        logger.warning(
                            "Invalid integer for %s: %s", env_var, value
                        )
                        continue
                elif key in [
                    "temperature",
                    "input_threshold",
                    "similarity_threshold",
                    "hotword_sensitivity",
                    "hotword_timeout",
                ]:
                    value = float(value)
                elif key == "fuel":
                    try:
                        value = int(value)
                    except ValueError:
                        logger.warning(
                            "Invalid value for MIA_SANDBOX_FUEL: %s", value
                        )
                        value = None

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

            with open(config_file, "w", encoding="utf-8") as f:
                if config_file.suffix.lower() == ".json":
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                elif config_file.suffix.lower() in [".yaml", ".yml"]:
                    yaml.safe_dump(config_dict, f, default_flow_style=False)
                else:
                    raise ConfigurationError(
                        f"Unsupported save format: {config_file.suffix}",
                        "UNSUPPORTED_SAVE_FORMAT",
                    )

            logger.info(f"Configuration saved to {config_file}")

        except Exception as e:
            raise ConfigurationError(
                f"Failed to save configuration: {str(e)}", "CONFIG_SAVE_FAILED"
            )

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert MIAConfig to dictionary."""
        if not self.config:
            return {}

        profiles_dict: Dict[str, Dict[str, Any]] = {}
        for name, profile in self.config.llm_profiles.items():
            profiles_dict[name] = {
                "label": profile.label,
                "description": profile.description,
                "provider": profile.provider,
                "model_id": profile.model_id,
                "api_key": profile.api_key,
                "url": profile.url,
                "max_tokens": profile.max_tokens,
                "temperature": profile.temperature,
                "timeout": profile.timeout,
                "system_prompt": profile.system_prompt,
                "stream": profile.stream,
                "scopes": profile.scopes,
                "metadata": profile.metadata,
            }

        return {
            "llm": {
                "provider": self.config.llm.provider,
                "model_id": self.config.llm.model_id,
                "api_key": self.config.llm.api_key,
                "url": self.config.llm.url,
                "max_tokens": self.config.llm.max_tokens,
                "temperature": self.config.llm.temperature,
                "timeout": self.config.llm.timeout,
            },
            "llm_profiles": profiles_dict,
            "default_llm_profile": self.config.default_llm_profile,
            "audio": {
                "enabled": self.config.audio.enabled,
                "sample_rate": self.config.audio.sample_rate,
                "chunk_size": self.config.audio.chunk_size,
                "device_id": self.config.audio.device_id,
                "input_threshold": self.config.audio.input_threshold,
                "speech_model": self.config.audio.speech_model,
                "tts_enabled": self.config.audio.tts_enabled,
                "tts_provider": self.config.audio.tts_provider,
                "tts_model_id": self.config.audio.tts_model_id,
                "tts_api_key": self.config.audio.tts_api_key,
                "tts_url": self.config.audio.tts_url,
                "llm_provider": self.config.audio.llm_provider,
                "llm_model_id": self.config.audio.llm_model_id,
                "llm_api_key": self.config.audio.llm_api_key,
                "llm_url": self.config.audio.llm_url,
                "vad_enabled": self.config.audio.vad_enabled,
                "vad_aggressiveness": self.config.audio.vad_aggressiveness,
                "vad_frame_duration_ms": self.config.audio.vad_frame_duration_ms,
                "vad_silence_duration_ms": self.config.audio.vad_silence_duration_ms,
                "playback_enabled": self.config.audio.playback_enabled,
                "playback_device_id": self.config.audio.playback_device_id,
                "push_to_talk": self.config.audio.push_to_talk,
                "hotword_enabled": self.config.audio.hotword_enabled,
                "hotword": self.config.audio.hotword,
                "hotword_sensitivity": self.config.audio.hotword_sensitivity,
                "hotword_timeout": self.config.audio.hotword_timeout,
            },
            "vision": {
                "enabled": self.config.vision.enabled,
                "model": self.config.vision.model,
                "device": self.config.vision.device,
                "max_image_size": self.config.vision.max_image_size,
                "supported_formats": self.config.vision.supported_formats,
            },
            "memory": {
                "enabled": self.config.memory.enabled,
                "vector_enabled": self.config.memory.vector_enabled,
                "graph_enabled": self.config.memory.graph_enabled,
                "long_term_enabled": self.config.memory.long_term_enabled,
                "vector_db_path": self.config.memory.vector_db_path,
                "max_memory_size": self.config.memory.max_memory_size,
                "embedding_dimension": self.config.memory.embedding_dimension,
                "similarity_threshold": self.config.memory.similarity_threshold,
                "max_results": self.config.memory.max_results,
            },
            "security": {
                "enabled": self.config.security.enabled,
                "max_file_size": self.config.security.max_file_size,
                "allowed_file_types": self.config.security.allowed_file_types,
                "blocked_commands": self.config.security.blocked_commands,
                "max_command_length": self.config.security.max_command_length,
                "audit_logging": self.config.security.audit_logging,
                "require_consent": self.config.security.require_consent,
                "scopes": self.config.security.scopes,
                "default_allow": self.config.security.default_allow,
                "api_enabled": self.config.security.api_enabled,
                "api_keys": self.config.security.api_keys,
            },
            "system": {
                "debug": self.config.system.debug,
                "log_level": self.config.system.log_level,
                "max_workers": self.config.system.max_workers,
                "request_timeout": self.config.system.request_timeout,
                "retry_attempts": self.config.system.retry_attempts,
                "cache_enabled": self.config.system.cache_enabled,
                "cache_ttl": self.config.system.cache_ttl,
            },
            "sandbox": {
                "enabled": self.config.sandbox.enabled,
                "work_dir": self.config.sandbox.work_dir,
                "log_dir": self.config.sandbox.log_dir,
                "runtime": self.config.sandbox.runtime,
                "max_memory_mb": self.config.sandbox.max_memory_mb,
                "timeout_ms": self.config.sandbox.timeout_ms,
                "fuel": self.config.sandbox.fuel,
            },
            "documents": {
                "template_dir": self.config.documents.template_dir,
                "output_dir": self.config.documents.output_dir,
                "default_template": self.config.documents.default_template,
            },
            "telegram": {
                "enabled": self.config.telegram.enabled,
                "api_id": self.config.telegram.api_id,
                "api_hash": self.config.telegram.api_hash,
                "bot_token": self.config.telegram.bot_token,
                "phone_number": self.config.telegram.phone_number,
                "session_dir": self.config.telegram.session_dir,
                "session_name": self.config.telegram.session_name,
                "default_peer": self.config.telegram.default_peer,
                "parse_mode": self.config.telegram.parse_mode,
                "request_timeout": self.config.telegram.request_timeout,
            },
        }

    def list_llm_profiles(self) -> List[str]:
        if not self.config:
            return []
        return list(self.config.llm_profiles.keys())

    def get_llm_profile(
        self, name: Optional[str]
    ) -> Optional[LLMProfileConfig]:
        if not self.config or not name:
            return None
        return self.config.llm_profiles.get(name)

    def activate_llm_profile(
        self, name: Optional[str]
    ) -> Optional[LLMProfileConfig]:
        if not self.config:
            raise ConfigurationError("No configuration loaded", "NO_CONFIG")
        if name is None:
            self.active_llm_profile = None
            return None
        profile = self.config.llm_profiles.get(name)
        if not profile:
            raise ConfigurationError(
                f"Unknown LLM profile: {name}", "UNKNOWN_LLM_PROFILE"
            )
        self.active_llm_profile = name
        return profile

    def resolve_llm_config(
        self, profile_name: Optional[str] = None
    ) -> Tuple[LLMConfig, Optional[LLMProfileConfig]]:
        if not self.config:
            raise ConfigurationError("No configuration loaded", "NO_CONFIG")

        base = self.config.llm
        chosen = (
            profile_name
            or self.active_llm_profile
            or self.config.default_llm_profile
        )
        if chosen and chosen in self.config.llm_profiles:
            profile = self.config.llm_profiles[chosen]
            merged = profile.apply_overrides(base)
            return merged, profile
        return base, None

    def get_config(self) -> Optional[MIAConfig]:
        """Get current configuration."""
        return self.config

    def update_config(self, section: str, key: str, value: Any) -> None:
        """Update a specific configuration value."""
        if not self.config:
            raise ConfigurationError("No configuration loaded", "NO_CONFIG")

        if not hasattr(self.config, section):
            raise ConfigurationError(
                f"Unknown configuration section: {section}", "UNKNOWN_SECTION"
            )

        section_obj = getattr(self.config, section)
        if not hasattr(section_obj, key):
            raise ConfigurationError(
                f"Unknown configuration key: {key}", "UNKNOWN_KEY"
            )

        setattr(section_obj, key, value)

        # Re-validate configuration
        try:
            self.config.validate()
        except ValidationError as e:
            raise ConfigurationError(
                f"Configuration update validation failed: {str(e)}",
                "UPDATE_VALIDATION_FAILED",
            )

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
            raise ConfigurationError(
                f"Configuration validation failed: {str(e)}",
                "VALIDATION_FAILED",
            )


# Lazy-loaded global configuration manager instance
_config_manager_instance: Optional[ConfigManager] = None


def get_config_manager(config_dir: str = "config") -> ConfigManager:
    """Get the global configuration manager instance (lazy-loaded)."""
    global _config_manager_instance
    if _config_manager_instance is None:
        _config_manager_instance = ConfigManager(config_dir)
    return _config_manager_instance


# For backward compatibility, provide a lazy-loaded instance
config_manager = property(lambda self: get_config_manager())
