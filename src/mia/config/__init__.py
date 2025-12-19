"""
MIA Config Module.

This module provides unified configuration management
with validation, environment support, and hot-reloading.
"""

from .config_manager import (
    ConfigManager,
    MIAConfig,
    ConfigEnvironment,
    ConfigSource,
    LogLevel,
    get_config,
    get_config_manager,
    init_config,
    export_schema,
    generate_example_config,
)

# Re-export subsystem configs if Pydantic is available
try:
    from .config_manager import (
        LLMConfig,
        OpenAIConfig,
        AnthropicConfig,
        OllamaConfig,
        GroqConfig,
        MemoryConfig,
        VisionConfig,
        AudioConfig,
        SecurityConfig,
        WebUIConfig,
        LoggingConfig,
        AgentConfig,
    )
    
    __all__ = [
        "ConfigManager",
        "MIAConfig",
        "ConfigEnvironment",
        "ConfigSource",
        "LogLevel",
        "get_config",
        "get_config_manager",
        "init_config",
        "export_schema",
        "generate_example_config",
        "LLMConfig",
        "OpenAIConfig",
        "AnthropicConfig",
        "OllamaConfig",
        "GroqConfig",
        "MemoryConfig",
        "VisionConfig",
        "AudioConfig",
        "SecurityConfig",
        "WebUIConfig",
        "LoggingConfig",
        "AgentConfig",
    ]
except ImportError:
    __all__ = [
        "ConfigManager",
        "MIAConfig",
        "ConfigEnvironment",
        "ConfigSource",
        "LogLevel",
        "get_config",
        "get_config_manager",
        "init_config",
    ]
