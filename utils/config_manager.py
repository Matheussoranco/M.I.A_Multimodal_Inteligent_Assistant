"""
Configuration validation and management for M.I.A.
"""
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


class ConfigManager:
    """Manages application configuration with validation."""
    
    def __init__(self, env_file: Optional[str] = None):
        self.env_file = env_file or ".env"
        self.config = {}
        self._load_config()
        self._validate_config()
    
    def _load_config(self):
        """Load configuration from environment and .env file."""
        # Load from .env file if it exists
        env_path = Path(self.env_file)
        if env_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
                logger.info(f"Loaded configuration from {env_path}")
            except ImportError:
                logger.warning("python-dotenv not installed, skipping .env file loading")
        
        # Core configuration with defaults
        self.config = {
            # LLM Configuration
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
            'gemini_api_key': os.getenv('GEMINI_API_KEY'),
            'ollama_url': os.getenv('OLLAMA_URL', 'http://localhost:11434'),
            
            # Audio Configuration
            'stt_model': os.getenv('STT_MODEL', 'openai/whisper-base.en'),
            'audio_device_index': int(os.getenv('AUDIO_DEVICE_INDEX', '0')),
            'audio_sample_rate': int(os.getenv('AUDIO_SAMPLE_RATE', '16000')),
            
            # Security Configuration
            'enable_file_access': os.getenv('ENABLE_FILE_ACCESS', 'false').lower() == 'true',
            'enable_system_commands': os.getenv('ENABLE_SYSTEM_COMMANDS', 'false').lower() == 'true',
            'enable_web_automation': os.getenv('ENABLE_WEB_AUTOMATION', 'true').lower() == 'true',
            
            # System Configuration
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'debug_mode': os.getenv('DEBUG_MODE', 'false').lower() == 'true',
            'cuda_device': os.getenv('CUDA_VISIBLE_DEVICES', '0'),
            
            # Memory Configuration
            'enable_long_term_memory': os.getenv('ENABLE_LONG_TERM_MEMORY', 'true').lower() == 'true',
            'memory_persistence_path': os.getenv('MEMORY_PERSISTENCE_PATH', './memory/'),
            'max_memory_entries': int(os.getenv('MAX_MEMORY_ENTRIES', '10000')),
        }
    
    def _validate_config(self):
        """Validate configuration and check requirements."""
        errors = []
        
        # Check for at least one LLM API key
        llm_keys = [
            self.config['openai_api_key'],
            self.config['anthropic_api_key'],
            self.config['gemini_api_key']
        ]
        
        if not any(llm_keys):
            logger.warning("No LLM API keys configured. Some features may be limited.")
        
        # Validate paths
        memory_path = Path(self.config['memory_persistence_path'])
        try:
            memory_path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            errors.append(f"Cannot create memory directory: {memory_path}")
        
        # Validate numeric values
        if self.config['audio_sample_rate'] not in [8000, 16000, 22050, 44100, 48000]:
            logger.warning(f"Unusual audio sample rate: {self.config['audio_sample_rate']}")
        
        if errors:
            raise ConfigurationError(f"Configuration errors: {'; '.join(errors)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM-specific configuration."""
        return {
            'openai_api_key': self.config['openai_api_key'],
            'anthropic_api_key': self.config['anthropic_api_key'],
            'gemini_api_key': self.config['gemini_api_key'],
            'ollama_url': self.config['ollama_url'],
        }
    
    def get_security_config(self) -> Dict[str, bool]:
        """Get security-specific configuration."""
        return {
            'enable_file_access': self.config['enable_file_access'],
            'enable_system_commands': self.config['enable_system_commands'],
            'enable_web_automation': self.config['enable_web_automation'],
        }
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self.config['debug_mode']
