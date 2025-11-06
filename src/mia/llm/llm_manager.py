"""
LLM Manager: Unified interface for multiple LLM APIs (OpenAI, HuggingFace, Local, etc.)
"""

import json
import logging
import os
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

import requests

from ..config_manager import ConfigManager, LLMConfig, LLMProfileConfig
from ..error_handler import global_error_handler, with_error_handling

# Import custom exceptions and error handling
from ..exceptions import (
    ConfigurationError,
    InitializationError,
    LLMProviderError,
    NetworkError,
)

# Optional imports with error handling
try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    OpenAI = None
    HAS_OPENAI = False

try:
    import warnings

    # Suppress transformers warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.pipelines import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    pipeline = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    HAS_TRANSFORMERS = False

# Async imports
try:
    import asyncio

    import aiohttp

    HAS_AIOHTTP = True
except ImportError:
    aiohttp = None
    asyncio = None
    HAS_AIOHTTP = False

logger = logging.getLogger(__name__)


class LLMManager:
    """Unified LLM manager supporting multiple providers."""

    @classmethod
    def detect_available_providers(
        cls, interactive: bool = True
    ) -> Dict[str, Any]:
        """
        Detect available LLM providers based on API keys and test connectivity.

        Args:
            interactive: Whether to allow user selection if multiple providers are available

        Returns:
            Dict with 'provider', 'model_id', 'api_key', 'url' for the selected provider
        """
        available_providers = []

        # Define providers and their environment variables
        provider_configs = {
            "openai": {
                "env_vars": ["OPENAI_API_KEY"],
                "api_key_env": "OPENAI_API_KEY",
                "default_model": "gpt-4.1",
                "url": "https://api.openai.com/v1",
            },
            "anthropic": {
                "env_vars": ["ANTHROPIC_API_KEY"],
                "api_key_env": "ANTHROPIC_API_KEY",
                "default_model": "claude-3-haiku-20240307",
                "url": "https://api.anthropic.com",
            },
            "gemini": {
                "env_vars": ["GOOGLE_API_KEY"],
                "api_key_env": "GOOGLE_API_KEY",
                "default_model": "gemini-pro",
                "url": "https://generativelanguage.googleapis.com",
            },
            "groq": {
                "env_vars": ["GROQ_API_KEY"],
                "api_key_env": "GROQ_API_KEY",
                "default_model": "llama2-70b-4096",
                "url": "https://api.groq.com",
            },
            "grok": {
                "env_vars": ["XAI_API_KEY"],
                "api_key_env": "XAI_API_KEY",
                "default_model": "grok-beta",
                "url": "https://api.x.ai",
            },
            "minimax": {
                "env_vars": ["MINIMAX_API_KEY"],
                "api_key_env": "MINIMAX_API_KEY",
                "default_model": "abab5.5-chat",
                "url": os.getenv("MINIMAX_URL")
                or "https://api.minimax.chat/v1/text/chatcompletion",
            },
            "nanochat": {
                "env_vars": [],
                "api_key_env": "NANOCHAT_API_KEY",
                "default_model": "nanochat-model",
                "url": os.getenv("NANOCHAT_URL")
                or "http://localhost:8081/api/generate",
            },
            "ollama": {
                "env_vars": [],  # Ollama doesn't need API key
                "default_model": "deepseek-r1:1.5b",
                "url": os.getenv("OLLAMA_URL")
                or "http://localhost:11434/api/generate",
            },
        }

        print("🔍 Detecting available LLM providers...")

        for provider_name, config in provider_configs.items():
            try:
                # Check if required environment variables are set
                has_keys = (
                    all(os.getenv(var) for var in config["env_vars"])
                    if config["env_vars"]
                    else True
                )

                if not has_keys:
                    print(f"❌ {provider_name}: No API key found")
                    continue

                # Test connectivity
                if cls._test_provider_connectivity(provider_name, config):
                    api_key_env = config.get("api_key_env")
                    api_key_value = (
                        os.getenv(api_key_env)
                        if api_key_env
                        else (
                            os.getenv(config["env_vars"][0])
                            if config["env_vars"]
                            else None
                        )
                    )
                    available_providers.append(
                        {
                            "name": provider_name,
                            "model": config["default_model"],
                            "url": config["url"],
                            "api_key": api_key_value,
                        }
                    )
                    print(f"✅ {provider_name}: Available")
                else:
                    print(f"❌ {provider_name}: Connection test failed")

            except Exception as e:
                print(f"❌ {provider_name}: Error - {str(e)}")

        if not available_providers:
            raise ConfigurationError(
                "No LLM providers available. Please set API keys for at least one provider.",
                "NO_PROVIDERS_AVAILABLE",
            )

        # If only one provider, use it
        if len(available_providers) == 1:
            selected = available_providers[0]
            print(f"📌 Using {selected['name']} (only available provider)")
            return {
                "provider": selected["name"],
                "model_id": selected["model"],
                "api_key": selected["api_key"],
                "url": selected["url"],
            }

        # Multiple providers available
        if interactive:
            print(
                f"\n📋 Found {len(available_providers)} available providers:"
            )
            for i, provider in enumerate(available_providers, 1):
                print(f"  {i}. {provider['name']} ({provider['model']})")

            while True:
                try:
                    choice = input(
                        "\nSelect provider (1-{}): ".format(
                            len(available_providers)
                        )
                    ).strip()
                    idx = int(choice) - 1
                    if 0 <= idx < len(available_providers):
                        selected = available_providers[idx]
                        print(f"📌 Selected: {selected['name']}")
                        return {
                            "provider": selected["name"],
                            "model_id": selected["model"],
                            "api_key": selected["api_key"],
                            "url": selected["url"],
                        }
                    else:
                        print("Invalid choice. Please select a valid number.")
                except (ValueError, KeyboardInterrupt):
                    print("Invalid input. Please enter a number.")
        else:
            # Non-interactive mode: prefer OpenAI, then others
            preferred_order = [
                "openai",
                "anthropic",
                "gemini",
                "groq",
                "grok",
                "minimax",
                "nanochat",
                "ollama",
            ]
            for pref in preferred_order:
                for provider in available_providers:
                    if provider["name"] == pref:
                        print(f"📌 Using {provider['name']} (preferred)")
                        return {
                            "provider": provider["name"],
                            "model_id": provider["model"],
                            "api_key": provider["api_key"],
                            "url": provider["url"],
                        }

            # Fallback to first available
            selected = available_providers[0]
            print(f"📌 Using {selected['name']} (fallback)")
            return {
                "provider": selected["name"],
                "model_id": selected["model"],
                "api_key": selected["api_key"],
                "url": selected["url"],
            }

    @classmethod
    def _test_provider_connectivity(
        cls, provider_name: str, config: Dict[str, Any]
    ) -> bool:
        """Test connectivity to a provider."""
        try:
            if provider_name == "openai" and HAS_OPENAI and OpenAI:
                # For OpenAI, just check if API key is set and looks valid
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key and api_key.startswith("sk-") and len(api_key) > 20:
                    # Optional: Try a minimal request to verify
                    try:
                        client = OpenAI(
                            api_key=api_key, max_retries=1, timeout=5.0
                        )
                        # Try a minimal request
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": "test"}],
                            max_tokens=1,
                        )
                        return True
                    except Exception:
                        # If API call fails, still consider it available if key looks valid
                        return True
                return False

            elif provider_name == "ollama":
                # Test Ollama connectivity
                import requests

                response = requests.get(
                    "http://localhost:11434/api/tags", timeout=5
                )
                return response.status_code == 200

            elif provider_name in [
                "anthropic",
                "gemini",
                "groq",
                "grok",
                "minimax",
            ]:
                # For other providers, just check if API key is set
                env_vars = config.get("env_vars", [])
                return all(os.getenv(var) for var in env_vars)
            elif provider_name == "nanochat":
                env_vars = config.get("env_vars", [])
                if env_vars and not all(os.getenv(var) for var in env_vars):
                    return False

                url = os.getenv("NANOCHAT_URL") or config.get("url")
                if not url:
                    return True

                # Attempt a lightweight connectivity check; accept common auth-required responses
                import requests

                try:
                    response = requests.get(url, timeout=5)
                    return response.status_code in (
                        200,
                        201,
                        202,
                        204,
                        401,
                        403,
                        405,
                    )
                except requests.exceptions.RequestException:
                    return False

            else:
                return False

        except Exception:
            return False

    def __init__(
        self,
        provider: Optional[str] = None,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        local_model_path: Optional[str] = None,
        config_manager: Optional[Any] = None,
        auto_detect: bool = True,
        profile: Optional[str] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        # Initialize configuration manager
        self.config_manager = config_manager or ConfigManager()
        if getattr(self.config_manager, "config", None) is None:
            try:
                self.config_manager.load_config()
            except ConfigurationError:
                pass

        self.profile_name: Optional[str] = profile
        self.profile_metadata: Dict[str, Any] = {}
        self.allowed_scopes: List[str] = []
        self.system_prompt: Optional[str] = None
        self.stream_enabled: bool = stream if stream is not None else True

        # Auto-detect provider if not specified and not in testing mode
        import sys

        is_testing = "pytest" in sys.modules or os.getenv("TESTING") == "true"

        if auto_detect and not provider and not is_testing:
            try:
                detected = self.detect_available_providers(interactive=True)
                provider = detected["provider"]
                model_id = detected["model_id"]
                api_key = detected["api_key"]
                url = detected["url"]
                logger.info(f"Auto-detected provider: {provider}")
            except Exception as e:
                logger.warning(
                    f"Auto-detection failed: {e}, falling back to config"
                )

        config = self.config_manager.config
        base_llm = (
            config.llm if config and hasattr(config, "llm") else LLMConfig()
        )

        resolved_llm = base_llm
        profile_config: Optional[LLMProfileConfig] = None
        if config and hasattr(self.config_manager, "resolve_llm_config"):
            try:
                resolved_llm, profile_config = (
                    self.config_manager.resolve_llm_config(profile)
                )
            except ConfigurationError:
                resolved_llm = base_llm

        self.provider = provider or resolved_llm.provider
        self.model_id = model_id or resolved_llm.model_id
        self.api_key = (
            api_key or resolved_llm.api_key or os.getenv("OPENAI_API_KEY")
        )
        self.url = (
            url
            or resolved_llm.url
            or self._fallback_url_for_provider(self.provider)
        )
        self.max_tokens = resolved_llm.max_tokens
        self.temperature = resolved_llm.temperature
        self.timeout = resolved_llm.timeout

        if profile_config:
            self.profile_name = profile_config.name
            self.system_prompt = profile_config.system_prompt
            self.profile_metadata = dict(profile_config.metadata or {})
            self.allowed_scopes = list(profile_config.scopes or [])
            if stream is None and profile_config.stream is not None:
                self.stream_enabled = bool(profile_config.stream)
            try:
                self.config_manager.activate_llm_profile(profile_config.name)
            except ConfigurationError:
                pass

        self.local_model_path = local_model_path

        self.client: Optional[Union[Any, object]] = None
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self._available = True

        try:
            self._initialize_provider()
        except Exception as e:
            logger.warning(
                f"Failed to initialize LLM provider {self.provider}: {e}"
            )
            # Try fallback provider if ollama fails
            if self.provider == "ollama":
                logger.info("Trying fallback to OpenAI provider...")
                try:
                    self.provider = "openai"
                    self.model_id = "gpt-3.5-turbo"
                    self.url = "https://api.openai.com/v1"
                    self._initialize_provider()
                    logger.info("Successfully fell back to OpenAI provider")
                except Exception as fallback_e:
                    logger.warning(
                        f"Fallback to OpenAI also failed: {fallback_e}"
                    )
                    self._available = False
            else:
                self._available = False
            # Re-raise exceptions during testing
            import sys

            if "pytest" in sys.modules or os.getenv("TESTING") == "true":
                raise e

    def _fallback_url_for_provider(
        self, provider: Optional[str]
    ) -> Optional[str]:
        mapping = {
            "openai": "https://api.openai.com/v1",
            "ollama": os.getenv("OLLAMA_URL")
            or "http://localhost:11434/api/generate",
            "nanochat": os.getenv("NANOCHAT_URL")
            or "http://localhost:8081/api/generate",
            "minimax": os.getenv("MINIMAX_URL")
            or "https://api.minimax.chat/v1/text/chatcompletion",
        }
        return mapping.get(provider or "", None)

    def supports_streaming(self) -> bool:
        return self.provider in {"openai", "ollama"}

    def _build_messages(
        self, prompt: str, kwargs: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        existing = kwargs.get("messages")
        if isinstance(existing, list) and existing:
            return existing
        messages: List[Dict[str, str]] = []
        system_prompt = kwargs.get("system_prompt") or self.system_prompt
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _apply_system_prompt_to_text(
        self, prompt: str, kwargs: Dict[str, Any]
    ) -> str:
        if kwargs.get("messages"):
            return prompt
        system_prompt = kwargs.get("system_prompt") or self.system_prompt
        if not system_prompt:
            return prompt
        prefix = str(system_prompt).strip()
        if not prefix:
            return prompt
        return f"{prefix}\n\n{prompt}"

    def _chunk_text(self, text: str, min_chars: int = 32) -> Iterable[str]:
        if not text:
            return []
        buffer: List[str] = []
        length = 0
        for token in text.split():
            buffer.append(token)
            length += len(token) + 1
            if length >= min_chars:
                yield " ".join(buffer)
                buffer = []
                length = 0
        if buffer:
            yield " ".join(buffer)

    def stream(self, prompt: str, **kwargs: Any) -> Iterable[str]:
        if not prompt:
            raise ValueError("Empty prompt provided")
        if not prompt.strip():
            raise ValueError("Prompt contains only whitespace")

        if kwargs.pop("stream", None) is False:
            response = self._query_sync_fallback(prompt, **kwargs)
            if response:
                yield response
            return

        try:
            if self.provider == "openai":
                yield from self._stream_openai(prompt, **kwargs)
                return
            if self.provider == "ollama":
                yield from self._stream_ollama(prompt, **kwargs)
                return
        except Exception as exc:
            logger.debug(
                "Streaming provider error (%s), falling back to sync query: %s",
                self.provider,
                exc,
            )

        response = self._query_sync_fallback(
            self._apply_system_prompt_to_text(prompt, kwargs), **kwargs
        )
        if response:
            yield from self._chunk_text(response)

    def _stream_openai(self, prompt: str, **kwargs: Any) -> Iterable[str]:
        if not HAS_OPENAI or self.client is None:
            raise LLMProviderError(
                "OpenAI streaming requires openai package",
                "CLIENT_NOT_AVAILABLE",
            )

        chat_attr = getattr(self.client, "chat", None)
        if chat_attr is None:
            raise LLMProviderError(
                "OpenAI client missing chat attribute", "CLIENT_MALFORMED"
            )

        completions_attr = getattr(chat_attr, "completions", None)
        if completions_attr is None:
            raise LLMProviderError(
                "OpenAI client missing completions attribute",
                "CLIENT_MALFORMED",
            )

        messages = self._build_messages(prompt, kwargs)
        stream_kwargs = {
            "model": self.model_id or "gpt-3.5-turbo",
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": True,
        }

        response = completions_attr.create(**stream_kwargs)
        for chunk in response:
            choices = getattr(chunk, "choices", [])
            for choice in choices:
                delta = getattr(choice, "delta", None)
                if not delta:
                    continue
                content = getattr(delta, "content", None)
                if content:
                    yield content

    def _stream_ollama(self, prompt: str, **kwargs: Any) -> Iterable[str]:
        target_url = self.url or self._fallback_url_for_provider("ollama")
        if not target_url:
            raise LLMProviderError(
                "Ollama URL not configured", "OLLAMA_NO_URL"
            )

        payload = {
            "model": self.model_id,
            "prompt": self._apply_system_prompt_to_text(prompt, kwargs),
            "stream": True,
        }
        if "options" in kwargs and isinstance(kwargs["options"], dict):
            payload["options"] = kwargs["options"]

        timeout = kwargs.get("timeout", self.timeout or 30)
        response = requests.post(
            target_url, json=payload, stream=True, timeout=timeout
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            try:
                decoded = line.decode("utf-8")
            except Exception:
                continue
            try:
                data = json.loads(decoded)
            except json.JSONDecodeError:
                logger.debug("Non-JSON line from Ollama stream: %s", decoded)
                continue
            token = data.get("response")
            if token:
                yield token
            if data.get("done"):
                break

    def _initialize_provider(self) -> None:
        """Initialize the specific provider with comprehensive error handling."""
        try:
            if self.provider == "openai":
                self._initialize_openai()
            elif self.provider == "huggingface":
                self._initialize_huggingface()
            elif self.provider == "local":
                self._initialize_local()
            elif self.provider in [
                "grok",
                "gemini",
                "ollama",
                "groq",
                "anthropic",
                "nanochat",
                "minimax",
            ]:
                self._initialize_api_provider()
            else:
                raise ConfigurationError(
                    f"Unknown provider: {self.provider}", "UNKNOWN_PROVIDER"
                )

        except Exception as e:
            error_context = {
                "provider": self.provider,
                "model_id": self.model_id,
                "has_api_key": bool(self.api_key),
            }

            # Convert to appropriate M.I.A exception
            if isinstance(e, (ConfigurationError, InitializationError)):
                raise e
            else:
                raise InitializationError(
                    f"Failed to initialize provider {self.provider}: {str(e)}",
                    "PROVIDER_INIT_FAILED",
                    error_context,
                )

    def _initialize_openai(self) -> None:
        """Initialize OpenAI provider with specific error handling."""
        if not HAS_OPENAI:
            raise InitializationError(
                "OpenAI package not installed. Run: pip install openai",
                "MISSING_DEPENDENCY",
            )
        if OpenAI is None:
            raise InitializationError(
                "OpenAI class not available", "IMPORT_ERROR"
            )

        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "OpenAI API key not provided", "MISSING_API_KEY"
            )

        try:
            # For OpenAI, use base URL without path
            base_url = self.url
            if self.url and "api.openai.com" in self.url:
                base_url = "https://api.openai.com/v1"

            self.client = OpenAI(base_url=base_url, api_key=api_key)
        except Exception as e:
            raise InitializationError(
                f"OpenAI client initialization failed: {str(e)}",
                "CLIENT_INIT_FAILED",
            )

    def _initialize_huggingface(self) -> None:
        """Initialize HuggingFace provider with specific error handling."""
        if not HAS_TRANSFORMERS:
            raise InitializationError(
                "Transformers package not installed. Run: pip install transformers",
                "MISSING_DEPENDENCY",
            )
        if pipeline is None:
            raise InitializationError(
                "Transformers pipeline not available", "IMPORT_ERROR"
            )

        if not self.model_id:
            raise ConfigurationError(
                "Model ID required for HuggingFace provider",
                "MISSING_MODEL_ID",
            )

        try:
            self.client = pipeline("text-generation", model=self.model_id)
        except Exception as e:
            raise InitializationError(
                f"HuggingFace pipeline initialization failed: {str(e)}",
                "PIPELINE_INIT_FAILED",
            )

    def _initialize_local(self) -> None:
        """Initialize local model provider with specific error handling."""
        if not HAS_TRANSFORMERS:
            raise InitializationError(
                "Transformers package not installed. Run: pip install transformers",
                "MISSING_DEPENDENCY",
            )
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise InitializationError(
                "Transformers models not available", "IMPORT_ERROR"
            )

        model_path = self.local_model_path or self.model_id
        if model_path is None:
            raise ConfigurationError(
                "Model path must be specified for local provider",
                "MISSING_MODEL_PATH",
            )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        except Exception as e:
            raise InitializationError(
                f"Local model initialization failed: {str(e)}",
                "MODEL_LOAD_FAILED",
            )

    def _initialize_api_provider(self) -> None:
        """Initialize API-based providers."""
        # API-based providers don't need special initialization
        # but we can validate configuration here
        if self.provider == "ollama":
            # Validate Ollama configuration
            if not self.url and not os.getenv("OLLAMA_URL"):
                self.url = "http://localhost:11434/api/generate"
        elif self.provider == "nanochat":
            if not self.url:
                self.url = (
                    os.getenv("NANOCHAT_URL")
                    or "http://localhost:8081/api/generate"
                )
        elif self.provider == "minimax":
            if not self.url:
                self.url = (
                    os.getenv("MINIMAX_URL")
                    or "https://api.minimax.chat/v1/text/chatcompletion"
                )

        logger.info(f"Initialized API provider: {self.provider}")

    def query_model(self, prompt: str, **kwargs) -> Optional[str]:
        """Alias for compatibility with main.py and other modules."""
        return self.query(prompt, **kwargs)

    @with_error_handling(global_error_handler, fallback_value=None)
    def query(self, prompt: str, **kwargs) -> Optional[str]:
        """Query the selected LLM provider with comprehensive error handling."""
        if not prompt:
            raise ValueError("Empty prompt provided")

        if not prompt.strip():
            raise ValueError("Prompt contains only whitespace")

        # Use async if available, otherwise fallback to sync
        if HAS_AIOHTTP and asyncio is not None:
            try:
                # Create event loop if needed
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If loop is already running, we need to handle differently
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run, self.query_async(prompt, **kwargs)
                            )
                            return future.result(timeout=30)
                    else:
                        return loop.run_until_complete(
                            self.query_async(prompt, **kwargs)
                        )
                except RuntimeError:
                    # No event loop, create new one
                    return asyncio.run(self.query_async(prompt, **kwargs))
            except Exception as e:
                logger.warning(
                    f"Async query failed, falling back to sync: {e}"
                )
                return self._query_sync_fallback(prompt, **kwargs)
        else:
            return self._query_sync_fallback(prompt, **kwargs)

    @with_error_handling(global_error_handler, fallback_value=None)
    async def query_async(self, prompt: str, **kwargs) -> Optional[str]:
        """Async version of query method for better performance."""
        if not prompt:
            raise ValueError("Empty prompt provided")

        if not prompt.strip():
            raise ValueError("Prompt contains only whitespace")

        try:
            if self.provider == "openai":
                return await self._query_openai_async(prompt, **kwargs)
            elif self.provider == "anthropic":
                return await self._query_anthropic_async(prompt, **kwargs)
            elif self.provider == "gemini":
                return await self._query_gemini_async(prompt, **kwargs)
            elif self.provider == "ollama":
                return await self._query_ollama_async(prompt, **kwargs)
            elif self.provider == "groq":
                return await self._query_groq_async(prompt, **kwargs)
            elif self.provider == "grok":
                return await self._query_grok_async(prompt, **kwargs)
            elif self.provider == "nanochat":
                return await self._query_nanochat_async(prompt, **kwargs)
            elif self.provider == "minimax":
                return await self._query_minimax_async(prompt, **kwargs)
            elif self.provider == "huggingface":
                return self._query_huggingface(
                    prompt, **kwargs
                )  # Sync for now
            elif self.provider == "local":
                return self._query_local(prompt, **kwargs)  # Sync for now
            else:
                raise LLMProviderError(
                    f"Provider {self.provider} not implemented",
                    "PROVIDER_NOT_IMPLEMENTED",
                )

        except Exception as e:
            # Re-raise M.I.A exceptions
            if isinstance(
                e, (LLMProviderError, NetworkError, ConfigurationError)
            ):
                raise e
            # Convert other exceptions to LLMProviderError
            raise LLMProviderError(
                f"Async query failed: {str(e)}",
                "ASYNC_QUERY_FAILED",
                {
                    "provider": self.provider,
                    "prompt_length": len(prompt),
                    "kwargs": kwargs,
                },
            )

    def _query_sync_fallback(self, prompt: str, **kwargs) -> Optional[str]:
        """Fallback synchronous query method."""
        try:
            if self.provider == "openai":
                return self._query_openai(prompt, **kwargs)
            elif self.provider == "anthropic":
                return self._query_anthropic(prompt, **kwargs)
            elif self.provider == "gemini":
                return self._query_gemini(prompt, **kwargs)
            elif self.provider == "ollama":
                return self._query_ollama(prompt, **kwargs)
            elif self.provider == "groq":
                return self._query_groq(prompt, **kwargs)
            elif self.provider == "grok":
                return self._query_grok(prompt, **kwargs)
            elif self.provider == "nanochat":
                return self._query_nanochat(prompt, **kwargs)
            elif self.provider == "minimax":
                return self._query_minimax(prompt, **kwargs)
            elif self.provider == "huggingface":
                return self._query_huggingface(prompt, **kwargs)
            elif self.provider == "local":
                return self._query_local(prompt, **kwargs)
            else:
                raise LLMProviderError(
                    f"Provider {self.provider} not implemented",
                    "PROVIDER_NOT_IMPLEMENTED",
                )
        except Exception as e:
            if isinstance(
                e, (LLMProviderError, NetworkError, ConfigurationError)
            ):
                raise e
            raise LLMProviderError(
                f"Sync query failed: {str(e)}",
                "SYNC_QUERY_FAILED",
                {
                    "provider": self.provider,
                    "prompt_length": len(prompt),
                    "kwargs": kwargs,
                },
            )

    def _query_openai(self, prompt: str, **kwargs) -> Optional[str]:
        """Query OpenAI with specific error handling."""
        if self.client is None or not HAS_OPENAI:
            raise LLMProviderError(
                "OpenAI client not available", "CLIENT_NOT_AVAILABLE"
            )

        try:
            # Use getattr for safer attribute access
            chat_attr = getattr(self.client, "chat", None)
            if chat_attr is None:
                raise LLMProviderError(
                    "OpenAI client missing chat attribute", "CLIENT_MALFORMED"
                )

            completions_attr = getattr(chat_attr, "completions", None)
            if completions_attr is None:
                raise LLMProviderError(
                    "OpenAI client missing completions attribute",
                    "CLIENT_MALFORMED",
                )

            formatted_messages = self._build_messages(prompt, kwargs)

            response = completions_attr.create(
                model=self.model_id or "gpt-3.5-turbo",
                messages=formatted_messages,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
            )

            if not response.choices:
                raise LLMProviderError(
                    "OpenAI returned no choices", "EMPTY_RESPONSE"
                )

            content = response.choices[0].message.content
            if content is None:
                raise LLMProviderError(
                    "OpenAI returned null content", "NULL_CONTENT"
                )

            return content

        except Exception as e:
            if isinstance(e, LLMProviderError):
                raise e
            raise LLMProviderError(f"OpenAI API error: {str(e)}", "API_ERROR")

    def _query_huggingface(self, prompt: str, **kwargs) -> Optional[str]:
        """Query HuggingFace with specific error handling."""
        if self.client is None or not HAS_TRANSFORMERS:
            raise LLMProviderError(
                "HuggingFace client not available", "CLIENT_NOT_AVAILABLE"
            )

        try:
            if not callable(self.client):
                raise LLMProviderError(
                    "HuggingFace pipeline not callable", "CLIENT_NOT_CALLABLE"
                )

            result = self.client(
                prompt, max_length=kwargs.get("max_length", 256)
            )

            if not isinstance(result, list) or len(result) == 0:
                raise LLMProviderError(
                    "HuggingFace returned invalid result format",
                    "INVALID_RESPONSE",
                )

            if "generated_text" not in result[0]:
                raise LLMProviderError(
                    "HuggingFace result missing generated_text", "MISSING_TEXT"
                )

            return result[0]["generated_text"]

        except Exception as e:
            if isinstance(e, LLMProviderError):
                raise e
            raise LLMProviderError(
                f"HuggingFace error: {str(e)}", "PIPELINE_ERROR"
            )

    def _query_local(self, prompt: str, **kwargs) -> Optional[str]:
        """Query local model with specific error handling."""
        if (
            self.model is None
            or self.tokenizer is None
            or not HAS_TRANSFORMERS
        ):
            raise LLMProviderError(
                "Local model not available", "MODEL_NOT_AVAILABLE"
            )

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                **inputs, max_length=kwargs.get("max_length", 256)
            )
            result = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

            if not result:
                raise LLMProviderError(
                    "Local model returned empty result", "EMPTY_RESULT"
                )

            return result

        except Exception as e:
            if isinstance(e, LLMProviderError):
                raise e
            raise LLMProviderError(
                f"Local model error: {str(e)}", "MODEL_ERROR"
            )

    def _query_anthropic(self, prompt: str, **kwargs) -> Optional[str]:
        """Query Anthropic Claude API."""
        try:
            headers = {
                "x-api-key": self.api_key or os.getenv("ANTHROPIC_API_KEY"),
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            data = {
                "model": self.model_id or "claude-3-opus-20240229",
                "max_tokens": kwargs.get("max_tokens", 1024),
                "messages": (
                    kwargs.get("messages")
                    if isinstance(kwargs.get("messages"), list)
                    and kwargs.get("messages")
                    else [{"role": "user", "content": prompt}]
                ),
            }
            url = self.url or "https://api.anthropic.com/v1/messages"
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()

            if (
                "content" in result
                and isinstance(result["content"], list)
                and result["content"]
            ):
                return result["content"][0].get("text", "")
            return ""
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return None

    def _query_gemini(self, prompt: str, **kwargs) -> Optional[str]:
        """Query Google Gemini API."""
        try:
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key or os.getenv("GEMINI_API_KEY"),
            }
            data = {"contents": [{"parts": [{"text": prompt}]}]}
            url = (
                self.url
                or "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
            )
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()

            candidates = result.get("candidates", [])
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                if parts and "text" in parts[0]:
                    return parts[0]["text"]
            return ""
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return None

    def _query_ollama(self, prompt: str, **kwargs) -> Optional[str]:
        """Query Ollama local API with enhanced error handling."""
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "model": self.model_id or "mistral:instruct",
                "prompt": self._apply_system_prompt_to_text(prompt, kwargs),
                "stream": False,
            }
            if "options" in kwargs and isinstance(kwargs["options"], dict):
                data["options"] = kwargs["options"]

            if self.api_key and self.api_key != "ollama":
                headers["Authorization"] = f"Bearer {self.api_key}"

            url = (
                self.url
                or self._fallback_url_for_provider("ollama")
                or "http://localhost:11434/api/generate"
            )

            try:
                response = requests.post(
                    url, headers=headers, json=data, timeout=30
                )
                response.raise_for_status()
            except requests.exceptions.Timeout:
                raise NetworkError("Ollama request timed out", "TIMEOUT")
            except requests.exceptions.ConnectionError:
                raise NetworkError(
                    "Cannot connect to Ollama. Make sure it's running: ollama serve",
                    "CONNECTION_ERROR",
                )
            except requests.exceptions.HTTPError as e:
                raise NetworkError(f"Ollama HTTP error: {e}", "HTTP_ERROR")

            try:
                result = response.json()
            except ValueError:
                raise LLMProviderError(
                    "Ollama returned invalid JSON", "INVALID_JSON"
                )

            if "response" not in result:
                raise LLMProviderError(
                    "Ollama response missing 'response' field",
                    "MISSING_RESPONSE",
                )

            return result["response"]

        except (NetworkError, LLMProviderError):
            raise
        except Exception as e:
            raise LLMProviderError(f"Ollama API error: {str(e)}", "API_ERROR")

    def _query_groq(self, prompt: str, **kwargs) -> Optional[str]:
        """Query Groq API."""
        try:
            headers = {
                "Authorization": f'Bearer {self.api_key or os.getenv("GROQ_API_KEY")}',
                "Content-Type": "application/json",
            }
            data = {
                "model": self.model_id or "llama2-70b-4096",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 1024),
            }
            url = self.url or "https://api.groq.com/openai/v1/chat/completions"
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return None

    def _query_grok(self, prompt: str, **kwargs) -> Optional[str]:
        """Query xAI Grok API."""
        try:
            headers = {
                "Authorization": f'Bearer {self.api_key or os.getenv("GROK_API_KEY")}',
                "Content-Type": "application/json",
            }
            data = {
                "model": self.model_id or "grok-1",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 1024),
            }
            url = self.url or "https://api.grok.com/v1/chat/completions"
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Grok API error: {e}")
            return None

    def _query_nanochat(self, prompt: str, **kwargs) -> Optional[str]:
        """Query Nanochat API."""
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "model": self.model_id or "nanochat-model",
                "prompt": prompt,
                "stream": False,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
            }

            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            url = self.url or "http://localhost:8081/api/generate"
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()

            if "response" not in result:
                raise LLMProviderError(
                    "Nanochat response missing 'response' field",
                    "MISSING_RESPONSE",
                )

            return result["response"]
        except Exception as e:
            logger.error(f"Nanochat API error: {e}")
            return None

    def _query_minimax(self, prompt: str, **kwargs) -> Optional[str]:
        """Query Minimax AI API."""
        api_key = self.api_key or os.getenv("MINIMAX_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "Minimax API key not provided", "MISSING_API_KEY"
            )

        url = (
            self.url
            or os.getenv("MINIMAX_URL")
            or "https://api.minimax.chat/v1/text/chatcompletion"
        )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        group_id = kwargs.get("group_id") or os.getenv("MINIMAX_GROUP_ID")
        if group_id:
            headers["X-Group-Id"] = group_id

        messages = kwargs.get("messages")
        if not messages:
            # Default to chat-completion structure expected by Minimax
            messages = [{"sender_type": "USER", "text": prompt}]

        payload: Dict[str, Any] = {
            "model": kwargs.get("model") or self.model_id or "abab5.5-chat",
            "messages": messages,
            "tokens_to_generate": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if kwargs.get("stream") is not None:
            payload["stream"] = kwargs["stream"]

        if kwargs.get("bot_setting"):
            payload["bot_setting"] = kwargs["bot_setting"]

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=kwargs.get("timeout", self.timeout),
            )
            response.raise_for_status()
            result = response.json()
            extracted = self._extract_minimax_text(result)
            if extracted is None:
                raise LLMProviderError(
                    "Minimax response did not contain textual output",
                    "EMPTY_RESPONSE",
                )
            return extracted
        except requests.exceptions.Timeout as exc:
            raise NetworkError("Minimax request timed out", "TIMEOUT") from exc
        except requests.exceptions.ConnectionError as exc:
            raise NetworkError(
                "Unable to reach Minimax API", "CONNECTION_ERROR"
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise NetworkError(
                f"Minimax HTTP error: {exc}", "HTTP_ERROR"
            ) from exc
        except (ConfigurationError, NetworkError, LLMProviderError):
            raise
        except Exception as exc:
            logger.error(f"Minimax API error: {exc}")
            return None

    # Async methods for better performance
    async def _query_openai_async(
        self, prompt: str, **kwargs
    ) -> Optional[str]:
        """Async query OpenAI."""
        if not HAS_AIOHTTP or aiohttp is None:
            return self._query_openai(prompt, **kwargs)

        if self.client is None or not HAS_OPENAI:
            raise LLMProviderError(
                "OpenAI client not available", "CLIENT_NOT_AVAILABLE"
            )

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                data = {
                    "model": self.model_id or "gpt-3.5-turbo",
                    "messages": (
                        kwargs.get("messages")
                        if isinstance(kwargs.get("messages"), list)
                        and kwargs.get("messages")
                        else [{"role": "user", "content": prompt}]
                    ),
                    "max_tokens": kwargs.get("max_tokens", 1024),
                    "temperature": kwargs.get("temperature", 0.7),
                }
                url = self.url or "https://api.openai.com/v1/chat/completions"

                async with session.post(
                    url, headers=headers, json=data
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenAI async error: {e}")
            return None

    async def _query_anthropic_async(
        self, prompt: str, **kwargs
    ) -> Optional[str]:
        """Async query Anthropic Claude API."""
        if not HAS_AIOHTTP or aiohttp is None:
            return self._query_anthropic(prompt, **kwargs)

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "x-api-key": self.api_key
                    or os.getenv("ANTHROPIC_API_KEY"),
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
                data = {
                    "model": self.model_id or "claude-3-opus-20240229",
                    "max_tokens": kwargs.get("max_tokens", 1024),
                    "messages": (
                        kwargs.get("messages")
                        if isinstance(kwargs.get("messages"), list)
                        and kwargs.get("messages")
                        else [{"role": "user", "content": prompt}]
                    ),
                }
                url = self.url or "https://api.anthropic.com/v1/messages"

                async with session.post(
                    url, headers=headers, json=data
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    if (
                        "content" in result
                        and isinstance(result["content"], list)
                        and result["content"]
                    ):
                        return result["content"][0].get("text", "")
                    return ""
        except Exception as e:
            logger.error(f"Anthropic async API error: {e}")
            return None

    async def _query_gemini_async(
        self, prompt: str, **kwargs
    ) -> Optional[str]:
        """Async query Google Gemini API."""
        if not HAS_AIOHTTP or aiohttp is None:
            return self._query_gemini(prompt, **kwargs)

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key
                    or os.getenv("GEMINI_API_KEY"),
                }
                data = {"contents": [{"parts": [{"text": prompt}]}]}
                url = (
                    self.url
                    or "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
                )

                async with session.post(
                    url, headers=headers, json=data
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    candidates = result.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts and "text" in parts[0]:
                            return parts[0]["text"]
                    return ""
        except Exception as e:
            logger.error(f"Gemini async API error: {e}")
            return None

    async def _query_ollama_async(
        self, prompt: str, **kwargs
    ) -> Optional[str]:
        """Async query Ollama local API."""
        if not HAS_AIOHTTP or aiohttp is None:
            return self._query_ollama(prompt, **kwargs)

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Content-Type": "application/json"}
                data = {
                    "model": self.model_id or "mistral:instruct",
                    "prompt": prompt,
                    "stream": False,
                }

                if self.api_key and self.api_key != "ollama":
                    headers["Authorization"] = f"Bearer {self.api_key}"

                url = self.url or "http://localhost:11434/api/generate"

                async with session.post(
                    url, headers=headers, json=data
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

                    if "response" not in result:
                        raise LLMProviderError(
                            "Ollama response missing 'response' field",
                            "MISSING_RESPONSE",
                        )

                    return result["response"]
        except Exception as e:
            logger.error(f"Ollama async API error: {e}")
            return None

    async def _query_groq_async(self, prompt: str, **kwargs) -> Optional[str]:
        """Async query Groq API."""
        if not HAS_AIOHTTP or aiohttp is None:
            return self._query_groq(prompt, **kwargs)

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f'Bearer {self.api_key or os.getenv("GROQ_API_KEY")}',
                    "Content-Type": "application/json",
                }
                data = {
                    "model": self.model_id or "llama2-70b-4096",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": kwargs.get("max_tokens", 1024),
                }
                url = (
                    self.url
                    or "https://api.groq.com/openai/v1/chat/completions"
                )

                async with session.post(
                    url, headers=headers, json=data
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Groq async API error: {e}")
            return None

    async def _query_grok_async(self, prompt: str, **kwargs) -> Optional[str]:
        """Async query Grok API."""
        if not HAS_AIOHTTP or aiohttp is None:
            return self._query_grok(prompt, **kwargs)

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f'Bearer {self.api_key or os.getenv("GROK_API_KEY")}',
                    "Content-Type": "application/json",
                }
                data = {
                    "model": self.model_id or "grok-1",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": kwargs.get("max_tokens", 1024),
                }
                url = self.url or "https://api.grok.com/v1/chat/completions"

                async with session.post(
                    url, headers=headers, json=data
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Grok async API error: {e}")
            return None

    async def _query_nanochat_async(
        self, prompt: str, **kwargs
    ) -> Optional[str]:
        """Async query Nanochat API."""
        if not HAS_AIOHTTP or aiohttp is None:
            return self._query_nanochat(prompt, **kwargs)

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Content-Type": "application/json"}
                data = {
                    "model": self.model_id or "nanochat-model",
                    "prompt": prompt,
                    "stream": False,
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                    "temperature": kwargs.get("temperature", self.temperature),
                }

                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                url = (
                    self.url or "http://localhost:8081/api/generate"
                )  # Default nanochat port

                async with session.post(
                    url, headers=headers, json=data
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

                    if "response" not in result:
                        raise LLMProviderError(
                            "Nanochat response missing 'response' field",
                            "MISSING_RESPONSE",
                        )

                    return result["response"]
        except Exception as e:
            logger.error(f"Nanochat async API error: {e}")
            return None

    async def _query_minimax_async(
        self, prompt: str, **kwargs
    ) -> Optional[str]:
        """Async query Minimax AI API."""
        if not HAS_AIOHTTP or aiohttp is None:
            return self._query_minimax(prompt, **kwargs)

        api_key = self.api_key or os.getenv("MINIMAX_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "Minimax API key not provided", "MISSING_API_KEY"
            )

        url = (
            self.url
            or os.getenv("MINIMAX_URL")
            or "https://api.minimax.chat/v1/text/chatcompletion"
        )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        group_id = kwargs.get("group_id") or os.getenv("MINIMAX_GROUP_ID")
        if group_id:
            headers["X-Group-Id"] = group_id

        messages = kwargs.get("messages")
        if not messages:
            messages = [{"sender_type": "USER", "text": prompt}]

        payload: Dict[str, Any] = {
            "model": kwargs.get("model") or self.model_id or "abab5.5-chat",
            "messages": messages,
            "tokens_to_generate": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if kwargs.get("stream") is not None:
            payload["stream"] = kwargs["stream"]

        if kwargs.get("bot_setting"):
            payload["bot_setting"] = kwargs["bot_setting"]

        timeout_val = kwargs.get("timeout", self.timeout)
        timeout = aiohttp.ClientTimeout(total=timeout_val)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url, headers=headers, json=payload
                ) as response:
                    if response.status >= 400:
                        error_body = await response.text()
                        raise NetworkError(
                            f"Minimax HTTP error {response.status}: {error_body}",
                            "HTTP_ERROR",
                            {"status": response.status},
                        )

                    result = await response.json()
                    extracted = self._extract_minimax_text(result)
                    if extracted is None:
                        raise LLMProviderError(
                            "Minimax response did not contain textual output",
                            "EMPTY_RESPONSE",
                        )
                    return extracted
        except asyncio.TimeoutError as exc:  # type: ignore[attr-defined]
            raise NetworkError("Minimax request timed out", "TIMEOUT") from exc
        except NetworkError:
            raise
        except Exception as exc:
            logger.error(f"Minimax async API error: {exc}")
            return None

    @staticmethod
    def _extract_minimax_text(payload: Dict[str, Any]) -> Optional[str]:
        """Extract textual content from Minimax responses."""
        if not payload:
            return None

        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict):
                    content = message.get("content") or message.get("text")
                    if isinstance(content, str):
                        return content

                messages = choice.get("messages")
                if isinstance(messages, list):
                    for item in messages:
                        if not isinstance(item, dict):
                            continue
                        sender_type = item.get("sender_type") or item.get(
                            "role"
                        )
                        if sender_type and str(sender_type).lower() in {
                            "bot",
                            "assistant",
                            "ai",
                        }:
                            content = item.get("text") or item.get("content")
                            if isinstance(content, str):
                                return content

        for key in ("output_text", "reply", "result"):
            content = payload.get(key)
            if isinstance(content, str) and content.strip():
                return content

        data_section = payload.get("data")
        if isinstance(data_section, dict):
            for key in ("text", "output_text", "content"):
                data_content = data_section.get(key)
                if isinstance(data_content, str) and data_content.strip():
                    return data_content

        return None

    def is_available(self) -> bool:
        """Check if the LLM provider is available and working."""
        try:
            if (
                self.provider == "openai"
                and self.client is not None
                and HAS_OPENAI
            ):
                # Type-safe OpenAI client checking
                try:
                    chat_attr = getattr(self.client, "chat", None)
                    if chat_attr is not None:
                        completions_attr = getattr(
                            chat_attr, "completions", None
                        )
                        if completions_attr is not None:
                            response = completions_attr.create(
                                model=self.model_id or "gpt-3.5-turbo",
                                messages=[
                                    {"role": "user", "content": "Hello"}
                                ],
                                max_tokens=10,
                            )
                            return (
                                response.choices[0].message.content is not None
                            )
                    return False
                except Exception:
                    return False

            elif self.provider == "ollama":
                # Test Ollama connection
                url = self.url or "http://localhost:11434/api/generate"
                headers = {"Content-Type": "application/json"}
                data = {
                    "model": self.model_id or "mistral:instruct",
                    "prompt": "Hello",
                    "stream": False,
                }
                response = requests.post(
                    url, headers=headers, json=data, timeout=5
                )
                return response.status_code == 200

            elif (
                self.provider == "huggingface"
                and self.client is not None
                and HAS_TRANSFORMERS
            ):
                return callable(self.client)

            elif (
                self.provider == "local"
                and self.model is not None
                and self.tokenizer is not None
                and HAS_TRANSFORMERS
            ):
                return True

            return False

        except Exception as e:
            logger.error(
                f"Error checking availability for {self.provider}: {e}"
            )
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "available": self.is_available(),
            "url": self.url,
            "has_api_key": bool(self.api_key),
        }
