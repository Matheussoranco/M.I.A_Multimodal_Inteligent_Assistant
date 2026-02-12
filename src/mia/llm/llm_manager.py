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

    import aiohttp

    HAS_AIOHTTP = True
except ImportError:
    aiohttp = None
    asyncio = None
    HAS_AIOHTTP = False

try:
    from llama_cpp import Llama  # type: ignore[import-not-found]
    HAS_LLAMA_CPP = True
except ImportError:
    Llama = None
    HAS_LLAMA_CPP = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class LLMManager:
    """Unified LLM manager supporting multiple providers."""

    @classmethod
    def detect_ollama_models(cls) -> List[Dict[str, Any]]:
        """Detect all models installed in Ollama."""
        models = []
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                for model in data.get("models", []):
                    models.append({
                        "name": model.get("name", "unknown"),
                        "size": model.get("size", 0),
                        "modified": model.get("modified_at", ""),
                        "provider": "ollama",
                    })
        except Exception:
            pass
        return models

    @classmethod
    def detect_local_models(cls) -> List[Dict[str, Any]]:
        """Detect local HuggingFace models in common directories."""
        models = []
        # Check common HuggingFace cache locations
        hf_cache_dirs = [
            os.path.expanduser("~/.cache/huggingface/hub"),
            os.path.expanduser("~/models"),
            os.path.join(os.getcwd(), "models"),
        ]
        
        for cache_dir in hf_cache_dirs:
            if os.path.exists(cache_dir):
                try:
                    for item in os.listdir(cache_dir):
                        item_path = os.path.join(cache_dir, item)
                        if os.path.isdir(item_path):
                            # Check if it looks like a model directory
                            config_file = os.path.join(item_path, "config.json")
                            if os.path.exists(config_file):
                                models.append({
                                    "name": item,
                                    "path": item_path,
                                    "provider": "local",
                                })
                except Exception:
                    pass
        return models

    @classmethod
    def detect_api_providers(cls) -> List[Dict[str, Any]]:
        """Detect available API providers based on environment variables."""
        providers = []
        
        api_configs = {
            "openai": {
                "env_key": "OPENAI_API_KEY",
                "default_model": "gpt-4o-mini",
                "url": "https://api.openai.com/v1",
            },
            "anthropic": {
                "env_key": "ANTHROPIC_API_KEY", 
                "default_model": "claude-3-haiku-20240307",
                "url": "https://api.anthropic.com",
            },
            "gemini": {
                "env_key": "GOOGLE_API_KEY",
                "default_model": "gemini-pro",
                "url": "https://generativelanguage.googleapis.com",
            },
            "groq": {
                "env_key": "GROQ_API_KEY",
                "default_model": "llama-3.1-70b-versatile",
                "url": "https://api.groq.com/openai/v1",
            },
            "grok": {
                "env_key": "XAI_API_KEY",
                "default_model": "grok-beta",
                "url": "https://api.x.ai/v1",
            },
        }
        
        for name, config in api_configs.items():
            api_key = os.getenv(config["env_key"])
            if api_key:
                providers.append({
                    "name": name,
                    "model": config["default_model"],
                    "url": config["url"],
                    "api_key": api_key,
                    "provider": "api",
                })
        
        return providers

    @classmethod
    def interactive_model_selection(cls) -> Dict[str, Any]:
        """
        Interactive model selection at startup.
        User chooses between local models or API providers.
        """
        print("\n" + "═" * 60)
        print("M.I.A - LLM Model Selection")
        print("═" * 60)
        
        # First, ask user preference
        print("\nHow would you like to use M.I.A?")
        print("  1. LOCAL Model (Ollama / HuggingFace)")
        print("  2. External API (OpenAI, Anthropic, etc.)")
        print("  3. Auto-detect (use first available)")
        
        while True:
            try:
                choice = input("\nChoice [1-3] (default: 1): ").strip() or "1"
                if choice in ["1", "2", "3"]:
                    break
                print("Invalid choice. Enter 1, 2, or 3.")
            except (KeyboardInterrupt, EOFError):
                print("\nUsing auto-detection...")
                choice = "3"
                break
        
        if choice == "1":
            return cls._select_local_model()
        elif choice == "2":
            return cls._select_api_provider()
        else:
            return cls._auto_detect_best()

    @classmethod
    def _select_local_model(cls) -> Dict[str, Any]:
        """Select a local model (Ollama or HuggingFace)."""
        print("\nDetecting local models...")
        
        ollama_models = cls.detect_ollama_models()
        local_models = cls.detect_local_models()
        
        all_models = []
        
        if ollama_models:
            print(f"\nOllama: {len(ollama_models)} model(s) found")
            for model in ollama_models:
                all_models.append({
                    "type": "ollama",
                    "name": model["name"],
                    "display": f"🦙 Ollama: {model['name']}",
                })
        else:
            print("\nOllama: No models found or service offline")
        
        if local_models:
            print(f"Local/HuggingFace: {len(local_models)} model(s) found")
            for model in local_models:
                all_models.append({
                    "type": "local",
                    "name": model["name"],
                    "path": model["path"],
                    "display": f"Local: {model['name']}",
                })
        
        if not all_models:
            print("\nNo local models found.")
            print("Install a model with: ollama pull qwen2.5:3b-instruct-q4_K_M")
            print("\nWould you like to use an external API? [y/N]: ", end="")
            try:
                if input().strip().lower() in ["s", "y", "sim", "yes"]:
                    return cls._select_api_provider()
            except (KeyboardInterrupt, EOFError):
                pass
            raise ConfigurationError(
                "No models available. Install a local model or configure an API.",
                "NO_MODELS_AVAILABLE",
            )
        
        print("\nAvailable models:")
        for i, model in enumerate(all_models, 1):
            print(f"  {i}. {model['display']}")
        
        while True:
            try:
                choice = input(f"\nSelect model [1-{len(all_models)}] (default: 1): ").strip() or "1"
                idx = int(choice) - 1
                if 0 <= idx < len(all_models):
                    selected = all_models[idx]
                    break
                print("Invalid choice.")
            except (ValueError, KeyboardInterrupt, EOFError):
                idx = 0
                selected = all_models[0]
                break
        
        print(f"\nSelected: {selected['display']}")
        
        if selected["type"] == "ollama":
            return {
                "provider": "ollama",
                "model_id": selected["name"],
                "api_key": None,
                "url": "http://localhost:11434/api/generate",
            }
        else:
            return {
                "provider": "local",
                "model_id": selected["name"],
                "api_key": None,
                "url": None,
                "local_model_path": selected.get("path"),
            }

    @classmethod
    def _select_api_provider(cls) -> Dict[str, Any]:
        """Select an API provider."""
        print("\nChecking available APIs...")
        
        api_providers = cls.detect_api_providers()
        
        if not api_providers:
            print("\nNo API configured.")
            print("Configure environment variables:")
            print("   - OPENAI_API_KEY")
            print("   - ANTHROPIC_API_KEY")
            print("   - GOOGLE_API_KEY")
            print("   - GROQ_API_KEY")
            print("\nWould you like to use a local model? [Y/n]: ", end="")
            try:
                if input().strip().lower() not in ["n", "no", "nao", "não"]:
                    return cls._select_local_model()
            except (KeyboardInterrupt, EOFError):
                pass
            raise ConfigurationError(
                "No API available. Configure an API key or use a local model.",
                "NO_API_AVAILABLE",
            )
        
        print(f"\n{len(api_providers)} API(s) available:")
        for i, provider in enumerate(api_providers, 1):
            print(f"  {i}. {provider['name'].upper()} ({provider['model']})")
        
        while True:
            try:
                choice = input(f"\nSelect API [1-{len(api_providers)}] (default: 1): ").strip() or "1"
                idx = int(choice) - 1
                if 0 <= idx < len(api_providers):
                    selected = api_providers[idx]
                    break
                print("Invalid choice.")
            except (ValueError, KeyboardInterrupt, EOFError):
                idx = 0
                selected = api_providers[0]
                break
        
        print(f"\nSelected: {selected['name'].upper()}")
        
        return {
            "provider": selected["name"],
            "model_id": selected["model"],
            "api_key": selected["api_key"],
            "url": selected["url"],
        }

    @classmethod
    def _auto_detect_best(cls) -> Dict[str, Any]:
        """Auto-detect the best available provider."""
        print("\nAuto-detecting best option...")
        
        # Try Ollama first (local, no cost)
        ollama_models = cls.detect_ollama_models()
        if ollama_models:
            model = ollama_models[0]
            print(f"Using Ollama: {model['name']}")
            return {
                "provider": "ollama",
                "model_id": model["name"],
                "api_key": None,
                "url": "http://localhost:11434/api/generate",
            }
        
        # Then try local models
        local_models = cls.detect_local_models()
        if local_models:
            model = local_models[0]
            print(f"Using local model: {model['name']}")
            return {
                "provider": "local",
                "model_id": model["name"],
                "api_key": None,
                "url": None,
                "local_model_path": model.get("path"),
            }
        
        # Finally try API providers
        api_providers = cls.detect_api_providers()
        if api_providers:
            provider = api_providers[0]
            print(f"Using API: {provider['name'].upper()}")
            return {
                "provider": provider["name"],
                "model_id": provider["model"],
                "api_key": provider["api_key"],
                "url": provider["url"],
            }
        
        raise ConfigurationError(
            "No LLM provider available.",
            "NO_PROVIDERS_AVAILABLE",
        )

    @classmethod
    def detect_available_providers(
        cls, interactive: bool = True
    ) -> Dict[str, Any]:
        """
        Detect available LLM providers with interactive selection.
        
        Args:
            interactive: Whether to show interactive selection menu
            
        Returns:
            Dict with 'provider', 'model_id', 'api_key', 'url' for the selected provider
        """
        if interactive:
            return cls.interactive_model_selection()
        else:
            return cls._auto_detect_best()

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

        self.client: Any = None
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

    def unload_model(self) -> None:
        """Unload the current model to free memory."""
        self.client = None
        self.model = None
        self.tokenizer = None
        
        # Aggressive cleanup
        import gc
        gc.collect()
        
        if HAS_TORCH and torch and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass
                
        logger.info(f"Model {self.model_id} unloaded from memory")
        self._available = False

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

    # ══════════════════════════════════════════════════════════════════
    # Native tool/function-calling API (chat)
    # ══════════════════════════════════════════════════════════════════

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = "auto",
        **kwargs: Any,
    ) -> Any:
        """Send a multi-turn conversation with optional tool definitions.

        Parameters
        ----------
        messages:
            List of ``{"role": …, "content": …}`` dicts, potentially
            including ``tool_calls`` and ``tool`` role messages.
        tools:
            OpenAI-format tool/function definitions.
        tool_choice:
            ``"auto"`` (default), ``"none"``, or a specific tool name.

        Returns
        -------
        ChatResponse
            Structured response (imported from ``core.agent``).
        """
        from ..core.agent import ChatResponse, ToolCall

        # ── OpenAI / Groq / Grok / OpenAI-compatible ────────────
        if self.provider in ("openai", "groq", "grok"):
            return self._chat_openai_compat(messages, tools, tool_choice, **kwargs)

        # ── Ollama (chat API with tools support) ─────────────────
        if self.provider == "ollama":
            return self._chat_ollama(messages, tools, tool_choice, **kwargs)

        # ── Anthropic ────────────────────────────────────────────
        if self.provider == "anthropic":
            return self._chat_anthropic(messages, tools, tool_choice, **kwargs)

        # ── Fallback: provider doesn't support chat() ────────────
        # Convert messages to a single prompt and route through query()
        prompt = self._messages_to_prompt(messages)
        content = self._query_sync_fallback(prompt, **kwargs)
        return ChatResponse(content=content or "", finish_reason="stop")

    def _chat_openai_compat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[str],
        **kwargs: Any,
    ) -> Any:
        """Chat via the OpenAI-compatible completions API (OpenAI/Groq/Grok)."""
        from ..core.agent import ChatResponse, ToolCall

        api_key = self.api_key
        url = self.url
        model = self.model_id

        if self.provider == "groq":
            api_key = api_key or os.getenv("GROQ_API_KEY")
            url = url or "https://api.groq.com/openai/v1"
            model = model or "llama-3.1-70b-versatile"
        elif self.provider == "grok":
            api_key = api_key or os.getenv("XAI_API_KEY")
            url = url or "https://api.x.ai/v1"
            model = model or "grok-beta"
        else:
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            url = url or "https://api.openai.com/v1"
            model = model or "gpt-4o-mini"

        # Use the openai SDK client if available
        if HAS_OPENAI and self.client is not None:
            try:
                request_kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                    "temperature": kwargs.get("temperature", self.temperature),
                }
                if tools:
                    request_kwargs["tools"] = tools
                    if tool_choice:
                        request_kwargs["tool_choice"] = tool_choice

                response = self.client.chat.completions.create(**request_kwargs)
                choice = response.choices[0]
                msg = choice.message

                # Parse tool calls from response
                parsed_calls: Optional[List[ToolCall]] = None
                if msg.tool_calls:
                    parsed_calls = []
                    for tc in msg.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments)
                        except (json.JSONDecodeError, TypeError):
                            args = {"raw": tc.function.arguments}
                        parsed_calls.append(ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=args,
                        ))

                return ChatResponse(
                    content=msg.content or "",
                    tool_calls=parsed_calls,
                    finish_reason=choice.finish_reason or "stop",
                )
            except Exception as exc:
                logger.error("OpenAI-compat chat failed: %s", exc)
                raise

        # Fallback: raw HTTP
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if tools:
            body["tools"] = tools
            if tool_choice:
                body["tool_choice"] = tool_choice

        chat_url = url.rstrip("/") + "/chat/completions"
        resp = requests.post(chat_url, headers=headers, json=body, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]

        parsed_calls = None
        if msg.get("tool_calls"):
            parsed_calls = []
            for tc in msg["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": tc["function"]["arguments"]}
                parsed_calls.append(ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=args,
                ))

        return ChatResponse(
            content=msg.get("content", ""),
            tool_calls=parsed_calls,
            finish_reason=choice.get("finish_reason", "stop"),
        )

    def _chat_ollama(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[str],
        **kwargs: Any,
    ) -> Any:
        """Chat via Ollama's /api/chat endpoint (supports tools since 0.4+)."""
        from ..core.agent import ChatResponse, ToolCall

        base_url = self.url or "http://localhost:11434"
        # Normalize to base URL
        for suffix in ("/api/generate", "/api/chat"):
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]
        chat_url = base_url.rstrip("/") + "/api/chat"

        body: Dict[str, Any] = {
            "model": self.model_id or "llama3",
            "messages": messages,
            "stream": False,
        }
        if tools:
            body["tools"] = tools

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "ollama":
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = requests.post(chat_url, headers=headers, json=body, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        msg = data.get("message", {})
        content = msg.get("content", "")

        parsed_calls = None
        if msg.get("tool_calls"):
            parsed_calls = []
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                parsed_calls.append(ToolCall(
                    id=f"call_{hash(func.get('name', '')) & 0xFFFFFFFF:08x}",
                    name=func.get("name", ""),
                    arguments=func.get("arguments", {}),
                ))

        return ChatResponse(
            content=content,
            tool_calls=parsed_calls,
            finish_reason="tool_calls" if parsed_calls else "stop",
        )

    def _chat_anthropic(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[str],
        **kwargs: Any,
    ) -> Any:
        """Chat via Anthropic's Messages API with tool use."""
        from ..core.agent import ChatResponse, ToolCall

        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ConfigurationError("Anthropic API key not set", "MISSING_API_KEY")

        # Convert OpenAI tool format to Anthropic format
        anthropic_tools = None
        if tools:
            anthropic_tools = []
            for t in tools:
                func = t.get("function", {})
                anthropic_tools.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                })

        # Separate system message from conversation
        system_text = ""
        conv_messages = []
        for m in messages:
            if m["role"] == "system":
                system_text += m.get("content", "") + "\n"
            elif m["role"] == "tool":
                # Anthropic uses "user" role with tool_result content blocks
                conv_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.get("tool_call_id", ""),
                        "content": m.get("content", ""),
                    }],
                })
            else:
                conv_messages.append(m)

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body: Dict[str, Any] = {
            "model": self.model_id or "claude-3-haiku-20240307",
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": conv_messages,
        }
        if system_text.strip():
            body["system"] = system_text.strip()
        if anthropic_tools:
            body["tools"] = anthropic_tools

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers, json=body, timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        # Parse Anthropic response content blocks
        content_text = ""
        parsed_calls = None
        for block in data.get("content", []):
            if block.get("type") == "text":
                content_text += block.get("text", "")
            elif block.get("type") == "tool_use":
                if parsed_calls is None:
                    parsed_calls = []
                parsed_calls.append(ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=block.get("input", {}),
                ))

        stop_reason = data.get("stop_reason", "end_turn")
        finish = "tool_calls" if stop_reason == "tool_use" else "stop"

        return ChatResponse(
            content=content_text,
            tool_calls=parsed_calls,
            finish_reason=finish,
        )

    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Flatten a multi-turn message list into a single text prompt."""
        parts: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"[System] {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            elif role == "tool":
                parts.append(f"[Tool Result] {content}")
        return "\n\n".join(parts)

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

    def _initialize_llama(self) -> None:
        """Initialize Llama.cpp model."""
        if not HAS_LLAMA_CPP:
            raise InitializationError(
                "Llama.cpp not available. Install with: pip install llama-cpp-python",
                "MISSING_DEPENDENCY",
            )
        if Llama is None:
            raise InitializationError(
                "Llama.cpp import failed.",
                "IMPORT_ERROR",
            )
        
        try:
            logger.info(f"Loading GGUF model from: {self.model_id}")
            # Detect GPU layers if cuda available
            n_gpu_layers = -1  # Default to all layers if supported
            model_path = self.local_model_path or self.model_id
            if model_path is None:
                raise ConfigurationError(
                    "Model path must be specified for GGUF provider",
                    "MISSING_MODEL_PATH",
                )
            context_window = getattr(self, "max_tokens", 2048)
            
            self.model = Llama(
                model_path=model_path,
                n_ctx=context_window,
                n_threads=os.cpu_count() or 4,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
            self._available = True
            logger.info("GGUF model loaded successfully")
        except Exception as e:
            raise InitializationError(
                f"Failed to load GGUF model: {str(e)}",
                "MODEL_LOAD_FAILED"
            )

    def _initialize_provider(self) -> None:
        """Initialize the specific provider with comprehensive error handling."""
        try:
            if self.provider == "openai":
                self._initialize_openai()
            elif self.provider == "huggingface":
                self._initialize_huggingface()
            elif self.provider == "local":
                self._initialize_local()
            elif self.provider == "gguf":
                self._initialize_llama()
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

    @with_error_handling(global_error_handler, fallback_value=None, reraise=True)
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

    @with_error_handling(global_error_handler, fallback_value=None, reraise=True)
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
            elif self.provider == "gguf":
                return self._query_llama(prompt, **kwargs)  # Sync for now
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
            elif self.provider == "gguf":
                return self._query_llama(prompt, **kwargs)
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

    def _query_llama(self, prompt: str, **kwargs) -> Optional[str]:
        """Query GGUF model via Llama.cpp."""
        if self.model is None:
            raise LLMProviderError(
                "GGUF model not initialized", "MODEL_NOT_AVAILABLE"
            )

        try:
            # Map common params to llama.cpp params
            params = {
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", 512),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.95),
                "echo": False
            }
            
            output = self.model(**params)
            
            # Extract text from standard OpenAI-compatible format
            if isinstance(output, dict) and "choices" in output:
                return output["choices"][0]["text"]
            
            return str(output)
            
        except Exception as e:
            raise LLMProviderError(
                f"Llama.cpp inference error: {str(e)}", "INFERENCE_ERROR"
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
                base_url = self.url or "http://localhost:11434"
                # Remove /api/generate suffix if present for tags endpoint
                if base_url.endswith("/api/generate"):
                    base_url = base_url.replace("/api/generate", "")
                tags_url = f"{base_url}/api/tags"
                try:
                    response = requests.get(tags_url, timeout=5)
                    return response.status_code == 200
                except Exception:
                    # Fallback to generate endpoint test
                    url = self.url or "http://localhost:11434/api/generate"
                    headers = {"Content-Type": "application/json"}
                    data = {
                        "model": self.model_id or "llama2",
                        "prompt": "Hi",
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
