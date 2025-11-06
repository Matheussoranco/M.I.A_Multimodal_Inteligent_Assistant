"""Unified inference helper built on top of :class:`LLMManager`."""

from __future__ import annotations

import os
import webbrowser
from typing import Any, Dict, List, Optional

from ..config_manager import ConfigManager
from .llm_manager import LLMManager

# Optional selenium import (kept for backwards compatibility with automation helpers)
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys

    HAS_SELENIUM = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_SELENIUM = False


class LLMInference:
    """High-level interface for querying LLM providers.

    This class wraps :class:`LLMManager` to expose a simplified interface that can
    seamlessly switch between local (e.g., Ollama) and remote providers
    (OpenAI, Anthropic, NanoChat, Minimax, etc.).
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        llama_model_path: Optional[str] = None,
        local_model_path: Optional[str] = None,
        config_manager: Optional[ConfigManager] = None,
        auto_detect: bool = True,
        **manager_kwargs: Any,
    ) -> None:
        """Create a new inference helper.

        Args:
            provider: Preferred provider override. Falls back to configuration
                or auto-detection when omitted.
            model_id: Optional model identifier to override configuration.
            api_key: Optional API key for remote providers.
            url: Optional base URL/endpoint for the provider.
            llama_model_path: Backwards compatible alias for ``local_model_path``.
            local_model_path: Filesystem path to a local model when using
                ``provider='local'`` or Ollama custom models.
            config_manager: Optional shared :class:`ConfigManager` instance.
            auto_detect: When ``True``, attempt to discover available providers
                if none are explicitly configured.
            **manager_kwargs: Additional keyword arguments forwarded to
                :class:`LLMManager`.
        """

        resolved_local_path = local_model_path or llama_model_path
        inferred_provider = provider

        if resolved_local_path and not inferred_provider:
            # Maintain backwards compatibility with legacy constructor usage.
            inferred_provider = "local"

        self.config_manager = config_manager or ConfigManager()
        self.manager = LLMManager(
            provider=inferred_provider,
            model_id=model_id,
            api_key=api_key,
            url=url,
            local_model_path=resolved_local_path,
            config_manager=self.config_manager,
            auto_detect=auto_detect,
            **manager_kwargs,
        )

    # ---------------------------------------------------------------------
    # Core inference helpers
    # ---------------------------------------------------------------------
    def query_model(self, text: str, **kwargs: Any) -> Optional[str]:
        """Query the configured model with a single prompt."""
        return self.manager.query(text, **kwargs)

    def generate_response(self, prompt: str, **kwargs: Any) -> Optional[str]:
        """Alias for :meth:`query_model` for semantic clarity."""
        return self.query_model(prompt, **kwargs)

    def generate_chat_completion(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> Optional[str]:
        """Support multi-turn chat style prompts by joining user content."""
        if not messages:
            return None

        # Delegate to provider with best-effort formatting.
        provider = self.manager.provider
        if provider in {"openai", "anthropic", "gemini", "groq", "grok", "minimax"}:
            kwargs.setdefault("messages", messages)
            return self.manager.query(messages[-1].get("content", ""), **kwargs)

        # For local providers, concatenate content for a pragmatic fallback.
        combined_prompt = "\n".join(
            msg.get("content", "") for msg in messages if msg.get("content")
        )
        return self.manager.query(combined_prompt, **kwargs)

    def switch_provider(self, provider: str, **overrides: Any) -> None:
        """Re-initialize the underlying manager with a new provider."""
        self.manager = LLMManager(
            provider=provider,
            model_id=overrides.get("model_id"),
            api_key=overrides.get("api_key"),
            url=overrides.get("url"),
            local_model_path=overrides.get("local_model_path"),
            config_manager=self.config_manager,
            auto_detect=False,
        )

    def update_config(self, config: Any) -> None:
        """Update the LLM configuration dynamically."""
        if hasattr(config, "provider"):
            self.manager.provider = config.provider
        if hasattr(config, "model_id"):
            self.manager.model_id = config.model_id
        if hasattr(config, "api_key"):
            self.manager.api_key = config.api_key
        if hasattr(config, "url"):
            self.manager.url = config.url
        if hasattr(config, "max_tokens"):
            self.manager.max_tokens = config.max_tokens
        if hasattr(config, "temperature"):
            self.manager.temperature = config.temperature
        if hasattr(config, "timeout"):
            self.manager.timeout = config.timeout

    def available_providers(self) -> Dict[str, Any]:
        """Inspect the currently active provider metadata."""
        return self.manager.get_model_info()

    # ---------------------------------------------------------------------
    # Legacy helper stubs (email, messaging, automation)
    # ---------------------------------------------------------------------
    def write_email(self, recipient: str, subject: str, body: str) -> str:
        return f"Email to {recipient} with subject '{subject}' sent successfully."

    def write_message(self, platform: str, recipient: str, message: str) -> str:
        return f"Message to {recipient} on {platform} sent successfully."

    def take_note(self, app: str, content: str) -> str:
        return f"Note saved in {app} with content: '{content}'"

    def open_program_or_website(self, target: str) -> str:
        if target.startswith("http"):
            webbrowser.open(target)
            return f"Website {target} opened."
        os.system(target)  # pragma: no cover - platform specific
        return f"Program {target} opened."

    def autofill_login(self, url: str, username: str, password: str) -> str:
        if not HAS_SELENIUM:
            return (
                "Selenium not available. Install selenium for web automation features."
            )

        driver = webdriver.Chrome()  # pragma: no cover - requires selenium runtime
        driver.get(url)

        try:
            user_field = driver.find_element(By.NAME, "username")
            pass_field = driver.find_element(By.NAME, "password")

            user_field.send_keys(username)
            pass_field.send_keys(password)
            pass_field.send_keys(Keys.RETURN)
            return "Login fields filled successfully."
        except Exception as exc:  # pragma: no cover - runtime dependent
            return f"Failed to autofill login: {exc}"


# Example usage:
# llm = LLMInference(provider="ollama", model_id="mistral:instruct")
# print(llm.query_model("What's the weather today?"))
