"""Provider registry setup and default registrations."""
from __future__ import annotations

from .registry import ProviderRegistry, ProviderLookupError, provider_registry
from .defaults import register_default_providers

register_default_providers()

__all__ = ["ProviderRegistry", "ProviderLookupError", "provider_registry", "register_default_providers"]
