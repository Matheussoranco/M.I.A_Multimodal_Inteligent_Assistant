"""Provider registry setup and default registrations."""

from __future__ import annotations

from .defaults import register_default_providers
from .registry import ProviderLookupError, ProviderRegistry, provider_registry

register_default_providers()

__all__ = [
    "ProviderRegistry",
    "ProviderLookupError",
    "provider_registry",
    "register_default_providers",
]
