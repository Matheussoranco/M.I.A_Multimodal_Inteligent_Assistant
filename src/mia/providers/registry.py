"""Lightweight provider registry for lazy component initialization."""
from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Optional


class ProviderLookupError(LookupError):
    """Raised when attempting to access a provider that is not registered."""


class ProviderRegistry:
    """Registry that supports lazy provider registration and instantiation."""

    def __init__(self) -> None:
        self._providers: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._defaults: Dict[str, str] = {}

    def register(
        self,
        domain: str,
        name: str,
        factory: Callable[..., Any],
        *,
        default: bool = False,
        singleton: bool = False,
    ) -> None:
        domain_map = self._providers.setdefault(domain, {})
        domain_map[name] = {"factory": factory, "singleton": singleton, "instance": None}
        if default or domain not in self._defaults:
            self._defaults[domain] = name

    def register_lazy(
        self,
        domain: str,
        name: str,
        import_path: str,
        attr: Optional[str] = None,
        *,
        default: bool = False,
        singleton: bool = False,
    ) -> None:
        """Register a provider without importing it upfront."""

        def factory(**kwargs: Any) -> Any:
            module = importlib.import_module(import_path)
            target = getattr(module, attr) if attr else module
            if callable(target):
                return target(**kwargs)
            if kwargs:
                raise TypeError(f"Provider '{domain}:{name}' does not accept arguments")
            return target

        self.register(domain, name, factory, default=default, singleton=singleton)

    def _get_entry(self, domain: str, name: Optional[str]) -> Dict[str, Any]:
        domain_map = self._providers.get(domain)
        if not domain_map:
            raise ProviderLookupError(f"No providers registered for domain '{domain}'")
        resolved_name = name or self._defaults.get(domain)
        if not resolved_name or resolved_name not in domain_map:
            raise ProviderLookupError(
                f"Provider '{name or ''}' not registered in domain '{domain}'"
            )
        return domain_map[resolved_name]

    def create(self, domain: str, name: Optional[str] = None, **kwargs: Any) -> Any:
        entry = self._get_entry(domain, name)
        if entry["singleton"]:
            if entry["instance"] is None:
                entry["instance"] = entry["factory"](**kwargs)
            return entry["instance"]
        return entry["factory"](**kwargs)

    def get_factory(self, domain: str, name: Optional[str] = None) -> Callable[..., Any]:
        entry = self._get_entry(domain, name)
        return entry["factory"]

    def is_registered(self, domain: str, name: Optional[str] = None) -> bool:
        try:
            self._get_entry(domain, name)
        except ProviderLookupError:
            return False
        return True

    def available(self, domain: str) -> Dict[str, Callable[..., Any]]:
        return {
            name: entry["factory"]
            for name, entry in self._providers.get(domain, {}).items()
        }

    def set_default(self, domain: str, name: str) -> None:
        if domain not in self._providers:
            raise ProviderLookupError(f"No providers registered for domain '{domain}'")
        if name not in self._providers[domain]:
            raise ProviderLookupError(f"Provider '{name}' not registered in domain '{domain}'")
        self._defaults[domain] = name


provider_registry = ProviderRegistry()

__all__ = ["ProviderRegistry", "ProviderLookupError", "provider_registry"]
