"""Shared CLI utility functions for M.I.A."""

from __future__ import annotations

import os
import re
import warnings
from typing import Any, Dict, List, Optional

try:
    import colorama
    colorama.init()
except ImportError:
    pass

from ..localization import _

# ═══════════════════════════════════════════════════════════════════════════════
# ANSI colour helpers
# ═══════════════════════════════════════════════════════════════════════════════


def bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"


def green(text: str) -> str:
    return f"\033[32m{text}\033[0m"


def yellow(text: str) -> str:
    return f"\033[33m{text}\033[0m"


def red(text: str) -> str:
    return f"\033[31m{text}\033[0m"


def cyan(text: str) -> str:
    return f"\033[36m{text}\033[0m"


# ═══════════════════════════════════════════════════════════════════════════════
# Localisation / text helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _msg(key: str, default: str, **kwargs: Any) -> str:
    """Return the localised message for *key*, falling back to *default*."""
    localised = _(key, **kwargs)
    if localised != key:
        return localised
    if kwargs:
        try:
            return default.format(**kwargs)
        except Exception:
            return default
    return default


def _extract_filepath(
    text: str, extensions: Optional[List[str]] = None
) -> Optional[str]:
    """Best-effort extraction of a file path from free-form text."""
    if not text:
        return None
    candidates = re.findall(r"[\w./\\:-]+", text)
    if not candidates:
        return None
    if extensions:
        lowered_exts = [ext.lower() for ext in extensions]
        for token in candidates:
            if any(token.lower().endswith(ext) for ext in lowered_exts):
                return token.strip("\"'")
    return candidates[-1].strip("\"'") if candidates else None


# ═══════════════════════════════════════════════════════════════════════════════
# User consent
# ═══════════════════════════════════════════════════════════════════════════════


def prompt_user_consent(
    action: str, params: Optional[Dict[str, Any]] = None
) -> bool:
    """Ask the user to confirm a sensitive action before execution."""
    auto = os.getenv("MIA_AUTO_CONSENT", "").strip().lower()
    if auto in {"1", "true", "yes", "y", "sim", "s"}:
        return True

    label = action.replace("_", " ")
    target = ""
    if params:
        for key in ("recipient", "to", "path", "url", "filename"):
            value = params.get(key)
            if value:
                target = str(value)
                break

    message = f"Confirm action '{label}'"
    if target:
        message += f" for '{target}'"
    message += "? [y/N]: "

    while True:
        try:
            choice = input(message).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if choice in {"y", "yes", "s", "sim"}:
            return True
        if choice in {"n", "no", "nao", "não", ""}:
            return False
        print("Please respond with 'y' or 'n'.")


# ═══════════════════════════════════════════════════════════════════════════════
# Warning suppression
# ═══════════════════════════════════════════════════════════════════════════════


def suppress_warnings() -> None:
    """Reduce noisy log output from ML frameworks (opt-in at startup)."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings(
        "ignore", message=".*slow.*processor.*", category=UserWarning
    )
    warnings.filterwarnings(
        "ignore", message=".*use_fast.*", category=UserWarning
    )
