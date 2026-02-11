"""Display and formatting functions for the M.I.A CLI."""

from __future__ import annotations

import logging
from typing import Any, Dict

from .utils import bold, green, yellow, red, cyan
from ..localization import _

logger = logging.getLogger(__name__)


def display_status(components: Dict[str, Any], args: Any) -> None:
    """Print a rich system-status banner to stdout."""
    llm = components.get("llm")
    audio_available = components.get("audio_available", False)
    device = components.get("device", "cpu")
    performance_monitor = components.get("performance_monitor")
    cache_manager = components.get("cache_manager")
    config_manager = components.get("config_manager")
    active_profile = (
        getattr(config_manager, "active_llm_profile", None)
        if config_manager
        else None
    )

    status = (
        green(_("status_connected"))
        if llm and hasattr(llm, "is_available") and llm.is_available()
        else red(_("status_issues"))
    )
    print(bold("[ STATUS ]") + f" {status}")
    print(bold("═" * 60))
    logger.info(_("welcome_message"))

    print(bold("\nM.I.A Status"))
    print(bold("─" * 40))
    mode_label = {
        "text": green("Text-only"),
        "audio": yellow("Audio"),
        "mixed": cyan("Mixed"),
        "auto": cyan("Auto"),
    }.get(getattr(args, "mode", "mixed"), cyan("Mixed"))
    print(f"  {bold('Mode:')} {mode_label}")
    if active_profile:
        print(f"  {bold('LLM Profile:')} {cyan(active_profile)}")
    print(
        f"  {bold('Model:')} "
        f"{yellow(llm.model_id if llm else getattr(args, 'model_id', ''))}"
    )
    print(
        f"  {bold('LLM:')} "
        f"{green('Connected') if llm and hasattr(llm, 'is_available') and llm.is_available() else red('Disconnected')}"
    )
    print(
        f"  {bold('Audio:')} "
        f"{green('Available') if audio_available else red('Not available')}"
    )
    print(f"  {bold('Device:')} {cyan(device)}")

    if performance_monitor and hasattr(performance_monitor, "get_current_metrics"):
        perf = performance_monitor.get_current_metrics()
        if perf:
            print(f"  {bold('CPU:')} {yellow(f'{perf.cpu_percent:.1f}%')}")
            print(f"  {bold('Memory:')} {yellow(f'{perf.memory_percent:.1f}%')}")

    if cache_manager and hasattr(cache_manager, "get_stats"):
        stats = cache_manager.get_stats()
        hit_rate = stats.get("memory_cache", {}).get("hit_rate", 0)
        print(f"  {bold('Cache Hit Rate:')} {green(f'{hit_rate:.1%}')}")

    print(bold("─" * 40))


def display_help(args: Any, components: Dict[str, Any]) -> None:
    """Print available commands and agent capabilities."""
    audio_available = components.get("audio_available", False)

    print(bold(_("help_title")))
    print(bold(_("help_separator")))
    print(green(_("quit_help")))
    print(green(_("help_help")))
    if args.mode != "text" and audio_available:
        print(yellow(_("audio_help")))
    if args.mode == "audio":
        print(cyan(_("text_help")))
    print(cyan(_("status_help")))
    print(cyan(_("models_help")))
    print(cyan("profiles - list configured LLM profiles"))
    print(cyan("switch-profile <name> - switch LLM profile at runtime"))
    print(cyan(_("clear_help")))
    print(cyan("reset   - clear conversation memory (fresh session)"))
    print(cyan("stats   - show session statistics"))
    print(cyan("memory  - show recent conversation history"))
    print()
    print(bold("Agent capabilities (just ask in natural language):"))
    print(cyan("  • Search the web, scrape pages, deep research"))
    print(cyan("  • Create, read, move, delete files & directories"))
    print(cyan("  • Run shell commands"))
    print(cyan("  • Create documents (Word, PDF), spreadsheets, presentations"))
    print(cyan("  • Send emails, WhatsApp, Telegram messages"))
    print(cyan("  • Analyse and generate code"))
    print(cyan("  • Remember and recall information (long-term memory)"))
    print(cyan("  • Open/close desktop apps, type text, keyboard shortcuts"))
    print(cyan("  • Clipboard operations, desktop notifications"))
    print(cyan("  • Calendar events, smart home control"))
    print(cyan("  • OCR text extraction from images"))
    print(bold(_("help_separator")))


def display_models(args: Any) -> None:
    """Print the current model and a hint for changing it."""
    print(bold("\nAvailable Models"))
    print(bold("─" * 40))
    model_id = getattr(args, "model_id", "")
    print(green(f"  {model_id} (current)"))
    print(cyan("  Use --model-id to change model"))
    print(bold("─" * 40))


def display_profiles(components: Dict[str, Any]) -> None:
    """Print all configured LLM profiles from config.yaml."""
    config_manager = components.get("config_manager")
    if not config_manager or not getattr(config_manager, "config", None):
        print(red("No profiles configured."))
        return

    profiles = config_manager.list_llm_profiles()
    active = getattr(config_manager, "active_llm_profile", None)
    if not profiles:
        print(yellow("No profiles found in config.yaml."))
        return

    print(bold("\nConfigured LLM Profiles"))
    print(bold("─" * 40))
    for name in profiles:
        profile = config_manager.get_llm_profile(name)
        label = getattr(profile, "label", None) or name
        marker = green(" (active)") if name == active else ""
        description = getattr(profile, "description", None)
        provider = (
            getattr(profile, "provider", "")
            or config_manager.config.llm.provider
        )
        model_id = (
            getattr(profile, "model_id", "")
            or config_manager.config.llm.model_id
        )
        print(cyan(f"  • {label}{marker}"))
        print(f"    Provider: {provider} — Model: {model_id}")
        if description:
            print(f"    {description}")
    print(bold("─" * 40))
