"""
M.I.A — Multimodal Intelligent Assistant
=========================================
Refactored main entry point.

This module is intentionally thin.  Heavy lifting is delegated to:
- ``cli.parser``   — argument parsing & logging setup
- ``cli.display``  — status banners, help, model/profile display
- ``cli.utils``    — ANSI colours, user-consent prompt, warning suppression
- ``core.agent``   — ToolCallingAgent (native function-calling + ReAct fallback)
"""

from __future__ import annotations

import logging
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, cast, TYPE_CHECKING

import numpy as np

# ── CLI modules (extracted from the former monolithic main.py) ──────
from .cli.parser import parse_arguments, setup_logging
from .cli.display import (
    display_help,
    display_models,
    display_profiles,
    display_status,
)
from .cli.utils import bold, cyan, green, red, yellow, _msg, prompt_user_consent

from .__version__ import __version__, get_full_version
from .localization import _, init_localization
from .providers import ProviderLookupError, provider_registry

if TYPE_CHECKING:
    from .tools.action_executor import ActionExecutor

# ── Soft imports (graceful degradation) ─────────────────────────────
try:
    from .cache_manager import CacheManager
    from .config_manager import ConfigManager
    from .error_handler import global_error_handler, safe_execute
    from .exceptions import *  # noqa: F403
    from .performance_monitor import PerformanceMonitor
    from .resource_manager import ResourceManager
except ImportError as exc:
    print(f"Warning: Some core modules could not be imported: {exc}")
    ConfigManager = None  # type: ignore[misc, assignment]
    ResourceManager = None  # type: ignore[misc, assignment]
    PerformanceMonitor = None  # type: ignore[misc, assignment]
    CacheManager = None  # type: ignore[misc, assignment]

try:
    from .audio.hotword_detector import HotwordDetector
except ImportError:
    HotwordDetector = None  # type: ignore[misc, assignment]

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Component initialisation
# ═══════════════════════════════════════════════════════════════════════════════


def initialize_components(args) -> Dict[str, Any]:
    """Bootstrap every M.I.A subsystem and return them as a dict."""
    components: Dict[str, Any] = {}
    configuration_error_cls = globals().get("ConfigurationError", Exception)

    # ── Configuration ───────────────────────────────────────────────
    config_manager = None
    if ConfigManager:
        try:
            config_manager = ConfigManager()
            config_manager.load_config()
            profile = getattr(args, "profile", None)
            if profile:
                try:
                    config_manager.activate_llm_profile(profile)
                except configuration_error_cls as exc:
                    logger.warning("Profile '%s' unavailable: %s", profile, exc)
            elif config_manager.config and config_manager.config.default_llm_profile:
                try:
                    config_manager.activate_llm_profile(
                        config_manager.config.default_llm_profile
                    )
                except configuration_error_cls:
                    pass
        except Exception as exc:
            logger.warning("Configuration manager failed: %s", exc)
            config_manager = None
    components["config_manager"] = config_manager

    # ── Device ──────────────────────────────────────────────────────
    device = "cpu"
    if HAS_TORCH and torch is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    components["device"] = device
    logger.info("Using device: %s", device)

    # ── LLM Manager ────────────────────────────────────────────────
    llm_kwargs: Dict[str, Any] = {}
    if getattr(args, "model_id", None):
        llm_kwargs["model_id"] = args.model_id
    if config_manager:
        llm_kwargs["config_manager"] = config_manager
        if config_manager.active_llm_profile:
            llm_kwargs["profile"] = config_manager.active_llm_profile
    try:
        components["llm"] = provider_registry.create("llm", **llm_kwargs)
        if components["llm"]:
            logger.info("LLM Manager initialised")
    except ProviderLookupError:
        logger.warning("LLM provider not registered — some features disabled")
        components["llm"] = None
    except Exception as exc:
        logger.error("Failed to initialise LLM Manager: %s", exc)
        components["llm"] = None

    # ── Security manager ────────────────────────────────────────────
    try:
        security_manager = provider_registry.create(
            "security", config_manager=config_manager
        )
    except (ProviderLookupError, Exception):
        security_manager = None
    components["security_manager"] = security_manager

    # ── RAG pipeline ────────────────────────────────────────────────
    try:
        rag_pipeline = provider_registry.create(
            "rag", "pipeline", config_manager=config_manager
        )
    except (ProviderLookupError, Exception):
        rag_pipeline = None
    components["rag_pipeline"] = rag_pipeline

    # ── Vision processor ────────────────────────────────────────────
    try:
        components["vision_processor"] = provider_registry.create("vision")
    except (ProviderLookupError, Exception):
        components["vision_processor"] = None

    # ── Action executor ─────────────────────────────────────────────
    try:
        components["action_executor"] = provider_registry.create(
            "actions",
            consent_callback=prompt_user_consent,
            config_manager=config_manager,
            security_manager=security_manager,
            rag_pipeline=rag_pipeline,
        )
        if components["action_executor"]:
            logger.info("Action executor initialised")
    except (ProviderLookupError, Exception) as exc:
        logger.warning("Action executor unavailable: %s", exc)
        components["action_executor"] = None

    # ── Tool-Calling Agent (with SOTA modules) ───────────────────────
    from .core.agent import ToolCallingAgent

    agent = None
    if components["llm"] and components["action_executor"]:
        try:
            # The agent auto-initialises CognitiveKernel, TaskPlanner,
            # and GuardrailsManager when auto_init_sota=True (default).
            agent = ToolCallingAgent(
                llm=components["llm"],
                action_executor=components["action_executor"],
                auto_init_sota=True,
            )
            logger.info(
                "Agent initialised (native_tools=%s, cognitive_kernel=%s)",
                agent.supports_native_tools,
                agent.cognitive_kernel is not None,
            )
        except Exception as exc:
            logger.warning("Agent init failed: %s", exc)
    components["agent"] = agent

    # ── Audio components ────────────────────────────────────────────
    components["audio_available"] = False
    components["audio_utils"] = None
    components["speech_processor"] = None
    components["speech_generator"] = None
    components["hotword_detector"] = None
    audio_config = getattr(
        getattr(config_manager, "config", None), "audio", None
    )
    components["audio_config"] = audio_config

    if getattr(args, "mode", "text") in ("audio", "mixed", "auto"):
        try:
            audio_utils = provider_registry.create("audio", "utils")
            if config_manager:
                speech_processor = provider_registry.create(
                    "audio", "processor", config_manager=config_manager
                )
                speech_generator = provider_registry.create(
                    "audio", "generator",
                    config_manager=config_manager,
                    audio_config=audio_config,
                )
            else:
                speech_processor = provider_registry.create("audio", "processor")
                speech_generator = provider_registry.create("audio", "generator")
            components["audio_utils"] = audio_utils
            components["speech_processor"] = speech_processor
            components["speech_generator"] = speech_generator
            if audio_utils and audio_config:
                audio_utils.configure(
                    sample_rate=getattr(audio_config, "sample_rate", None),
                    chunk_size=getattr(audio_config, "chunk_size", None),
                    device_id=getattr(audio_config, "device_id", None),
                    input_threshold=getattr(audio_config, "input_threshold", None),
                )
            components["audio_available"] = bool(audio_utils and speech_processor)
            if components["audio_available"] and HotwordDetector:
                if audio_config and audio_config.hotword:
                    try:
                        components["hotword_detector"] = HotwordDetector(
                            audio_config.hotword,
                            sensitivity=audio_config.hotword_sensitivity or 0.5,
                            energy_floor=getattr(audio_config, "input_threshold", 0.01) or 0.01,
                        )
                    except Exception:
                        pass
        except (ProviderLookupError, Exception):
            components["audio_available"] = False

    # ── Monitoring / Cache / Resources ──────────────────────────────
    try:
        components["performance_monitor"] = PerformanceMonitor() if PerformanceMonitor else None
        components["cache_manager"] = CacheManager() if CacheManager else None
        components["resource_manager"] = ResourceManager() if ResourceManager else None
    except Exception:
        components["performance_monitor"] = None
        components["cache_manager"] = None
        components["resource_manager"] = None

    # Register memory-pressure callback
    rm = components.get("resource_manager")
    llm = components.get("llm")
    if llm and hasattr(llm, "unload_model") and rm:
        try:
            rm.register_memory_pressure_callback(llm.unload_model)
        except Exception:
            pass

    return components


# ═══════════════════════════════════════════════════════════════════════════════
# Command handling
# ═══════════════════════════════════════════════════════════════════════════════


def handle_command(cmd: str, args, components: Dict[str, Any]):
    """Handle slash-style commands.  Returns ``(should_continue, handled)``."""
    cmd_lower = cmd.strip().lower()

    if cmd_lower in ("quit", "exit", "sair"):
        print(bold(_("exiting")))
        return False, True

    if cmd_lower == "help":
        display_help(args, components)
        return True, True

    if cmd_lower == "status":
        display_status(components, args)
        return True, True

    if cmd_lower == "models":
        display_models(args)
        return True, True

    if cmd_lower == "profiles":
        display_profiles(components)
        return True, True

    if cmd_lower.startswith("switch-profile"):
        _switch_profile(cmd, components)
        return True, True

    if cmd_lower == "clear":
        _clear_context(components)
        return True, True

    if cmd_lower == "reset":
        agent = components.get("agent")
        if agent and hasattr(agent, "reset_conversation"):
            agent.reset_conversation()
            print(yellow("Conversation memory cleared — fresh session started."))
        else:
            _clear_context(components)
        return True, True

    if cmd_lower == "stats":
        agent = components.get("agent")
        if agent and hasattr(agent, "session_stats"):
            s = agent.session_stats
            print(cyan("─── Session Stats ───"))
            print(f"  Turns:       {s.get('turns', 0)}")
            print(f"  Tool calls:  {s.get('tool_calls', 0)}")
            print(f"  Tokens:      {s.get('tokens_used', 0)}")
            print(f"  Uptime:      {s.get('uptime_seconds', 0)}s")
        else:
            print(yellow("Agent stats not available."))
        return True, True

    if cmd_lower == "memory":
        agent = components.get("agent")
        if agent and hasattr(agent, "memory"):
            msgs = agent.memory.get_messages_for_llm()
            print(cyan(f"─── Conversation Memory ({len(msgs)} messages) ───"))
            for m in msgs[-10:]:
                role = m.get("role", "?")
                content = m.get("content", "")[:120]
                print(f"  [{role}] {content}")
        else:
            print(yellow("Conversation memory not available."))
        return True, True

    if cmd_lower == "audio" and components.get("speech_processor"):
        args.mode = "audio"
        print(yellow("Switched to audio mode."))
        return True, True

    if cmd_lower == "text" and getattr(args, "mode", "") == "audio":
        args.mode = "text"
        print(cyan("Switched to text mode."))
        return True, True

    return True, False  # not a command


def _switch_profile(cmd: str, components: Dict[str, Any]) -> None:
    config_manager = components.get("config_manager")
    if not config_manager or not getattr(config_manager, "config", None):
        print(red("Configuration not available."))
        return
    parts = cmd.split()
    if len(parts) < 2:
        print(red("Usage: switch-profile <name>"))
        return
    name = parts[1]
    profiles = config_manager.list_llm_profiles()
    if name not in profiles:
        print(red(f"Profile '{name}' not found. Available: {', '.join(profiles)}"))
        return
    try:
        config_manager.active_llm_profile = name
        profile = config_manager.get_llm_profile(name)
        llm = components.get("llm")
        if llm and hasattr(llm, "update_config"):
            llm.update_config(profile.apply_overrides(config_manager.config.llm))
        print(green(f"Profile switched to: {getattr(profile, 'label', name)}"))
    except Exception as exc:
        print(red(f"Error switching profile: {exc}"))


def _clear_context(components: Dict[str, Any]) -> None:
    cm = components.get("cache_manager")
    pm = components.get("performance_monitor")
    if cm and hasattr(cm, "clear_all"):
        cm.clear_all()
    if pm and hasattr(pm, "optimize_performance"):
        pm.optimize_performance()
    print(yellow("Context cleared."))


# ═══════════════════════════════════════════════════════════════════════════════
# Audio input helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _capture_audio(args, components) -> Optional[str]:
    """Try to capture speech from the microphone and return transcription."""
    au = components.get("audio_utils")
    sp = components.get("speech_processor")
    ac = components.get("audio_config")
    if not (au and sp and components.get("audio_available")):
        return None

    try:
        print(bold(_msg("audio_listening", "Listening… (Ctrl+C for text)")))
        buf = au.capture_with_vad(speech_processor=sp, audio_config=ac)
        if buf is None or buf.size == 0:
            if hasattr(sp, "listen_microphone"):
                tx = sp.listen_microphone()
                if tx:
                    print(green(f"You said: {tx.strip()}"))
                    return tx.strip()
            return None
        tx = sp.transcribe_audio_data(buf.astype(np.float32).tobytes(), au.sample_rate)
        if tx and tx.strip():
            print(green(f"You said: {tx.strip()}"))
            return tx.strip()
        return None
    except KeyboardInterrupt:
        print(bold("\nSwitching to text…"))
        args.mode = "text"
        return None
    except Exception as exc:
        logger.error("Audio capture error: %s", exc)
        args.mode = "text"
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Main interaction loop
# ═══════════════════════════════════════════════════════════════════════════════


def run_interaction_loop(args, components: Dict[str, Any]) -> None:
    """Interactive REPL — delegates all user requests to ToolCallingAgent."""
    agent = components.get("agent")

    while True:
        try:
            # ── Collect input ───────────────────────────────────────
            user_input: Optional[str] = None
            context: Dict[str, Any] = {}

            if args.mode == "audio" and components.get("audio_available"):
                user_input = _capture_audio(args, components)
                if user_input:
                    context["audio"] = user_input

            if user_input is None:
                if args.mode == "audio":
                    continue
                try:
                    user_input = input(bold("You: ")).strip()
                except (EOFError, KeyboardInterrupt):
                    print(bold("\nGoodbye!"))
                    return

            if not user_input:
                continue

            # ── Commands ────────────────────────────────────────────
            cont, handled = handle_command(user_input, args, components)
            if not cont:
                return
            if handled:
                continue

            # ── Image context ───────────────────────────────────────
            vp = components.get("vision_processor")
            img = getattr(args, "image_input", None)
            if img and vp:
                try:
                    result = vp.process_image(img)
                    context["image"] = result
                    args.image_input = None
                except Exception:
                    args.image_input = None

            # ── Agent execution ─────────────────────────────────────
            if agent:
                print(cyan("M.I.A: "), end="", flush=True)
                try:
                    response = agent.run(user_input, context=context or None)
                    print(response)
                except Exception as exc:
                    logger.error("Agent error: %s", exc)
                    print(red(f"Error: {exc}"))
            elif components.get("llm"):
                # Fallback: direct LLM query (no tool use)
                llm = components["llm"]
                print(cyan(_("thinking")))
                try:
                    resp = llm.query(user_input) or ""
                    print(cyan("M.I.A: ") + resp)
                except Exception as exc:
                    print(red(f"LLM error: {exc}"))
            else:
                print(red("No LLM available. Check your configuration."))

            # ── Optional TTS ────────────────────────────────────────
            ac = components.get("audio_config")
            sg = components.get("speech_generator")
            if ac and getattr(ac, "tts_enabled", False) and sg and agent:
                try:
                    sg.enqueue_speech(response)
                except Exception:
                    pass

        except KeyboardInterrupt:
            print(bold("\nGoodbye!"))
            return
        except Exception as exc:
            logger.error("Unexpected error: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════════
# Resource cleanup
# ═══════════════════════════════════════════════════════════════════════════════


def cleanup_resources(components: Dict[str, Any]) -> None:
    """Shut down subsystems and release resources."""
    logger.info("Cleaning up resources…")
    try:
        pm = components.get("performance_monitor")
        cm = components.get("cache_manager")
        rm = components.get("resource_manager")

        if pm and hasattr(pm, "stop_monitoring"):
            pm.stop_monitoring()
        if pm and hasattr(pm, "cleanup"):
            pm.cleanup()
        if cm and hasattr(cm, "clear_all"):
            cm.clear_all()
        if rm and hasattr(rm, "stop"):
            rm.stop()

        logger.info("Resource cleanup completed")
    except Exception as exc:
        logger.error("Error during cleanup: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Top-level entry point for ``python -m mia``."""
    components: Dict[str, Any] = {}
    try:
        args = parse_arguments()
        setup_logging(args)
        init_localization(getattr(args, "language", "en"))

        # ── Info flag ───────────────────────────────────────────────
        if getattr(args, "info", False):
            info = get_full_version()
            print(bold(f"{info['title']} v{info['version']}"))
            print(f"Description: {info['description']}")
            print(f"Author: {info['author']}")
            print(f"License: {info['license']}")
            print(f"Build: {info['build']}")
            print(f"Status: {info['status']}")
            print(f"Python: {sys.version}")
            print(f"Platform: {sys.platform}")
            return

        # ── Web UI mode ─────────────────────────────────────────────
        if getattr(args, "web", False) or getattr(args, "mode", "text") == "web":
            try:
                from .web.webui import run_webui

                host = getattr(args, "host", "0.0.0.0")
                port = getattr(args, "port", 8080)
                run_webui(host=host, port=port)
                return
            except ImportError as exc:
                print(red(f"Web UI requires additional dependencies: {exc}"))
                print("Install with: pip install fastapi uvicorn")
                return

        # ── Deprecated flag migration ───────────────────────────────
        if getattr(args, "text_only", False):
            args.mode = "text"
        elif getattr(args, "audio_mode", False):
            args.mode = "audio"

        # ── Bootstrap ───────────────────────────────────────────────
        print(yellow(_("initializing")))
        components = initialize_components(args)
        display_status(components, args)
        run_interaction_loop(args, components)

    except Exception as exc:
        logger.error("Critical error: %s", exc)
        sys.exit(1)
    finally:
        cleanup_resources(components)


if __name__ == "__main__":
    main()

