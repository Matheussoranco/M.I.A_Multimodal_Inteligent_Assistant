import argparse
import json
import logging
import os
import re
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import colorama

    colorama.init()
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False

from .__version__ import __version__, get_full_version
from .audio.hotword_detector import HotwordDetector
from .localization import _, init_localization
from .providers import ProviderLookupError, provider_registry

try:
    from .cache_manager import CacheManager
    from .config_manager import ConfigManager
    from .error_handler import (
        global_error_handler,
        safe_execute,
        with_error_handling,
    )
    from .exceptions import *
    from .performance_monitor import PerformanceMonitor
    from .resource_manager import ResourceManager
except ImportError as e:
    print(f"Warning: Some core modules could not be imported: {e}")
    ConfigManager = None
    ResourceManager = None
    PerformanceMonitor = None
    CacheManager = None

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


# Color formatting functions
def bold(text):
    """Make text bold using ANSI escape codes"""
    return f"\033[1m{text}\033[0m"


def green(text):
    """Make text green using ANSI escape codes"""
    return f"\033[32m{text}\033[0m"


def yellow(text):
    """Make text yellow using ANSI escape codes"""
    return f"\033[33m{text}\033[0m"


def red(text):
    """Make text red using ANSI escape codes"""
    return f"\033[31m{text}\033[0m"


def cyan(text):
    """Make text cyan using ANSI escape codes"""
    return f"\033[36m{text}\033[0m"


def _msg(key: str, default: str, **kwargs) -> str:
    """Return localized message or provided default when missing."""
    localized = _(key, **kwargs)
    if localized != key:
        return localized

    if kwargs:
        try:
            return default.format(**kwargs)
        except Exception:
            return default
    return default


def _extract_filepath(
    text: str, extensions: Optional[List[str]] = None
) -> Optional[str]:
    """Extract a probable file path from free-form text."""

    if not text:
        return None

    candidates = re.findall(r"[\w./\\:-]+", text)
    if not candidates:
        return None

    if extensions:
        lowered_exts = [ext.lower() for ext in extensions]
        for token in candidates:
            lowered = token.lower()
            if any(lowered.endswith(ext) for ext in lowered_exts):
                return token.strip("\"'")

    return candidates[-1].strip("\"'") if candidates else None


def prompt_user_consent(
    action: str, params: Optional[Dict[str, Any]] = None
) -> bool:
    """Ask the user to confirm sensitive actions before execution."""
    auto_consent = os.getenv("MIA_AUTO_CONSENT", "").strip().lower()
    if auto_consent in {"1", "true", "yes", "y", "sim", "s"}:
        return True

    action_label = action.replace("_", " ")
    target = ""
    if params:
        for key in ("recipient", "to", "path", "url", "filename"):
            value = params.get(key)
            if value:
                target = str(value)
                break

    message = f"⚠️ Confirmar ação '{action_label}'"
    if target:
        message += f" para '{target}'"
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
        print("Por favor, responda com 'y' ou 'n'.")


def _suppress_warnings_env() -> None:
    """Set environment variables and warning filters to reduce noise (opt-in)."""
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


logger = logging.getLogger(__name__)


def choose_mode():
    """Interactive mode selection for M.I.A initialization."""
    print("\n" + "=" * 60)
    print("          M.I.A - Multimodal Intelligent Assistant")
    print("=" * 60)
    print("\nChoose your interaction mode:")
    print("1. Text-only mode")
    print("2. Audio mode")
    print("3. Mixed mode (default)")

    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            if choice == "1":
                return "text"
            elif choice == "2":
                return "audio"
            elif choice == "3" or choice == "":
                return "mixed"
            else:
                print("Invalid choice. Please select 1, 2, or 3.")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            sys.exit(0)


def detect_and_execute_agent_commands(
    user_input: str, action_executor
) -> Tuple[bool, str]:
    """
    Detects and executes agent commands based on user input.

    Args:
        user_input: User input text
        action_executor: ActionExecutor instance

    Returns:
        Tuple (command_executed, result_message)
    """
    if not action_executor or not user_input:
        return False, ""

    user_input_lower = user_input.lower()

    # File creation command detection
    if any(
        keyword in user_input_lower
        for keyword in ["criar arquivo", "create file", "novo arquivo"]
    ):
        try:
            # Extract filename - take last word that could be a filename
            words = user_input.split()
            filename = None

            # Look for word that seems like a filename
            for word in reversed(words):
                if word.lower() not in [
                    "criar",
                    "arquivo",
                    "create",
                    "file",
                    "novo",
                ]:
                    filename = word.replace('"', "").replace("'", "")
                    break

            if not filename:
                # If not found, use default name
                filename = "novo_arquivo.txt"

            # Extract content if specified
            content = _(
                "file_created_timestamp",
                timestamp=__import__("datetime")
                .datetime.now()
                .strftime("%Y-%m-%d %H:%M:%S"),
            )
            if (
                "conteúdo" in user_input_lower
                or "content" in user_input_lower
                or "com" in user_input_lower
            ):
                content_start = user_input.lower().find("conteúdo")
                if content_start == -1:
                    content_start = user_input.lower().find("content")
                if content_start == -1:
                    content_start = user_input.lower().find("com")

                if content_start != -1:
                    remaining = user_input[content_start:].strip()
                    if '"' in remaining or "'" in remaining:
                        # Extract text between quotes

                        import re

                        match = re.search(r'["\']([^"\']*)["\']', remaining)
                        if match:
                            content = match.group(1)

            result = action_executor.execute(
                "create_file", {"path": filename, "content": content}
            )
            return True, _("agent_file_created", filename=filename)
        except Exception as e:
            return True, _("agent_file_error", error=e)

    # Document commands (DOCX)
    elif any(
        keyword in user_input_lower
        for keyword in ["docx", "documento", "word", "gera doc"]
    ) and ("criar" in user_input_lower or "gerar" in user_input_lower):
        try:
            import re

            quote_pairs = re.findall(r'"([^\"]+)"|\'([^\']+)\'', user_input)
            extracted = [first or second for first, second in quote_pairs]
            title = extracted[0] if extracted else None
            summary = extracted[1] if len(extracted) > 1 else None

            template = "proposal"
            if "relatorio" in user_input_lower or "report" in user_input_lower:
                template = "report"

            result = action_executor.execute(
                "create_docx",
                {
                    "title": title,
                    "summary": summary,
                    "template": template,
                },
            )
            return True, result
        except Exception as e:
            return True, _("agent_analysis_error", error=e)

    # Document commands (PDF)
    elif "pdf" in user_input_lower and (
        "criar" in user_input_lower or "gerar" in user_input_lower
    ):
        try:
            import re

            quote_pairs = re.findall(r'"([^\"]+)"|\'([^\']+)\'', user_input)
            extracted = [first or second for first, second in quote_pairs]
            title = extracted[0] if extracted else None
            summary = extracted[1] if len(extracted) > 1 else None

            template = (
                "report"
                if "relatorio" in user_input_lower
                or "report" in user_input_lower
                else "proposal"
            )

            result = action_executor.execute(
                "create_pdf",
                {
                    "title": title,
                    "summary": summary,
                    "template": template,
                },
            )
            return True, result
        except Exception as e:
            return True, _("agent_analysis_error", error=e)

    # Sandbox command
    elif any(
        keyword in user_input_lower for keyword in ["sandbox", "wasm", "wasi"]
    ):
        try:
            result = action_executor.execute(
                "run_sandboxed",
                {
                    "module_path": _extract_filepath(
                        user_input, extensions=[".wasm", ".wat"]
                    ),
                },
            )
            return True, result
        except Exception as e:
            return True, _("agent_analysis_error", error=e)

    # Code analysis command detection
    elif any(
        keyword in user_input_lower
        for keyword in [
            "analisar c?digo",
            "analisar código",
            "analyze code",
            "analisar arquivo",
        ]
    ):
        try:
            # Extract filename
            words = user_input.split()
            filepath = None

            for word in words:
                if "." in word and any(
                    ext in word.lower()
                    for ext in [".py", ".js", ".ts", ".java", ".cpp", ".c"]
                ):
                    filepath = word
                    break

            if not filepath:
                return True, _("agent_specify_file")

            result = action_executor.execute(
                "analyze_code", {"path": filepath}
            )
            return True, _("agent_code_analysis", result=result)
        except Exception as e:
            return True, _("agent_analysis_error", error=e)

    # WhatsApp message command detection
    elif any(
        keyword in user_input_lower
        for keyword in [
            "whatsapp",
            "enviar whatsapp",
            "send whatsapp",
            "mensagem whatsapp",
        ]
    ):
        try:
            import re

            # Extract phone number (e.g., +5511999999999) if present
            phone_match = re.search(r"(\+\d{10,15}|\d{10,15})", user_input)
            recipient = phone_match.group(1) if phone_match else ""

            # Extract message between quotes, else take remainder
            msg_match = re.search(r'["\']([^"\']+)["\']', user_input)
            message = msg_match.group(1) if msg_match else user_input

            result = action_executor.execute(
                "send_whatsapp",
                {"recipient": recipient, "message": message},
            )
            return True, result
        except Exception as e:
            return True, _("agent_analysis_error", error=e)

    # Telegram message command detection
    elif any(
        keyword in user_input_lower
        for keyword in [
            "telegram",
            "enviar telegram",
            "send telegram",
            "mensagem telegram",
        ]
    ):
        try:
            import re

            # Extract chat id or username (numeric id recommended)
            chat_match = re.search(
                r"(?:para|to)\s+(@?[\w\d_\-]+)", user_input_lower
            )
            to_id = chat_match.group(1) if chat_match else ""

            msg_match = re.search(r'["\']([^"\']+)["\']', user_input)
            message = msg_match.group(1) if msg_match else user_input

            result = action_executor.execute(
                "send_telegram",
                {"to": to_id, "message": message},
            )
            return True, result
        except Exception as e:
            return True, _("agent_analysis_error", error=e)

    # Email command detection
    elif any(
        keyword in user_input_lower
        for keyword in ["enviar email", "send email", "mandar email", "email"]
    ):
        try:
            import re

            to_match = re.search(
                r"(?:para|to)\s+([\w\.-]+@[\w\.-]+)", user_input_lower
            )
            to_addr = to_match.group(1) if to_match else ""

            subj_match = re.search(
                r"assunto\s*[\:\-]?\s*[\"\']([^\"\']+)[\"\']|subject\s*[\:\-]?\s*[\"\']([^\"\']+)[\"\']",
                user_input_lower,
            )
            subject = next(
                (
                    g
                    for g in (
                        subj_match.group(1) if subj_match else None,
                        subj_match.group(2) if subj_match else None,
                    )
                    if g
                ),
                "(No Subject)",
            )

            body_match = re.search(
                r"(?:corpo|body)\s*[\:\-]?\s*[\"\']([^\"\']+)[\"\']",
                user_input,
            )
            body = body_match.group(1) if body_match else user_input

            params = {"to": to_addr, "subject": subject, "body": body}
            result = action_executor.execute("send_email", params)
            return True, result
        except Exception as e:
            return True, _("agent_analysis_error", error=e)

    # Web navigation command detection
    elif any(
        keyword in user_input_lower
        for keyword in ["abrir ", "navegar", "open url", "visit", "acessar"]
    ):
        try:
            import re

            url_match = re.search(r"(https?://\S+)", user_input)
            url = url_match.group(1) if url_match else None
            if not url:
                return True, _("agent_specify_file")
            result = action_executor.execute("web_automation", {"url": url})
            return True, result
        except Exception as e:
            return True, _("agent_analysis_error", error=e)

    # Spreadsheet creation command detection
    elif any(
        keyword in user_input_lower
        for keyword in [
            "criar planilha",
            "create spreadsheet",
            "create sheet",
            "nova planilha",
        ]
    ):
        try:
            import re

            # look for filename with .xlsx or .csv
            file_match = re.search(
                r"\b(\S+\.(?:xlsx|csv))\b", user_input_lower
            )
            filename = file_match.group(1) if file_match else "planilha.xlsx"
            result = action_executor.execute(
                "create_sheet", {"filename": filename}
            )
            return True, result
        except Exception as e:
            return True, _("agent_analysis_error", error=e)

    # Presentation creation command detection
    elif any(
        keyword in user_input_lower
        for keyword in [
            "criar apresentação",
            "criar apresentacao",
            "create presentation",
            "criar powerpoint",
            "create powerpoint",
        ]
    ):
        try:
            import re

            file_match = re.search(r"\b(\S+\.pptx)\b", user_input_lower)
            filename = (
                file_match.group(1) if file_match else "apresentacao.pptx"
            )
            # Optional title in quotes
            title_match = re.search(r'["\']([^"\']+)["\']', user_input)
            title = title_match.group(1) if title_match else ""
            result = action_executor.execute(
                "create_presentation", {"filename": filename, "title": title}
            )
            return True, result
        except Exception as e:
            return True, _("agent_analysis_error", error=e)

    return False, ""


def parse_arguments():
    """Parse command line arguments and return args object."""
    parser = argparse.ArgumentParser(
        description="M.I.A - Multimodal Intelligent Assistant"
    )
    parser.add_argument(
        "--mode",
        choices=["text", "audio", "mixed", "auto"],
        default="mixed",
        help="Interaction mode: text|audio|mixed|auto",
    )
    parser.add_argument(
        "--text-only", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--audio-mode", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--language",
        choices=["en", "pt"],
        default=None,
        help="Interface language (en=English, pt=Portuguese)",
    )
    parser.add_argument(
        "--image-input", type=str, default=None, help="Image to process"
    )
    parser.add_argument(
        "--model-id", type=str, default="gpt-oss:latest", help="Model ID"
    )
    parser.add_argument(
        "--profile",
        type=str,
        help="Nome do perfil de LLM configurado em config.yaml",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"M.I.A {__version__}",
        help="Show version information",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show detailed version and system information",
    )
    return parser.parse_args()


def setup_logging(args):
    """Setup logging configuration based on arguments."""
    level = logging.DEBUG if getattr(args, "debug", False) else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    for name in (
        "transformers",
        "torch",
        "tensorflow",
        "numba",
        "chromadb",
        "urllib3",
        "requests",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
    _suppress_warnings_env()


def initialize_components(args):
    """Initialize all M.I.A components and return them in a dictionary."""
    components = {}
    configuration_error_cls = globals().get("ConfigurationError", Exception)

    # Load configuration first so the rest of the pipeline can consume overrides
    config_manager = None
    if ConfigManager:
        try:
            config_manager = ConfigManager()
            config_manager.load_config()
            if getattr(args, "profile", None):
                try:
                    config_manager.activate_llm_profile(args.profile)
                except configuration_error_cls as exc:
                    logger.warning(
                        "Requested profile '%s' unavailable: %s",
                        args.profile,
                        exc,
                    )
            elif (
                config_manager.config
                and config_manager.config.default_llm_profile
            ):
                try:
                    config_manager.activate_llm_profile(
                        config_manager.config.default_llm_profile
                    )
                except configuration_error_cls as exc:
                    logger.warning(
                        "Failed to activate default profile '%s': %s",
                        config_manager.config.default_llm_profile,
                        exc,
                    )
        except Exception as exc:  # pragma: no cover - defensive load
            logger.warning(
                "Configuration manager failed to load config: %s", exc
            )
            config_manager = None
    components["config_manager"] = config_manager

    # Initialize device
    device = "cpu"
    if HAS_TORCH and torch is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    components["device"] = device
    logger.info(f"Using device: {device}")

    # Initialize LLM Manager lazily via provider registry
    llm_kwargs: Dict[str, Any] = {}
    if getattr(args, "model_id", None):
        llm_kwargs["model_id"] = args.model_id
    if config_manager:
        llm_kwargs["config_manager"] = config_manager
        if config_manager.active_llm_profile:
            llm_kwargs["profile"] = config_manager.active_llm_profile
    elif getattr(args, "profile", None):
        llm_kwargs["profile"] = args.profile
    try:
        components["llm"] = provider_registry.create("llm", **llm_kwargs)
        if components["llm"]:
            logger.info("LLM Manager initialized successfully")
        else:
            logger.warning(
                "LLM provider returned no instance - text processing disabled"
            )
    except ProviderLookupError:
        logger.warning(
            "LLM provider not registered - some features will be disabled"
        )
        components["llm"] = None
    except Exception as e:
        logger.error(f"Failed to initialize LLM Manager: {e}")
        components["llm"] = None

    # Initialize audio components if not text-only mode
    components["audio_available"] = False
    components["audio_utils"] = None
    components["speech_processor"] = None
    components["speech_generator"] = None
    components["hotword_detector"] = None
    audio_config = getattr(
        getattr(config_manager, "config", None), "audio", None
    )
    components["audio_config"] = audio_config

    if args.mode in ("audio", "mixed", "auto"):
        try:
            audio_utils = provider_registry.create("audio", "utils")
            if config_manager:
                speech_processor = provider_registry.create(
                    "audio", "processor", config_manager=config_manager
                )
                speech_generator = provider_registry.create(
                    "audio",
                    "generator",
                    config_manager=config_manager,
                    audio_config=audio_config,
                )
            else:
                speech_processor = provider_registry.create(
                    "audio", "processor"
                )
                speech_generator = provider_registry.create(
                    "audio", "generator"
                )
            components["audio_utils"] = audio_utils
            components["speech_processor"] = speech_processor
            components["speech_generator"] = speech_generator
            if audio_utils and audio_config:
                audio_utils.configure(
                    sample_rate=getattr(audio_config, "sample_rate", None),
                    chunk_size=getattr(audio_config, "chunk_size", None),
                    device_id=getattr(audio_config, "device_id", None),
                    input_threshold=getattr(
                        audio_config, "input_threshold", None
                    ),
                )
            components["audio_available"] = bool(
                audio_utils and speech_processor
            )
            if components["audio_available"]:
                if audio_config and audio_config.hotword:
                    try:
                        components["hotword_detector"] = HotwordDetector(
                            audio_config.hotword,
                            sensitivity=audio_config.hotword_sensitivity
                            or 0.5,
                            energy_floor=getattr(
                                audio_config, "input_threshold", 0.01
                            )
                            or 0.01,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Hotword detector initialization failed: %s", exc
                        )
                logger.info("Audio components initialized successfully")
            else:
                logger.warning("Audio components unavailable or returned None")
        except ProviderLookupError:
            logger.warning("Audio providers not registered")
            components["audio_available"] = False
        except Exception as e:
            logger.warning(f"Audio components failed to initialize: {e}")
            components["audio_available"] = False

    # Initialize security manager
    try:
        security_manager = provider_registry.create(
            "security", config_manager=config_manager
        )
    except ProviderLookupError:
        security_manager = None
    except Exception as exc:
        logger.warning("Security manager failed to initialize: %s", exc)
        security_manager = None
    components["security_manager"] = security_manager

    # Initialize RAG pipeline
    try:
        rag_pipeline = provider_registry.create(
            "rag", "pipeline", config_manager=config_manager
        )
    except ProviderLookupError:
        rag_pipeline = None
    except Exception as exc:
        logger.warning("RAG pipeline unavailable: %s", exc)
        rag_pipeline = None
    components["rag_pipeline"] = rag_pipeline

    # Initialize vision processor
    try:
        components["vision_processor"] = provider_registry.create("vision")
        if components["vision_processor"]:
            logger.info("Vision processor initialized successfully")
        else:
            logger.warning("Vision provider returned no instance")
    except ProviderLookupError:
        logger.warning("Vision provider not registered")
        components["vision_processor"] = None
    except Exception as e:
        logger.warning(f"Vision processor failed to initialize: {e}")
        components["vision_processor"] = None

    # Initialize action executor
    try:
        components["action_executor"] = provider_registry.create(
            "actions",
            consent_callback=prompt_user_consent,
            config_manager=config_manager,
            security_manager=security_manager,
            rag_pipeline=rag_pipeline,
        )
        if components["action_executor"]:
            logger.info("Action executor initialized successfully")
        else:
            logger.warning("Action executor provider returned no instance")
    except ProviderLookupError:
        logger.warning("Action executor provider not registered")
        components["action_executor"] = None
    except Exception as e:
        logger.warning(f"Action executor failed to initialize: {e}")
        components["action_executor"] = None

    # Initialize other components
    try:
        if PerformanceMonitor:
            components["performance_monitor"] = PerformanceMonitor()
        else:
            components["performance_monitor"] = None

        if CacheManager:
            components["cache_manager"] = CacheManager()
        else:
            components["cache_manager"] = None

        if ResourceManager:
            components["resource_manager"] = ResourceManager()
        else:
            components["resource_manager"] = None

        logger.info("Additional components initialized successfully")
    except Exception as e:
        logger.warning(f"Some additional components failed to initialize: {e}")
        components["performance_monitor"] = None
        components["cache_manager"] = None
        components["resource_manager"] = None

    return components


def display_status(components, args):
    """Display system status information."""
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

    # Display detailed status
    print(bold("\n📊 M.I.A Status"))
    print(bold("─" * 40))
    mode_label = {
        "text": green("Text-only"),
        "audio": yellow("Audio"),
        "mixed": cyan("Mixed"),
        "auto": cyan("Auto"),
    }.get(args.mode, cyan("Mixed"))
    print(f"  {bold('Mode:')} {mode_label}")
    if active_profile:
        print(f"  {bold('Perfil LLM:')} {cyan(active_profile)}")
    print(
        f"  {bold('Model:')} {yellow(llm.model_id if llm else getattr(args, 'model_id', ''))}"
    )
    print(
        f"  {bold('LLM:')} {(green('Connected') if llm and hasattr(llm, 'is_available') and llm.is_available() else red('Disconnected'))}"
    )
    print(
        f"  {bold('Audio:')} {(green('Available') if audio_available else red('Not available'))}"
    )
    print(f"  {bold('Device:')} {cyan(device)}")

    if performance_monitor and hasattr(
        performance_monitor, "get_current_metrics"
    ):
        perf_metrics = performance_monitor.get_current_metrics()
        if perf_metrics:
            print(
                f"  {bold('CPU:')} {yellow(f'{perf_metrics.cpu_percent:.1f}%')}"
            )
            print(
                f"  {bold('Memory:')} {yellow(f'{perf_metrics.memory_percent:.1f}%')}"
            )

    if cache_manager and hasattr(cache_manager, "get_stats"):
        cache_stats = cache_manager.get_stats()
        hit_rate = cache_stats.get("memory_cache", {}).get("hit_rate", 0)
        print(
            f"  {bold('Cache Hit Rate:')} {green('{:.1%}'.format(hit_rate))}"
        )

    print(bold("─" * 40))


def process_image_input(args, components):
    """Process image input if provided."""
    vision_processor = components.get("vision_processor")
    if hasattr(args, "image_input") and args.image_input and vision_processor:
        try:
            result = vision_processor.process_image(args.image_input)
            args.image_input = None  # Clear after processing
            logger.info("Image processed successfully")
            return {"image": result}
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            args.image_input = None  # Clear even on error
            return {}
    return {}


def process_audio_input(args, components):
    """Process audio input and return transcribed text."""
    audio_utils = components.get("audio_utils")
    speech_processor = components.get("speech_processor")
    audio_config = components.get("audio_config")
    hotword_detector = components.get("hotword_detector")

    if not (
        args.mode == "audio"
        and components.get("audio_available")
        and audio_utils
        and speech_processor
    ):
        return None, {}

    try:
        if audio_config and audio_config.hotword_enabled and hotword_detector:
            print(
                bold(
                    _msg(
                        "audio_waiting_hotword",
                        "🪄 Diga '{hotword}' para ativar",
                        hotword=audio_config.hotword,
                    )
                )
            )
            if not _await_hotword(
                audio_utils, speech_processor, hotword_detector, audio_config
            ):
                print(
                    yellow(
                        _msg(
                            "audio_hotword_timeout",
                            "⏱️ Tempo limite aguardando hotword.",
                        )
                    )
                )
                return None, {}

        if audio_config and audio_config.push_to_talk:
            prompt = _msg(
                "audio_push_to_talk", "Pressione e segure espaço para falar"
            )
            if not audio_utils.wait_for_push_to_talk(prompt=prompt):
                print(
                    yellow(
                        _msg(
                            "audio_push_to_talk_cancel",
                            "🔕 Captura cancelada.",
                        )
                    )
                )
                return None, {}

        print(
            bold(
                _msg("audio_listening", "🎤 Escutando... (Ctrl+C para texto)")
            )
        )
        audio_buffer = audio_utils.capture_with_vad(
            speech_processor=speech_processor,
            audio_config=audio_config,
        )

        if audio_buffer is None or audio_buffer.size == 0:
            print(red(_msg("audio_no_speech", "🔇 Nenhuma fala detectada.")))
            return None, {}

        transcription = speech_processor.transcribe_audio_data(
            audio_buffer.astype(np.float32).tobytes(),
            audio_utils.sample_rate,
        )

        user_input = (
            transcription.strip() if isinstance(transcription, str) else ""
        )
        if not user_input:
            print(
                red(
                    _msg(
                        "audio_transcription_failed",
                        "❌ Não foi possível transcrever o áudio.",
                    )
                )
            )
            return None, {}

        print(green(f"🎙️  You said: {user_input}"))
        return user_input, {"audio": user_input}

    except KeyboardInterrupt:
        print(bold("\n🔥 Switching to text mode..."))
        args.mode = "text"
        return None, {}
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        print(red("❌ Audio processing failed. Switching to text mode."))
        args.mode = "text"
        return None, {}


def _await_hotword(
    audio_utils, speech_processor, hotword_detector, audio_config
) -> bool:
    """Listen for the configured hotword within the timeout window."""
    timeout = getattr(audio_config, "hotword_timeout", 15.0) or 15.0
    window_s = max(
        0.75,
        getattr(audio_config, "chunk_size", 1024)
        / float(audio_utils.sample_rate),
    )
    deadline = time.time() + timeout

    while time.time() < deadline:
        chunk = audio_utils.capture_chunk(duration_s=window_s)
        if chunk is None or chunk.size == 0:
            continue

        transcription = speech_processor.transcribe_audio_data(
            chunk.astype(np.float32).tobytes(),
            audio_utils.sample_rate,
        )
        if not transcription:
            continue

        detection = hotword_detector.detect(transcription, chunk)
        if detection:
            confidence_pct = int(detection.confidence * 100)
            print(
                green(
                    _msg(
                        "audio_hotword_detected",
                        "🔊 Hotword detectada ({confidence}%)",
                        confidence=confidence_pct,
                    )
                )
            )
            return True

    return False


def get_text_input(args):
    """Get text input from user."""
    prompt = (
        bold("💬 You: ") if args.mode != "audio" else bold("🎤 You (audio): ")
    )
    try:
        user_input = input(prompt).strip()
        return user_input
    except (EOFError, KeyboardInterrupt):
        print(bold("Saindo do M.I.A... Até logo!"))
        logger.info("Shutting down M.I.A...")
        return "quit"


def process_command(cmd, args, components):
    """Process special commands and return (should_continue, response_text)."""
    cmd = cmd.lower()

    if cmd == "quit":
        print(bold(_("exiting")))
        return False, ""
    elif cmd == "help":
        return True, display_help(args, components)
    elif cmd == "status":
        return True, display_status(components, args)
    elif cmd == "models":
        return True, display_models(args)
    elif cmd == "profiles":
        return True, display_profiles(components)
    elif cmd.startswith("switch-profile"):
        return True, switch_llm_profile(cmd, components)
    elif cmd == "clear":
        return True, clear_context(components)
    elif cmd == "audio" and components.get("speech_processor"):
        args.mode = "audio"
        return True, yellow(
            "🎤 Switched to audio input mode. Say something..."
        )
    elif cmd == "text" and args.mode == "audio":
        args.mode = "text"
        return True, cyan("🔤 Switched to text input mode.")
    else:
        return True, None


def display_help(args, components):
    """Display help information."""
    audio_available = components.get("audio_available", False)
    action_executor = components.get("action_executor")

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
    print(cyan("profiles - listar perfis LLM configurados"))
    print(cyan("switch-profile <nome> - trocar perfil LLM em tempo real"))
    print(cyan(_("clear_help")))

    # Agent commands
    if action_executor:
        print(bold(_("agent_title")))
        print(bold(_("help_separator")))
        print(cyan(_("create_file_help")))
        print(cyan(_("make_note_help")))
        print(cyan(_("analyze_code_help")))
        print(cyan(_("search_file_help")))

    print(bold(_("help_separator")))


def display_models(args):
    """Display available models."""
    print(bold("\n🤖 Available Models"))
    print(bold("─" * 40))
    print(
        green(
            "  gpt-oss:latest"
            + (
                " (current)"
                if getattr(args, "model_id", "") == "gpt-oss:latest"
                else ""
            )
        )
    )
    print(
        green(
            "  gemma3:4b-it-qat"
            + (
                " (current)"
                if getattr(args, "model_id", "") == "gemma3:4b-it-qat"
                else ""
            )
        )
    )
    print(cyan("  Use --model-id to change model"))
    print(bold("─" * 40))


def display_profiles(components):
    """Display configured LLM profiles."""
    config_manager = components.get("config_manager")
    if not config_manager or not getattr(config_manager, "config", None):
        print(red("Nenhum perfil configurado."))
        return

    profiles = config_manager.list_llm_profiles()
    active = getattr(config_manager, "active_llm_profile", None)
    if not profiles:
        print(yellow("Nenhum perfil foi encontrado em config.yaml."))
        return

    print(bold("\n🎯 Perfis de LLM configurados"))
    print(bold("─" * 40))
    for name in profiles:
        profile = config_manager.get_llm_profile(name)
        label = getattr(profile, "label", None) or name
        marker = green(" (ativo)") if name == active else ""
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


def switch_llm_profile(cmd, components):
    """Switch to a different LLM profile at runtime."""
    config_manager = components.get("config_manager")
    if not config_manager or not getattr(config_manager, "config", None):
        return red("Configuração não disponível para troca de perfil.")

    # Parse command: "switch-profile <profile_name>"
    parts = cmd.split()
    if len(parts) < 2:
        return red("Uso: switch-profile <nome_do_perfil>")

    profile_name = parts[1]
    profiles = config_manager.list_llm_profiles()

    if profile_name not in profiles:
        available = ", ".join(profiles) if profiles else "nenhum"
        return red(
            f"Perfil '{profile_name}' não encontrado. Disponíveis: {available}"
        )

    try:
        # Apply profile to config manager
        config_manager.active_llm_profile = profile_name
        profile = config_manager.get_llm_profile(profile_name)

        # Update LLM component if available
        llm = components.get("llm")
        if llm and hasattr(llm, "update_config"):
            profile_config = profile.apply_overrides(config_manager.config.llm)
            llm.update_config(profile_config)

        label = getattr(profile, "label", None) or profile_name
        description = getattr(profile, "description", None)
        provider = (
            getattr(profile, "provider", "")
            or config_manager.config.llm.provider
        )
        model_id = (
            getattr(profile, "model_id", "")
            or config_manager.config.llm.model_id
        )

        result = green(f"🔄 Perfil alterado para: {label}")
        result += f"\n   Provider: {provider} — Model: {model_id}"
        if description:
            result += f"\n   {description}"

        return result

    except Exception as exc:
        logger.error(f"Failed to switch LLM profile: {exc}")
        return red(f"Erro ao trocar perfil: {exc}")


def clear_context(components):
    """Clear conversation context and optimize performance."""
    cache_manager = components.get("cache_manager")
    performance_monitor = components.get("performance_monitor")

    if cache_manager and hasattr(cache_manager, "clear_all"):
        cache_manager.clear_all()

    if performance_monitor and hasattr(
        performance_monitor, "optimize_performance"
    ):
        performance_monitor.optimize_performance()

    return yellow("🧹 Conversation context cleared.")


def process_with_llm(user_input, inputs, components):
    """Process user input with LLM and return a structured response payload."""
    llm = components.get("llm")
    action_executor = components.get("action_executor")

    result: Dict[str, Any] = {
        "response": None,
        "streamed": False,
        "citations": None,
        "error": None,
    }

    if not user_input:
        result["error"] = yellow(_("no_input"))
        return result

    # Check for agent commands first
    if user_input and action_executor:
        agent_executed, agent_result = detect_and_execute_agent_commands(
            user_input, action_executor
        )
        if agent_executed:
            result["response"] = agent_result
            result["streamed"] = False
            return result

    if not llm or not hasattr(llm, "query"):
        result["error"] = red(_("llm_unavailable"))
        return result

    rag_pipeline = components.get("rag_pipeline")
    rag_prompt = None
    if rag_pipeline and hasattr(rag_pipeline, "build_prompt"):
        try:
            rag_prompt = rag_pipeline.build_prompt(user_input)
            if rag_prompt and rag_prompt.citations:
                result["citations"] = rag_prompt.format_citations()
        except Exception as exc:
            logger.debug("RAG prompt construction failed: %s", exc)
            rag_prompt = None

    stream_kwargs: Dict[str, Any] = {}
    if rag_prompt:
        stream_kwargs["messages"] = rag_prompt.messages

    print(cyan(_("thinking")))

    try:
        supports_stream = hasattr(llm, "stream") and callable(
            getattr(llm, "stream")
        )
        can_stream = supports_stream and getattr(llm, "stream_enabled", True)
        supports_stream_fn = getattr(llm, "supports_streaming", None)
        if callable(supports_stream_fn):
            can_stream = can_stream and supports_stream_fn()
        else:
            can_stream = False

        if can_stream:
            print(cyan("🤖 M.I.A: "), end="", flush=True)
            tokens: List[str] = []
            for chunk in llm.stream(user_input, **stream_kwargs):
                if chunk is None:
                    continue
                tokens.append(str(chunk))
                sys.stdout.write(str(chunk))
                sys.stdout.flush()
            print()
            response_text = "".join(tokens).strip()
            result["streamed"] = True
        else:
            response_text = llm.query(user_input, **stream_kwargs) or ""

        if not response_text:
            result["error"] = red(_("no_response"))
            return result

        result["response"] = response_text

        if rag_pipeline and hasattr(rag_pipeline, "remember"):
            try:
                rag_pipeline.remember(
                    response_text,
                    metadata={
                        "source": "conversation",
                        "question": user_input,
                        "timestamp": time.time(),
                    },
                )
            except Exception as exc:
                logger.debug("Failed to store conversation in RAG: %s", exc)

        # Optionally speak the response
        audio_config = components.get("audio_config")
        speech_generator = components.get("speech_generator")
        audio_utils = components.get("audio_utils")
        if (
            audio_config
            and getattr(audio_config, "tts_enabled", False)
            and speech_generator
        ):
            try:
                speech_generator.enqueue_speech(response_text)
            except Exception as exc:
                logger.debug("TTS enqueue failed: %s", exc)

        return result

    except Exception as exc:
        logger.error(f"LLM processing error: {exc}")
        result["error"] = red(_("llm_error", error=str(exc)))
        return result


def cleanup_resources(components):
    """Cleanup system resources."""
    logger.info(_("cleanup_message"))

    try:
        performance_monitor = components.get("performance_monitor")
        cache_manager = components.get("cache_manager")
        resource_manager = components.get("resource_manager")

        if performance_monitor and hasattr(
            performance_monitor, "stop_monitoring"
        ):
            performance_monitor.stop_monitoring()

        if performance_monitor and hasattr(performance_monitor, "cleanup"):
            performance_monitor.cleanup()

        if cache_manager and hasattr(cache_manager, "clear_all"):
            cache_manager.clear_all()

        if resource_manager and hasattr(resource_manager, "stop"):
            resource_manager.stop()

        logger.info("Resource cleanup completed")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


def collect_user_inputs(args, components):
    """
    Collect all types of user inputs (image, audio, text).

    Returns:
        tuple: (user_input, inputs_dict) where user_input is the text input
               and inputs_dict contains all collected inputs
    """
    inputs = {}

    # Process image input
    image_inputs = process_image_input(args, components)
    inputs.update(image_inputs)

    # Process audio input
    audio_text, audio_inputs = process_audio_input(args, components)
    if audio_inputs:
        inputs.update(audio_inputs)

    user_input = audio_text

    # If audio mode is active but yielded no input, prompt again next loop
    if user_input is None and args.mode == "audio":
        return None, {}

    # Get text input if still empty
    if user_input is None:
        user_input = get_text_input(args)
        if not user_input:
            return None, {}

    return user_input, inputs


def handle_user_command(user_input, args, components):
    """
    Handle user commands and return whether to continue and any response.

    Returns:
        tuple: (should_continue, command_response)
    """
    should_continue, command_response = process_command(
        user_input, args, components
    )
    if not should_continue:
        return False, None
    if command_response is not None:
        print(command_response)
        return True, command_response
    return True, None


def process_user_request(user_input, inputs, components):
    """
    Process user request with LLM and handle the response.

    Args:
        user_input: The user's text input
        inputs: Dictionary of all collected inputs
        components: Application components

    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        payload = process_with_llm(user_input, inputs, components)
        if not payload:
            return False

        error_msg = payload.get("error") if isinstance(payload, dict) else None
        if error_msg:
            print(error_msg)
            return False

        if isinstance(payload, dict):
            response_text = payload.get("response")
            streamed = payload.get("streamed", False)
            citations = payload.get("citations")

            if response_text and not streamed:
                text_str = str(response_text)
                if text_str.startswith("🤖"):
                    print(cyan(text_str))
                else:
                    print(cyan("🤖 M.I.A: ") + text_str)

            if citations:
                print(cyan(str(citations)))

        return True
    except Exception as e:
        logger.error(f"Error processing with LLM: {e}")
        print(red(_("processing_error", error=e)))
        return False


def handle_interaction_error(error, context="main loop"):
    """
    Handle errors that occur during user interaction.

    Args:
        error: The exception that occurred
        context: Context string for logging

    Returns:
        bool: True if should continue, False if should exit
    """
    if isinstance(error, KeyboardInterrupt):
        logger.info("Shutting down M.I.A...")
        return False
    else:
        logger.error(f"Unexpected error in {context}: {error}")
        return True


def run_interaction_loop(args, components):
    """
    Run the main interaction loop.

    Args:
        args: Parsed command line arguments
        components: Initialized application components

    Returns:
        bool: True if loop completed normally, False if interrupted
    """
    while True:
        try:
            # Collect user inputs
            user_input, inputs = collect_user_inputs(args, components)
            if user_input is None:
                continue

            # Handle commands
            should_continue, command_handled = handle_user_command(
                user_input, args, components
            )
            if not should_continue:
                return False
            if command_handled is not None:
                continue

            # Add text input to inputs
            inputs["text"] = user_input

            # Process the request
            if inputs:
                process_user_request(user_input, inputs, components)

        except Exception as e:
            if not handle_interaction_error(e, "interaction loop"):
                return False

    return True


def main():
    """Main function for M.I.A application"""
    components = (
        {}
    )  # Initialize components to avoid UnboundLocalError in finally block
    try:
        # Parse arguments
        args = parse_arguments()

        # Apply logging and warning settings after CLI is parsed
        setup_logging(args)

        # Initialize localization
        init_localization(args.language)

        # Handle info command
        if args.info:
            info = get_full_version()
            print(bold(f"🤖 {info['title']} v{info['version']}"))
            print(f"📝 {info['description']}")
            print(f"👤 Author: {info['author']}")
            print(f"📄 License: {info['license']}")
            print(f"🏗️  Build: {info['build']}")
            print(f"📊 Status: {info['status']}")
            print(f"🐍 Python: {sys.version}")
            print(f"💻 Platform: {sys.platform}")
            return

        # Determine effective mode (support deprecated flags)
        if getattr(args, "text_only", False):
            args.mode = "text"
        elif getattr(args, "audio_mode", False):
            args.mode = "audio"

        # Initialize core components
        print(yellow(_("initializing")))
        components = initialize_components(args)

        # Display status
        display_status(components, args)

        # Run main interaction loop
        run_interaction_loop(args, components)

    except Exception as e:
        logger.error(f"Critical error during initialization: {e}")
        sys.exit(1)

    finally:
        # Cleanup resources
        cleanup_resources(components)


if __name__ == "__main__":
    main()
