import argparse
import sys
import logging
import os
import warnings
from typing import Tuple

try:
    import colorama
    colorama.init()
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False

from .__version__ import __version__, get_full_version
from .localization import init_localization, _

try:
    from .exceptions import *
    from .error_handler import global_error_handler, with_error_handling, safe_execute
    from .config_manager import ConfigManager
    from .resource_manager import ResourceManager
    from .performance_monitor import PerformanceMonitor
    from .cache_manager import CacheManager
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

OPTIONAL_MODULES = {}

def _import_optional_module(module_path: str, module_name: str):
    try:
        if module_path.startswith('.'):
            full_module_path = f"mia{module_path}"
        else:
            full_module_path = module_path
        module = __import__(full_module_path, fromlist=[module_name])
        OPTIONAL_MODULES[module_name] = getattr(module, module_name, None)
        return True
    except (ImportError, ModuleNotFoundError, AttributeError):
        OPTIONAL_MODULES[module_name] = None
        return False

_import_optional_module('.audio.audio_utils', 'AudioUtils')
_import_optional_module('.audio.speech_processor', 'SpeechProcessor')
_import_optional_module('.audio.speech_generator', 'SpeechGenerator')
_import_optional_module('.llm.llm_manager', 'LLMManager')
_import_optional_module('.core.cognitive_architecture', 'MIACognitiveCore')
_import_optional_module('.multimodal.processor', 'MultimodalProcessor')
_import_optional_module('.multimodal.vision_processor', 'VisionProcessor')
_import_optional_module('.memory.knowledge_graph', 'AgentMemory')
_import_optional_module('.memory.long_term_memory', 'LongTermMemory')
_import_optional_module('.langchain.langchain_verifier', 'LangChainVerifier')
_import_optional_module('.system.system_control', 'SystemControl')
_import_optional_module('.utils.automation_util', 'AutomationUtil')
_import_optional_module('.tools.action_executor', 'ActionExecutor')
_import_optional_module('.plugins.plugin_manager', 'PluginManager')
_import_optional_module('.security.security_manager', 'SecurityManager')
_import_optional_module('.security.security_manager', 'SecurityManager')

# Extract imported modules for easier access
AudioUtils = OPTIONAL_MODULES.get('AudioUtils')
SpeechProcessor = OPTIONAL_MODULES.get('SpeechProcessor')
SpeechGenerator = OPTIONAL_MODULES.get('SpeechGenerator')
LLMManager = OPTIONAL_MODULES.get('LLMManager')
MIACognitiveCore = OPTIONAL_MODULES.get('MIACognitiveCore')
MultimodalProcessor = OPTIONAL_MODULES.get('MultimodalProcessor')
VisionProcessor = OPTIONAL_MODULES.get('VisionProcessor')
AgentMemory = OPTIONAL_MODULES.get('AgentMemory')
LongTermMemory = OPTIONAL_MODULES.get('LongTermMemory')
LangChainVerifier = OPTIONAL_MODULES.get('LangChainVerifier')
SystemControl = OPTIONAL_MODULES.get('SystemControl')
AutomationUtil = OPTIONAL_MODULES.get('AutomationUtil')
ActionExecutor = OPTIONAL_MODULES.get('ActionExecutor')
PluginManager = OPTIONAL_MODULES.get('PluginManager')
SecurityManager = OPTIONAL_MODULES.get('SecurityManager')


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
    warnings.filterwarnings("ignore", message=".*use_fast.*", category=UserWarning)


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

            result = action_executor.execute("analyze_code", {"path": filepath})
            return True, _("agent_code_analysis", result=result)
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
    parser.add_argument("--text-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--audio-mode", action="store_true", help=argparse.SUPPRESS)
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
        "--model-id", type=str, default="deepseek-r1:1.5b", help="Model ID"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
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

    # Initialize device
    device = "cpu"
    if HAS_TORCH and torch is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    components['device'] = device
    logger.info(f"Using device: {device}")

    # Initialize LLM Manager
    try:
        if LLMManager:
            components['llm'] = LLMManager(model_id=args.model_id)
            logger.info("LLM Manager initialized successfully")
        else:
            logger.warning("LLM Manager not available - some features will be disabled")
            components['llm'] = None
    except Exception as e:
        logger.error(f"Failed to initialize LLM Manager: {e}")
        components['llm'] = None

    # Initialize audio components if not text-only mode
    components['audio_available'] = False
    components['audio_utils'] = None
    components['speech_processor'] = None

    if args.mode in ("audio", "mixed", "auto"):
        try:
            if AudioUtils and SpeechProcessor:
                components['audio_utils'] = AudioUtils()
                components['speech_processor'] = SpeechProcessor()
                components['audio_available'] = True
                logger.info("Audio components initialized successfully")
            else:
                logger.warning("Audio components not available")
                components['audio_available'] = False
        except Exception as e:
            logger.warning(f"Audio components failed to initialize: {e}")
            components['audio_available'] = False

    # Initialize vision processor
    try:
        if VisionProcessor:
            components['vision_processor'] = VisionProcessor()
            logger.info("Vision processor initialized successfully")
        else:
            logger.warning("Vision processor not available")
            components['vision_processor'] = None
    except Exception as e:
        logger.warning(f"Vision processor failed to initialize: {e}")
        components['vision_processor'] = None

    # Initialize action executor
    try:
        if ActionExecutor:
            components['action_executor'] = ActionExecutor()
            logger.info("Action executor initialized successfully")
        else:
            logger.warning("Action executor not available")
            components['action_executor'] = None
    except Exception as e:
        logger.warning(f"Action executor failed to initialize: {e}")
        components['action_executor'] = None

    # Initialize other components
    try:
        if PerformanceMonitor:
            components['performance_monitor'] = PerformanceMonitor()
        else:
            components['performance_monitor'] = None

        if CacheManager:
            components['cache_manager'] = CacheManager()
        else:
            components['cache_manager'] = None

        if ResourceManager:
            components['resource_manager'] = ResourceManager()
        else:
            components['resource_manager'] = None

        logger.info("Additional components initialized successfully")
    except Exception as e:
        logger.warning(f"Some additional components failed to initialize: {e}")
        components['performance_monitor'] = None
        components['cache_manager'] = None
        components['resource_manager'] = None

    return components


def display_status(components, args):
    """Display system status information."""
    llm = components.get('llm')
    audio_available = components.get('audio_available', False)
    device = components.get('device', 'cpu')
    performance_monitor = components.get('performance_monitor')
    cache_manager = components.get('cache_manager')

    status = (
        green(_('status_connected'))
        if llm and hasattr(llm, 'is_available') and llm.is_available()
        else red(_('status_issues'))
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
    print(f"  {bold('Model:')} {yellow(getattr(args, 'model_id', ''))}")
    print(
        f"  {bold('LLM:')} {(green('Connected') if llm and hasattr(llm, 'is_available') and llm.is_available() else red('Disconnected'))}"
    )
    print(
        f"  {bold('Audio:')} {(green('Available') if audio_available else red('Not available'))}"
    )
    print(f"  {bold('Device:')} {cyan(device)}")

    if performance_monitor and hasattr(performance_monitor, "get_current_metrics"):
        perf_metrics = performance_monitor.get_current_metrics()
        if perf_metrics:
            print(f"  {bold('CPU:')} {yellow(f'{perf_metrics.cpu_percent:.1f}%')}")
            print(f"  {bold('Memory:')} {yellow(f'{perf_metrics.memory_percent:.1f}%')}")

    if cache_manager and hasattr(cache_manager, "get_stats"):
        cache_stats = cache_manager.get_stats()
        hit_rate = cache_stats.get("memory_cache", {}).get("hit_rate", 0)
        print(f"  {bold('Cache Hit Rate:')} {green('{:.1%}'.format(hit_rate))}")

    print(bold("─" * 40))


def process_image_input(args, components):
    """Process image input if provided."""
    vision_processor = components.get('vision_processor')
    if (hasattr(args, "image_input") and args.image_input and vision_processor):
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
    if (args.mode == "audio" and
        components.get('speech_processor') and
        components.get('audio_available') and
        components.get('audio_utils')):

        try:
            print(bold("🎤 Listening... (speak now or press Ctrl+C to switch to text)"))
            audio_utils = components['audio_utils']
            speech_processor = components['speech_processor']

            mic = audio_utils.record_audio(speech_processor, 2.0, 0.25)
            audio_chunk = next(mic)
            transcription = speech_processor.transcribe_audio_data(
                audio_chunk.tobytes(), 16000
            )
            user_input = (
                transcription.strip()
                if isinstance(transcription, str)
                else ""
            )

            if not user_input:
                print(red("🔇 No speech detected, try speaking louder or closer to the microphone"))
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

    return None, {}


def get_text_input(args):
    """Get text input from user."""
    prompt = (
        bold("💬 You: ")
        if args.mode != "audio"
        else bold("🎤 You (audio): ")
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
    elif cmd == "clear":
        return True, clear_context(components)
    elif cmd == "audio" and components.get('speech_processor'):
        args.mode = "audio"
        return True, yellow("🎤 Switched to audio input mode. Say something...")
    elif cmd == "text" and args.mode == "audio":
        args.mode = "text"
        return True, cyan("🔤 Switched to text input mode.")
    else:
        return True, None


def display_help(args, components):
    """Display help information."""
    audio_available = components.get('audio_available', False)
    action_executor = components.get('action_executor')

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
            "  deepseek-r1:1.5b"
            + (
                " (current)"
                if getattr(args, "model_id", "") == "deepseek-r1:1.5b"
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


def clear_context(components):
    """Clear conversation context and optimize performance."""
    cache_manager = components.get('cache_manager')
    performance_monitor = components.get('performance_monitor')

    if cache_manager and hasattr(cache_manager, "clear_all"):
        cache_manager.clear_all()

    if performance_monitor and hasattr(performance_monitor, "optimize_performance"):
        performance_monitor.optimize_performance()

    return yellow("🧹 Conversation context cleared.")


def process_with_llm(user_input, inputs, components):
    """Process user input with LLM and return response."""
    llm = components.get('llm')
    action_executor = components.get('action_executor')

    # Check for agent commands first
    if user_input and action_executor:
        agent_executed, agent_result = detect_and_execute_agent_commands(
            user_input, action_executor
        )
        if agent_executed:
            return agent_result

    # Regular LLM processing
    if llm and hasattr(llm, "query"):
        if user_input:
            print(cyan(_("thinking")))
            try:
                response = llm.query(user_input)
                if response:
                    return cyan("🤖 M.I.A: ") + response
                else:
                    return red(_("no_response"))
            except Exception as e:
                logger.error(f"LLM query error: {e}")
                return red(_("llm_error", error=str(e)))
        else:
            return yellow(_("no_input"))
    else:
        return red(_("llm_unavailable"))


def cleanup_resources(components):
    """Cleanup system resources."""
    logger.info(_("cleanup_message"))

    try:
        performance_monitor = components.get('performance_monitor')
        cache_manager = components.get('cache_manager')
        resource_manager = components.get('resource_manager')

        if (performance_monitor and hasattr(performance_monitor, "stop_monitoring")):
            performance_monitor.stop_monitoring()

        if (performance_monitor and hasattr(performance_monitor, "cleanup")):
            performance_monitor.cleanup()

        if (cache_manager and hasattr(cache_manager, "clear_all")):
            cache_manager.clear_all()

        if (resource_manager and hasattr(resource_manager, "stop")):
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
    user_input, audio_inputs = process_audio_input(args, components)
    if user_input is None:  # No speech detected
        return None, {}
    inputs.update(audio_inputs)
    
    # Get text input if no audio input
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
    should_continue, command_response = process_command(user_input, args, components)
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
        response = process_with_llm(user_input, inputs, components)
        if response:
            print(response)
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
            should_continue, command_handled = handle_user_command(user_input, args, components)
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
    components = {}  # Initialize components to avoid UnboundLocalError in finally block
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
