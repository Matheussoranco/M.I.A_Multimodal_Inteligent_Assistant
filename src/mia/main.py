import argparse
import sys
import logging
import os
import warnings
from typing import Optional

# Import version information
from .__version__ import __version__, get_full_version

# Import localization
from .localization import init_localization, get_localization, _

# Optional imports with error handling
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

# Import project modules with error handling
try:
    from .audio.audio_utils import AudioUtils
except ImportError:
    AudioUtils = None

try:
    from .audio.speech_processor import SpeechProcessor
except ImportError:
    SpeechProcessor = None

try:
    from .llm.llm_manager import LLMManager
except ImportError:
    LLMManager = None

try:
    from .audio.speech_generator import SpeechGenerator
except ImportError:
    SpeechGenerator = None

try:
    from .core.cognitive_architecture import MIACognitiveCore
except ImportError:
    MIACognitiveCore = None

try:
    from .multimodal.processor import MultimodalProcessor
except ImportError:
    MultimodalProcessor = None

try:
    from .memory.knowledge_graph import AgentMemory
except ImportError:
    AgentMemory = None

try:
    from .langchain.langchain_verifier import LangChainVerifier
except ImportError:
    LangChainVerifier = None

try:
    from .system.system_control import SystemControl
except ImportError:
    SystemControl = None

try:
    from .utils.automation_util import AutomationUtil
except ImportError:
    AutomationUtil = None

try:
    from .tools.action_executor import ActionExecutor
except ImportError:
    ActionExecutor = None

try:
    from .learning.user_learning import UserLearning
except ImportError:
    UserLearning = None

try:
    from .plugins.plugin_manager import PluginManager
except ImportError:
    PluginManager = None

try:
    from .security.security_manager import SecurityManager
except ImportError:
    SecurityManager = None

try:
    from .deployment.deployment_manager import DeploymentManager
except ImportError:
    DeploymentManager = None

try:
    from .multimodal.vision_processor import VisionProcessor
except ImportError:
    VisionProcessor = None

try:
    from .memory.long_term_memory import LongTermMemory
except ImportError:
    LongTermMemory = None

try:
    from .planning.calendar_integration import CalendarIntegration
except ImportError:
    CalendarIntegration = None

# Import custom exceptions and error handling
try:
    from .exceptions import *
    from .error_handler import global_error_handler, with_error_handling, safe_execute
    from .config_manager import ConfigManager
    from .resource_manager import ResourceManager
    from .performance_monitor import PerformanceMonitor
    from .cache_manager import CacheManager
except ImportError as e:
    print(f"Warning: Some modules could not be imported: {e}")
    # Create dummy classes for missing modules
    ConfigManager = None
    ResourceManager = None
    PerformanceMonitor = None
    CacheManager = None


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
    # Parse arguments
    parser = argparse.ArgumentParser(description="M.I.A - Multimodal Intelligent Assistant")
    # Unified mode selector (with backward-compatible flags)
    parser.add_argument('--mode', choices=['text', 'audio', 'mixed', 'auto'], default='mixed',
                help='Interaction mode: text|audio|mixed|auto')
    parser.add_argument('--text-only', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--audio-mode', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--language', choices=['en', 'pt'], default=None,
                help='Interface language (en=English, pt=Portuguese)')
    parser.add_argument('--image-input', type=str, default=None, help='Image to process')
    parser.add_argument('--model-id', type=str, default='deepseek-r1:1.5b', help='Model ID')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--version', action='version', version=f'M.I.A {__version__}', help='Show version information')
    parser.add_argument('--info', action='store_true', help='Show detailed version and system information')
    args = parser.parse_args()

    # Apply logging and warning settings after CLI is parsed
    setup_logging(getattr(args, 'debug', False))
    _suppress_warnings_env()

def _suppress_warnings_env() -> None:
    """Set environment variables and warning filters to reduce noise (opt-in)."""
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*slow.*processor.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*use_fast.*", category=UserWarning)


logger = logging.getLogger(__name__)


def choose_mode():
    """Interactive mode selection for M.I.A initialization."""
    print("\n" + "="*60)
    print("          M.I.A - Multimodal Intelligent Assistant")
    print("="*60)
    print("\nChoose your interaction mode:")
    print("1. Text-only mode")
    print("2. Audio mode")
    print("3. Mixed mode (default)")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            if choice == '1':
                return 'text'
            elif choice == '2':
                return 'audio'
            elif choice == '3' or choice == '':
                return 'mixed'
            else:
                print("Invalid choice. Please select 1, 2, or 3.")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            sys.exit(0)

def detect_and_execute_agent_commands(user_input: str, action_executor) -> tuple[bool, str]:
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
    if any(keyword in user_input_lower for keyword in ['criar arquivo', 'create file', 'novo arquivo']):
        try:
            # Extract filename - take last word that could be a filename
            words = user_input.split()
            filename = None
            
            # Look for word that seems like a filename (has extension or isn't a command word)
            for word in reversed(words):
                if word.lower() not in ['criar', 'arquivo', 'create', 'file', 'novo']:
                    filename = word.replace('"', '').replace("'", '')
                    break
            
            if not filename:
                # If not found, use default name
                filename = "novo_arquivo.txt"
            
            # Extract content if specified
            content = _("file_created_timestamp", timestamp=__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            if 'conteúdo' in user_input_lower or 'content' in user_input_lower or 'com' in user_input_lower:
                content_start = user_input.lower().find('conteúdo')
                if content_start == -1:
                    content_start = user_input.lower().find('content')
                if content_start == -1:
                    content_start = user_input.lower().find('com')
                
                if content_start != -1:
                    remaining = user_input[content_start:].strip()
                    if '"' in remaining or "'" in remaining:
                        # Extract text between quotes
                        import re
                        match = re.search(r'["\']([^"\']*)["\']', remaining)
                        if match:
                            content = match.group(1)
            
            result = action_executor.execute('create_file', {'path': filename, 'content': content})
            return True, _("agent_file_created", filename=filename)
        except Exception as e:
            return True, _("agent_file_error", error=e)
    
    # Note command detection
    elif any(keyword in user_input_lower for keyword in ['fazer nota', 'criar nota', 'make note', 'anotar']):
        try:
            # Extract note content
            content = user_input
            title = "Nota M.I.A"
            
            # Try to extract title
            if 'título' in user_input_lower or 'title' in user_input_lower:
                title_start = user_input.lower().find('título')
                if title_start == -1:
                    title_start = user_input.lower().find('title')
                if title_start != -1:
                    remaining = user_input[title_start:].strip()
                    import re
                    match = re.search(r'["\']([^"\']*)["\']', remaining)
                    if match:
                        title = match.group(1)
            
            result = action_executor.execute('make_note', {'content': content, 'title': title})
            return True, _("agent_note_saved")
        except Exception as e:
            return True, _("agent_note_error", error=e)
    
    # Code analysis command detection
    elif any(keyword in user_input_lower for keyword in ['analisar c?digo', 'analisar código', 'analyze code', 'analisar arquivo']):
        try:
            # Extract filename
            words = user_input.split()
            filepath = None
            
            for word in words:
                if '.' in word and any(ext in word.lower() for ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c']):
                    filepath = word
                    break
            
            if not filepath:
                return True, _("agent_specify_file")
            
            result = action_executor.execute('analyze_code', {'path': filepath})
            return True, _("agent_code_analysis", result=result)
        except Exception as e:
            return True, _("agent_analysis_error", error=e)
    
    # File search command detection
    elif any(keyword in user_input_lower for keyword in ['buscar arquivo', 'search file', 'encontrar arquivo']):
        try:
            # Extract filename
            words = user_input.split()
            filename = None
            
            for i, word in enumerate(words):
                if word.lower() in ['arquivo', 'file'] and i + 1 < len(words):
                    filename = words[i + 1]
                    break
            
            if not filename:
                return True, _("agent_specify_search")
            
            result = action_executor.execute('search_file', {'name': filename})
            return True, _("agent_code_analysis", result=result)
        except Exception as e:
            return True, _("agent_search_error", error=e)
    
    return False, ""

def main():
    """Main function for M.I.A application"""
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description="M.I.A - Multimodal Intelligent Assistant")
        parser.add_argument('--mode', choices=['text', 'audio', 'mixed', 'auto'], default='mixed',
                            help='Interaction mode: text|audio|mixed|auto')
        parser.add_argument('--text-only', action='store_true', help=argparse.SUPPRESS)
        parser.add_argument('--audio-mode', action='store_true', help=argparse.SUPPRESS)
        parser.add_argument('--language', choices=['en', 'pt'], default=None,
                            help='Interface language (en=English, pt=Portuguese)')
        parser.add_argument('--image-input', type=str, default=None, help='Image to process')
        parser.add_argument('--model-id', type=str, default='deepseek-r1:1.5b', help='Model ID')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        parser.add_argument('--version', action='version', version=f'M.I.A {__version__}', help='Show version information')
        parser.add_argument('--info', action='store_true', help='Show detailed version and system information')
        args = parser.parse_args()

        # Apply logging and warning settings after CLI is parsed
        level = logging.DEBUG if getattr(args, 'debug', False) else logging.INFO
        logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.StreamHandler(sys.stdout)])
        for name in ("transformers", "torch", "tensorflow", "numba", "chromadb", "urllib3", "requests"):
            logging.getLogger(name).setLevel(logging.WARNING)
        _suppress_warnings_env()

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
        if getattr(args, 'text_only', False):
            args.mode = 'text'
        elif getattr(args, 'audio_mode', False):
            args.mode = 'audio'

        # Initialize core components
        print(yellow(_("initializing")))

        # Initialize device
        device = 'cpu'
        if HAS_TORCH and torch is not None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")

        # Initialize LLM Manager
        try:
            if LLMManager:
                llm = LLMManager(model_id=args.model_id)
                logger.info("LLM Manager initialized successfully")
            else:
                logger.warning("LLM Manager not available - some features will be disabled")
                llm = None
        except Exception as e:
            logger.error(f"Failed to initialize LLM Manager: {e}")
            llm = None

        # Initialize audio components if not text-only mode
        audio_available = False
        audio_utils = None
        speech_processor = None

        if args.mode in ('audio', 'mixed', 'auto'):
            try:
                if AudioUtils and SpeechProcessor:
                    audio_utils = AudioUtils()
                    speech_processor = SpeechProcessor()
                    audio_available = True
                    logger.info("Audio components initialized successfully")
                else:
                    logger.warning("Audio components not available")
                    audio_available = False
            except Exception as e:
                logger.warning(f"Audio components failed to initialize: {e}")
                audio_available = False

        # Initialize vision processor
        vision_processor = None
        try:
            if VisionProcessor:
                vision_processor = VisionProcessor()
                logger.info("Vision processor initialized successfully")
            else:
                logger.warning("Vision processor not available")
        except Exception as e:
            logger.warning(f"Vision processor failed to initialize: {e}")

        # Initialize action executor
        action_executor = None
        try:
            if ActionExecutor:
                action_executor = ActionExecutor()
                logger.info("Action executor initialized successfully")
            else:
                logger.warning("Action executor not available")
        except Exception as e:
            logger.warning(f"Action executor failed to initialize: {e}")

        # Initialize other components
        performance_monitor = None
        cache_manager = None
        resource_manager = None

        try:
            if PerformanceMonitor:
                performance_monitor = PerformanceMonitor()
            if CacheManager:
                cache_manager = CacheManager()
            if ResourceManager:
                resource_manager = ResourceManager()
            logger.info("Additional components initialized successfully")
        except Exception as e:
            logger.warning(f"Some additional components failed to initialize: {e}")

        # Status display
        print(bold("[ STATUS ]") + f" {(green(_('status_connected')) if llm and hasattr(llm, 'is_available') and llm.is_available() else red(_('status_issues')))}")
        print(bold("═"*60))
        logger.info(_("welcome_message"))

        # Main interaction loop
        while True:
            inputs = {}
            user_input = ""

            try:
                # Process image input
                if hasattr(args, 'image_input') and args.image_input and vision_processor:
                    try:
                        inputs['image'] = vision_processor.process_image(args.image_input)
                        args.image_input = None
                        logger.info("Image processed successfully")
                    except Exception as e:
                        logger.error(f"Error processing image: {e}")

                # Process audio input
                if args.mode == 'audio' and speech_processor and audio_available and audio_utils:
                    try:
                        print(bold("🎤 Listening... (speak now or press Ctrl+C to switch to text)"))
                        mic = audio_utils.record_audio(speech_processor, 2.0, 0.25)
                        audio_chunk = next(mic)
                        transcription = speech_processor.transcribe_audio_data(audio_chunk.tobytes(), 16000)
                        user_input = transcription.strip() if isinstance(transcription, str) else ''
                        if not user_input:
                            print(red("🔇 No speech detected, try speaking louder or closer to the microphone"))
                            continue
                        print(green(f"🎙️  You said: {user_input}"))
                        inputs['audio'] = user_input
                    except KeyboardInterrupt:
                        print(bold("\n🔥 Switching to text mode..."))
                        args.mode = 'text'
                        continue
                    except Exception as e:
                        logger.error(f"Error processing audio: {e}")
                        print(red("❌ Audio processing failed. Switching to text mode."))
                        args.mode = 'text'
                        continue
                else:
                    # Text input
                    prompt = bold("💬 You: ") if args.mode != 'audio' else bold("🎤 You (audio): ")
                    try:
                        user_input = input(prompt).strip()
                    except (EOFError, KeyboardInterrupt):
                        print(bold("Saindo do M.I.A... Até logo!"))
                        logger.info("Shutting down M.I.A...")
                        break

                    if not user_input:
                        continue

                    # Process commands
                    cmd = user_input.lower()
                    if cmd == 'quit':
                        print(bold(_("exiting")))
                        break
                    elif cmd == 'help':
                        print(bold(_("help_title")))
                        print(bold(_("help_separator")))
                        print(green(_("quit_help")))
                        print(green(_("help_help")))
                        if args.mode != 'text' and audio_available:
                            print(yellow(_("audio_help")))
                        if args.mode == 'audio':
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
                        continue
                    elif cmd == 'status':
                        print(bold("\n📊 M.I.A Status"))
                        print(bold("─"*40))
                        mode_label = {
                            'text': green('Text-only'),
                            'audio': yellow('Audio'),
                            'mixed': cyan('Mixed'),
                            'auto': cyan('Auto'),
                        }.get(args.mode, cyan('Mixed'))
                        print(f"  {bold('Mode:')} {mode_label}")
                        print(f"  {bold('Model:')} {yellow(getattr(args, 'model_id', ''))}")
                        print(f"  {bold('LLM:')} {(green('Connected') if llm and hasattr(llm, 'is_available') and llm.is_available() else red('Disconnected'))}")
                        print(f"  {bold('Audio:')} {(green('Available') if audio_available else red('Not available'))}")
                        print(f"  {bold('Device:')} {cyan(device)}")
                        if performance_monitor and hasattr(performance_monitor, 'get_current_metrics'):
                            perf_metrics = performance_monitor.get_current_metrics()
                            if perf_metrics:
                                print(f"  {bold('CPU:')} {yellow(f'{perf_metrics.cpu_percent:.1f}%')}")
                                print(f"  {bold('Memory:')} {yellow(f'{perf_metrics.memory_percent:.1f}%')}")
                        if cache_manager and hasattr(cache_manager, 'get_stats'):
                            cache_stats = cache_manager.get_stats()
                            hit_rate = cache_stats.get('memory_cache', {}).get('hit_rate', 0)
                            print(f"  {bold('Cache Hit Rate:')} {green('{:.1%}'.format(hit_rate))}")
                        print(bold("─"*40))
                        continue
                    elif cmd == 'models':
                        print(bold("\n🤖 Available Models"))
                        print(bold("─"*40))
                        print(green("  deepseek-r1:1.5b" + (" (current)" if getattr(args, 'model_id', '') == 'deepseek-r1:1.5b' else "")))
                        print(green("  gemma3:4b-it-qat" + (" (current)" if getattr(args, 'model_id', '') == 'gemma3:4b-it-qat' else "")))
                        print(cyan("  Use --model-id to change model"))
                        print(bold("─"*40))
                        continue
                    elif cmd == 'clear':
                        print(yellow("🧹 Conversation context cleared."))
                        if cache_manager and hasattr(cache_manager, 'clear_all'):
                            cache_manager.clear_all()
                        if performance_monitor and hasattr(performance_monitor, 'optimize_performance'):
                            performance_monitor.optimize_performance()
                        continue
                    elif cmd == 'audio' and speech_processor:
                        args.mode = 'audio'
                        print(yellow("🎤 Switched to audio input mode. Say something..."))
                        continue
                    elif cmd == 'text' and args.mode == 'audio':
                        args.mode = 'text'
                        print(cyan("🔤 Switched to text input mode."))
                        continue
                    else:
                        inputs['text'] = user_input

                # Process the inputs with the LLM
                if inputs:
                    try:
                        # Check for agent commands first
                        input_text = inputs.get('text', inputs.get('audio', ''))

                        if input_text and action_executor:
                            agent_executed, agent_result = detect_and_execute_agent_commands(input_text, action_executor)
                            if agent_executed:
                                print(agent_result)
                                continue

                        # Regular LLM processing
                        if llm and hasattr(llm, 'query'):
                            if input_text:
                                print(cyan(_("thinking")))
                                response = llm.query(input_text)
                                if response:
                                    print(cyan("🤖 M.I.A: ") + response)
                                else:
                                    print(red(_("no_response")))
                            else:
                                print(yellow(_("no_input")))
                        else:
                            print(red(_("llm_unavailable")))
                    except Exception as e:
                        logger.error(f"Error processing with LLM: {e}")
                        print(red(_("processing_error", error=e)))

            except KeyboardInterrupt:
                logger.info("Shutting down M.I.A...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                continue

    except Exception as e:
        logger.error(f"Critical error during initialization: {e}")
        sys.exit(1)

    finally:
        # Cleanup resources
        logger.info(_("cleanup_message"))
        try:
            # Check if variables exist before using them
            if 'performance_monitor' in locals() and performance_monitor and hasattr(performance_monitor, 'stop_monitoring'):
                performance_monitor.stop_monitoring()
            if 'performance_monitor' in locals() and performance_monitor and hasattr(performance_monitor, 'cleanup'):
                performance_monitor.cleanup()
            if 'cache_manager' in locals() and cache_manager and hasattr(cache_manager, 'clear_all'):
                cache_manager.clear_all()
            if 'resource_manager' in locals() and resource_manager and hasattr(resource_manager, 'stop'):
                resource_manager.stop()
            logger.info("Resource cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    main()