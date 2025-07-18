import argparse
import sys
import logging
import os
import warnings
from typing import Optional

# Suppress TensorFlow and other library warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformers warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*slow.*processor.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*use_fast.*", category=UserWarning)

# Optional imports with error handling
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

# Import project modules
from .audio.audio_utils import AudioUtils
from .audio.speech_processor import SpeechProcessor
from .llm.llm_manager import LLMManager
from .audio.speech_generator import SpeechGenerator
from .core.cognitive_architecture import MIACognitiveCore
from .multimodal.processor import MultimodalProcessor
from .memory.knowledge_graph import AgentMemory
from .langchain.langchain_verifier import LangChainVerifier
from .system.system_control import SystemControl
from .utils.automation_util import AutomationUtil
from .tools.action_executor import ActionExecutor
from .learning.user_learning import UserLearning
from .plugins.plugin_manager import PluginManager
from .security.security_manager import SecurityManager
from .deployment.deployment_manager import DeploymentManager
from .multimodal.vision_processor import VisionProcessor
from .memory.long_term_memory import LongTermMemory
from .planning.calendar_integration import CalendarIntegration

# Import custom exceptions and error handling
from .exceptions import *
from .error_handler import global_error_handler, with_error_handling, safe_execute
from .config_manager import ConfigManager
from .resource_manager import ResourceManager
from .performance_monitor import PerformanceMonitor
from .cache_manager import CacheManager


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


def blue(text):
    """Make text blue using ANSI escape codes"""
    return f"\033[34m{text}\033[0m"

# Configure logging
def setup_logging(debug_mode=False):
    """Setup logging configuration with proper levels"""
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress noisy third-party loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


# Configure logging
setup_logging()
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
def main():
    """Main function for M.I.A application"""
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description="M.I.A - Multimodal Intelligent Assistant")
        parser.add_argument('--text-only', action='store_true', help='Text-only mode')
        parser.add_argument('--audio-mode', action='store_true', help='Audio mode')
        parser.add_argument('--image-input', type=str, default=None, help='Image to process')
        parser.add_argument('--model-id', type=str, default='deepseek-r1:1.5b', help='Model ID')
        args = parser.parse_args()

        # Initialize components
        vision_processor = None
        audio_utils = None
        speech_processor = None
        audio_available = False
        llm = None
        performance_monitor = None
        cache_manager = None
        device = 'cpu'
        resource_manager = None

        # Status display
        print(bold("[ STATUS ]") + f" {(green('LLM Connected') if llm and hasattr(llm, 'is_available') and llm.is_available() else red('LLM Connection Issues'))}")
        print(bold("═"*60))
        logger.info("Welcome to M.I.A 2.0 - Multimodal Intelligent Assistant!")

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
                if hasattr(args, 'audio_mode') and args.audio_mode and speech_processor and audio_available and audio_utils:
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
                        print(bold("\n🔤 Switching to text mode..."))
                        args.audio_mode = False
                        continue
                    except Exception as e:
                        logger.error(f"Error processing audio: {e}")
                        print(red("❌ Audio processing failed. Switching to text mode."))
                        args.audio_mode = False
                        continue
                else:
                    # Text input
                    prompt = bold("💬 You: ") if not (hasattr(args, 'audio_mode') and args.audio_mode) else bold("🎤 You (audio): ")
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
                        print(bold("Exiting M.I.A. Até logo!"))
                        break
                    elif cmd == 'help':
                        print(bold("\n📚 M.I.A Commands"))
                        print(bold("─"*40))
                        print(green("  quit   ") + "- Exit M.I.A")
                        print(green("  help   ") + "- Show this help message")
                        if not getattr(args, 'text_only', False) and audio_available:
                            print(yellow("  audio  ") + "- Switch to audio input mode")
                        if getattr(args, 'audio_mode', False):
                            print(cyan("  text   ") + "- Switch to text input mode")
                        print(blue("  status ") + "- Show current system status")
                        print(blue("  models ") + "- List available models")
                        print(blue("  clear  ") + "- Clear conversation context")
                        print(bold("─"*40))
                        continue
                    elif cmd == 'status':
                        print(bold("\n📊 M.I.A Status"))
                        print(bold("─"*40))
                        print(f"  {bold('Mode:')} {green('Text-only') if getattr(args, 'text_only', False) else yellow('Audio') if getattr(args, 'audio_mode', False) else blue('Mixed')}")
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
                    elif cmd == 'audio' and not getattr(args, 'text_only', False) and speech_processor:
                        args.audio_mode = True
                        print(yellow("🎤 Switched to audio input mode. Say something..."))
                        continue
                    elif cmd == 'text' and getattr(args, 'audio_mode', False):
                        args.audio_mode = False
                        print(cyan("🔤 Switched to text input mode."))
                        continue
                    else:
                        inputs['text'] = user_input

                # Process the inputs here (add your main processing logic)
                # This is where you would handle the actual conversation logic
                
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
        logger.info("Cleaning up resources...")
        try:
            if performance_monitor and hasattr(performance_monitor, 'stop_monitoring'):
                performance_monitor.stop_monitoring()
            if performance_monitor and hasattr(performance_monitor, 'cleanup'):
                performance_monitor.cleanup()
            if cache_manager and hasattr(cache_manager, 'clear_all'):
                cache_manager.clear_all()
            if resource_manager and hasattr(resource_manager, 'cleanup'):
                resource_manager.cleanup()
            logger.info("Resource cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    main()