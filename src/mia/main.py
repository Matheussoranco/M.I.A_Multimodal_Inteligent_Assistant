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
    print("1. Text Mode (Recommended) - Type your messages")
    print("2. Audio Mode - Speak your messages")
    print("3. Mixed Mode - Switch between text and audio")
    print("4. Auto-detect - Let M.I.A choose based on available hardware")
    print("-" * 60)
    
    while True:
        try:
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == '1':
                return 'text-only', "Text Mode selected - Audio features disabled"
            elif choice == '2':
                return 'audio-mode', "Audio Mode selected - Voice input enabled"
            elif choice == '3':
                return 'mixed', "Mixed Mode selected - Both text and audio available"
            elif choice == '4':
                return 'auto', "Auto-detect mode - M.I.A will detect available features"
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except (EOFError, KeyboardInterrupt):
            print("\nDefaulting to Text Mode...")
            return 'text-only', "Text Mode selected (default)"

def main():
    parser = argparse.ArgumentParser(description="M.I.A - Multimodal Intelligent Assistant")
    parser.add_argument("--url", default='http://localhost:11434/api/generate', help="LLM API URL")
    parser.add_argument("--model-id", default='deepseek-r1:1.5b', help="Model ID to use")
    parser.add_argument("--api-key", default='ollama', help="API key for the LLM service")
    parser.add_argument("--stt-model", default="openai/whisper-base.en", help="Speech-to-text model")
    parser.add_argument("--image-input", help="Path to image file for multimodal processing")
    parser.add_argument("--enable-reasoning", action="store_true", help="Enable advanced reasoning")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--mode", choices=['text', 'audio', 'mixed', 'auto'], help="Interaction mode (text, audio, mixed, auto)")
    parser.add_argument("--audio-mode", action="store_true", help="Enable audio input mode")
    parser.add_argument("--text-only", action="store_true", help="Text-only mode (disable audio completely)")
    parser.add_argument("--skip-mode-selection", action="store_true", help="Skip interactive mode selection")

    args = parser.parse_args()
    
    # Handle --mode argument
    if args.mode:
        args.skip_mode_selection = True
        if args.mode == 'text':
            args.text_only = True
        elif args.mode == 'audio':
            args.audio_mode = True
        elif args.mode == 'mixed':
            # Keep defaults - both text and audio available
            pass
        elif args.mode == 'auto':
            # Auto-detect mode
            pass
    
    # Initialize mode selection variables
    selected_mode = None
    mode_message = ""
    
    # Interactive mode selection if not skipped and no mode specified
    if not args.skip_mode_selection and not args.text_only and not args.audio_mode:
        selected_mode, mode_message = choose_mode()
        print(f"\n✓ {mode_message}")
        
        # Apply the selected mode
        if selected_mode == 'text-only':
            args.text_only = True
        elif selected_mode == 'audio-mode':
            args.audio_mode = True
        elif selected_mode == 'mixed':
            # Keep defaults - both text and audio available
            pass
        elif selected_mode == 'auto':
            # Auto-detect will be handled later
            pass

    # Setup logging based on debug mode
    setup_logging(args.debug)

    # Determine device
    if HAS_TORCH and torch is not None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
    else:
        device = "cpu"
        logger.warning("PyTorch not available, using CPU mode")

    try:
        # Initialize core components
        logger.info("Initializing M.I.A components...")
        
        # Initialize LLM with error handling
        llm = LLMManager(provider='ollama', model_id=args.model_id, api_key=args.api_key, url=args.url)
        if not llm.is_available():
            logger.warning(f"LLM provider {llm.provider} not available, some features may be limited")
        
        # Initialize other components
        cognitive_core = MIACognitiveCore(llm)
        multimodal_processor = MultimodalProcessor()
        memory = AgentMemory()
        
        audio_utils = AudioUtils()
        langchain_verifier = LangChainVerifier(llm=llm)

        # Initialize additional modules
        action_executor = ActionExecutor({"open_file": True, "web_search": True})
        user_learning = UserLearning()
        plugin_manager = PluginManager()
        plugin_manager.load_plugins()
        security_manager = SecurityManager()
        deployment_manager = DeploymentManager()
        vision_processor = VisionProcessor()
        long_term_memory = LongTermMemory()
        calendar = CalendarIntegration()

        # Initialize audio components only if not in text-only mode
        speech_processor = None
        audio_model = None
        audio_available = False
        
        if not args.text_only:
            try:
                speech_processor = SpeechProcessor(model_name=args.stt_model or "base")
                audio_model = SpeechGenerator(device) if HAS_TORCH else None
                audio_available = True
                logger.info("Audio components initialized successfully")
            except Exception as e:
                logger.warning(f"Speech recognition not available: {e}")
                speech_processor = None
                audio_model = None
                audio_available = False
        
        # Auto-detect mode handling
        if selected_mode == 'auto':
            if audio_available:
                print("✓ Auto-detect: Audio hardware detected - Mixed mode enabled")
                logger.info("Auto-detect: Audio capabilities available")
            else:
                print("✓ Auto-detect: Audio hardware not available - Text mode enabled")
                logger.info("Auto-detect: Falling back to text-only mode")
                args.text_only = True

        # Display final mode configuration
        print("\n" + "="*50)
        print("          M.I.A INITIALIZATION COMPLETE")
        print("="*50)
        
        if args.text_only:
            print("🔤 MODE: Text-only")
            print("📝 INPUT: Type your messages")
            print("🔊 AUDIO: Disabled")
        elif args.audio_mode:
            print("🎤 MODE: Audio-first")
            print("🗣️  INPUT: Voice commands")
            print("🔊 AUDIO: Enabled")
        else:
            print("🔀 MODE: Mixed (Text + Audio)")
            print("📝 INPUT: Type messages or say 'audio' to switch")
            print("🔊 AUDIO: Available" if audio_available else "Not available")
        
        print(f"🤖 MODEL: {args.model_id}")
        print(f"🌐 PROVIDER: Ollama")
        print(f"✅ STATUS: {'LLM Connected' if llm.is_available() else 'LLM Connection Issues'}")
        print("="*50)

        logger.info("Welcome to M.I.A 2.0 - Multimodal Intelligent Assistant!")
        
        # Show input mode information
        if args.text_only:
            logger.info("Running in text-only mode")
        elif args.audio_mode:
            logger.info("Running in audio input mode")
        else:
            logger.info("Running in mixed mode (text + audio)")
        
        print("\n🚀 M.I.A is ready! Type your message or 'quit' to exit.")
        if not args.text_only and audio_available:
            print("💡 Tip: Type 'audio' to switch to voice mode, 'text' to switch back")
        print("🆘 Type 'help' for available commands")
        print("-" * 50)

        while True:
            try:
                inputs = {}
                user_input = ""
                
                # Handle image input if provided
                if args.image_input:
                    try:
                        inputs['image'] = vision_processor.process_image(args.image_input)
                        args.image_input = None  # Process only once
                        logger.info("Image processed successfully")
                    except Exception as e:
                        logger.error(f"Error processing image: {e}")
                
                # Input handling based on mode
                if args.audio_mode and speech_processor and audio_available:
                    # Audio input mode
                    try:
                        print("🎤 Listening... (speak now or press Ctrl+C to switch to text)")
                        mic = audio_utils.record_audio(speech_processor, 2.0, 0.25)
                        audio_chunk = next(mic)
                        transcription = speech_processor.transcribe_audio_data(audio_chunk.tobytes(), 16000)
                        user_input = transcription.strip() if isinstance(transcription, str) else ''
                        
                        if not user_input:
                            print("🔇 No speech detected, try speaking louder or closer to the microphone")
                            continue
                            
                        print(f"🎙️  You said: {user_input}")
                        inputs['audio'] = user_input
                    except KeyboardInterrupt:
                        print("\n🔤 Switching to text mode...")
                        args.audio_mode = False
                        continue
                    except Exception as e:
                        logger.error(f"Error processing audio: {e}")
                        print("❌ Audio processing failed. Switching to text mode.")
                        args.audio_mode = False
                        continue
                else:
                    # Text input mode (default)
                    try:
                        prompt = "🎤 You (audio): " if args.audio_mode else "💬 You: "
                        user_input = input(prompt).strip()
                        
                        if not user_input:
                            continue
                            
                        # Handle special commands
                        if user_input.lower() == 'quit':
                            break
                        elif user_input.lower() == 'help':
                            print("\n📚 M.I.A Commands:")
                            print("  quit         - Exit M.I.A")
                            print("  help         - Show this help message")
                            if not args.text_only and audio_available:
                                print("  audio        - Switch to audio input mode")
                            if args.audio_mode:
                                print("  text         - Switch to text input mode")
                            print("  status       - Show current system status")
                            print("  models       - List available models")
                            print("  clear        - Clear conversation context")
                            print("-" * 40)
                            continue
                        elif user_input.lower() == 'status':
                            print(f"\n📊 M.I.A Status:")
                            print(f"  Mode: {'Text-only' if args.text_only else 'Audio' if args.audio_mode else 'Mixed'}")
                            print(f"  Model: {args.model_id}")
                            print(f"  LLM: {'Connected' if llm.is_available() else 'Disconnected'}")
                            print(f"  Audio: {'Available' if audio_available else 'Not available'}")
                            print(f"  Device: {device}")
                            print("-" * 40)
                            continue
                        elif user_input.lower() == 'models':
                            print("\n🤖 Available Models:")
                            print("  deepseek-r1:1.5b (current)" if args.model_id == 'deepseek-r1:1.5b' else "  deepseek-r1:1.5b")
                            print("  gemma3:4b-it-qat" + (" (current)" if args.model_id == 'gemma3:4b-it-qat' else ""))
                            print("  Use --model-id to change model")
                            print("-" * 40)
                            continue
                        elif user_input.lower() == 'clear':
                            print("🧹 Conversation context cleared.")
                            # Reset memory if needed
                            continue
                        elif user_input.lower() == 'audio' and not args.text_only and speech_processor:
                            args.audio_mode = True
                            print("🎤 Switched to audio input mode. Say something...")
                            continue
                        elif user_input.lower() == 'text' and args.audio_mode:
                            args.audio_mode = False
                            print("🔤 Switched to text input mode.")
                            continue
                            
                        inputs['text'] = user_input
                    except (EOFError, KeyboardInterrupt):
                        logger.info("Shutting down M.I.A...")
                        break

                # Cognitive processing
                response = None
                try:
                    if args.enable_reasoning:
                        # Use the appropriate input for processing
                        input_text = inputs.get('audio', inputs.get('text', ''))
                        processed = cognitive_core.process_multimodal_input({'text': input_text, **inputs})
                        memory.store_experience(processed['text'], processed['embedding'])
                        long_term_memory.remember(processed['text'])
                        response = llm.query_model(
                            processed['text'], 
                            context=memory.retrieve_context(processed['embedding'])
                        )
                    else:
                        # Use the appropriate input for LLM query
                        input_text = inputs.get('audio', inputs.get('text', ''))
                        response = llm.query_model(input_text)
                except Exception as e:
                    logger.error(f"Error querying LLM: {e}")
                    response = "Sorry, I encountered an error processing your request."

                # Validate response
                if not response:
                    logger.warning("Received empty response from LLM")
                    response = "I didn't understand that. Could you please try again?"

                # Action execution with security and plugin support
                try:
                    if response.startswith("ACTION:"):
                        action = response[7:].strip()
                        if security_manager.check_permission(action):
                            # Try plugin first
                            plugin = plugin_manager.get_plugin(action)
                            if plugin:
                                action_result = plugin.run()
                            else:
                                action_result = action_executor.execute(action, {})
                            logger.info(f"Action Result: {action_result}")
                        else:
                            logger.warning(f"Action denied: {action}")
                            print(security_manager.explain_action(action))
                            
                    elif response.startswith("CALENDAR:"):
                        event = response[9:].strip()
                        logger.info(calendar.add_event(event))
                        
                    else:
                        # Regular response - always show text
                        print(f"🤖 M.I.A: {response}")
                        
                        # Convert to speech only if audio mode is enabled and available
                        if audio_model and not args.text_only and args.audio_mode and audio_available:
                            try:
                                print("🔊 Converting to speech...")
                                speech = audio_model.generate_speech(response)
                                if isinstance(speech, dict) and 'audio' in speech and 'sampling_rate' in speech:
                                    audio_utils.play_audio(speech['audio'], speech['sampling_rate'])
                                    print("✅ Speech playback complete")
                                else:
                                    logger.warning("Speech generation returned unexpected format")
                            except Exception as e:
                                logger.error(f"Error generating speech: {e}")
                                print("❌ Speech generation failed")
                        
                except Exception as e:
                    logger.error(f"Error processing response: {e}")
                    print(f"🤖 M.I.A: {response}")  # Fallback to text output

            except KeyboardInterrupt:
                logger.info("Shutting down M.I.A...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                continue

    except Exception as e:
        logger.error(f"Critical error during initialization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()