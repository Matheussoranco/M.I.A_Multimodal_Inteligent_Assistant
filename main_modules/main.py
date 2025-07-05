import argparse
import sys
import logging
from typing import Optional

# Optional imports with error handling
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

# Import project modules
from audio.audio_utils import AudioUtils
from audio.speech_processor import SpeechProcessor
from llm.llm_manager import LLMManager
from audio.speech_generator import SpeechGenerator
from core.cognitive_architecture import MIACognitiveCore
from multimodal.processor import MultimodalProcessor
from memory.knowledge_graph import AgentMemory
from langchain.langchain_verifier import LangChainVerifier
from system.system_control import SystemControl
from utils.automation_util import AutomationUtil
from tools.action_executor import ActionExecutor
from learning.user_learning import UserLearning
from plugins.plugin_manager import PluginManager
from security.security_manager import SecurityManager
from deployment.deployment_manager import DeploymentManager
from multimodal.vision_processor import VisionProcessor
from memory.long_term_memory import LongTermMemory
from planning.calendar_integration import CalendarIntegration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="M.I.A - Multimodal Intelligent Assistant")
    parser.add_argument("--url", default='http://localhost:11434/v1', help="LLM API URL")
    parser.add_argument("--model-id", default='mistral:instruct', help="Model ID to use")
    parser.add_argument("--api-key", default='ollama', help="API key for the LLM service")
    parser.add_argument("--stt-model", default="openai/whisper-base.en", help="Speech-to-text model")
    parser.add_argument("--image-input", help="Path to image file for multimodal processing")
    parser.add_argument("--enable-reasoning", action="store_true", help="Enable advanced reasoning")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine device
    if HAS_TORCH:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
    else:
        device = "cpu"
        logger.warning("PyTorch not available, using CPU mode")

    try:
        # Initialize core components
        logger.info("Initializing M.I.A components...")
        
        # Initialize LLM with error handling
        llm = LLMManager(provider='openai', model_id=args.model_id, api_key=args.api_key, url=args.url)
        if not llm.is_available():
            logger.warning(f"LLM provider {llm.provider} not available, some features may be limited")
        
        # Initialize other components
        cognitive_core = MIACognitiveCore(llm)
        multimodal_processor = MultimodalProcessor()
        memory = AgentMemory()
        
        audio_utils = AudioUtils()
        speech_processor = SpeechProcessor(model_name=args.stt_model or "base")
        audio_model = SpeechGenerator(device) if HAS_TORCH else None
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

        logger.info("Welcome to M.I.A 2.0 - Multimodal Intelligent Assistant!")

        while True:
            try:
                inputs = {}
                
                # Handle image input if provided
                if args.image_input:
                    try:
                        inputs['image'] = vision_processor.process_image(args.image_input)
                        args.image_input = None  # Process only once
                        logger.info("Image processed successfully")
                    except Exception as e:
                        logger.error(f"Error processing image: {e}")
                
                # Audio processing
                try:
                    mic = audio_utils.record_audio(speech_processor, 2.0, 0.25)
                    audio_chunk = next(mic)
                    transcription = speech_processor.transcribe_audio_data(audio_chunk.tobytes(), 16000)
                    inputs['audio'] = transcription.strip() if isinstance(transcription, str) else ''
                    
                    if not inputs['audio']:
                        logger.info("No speech detected, continuing...")
                        continue
                        
                    logger.info(f"Transcribed: {inputs['audio']}")
                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                    continue

                # Cognitive processing
                response = None
                try:
                    if args.enable_reasoning:
                        processed = cognitive_core.process_multimodal_input(inputs)
                        memory.store_experience(processed['text'], processed['embedding'])
                        long_term_memory.remember(processed['text'])
                        response = llm.query_model(
                            processed['text'], 
                            context=memory.retrieve_context(processed['embedding'])
                        )
                    else:
                        response = llm.query_model(inputs['audio'])
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
                        # Regular response - convert to speech if possible
                        print(f"M.I.A: {response}")
                        
                        if audio_model:
                            try:
                                speech = audio_model.generate_speech(response)
                                if isinstance(speech, dict) and 'audio' in speech and 'sampling_rate' in speech:
                                    audio_utils.play_audio(speech['audio'], speech['sampling_rate'])
                                else:
                                    logger.warning("Speech generation returned unexpected format")
                            except Exception as e:
                                logger.error(f"Error generating speech: {e}")
                        
                except Exception as e:
                    logger.error(f"Error processing response: {e}")
                    print(f"M.I.A: {response}")  # Fallback to text output

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