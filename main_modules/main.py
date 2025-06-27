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
import argparse
import torch
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default='http://localhost:11434/v1')
    parser.add_argument("--model-id", default='mistral:instruct')
    parser.add_argument("--api-key", default='ollama')
    parser.add_argument("--stt-model", default="openai/whisper-base.en")
    
    parser.add_argument("--image-input", help="Path to image file for multimodal processing")
    parser.add_argument("--enable-reasoning", action="store_true", help="Enable advanced reasoning")

    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Replace LLMInference with LLMManager for unified LLM API
    llm = LLMManager(provider='openai', model_id=args.model_id, api_key=args.api_key, url=args.url)
    cognitive_core = MIACognitiveCore(llm)
    multimodal_processor = MultimodalProcessor()
    memory = AgentMemory()
    
    audio_utils = AudioUtils()
    speech_processor = SpeechProcessor(model_path=args.stt_model)
    audio_model = SpeechGenerator(device)
    langchain_verifier = LangChainVerifier(llm=llm)

    # Initialize new modules
    action_executor = ActionExecutor({"open_file": True, "web_search": True})
    user_learning = UserLearning()
    plugin_manager = PluginManager()
    plugin_manager.load_plugins()
    security_manager = SecurityManager()
    deployment_manager = DeploymentManager()
    vision_processor = VisionProcessor()
    long_term_memory = LongTermMemory()
    calendar = CalendarIntegration()

    print("Welcome to M.I.A 2.0 - Multimodal Intelligent Assistant!")

    while True:
        try:
            inputs = {}
            if args.image_input:
                # Use VisionProcessor for advanced image processing
                inputs['image'] = vision_processor.process_image(args.image_input)
                args.image_input = None
            # Audio processing
            mic = audio_utils.record_audio(speech_processor, 2.0, 0.25)
            audio_chunk = next(mic)
            transcription = speech_processor.transcribe_audio(audio_chunk)
            inputs['audio'] = transcription.strip() if isinstance(transcription, str) else ''
            # Cognitive processing
            if args.enable_reasoning:
                processed = cognitive_core.process_multimodal_input(inputs)
                memory.store_experience(processed['text'], processed['embedding'])
                # Store in long-term memory
                long_term_memory.remember(processed['text'])
                response = llm.query_model(
                    processed['text'], 
                    context=memory.retrieve_context(processed['embedding'])
                )
            else:
                response = llm.query_model(inputs['audio'])
            # Action execution with security and plugin support
            if response.startswith("ACTION:"):
                action = response[7:].strip()
                if security_manager.check_permission(action):
                    # Try plugin first
                    plugin = plugin_manager.get_plugin(action)
                    if plugin:
                        action_result = plugin.run()
                    else:
                        action_result = action_executor.execute(action, {})
                    print(f"Action Result: {action_result}")
                else:
                    print(security_manager.explain_action(action))
            elif response.startswith("CALENDAR:"):
                event = response[9:].strip()
                print(calendar.add_event(event))
            else:
                # User learning feedback
                user_learning.update_profile({"last_response": response})
                # Synthesize and play audio
                speech = audio_model.synthesize_audio(response)
                # If speech is a dict with 'audio' and 'sampling_rate', use them; else, try to play directly
                if isinstance(speech, dict) and 'audio' in speech and 'sampling_rate' in speech:
                    audio_utils.play_audio(speech['audio'], speech['sampling_rate'])
                elif hasattr(speech, 'audio') and hasattr(speech, 'sampling_rate'):
                    audio_utils.play_audio(speech.audio, speech.sampling_rate)
                else:
                    print("[Warning] Could not play synthesized audio: unexpected format.")

            # Example: Use system control for file actions
            # SystemControl.open_file('example.txt')
            # Example: Use LangChain for verification
            # verification_result = langchain_verifier.verify(inputs['audio'])

        except KeyboardInterrupt:
            print("\nM.I.A: Session ended. Goodbye!")
            break

if __name__ == "__main__":
    main()