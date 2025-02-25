from audio_utils import AudioUtils
from speech_processor import SpeechProcessor
from llm_inference import LLMInference
from speech_generator import SpeechGenerator
from cognitive_architecture import MIACognitiveCore
from multimodal.processor import MultimodalProcessor
from memory.knowledge_graph import AgentMemory
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

    model = LLMInference(
        model_id=args.model_id,
        url=args.url,
        api_key=args.api_key
    )
    cognitive_core = MIACognitiveCore(model)
    multimodal_processor = MultimodalProcessor()
    memory = AgentMemory()
    
    audio_utils = AudioUtils()
    speech_processor = SpeechProcessor(device, args.stt_model)
    audio_model = SpeechGenerator(device)

    print("Welcome to M.I.A 2.0 - Multimodal Intelligent Assistant!")

    while True:
        try:
            inputs = {}
            if args.image_input:
                inputs['image'] = multimodal_processor.process_image(args.image_input)
                args.image_input = None  # Reset after processing
            
            # Audio processing
            mic = audio_utils.record_audio(speech_processor.transcriber, 2.0, 0.25)
            audio_input = next(speech_processor.transcribe_audio(mic))
            inputs['audio'] = audio_input['text']
            
            # Cognitive processing
            if args.enable_reasoning:
                processed = cognitive_core.process_multimodal_input(inputs)
                memory.store_experience(processed['text'], processed['embedding'])
                response = model.query_model(
                    processed['text'], 
                    context=memory.retrieve_context(processed['embedding'])
                )
            else:
                response = model.query_model(inputs['audio'])

            # Action execution
            if response.startswith("ACTION:"):
                action_result = AutomationUtil.execute_action(response[7:])
                print(f"Action Result: {action_result}")
            else:
                speech = audio_model.synthesize_audio(response)
                audio_utils.play_audio(speech['audio'], speech['sampling_rate'])

        except KeyboardInterrupt:
            print("\nM.I.A: Session ended. Goodbye!")
            break