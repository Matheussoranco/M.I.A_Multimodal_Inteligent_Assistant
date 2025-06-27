import threading
import queue
import sys
from llm.llm_manager import LLMManager
from audio.audio_utils import AudioUtils
from audio.speech_processor import SpeechProcessor
import torch

class ChatInterface:
    def __init__(self, llm_provider='openai', model_id='gpt-3.5-turbo', api_key=None, url=None, voice_mode=False, stt_model='openai/whisper-base.en'):
        self.llm = LLMManager(provider=llm_provider, model_id=model_id, api_key=api_key, url=url)
        self.voice_mode = voice_mode
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if voice_mode:
            self.audio_utils = AudioUtils()
            self.speech_processor = SpeechProcessor(stt_model)
        self.input_queue = queue.Queue()
        self.stop_event = threading.Event()

    def run(self):
        print("Welcome to M.I.A Chat! Type your message or say it (voice mode). Type 'exit' to quit.")
        if self.voice_mode:
            threading.Thread(target=self._voice_input_loop, daemon=True).start()
        while not self.stop_event.is_set():
            if not self.voice_mode:
                user_input = input("You: ")
                if user_input.lower() == 'exit':
                    break
                self._handle_input(user_input)
            else:
                try:
                    user_input = self.input_queue.get(timeout=0.1)
                    if user_input.lower() == 'exit':
                        self.stop_event.set()
                        break
                    self._handle_input(user_input)
                except queue.Empty:
                    continue
        print("Goodbye!")

    def _handle_input(self, user_input):
        print("MIA is thinking...")
        try:
            response = self.llm.query(user_input)
            print(f"MIA: {response}")
        except Exception as e:
            print(f"Error: {e}")

    def _voice_input_loop(self):
        print("[Voice mode enabled. Speak now.]")
        while not self.stop_event.is_set():
            # For compatibility, use the speech_processor's sampling_rate
            mic = self.audio_utils.record_audio(self.speech_processor, 2.0, 0.25)
            # mic is a generator of audio chunks; take the first chunk for demo
            audio_chunk = next(mic)
            transcription = self.speech_processor.transcribe_audio(audio_chunk)
            text = transcription.strip() if isinstance(transcription, str) else ''
            if text:
                print(f"[You said]: {text}")
                self.input_queue.put(text)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', default='openai')
    parser.add_argument('--model-id', default='gpt-3.5-turbo')
    parser.add_argument('--api-key', default=None)
    parser.add_argument('--url', default=None)
    parser.add_argument('--voice', action='store_true')
    parser.add_argument('--stt-model', default='openai/whisper-base.en')
    args = parser.parse_args()
    chat = ChatInterface(llm_provider=args.provider, model_id=args.model_id, api_key=args.api_key, url=args.url, voice_mode=args.voice, stt_model=args.stt_model)
    chat.run()
