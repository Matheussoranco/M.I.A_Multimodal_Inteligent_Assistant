from transformers import pipeline
from datasets import load_dataset
import sounddevice
import torch
from LLM_inference import LLMInference

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class SpeechGenerator:
    def __init__(self, device, model_id="microsoft/speecht5_tts", speaker=7306, llama_model_path=None):
        """
        Initialize the SpeechGenerator with TTS and LLM capabilities.

        :param device: Device for computation (e.g., 'cpu' or 'cuda:0').
        :param model_id: Model identifier for text-to-speech.
        :param speaker: Speaker ID for voice synthesis.
        :param llama_model_path: Path to local LLama model for text generation.
        """
        self.synthesiser = pipeline("text-to-speech", model=model_id, device=device)
        self.embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embedding = torch.tensor(self.embeddings_dataset[speaker]["xvector"]).unsqueeze(0)
        self.llm_inference = LLMInference(llama_model_path=llama_model_path)

    def generate_text(self, prompt):
        """
        Generate text using the integrated LLM.

        :param prompt: Text prompt for the LLM.
        :return: Generated text.
        """
        return self.llm_inference.query_model(prompt)

    def synthesize_audio(self, text):
        """
        Generate audio for the provided text with a female voice.

        :param text: Input text to convert to speech.
        :return: Synthesized speech audio.
        """
        female_speaker_id = 7317  # Example ID for a female voice; adjust as needed
        self.speaker_embedding = torch.tensor(self.embeddings_dataset[female_speaker_id]["xvector"]).unsqueeze(0)
        speech = self.synthesiser(text, 
                                  forward_params={"speaker_embeddings": self.speaker_embedding})
        return speech

# Example usage:
# generator = SpeechGenerator(device=device, llama_model_path="path/to/llama/model")
# generated_text = generator.generate_text("What is the weather today?")
# audio = generator.synthesize_audio(generated_text)
