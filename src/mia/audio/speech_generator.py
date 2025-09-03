"""Speech generation module with TTS and LLM capabilities."""
import logging
from typing import Optional, Any

# Optional imports with fallbacks
try:
    import warnings
    # Suppress transformers warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        from transformers.pipelines import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    pipeline = None

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    load_dataset = None

try:
    import sounddevice
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    sounddevice = None

try:
    import torch
    TORCH_AVAILABLE = True
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    device = "cpu"

try:
    from mia.llm.llm_inference import LLMInference
    LLM_AVAILABLE = True
except ImportError:
    LLMInference = None
    LLM_AVAILABLE = False

logger = logging.getLogger(__name__)

class SpeechGenerator:
    def __init__(self, device=None, model_id="microsoft/speecht5_tts", speaker=7306, llama_model_path=None):
        """
        Initialize the SpeechGenerator with TTS and LLM capabilities.

        :param device: Device for computation (e.g., 'cpu' or 'cuda:0').
        :param model_id: Model identifier for text-to-speech.
        :param speaker: Speaker ID for voice synthesis.
        :param llama_model_path: Path to local LLama model for text generation.
        """
        self.device = device or globals().get('device', 'cpu')
        self.model_id = model_id
        self.speaker = speaker
        self.llama_model_path = llama_model_path
        
        # Initialize components if available
        self.synthesiser = None
        self.embeddings_dataset = None
        self.speaker_embedding = None
        self.llm_inference = None
        
        self._init_tts()
        self._init_embeddings()
        self._init_llm()
        
    def _init_tts(self):
        """Initialize TTS pipeline."""
        # Disable TTS pipeline due to unsupported task
        logger.warning("TTS pipeline disabled - unsupported task 'text-to-speech'")
        self.synthesiser = None
            
    def _init_embeddings(self):
        """Initialize speaker embeddings."""
        if DATASETS_AVAILABLE and load_dataset:
            try:
                self.embeddings_dataset = list(load_dataset("Matthijs/cmu-arctic-xvectors", split="validation"))
                logger.info("Speaker embeddings dataset loaded")
                
                # Initialize speaker embedding if torch is available
                if TORCH_AVAILABLE and torch and self.embeddings_dataset:
                    self.speaker_embedding = torch.tensor(self.embeddings_dataset[self.speaker]["xvector"]).unsqueeze(0)
                    
            except Exception as e:
                logger.error(f"Failed to load embeddings dataset: {e}")
                self.embeddings_dataset = None
        else:
            logger.warning("Datasets not available - speaker embeddings disabled")
            
    def _init_llm(self):
        """Initialize LLM inference."""
        if LLM_AVAILABLE and LLMInference:
            try:
                self.llm_inference = LLMInference(llama_model_path=self.llama_model_path)
                logger.info("LLM inference initialized")
            except Exception as e:
                logger.error(f"Failed to initialize LLM inference: {e}")
                self.llm_inference = None
        else:
            logger.warning("LLM inference not available")

    def set_speaker(self, speaker_id):
        """Set the speaker for TTS voice."""
        if not TORCH_AVAILABLE or not torch or not self.embeddings_dataset:
            logger.warning("Cannot set speaker - dependencies not available")
            return False
            
        try:
            self.speaker = speaker_id
            self.speaker_embedding = torch.tensor(self.embeddings_dataset[speaker_id]["xvector"]).unsqueeze(0)
            logger.info(f"Speaker set to {speaker_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to set speaker {speaker_id}: {e}")
            return False

    def set_female_speaker(self):
        """Set a female speaker for TTS voice."""
        if not self.embeddings_dataset:
            logger.warning("Cannot set female speaker - embeddings not available")
            return False
            
        female_speaker_id = 7306  # Default female speaker
        return self.set_speaker(female_speaker_id)

    def generate_speech(self, text):
        """Generate speech from text."""
        if not self.synthesiser:
            logger.error("TTS synthesizer not available")
            return None
            
        if not text or not text.strip():
            logger.warning("Empty text provided for speech generation")
            return None
            
        try:
            if self.speaker_embedding is not None:
                speech = self.synthesiser(text, forward_params={"speaker_embeddings": self.speaker_embedding})
            else:
                speech = self.synthesiser(text)
            return speech
        except Exception as e:
            logger.error(f"Failed to generate speech: {e}")
            return None

    def generate_response_and_speech(self, prompt):
        """Generate text response using LLM and convert to speech."""
        if not self.llm_inference:
            logger.error("LLM inference not available")
            return None, None
            
        try:
            response = self.llm_inference.generate_response(prompt)
            if response:
                speech = self.generate_speech(response)
                return response, speech
            else:
                return None, None
        except Exception as e:
            logger.error(f"Failed to generate response and speech: {e}")
            return None, None

    def is_available(self):
        """Check if TTS functionality is available."""
        return self.synthesiser is not None
        
    def get_status(self):
        """Get status of all components."""
        return {
            "tts_available": self.synthesiser is not None,
            "embeddings_available": self.embeddings_dataset is not None,
            "llm_available": self.llm_inference is not None,
            "torch_available": TORCH_AVAILABLE,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "datasets_available": DATASETS_AVAILABLE,
            "sounddevice_available": SOUNDDEVICE_AVAILABLE
        }
