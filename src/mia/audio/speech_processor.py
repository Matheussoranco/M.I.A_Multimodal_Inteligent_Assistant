"""Speech processing module for audio transcription and recognition."""
import logging
from typing import Optional, Any, Dict
import io

# Import configuration manager
from ..config_manager import ConfigManager

# Optional imports with fallbacks
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    sr = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    sf = None

logger = logging.getLogger(__name__)

class SpeechProcessor:
    """Speech processing class for audio transcription and recognition."""
    
    def __init__(self, model_name=None, use_whisper=True, config_manager=None):
        """
        Initialize the SpeechProcessor.
        
        :param model_name: Whisper model name (tiny, base, small, medium, large)
        :param use_whisper: Whether to use Whisper as primary transcription method
        :param config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        
        # Use configuration values if not provided
        self.model_name = model_name or self.config_manager.config.audio.speech_model
        self.use_whisper = use_whisper
        self.whisper_model = None
        self.recognizer = None
        
        # Configuration values
        self.sample_rate = self.config_manager.config.audio.sample_rate
        self.chunk_size = self.config_manager.config.audio.chunk_size
        self.device_id = self.config_manager.config.audio.device_id
        self.input_threshold = self.config_manager.config.audio.input_threshold
        self.microphone = None
        
        self._init_whisper()
        self._init_speech_recognition()
        
    def _init_whisper(self):
        """Initialize Whisper model."""
        if WHISPER_AVAILABLE and whisper and self.use_whisper:
            try:
                logger.info(f"Loading Whisper model: {self.model_name}")
                self.whisper_model = whisper.load_model(self.model_name)
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                self.whisper_model = None
        else:
            logger.warning("Whisper not available or disabled")
            
    def _init_speech_recognition(self):
        """Initialize speech recognition as fallback."""
        if SPEECH_RECOGNITION_AVAILABLE and sr:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                logger.info("Speech recognition initialized")
            except Exception as e:
                logger.error(f"Failed to initialize speech recognition: {e}")
                self.recognizer = None
        else:
            logger.warning("Speech recognition not available")
    
    def transcribe_audio_file(self, audio_file_path: str) -> Optional[str]:
        """
        Transcribe audio from file.
        
        :param audio_file_path: Path to audio file
        :return: Transcribed text or None if failed
        """
        # Try Whisper first
        if self.whisper_model:
            try:
                result = self.whisper_model.transcribe(audio_file_path)
                text = result.get("text", "").strip()
                if text:
                    logger.info(f"Whisper transcription: {text[:100]}...")
                    return text
            except Exception as e:
                logger.error(f"Whisper transcription failed: {e}")
        
        # Fallback to speech recognition
        if self.recognizer:
            try:
                with sr.AudioFile(audio_file_path) as source:
                    audio = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio)
                    if text:
                        logger.info(f"Speech recognition transcription: {text[:100]}...")
                        return text
            except Exception as e:
                logger.error(f"Speech recognition transcription failed: {e}")
        
        logger.error("All transcription methods failed")
        return None
    
    def transcribe_audio_data(self, audio_data: bytes, sample_rate: int = 16000) -> Optional[str]:
        """
        Transcribe audio from raw audio data.
        
        :param audio_data: Raw audio data as bytes
        :param sample_rate: Sample rate of the audio
        :return: Transcribed text or None if failed
        """
        if not audio_data:
            logger.warning("Empty audio data provided")
            return None
            
        # Try Whisper first
        if self.whisper_model and NUMPY_AVAILABLE and np:
            try:
                # Convert bytes to numpy array for Whisper
                if SOUNDFILE_AVAILABLE and sf:
                    audio_array, _ = sf.read(io.BytesIO(audio_data))
                else:
                    # Simple conversion assuming 16-bit PCM
                    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                result = self.whisper_model.transcribe(audio_array)
                text = result.get("text", "").strip()
                if text:
                    logger.info(f"Whisper transcription: {text[:100]}...")
                    return text
            except Exception as e:
                logger.error(f"Whisper transcription from data failed: {e}")
        
        # Fallback to speech recognition
        if self.recognizer and SPEECH_RECOGNITION_AVAILABLE:
            try:
                audio = sr.AudioData(audio_data, sample_rate, 2)  # Assuming 16-bit
                text = self.recognizer.recognize_google(audio)
                if text:
                    logger.info(f"Speech recognition transcription: {text[:100]}...")
                    return text
            except Exception as e:
                logger.error(f"Speech recognition from data failed: {e}")
        
        logger.error("All transcription methods failed for audio data")
        return None
    
    def listen_microphone(self, timeout: float = 5.0, phrase_time_limit: float = 10.0) -> Optional[str]:
        """
        Listen to microphone and transcribe speech.
        
        :param timeout: Maximum time to wait for speech to start
        :param phrase_time_limit: Maximum time to record a phrase
        :return: Transcribed text or None if failed
        """
        if not self.recognizer or not self.microphone:
            logger.error("Speech recognition not available for microphone input")
            return None
            
        try:
            logger.info("Listening for speech...")
            with self.microphone as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
            # Listen for audio
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            # Try Whisper first if available
            if self.whisper_model:
                try:
                    # Convert to format suitable for Whisper
                    audio_data = audio.get_wav_data()
                    return self.transcribe_audio_data(audio_data, audio.sample_rate)
                except Exception as e:
                    logger.error(f"Whisper microphone transcription failed: {e}")
            
            # Fallback to Google Speech Recognition
            try:
                text = self.recognizer.recognize_google(audio)
                if text:
                    logger.info(f"Microphone transcription: {text}")
                    return text
            except sr.UnknownValueError:
                logger.warning("Could not understand audio")
            except sr.RequestError as e:
                logger.error(f"Speech recognition service error: {e}")
                
        except sr.WaitTimeoutError:
            logger.warning("Listening timeout - no speech detected")
        except Exception as e:
            logger.error(f"Microphone listening failed: {e}")
        
        return None
    
    def is_available(self) -> bool:
        """Check if any transcription method is available."""
        return self.whisper_model is not None or self.recognizer is not None
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all components."""
        return {
            "whisper_available": self.whisper_model is not None,
            "speech_recognition_available": self.recognizer is not None,
            "microphone_available": self.microphone is not None,
            "whisper_model": self.model_name if self.whisper_model else None,
            "dependencies": {
                "whisper": WHISPER_AVAILABLE,
                "speech_recognition": SPEECH_RECOGNITION_AVAILABLE,
                "numpy": NUMPY_AVAILABLE,
                "soundfile": SOUNDFILE_AVAILABLE
            }
        }
