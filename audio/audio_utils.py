import logging
import numpy as np
import time
from typing import Generator, Optional

# Optional imports with error handling
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None
    HAS_SOUNDDEVICE = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    sf = None
    HAS_SOUNDFILE = False

try:
    from pydub import AudioSegment
    from pydub.playback import play
    HAS_PYDUB = True
except ImportError:
    AudioSegment = None
    play = None
    HAS_PYDUB = False

logger = logging.getLogger(__name__)

class AudioUtils:
    """Utility class for audio recording and playback."""
    
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 1
        
    def record_audio(self, transcriber, chunk_length_s=2.0, stream_chunk_s=0.25, save_to_file=None) -> Generator[np.ndarray, None, None]:
        """
        Record audio using a live microphone stream.

        :param transcriber: Transcriber object to get sampling rate.
        :param chunk_length_s: Length of audio chunks in seconds.
        :param stream_chunk_s: Stream chunk duration in seconds.
        :param save_to_file: Optional path to save the recorded audio.
        :return: Recorded audio data as a generator.
        """
        if not HAS_SOUNDDEVICE:
            logger.warning("sounddevice not available, returning simulated audio")
            yield self._create_simulated_audio(chunk_length_s)
            return
            
        try:
            logger.info("Starting audio recording...")
            chunk_samples = int(chunk_length_s * self.sample_rate)
            
            # Record audio
            audio_data = sd.rec(
                frames=chunk_samples,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32
            )
            
            # Wait for recording to complete
            sd.wait()
            
            # Flatten if multi-channel
            if audio_data.ndim > 1:
                audio_data = audio_data.flatten()
                
            logger.info(f"Recorded {len(audio_data)} samples")
            yield audio_data
            
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            yield self._create_simulated_audio(chunk_length_s)

    def _create_simulated_audio(self, duration_s: float) -> np.ndarray:
        """Create simulated audio data for testing."""
        samples = int(duration_s * self.sample_rate)
        # Create some random noise as placeholder
        return np.random.normal(0, 0.1, samples).astype(np.float32)

    def play_audio(self, audio_data: np.ndarray, sample_rate: int = None):
        """
        Play audio data.
        
        :param audio_data: Audio data as numpy array
        :param sample_rate: Sample rate for playback
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        try:
            if HAS_SOUNDDEVICE:
                logger.info("Playing audio with sounddevice...")
                sd.play(audio_data, samplerate=sample_rate)
                sd.wait()  # Wait until audio is finished
            else:
                logger.warning("Audio playback not available (sounddevice not installed)")
                
        except Exception as e:
            logger.error(f"Error playing audio: {e}")

    def save_audio(self, audio_data: np.ndarray, filename: str, sample_rate: int = None):
        """
        Save audio data to file.
        
        :param audio_data: Audio data as numpy array
        :param filename: Output filename
        :param sample_rate: Sample rate
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        try:
            if HAS_SOUNDFILE:
                sf.write(filename, audio_data, sample_rate)
                logger.info(f"Audio saved to {filename}")
            else:
                logger.warning("Cannot save audio (soundfile not installed)")
                
        except Exception as e:
            logger.error(f"Error saving audio: {e}")

    def load_audio(self, filename: str) -> tuple[np.ndarray, int]:
        """
        Load audio from file.
        
        :param filename: Input filename
        :return: Tuple of (audio_data, sample_rate)
        """
        try:
            if HAS_SOUNDFILE:
                audio_data, sample_rate = sf.read(filename)
                logger.info(f"Audio loaded from {filename}")
                return audio_data, sample_rate
            else:
                logger.warning("Cannot load audio (soundfile not installed)")
                return self._create_simulated_audio(2.0), self.sample_rate
                
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return self._create_simulated_audio(2.0), self.sample_rate

    def get_audio_info(self) -> dict:
        """Get information about audio capabilities."""
        return {
            "has_sounddevice": HAS_SOUNDDEVICE,
            "has_soundfile": HAS_SOUNDFILE,
            "has_pydub": HAS_PYDUB,
            "sample_rate": self.sample_rate,
            "channels": self.channels
        }
