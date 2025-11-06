import io
import logging
import time
from typing import Generator, List, Optional

import numpy as np

# Optional imports with error handling
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None
    HAS_SOUNDDEVICE = False

try:
    import keyboard  # type: ignore
    HAS_KEYBOARD = True
except ImportError:  # pragma: no cover - optional dependency
    keyboard = None  # type: ignore
    HAS_KEYBOARD = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    sf = None
    HAS_SOUNDFILE = False

try:
    import warnings
    # Suppress pydub ffmpeg warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")
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
        self.chunk_size = 1024
        self.device_id: Optional[int] = None
        self.input_threshold = 0.01

    def configure(
        self,
        sample_rate: Optional[int] = None,
        chunk_size: Optional[int] = None,
        device_id: Optional[int] = None,
        input_threshold: Optional[float] = None,
    ) -> None:
        """Adjust runtime audio parameters from configuration."""
        if sample_rate is not None:
            self.sample_rate = int(sample_rate)
        if chunk_size is not None:
            self.chunk_size = int(chunk_size)
        if device_id is not None:
            self.device_id = int(device_id)
        if input_threshold is not None:
            self.input_threshold = float(input_threshold)
        
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
            logger.error("sounddevice not available - audio recording disabled")
            logger.error("Install sounddevice: pip install sounddevice")
            raise RuntimeError("Audio recording not available - sounddevice not installed")
            
        try:
            logger.info("Starting audio recording...")
            chunk_samples = int(chunk_length_s * self.sample_rate)
            
            # Record audio
            if sd is not None:
                audio_data = sd.rec(  # type: ignore
                    frames=chunk_samples,
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=np.float32,
                    device=self.device_id,
                )
                sd.wait()  # type: ignore
            else:
                # Fallback simulation
                audio_data = self._create_simulated_audio(chunk_length_s).reshape(-1, 1)
            
            # Flatten if multi-channel
            if audio_data.ndim > 1:
                audio_data = audio_data.flatten()
                
            logger.info(f"Recorded {len(audio_data)} samples")
            yield audio_data
            
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            raise RuntimeError(f"Audio recording failed: {e}")

    def _create_simulated_audio(self, duration_s: float) -> np.ndarray:
        """Create simulated audio data for testing."""
        samples = int(duration_s * self.sample_rate)
        # Create some random noise as placeholder
        return np.random.normal(0, 0.1, samples).astype(np.float32)

    def capture_chunk(
        self,
        duration_s: float = 0.5,
        device_index: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """Capture a short chunk from the microphone."""
        frames = max(1, int(duration_s * self.sample_rate))
        if not HAS_SOUNDDEVICE or sd is None:
            return self._create_simulated_audio(duration_s)

        try:
            audio = sd.rec(  # type: ignore
                frames=frames,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                device=device_index if device_index is not None else self.device_id,
            )
            sd.wait()  # type: ignore
            if audio.ndim > 1:
                audio = audio[:, 0]
            return audio.astype(np.float32)
        except Exception as exc:  # pragma: no cover - hardware dependent
            logger.error("Audio chunk capture failed: %s", exc)
            return None

    def play_audio(self, audio_data: np.ndarray, sample_rate: Optional[int] = None):
        """
        Play audio data.
        
        :param audio_data: Audio data as numpy array
        :param sample_rate: Sample rate for playback
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        try:
            if HAS_SOUNDDEVICE and sd is not None:
                logger.info("Playing audio with sounddevice...")
                sd.play(audio_data, samplerate=sample_rate)  # type: ignore
                sd.wait()  # type: ignore # Wait until audio is finished
            elif HAS_PYDUB and AudioSegment is not None and play is not None:
                logger.info("Playing audio via pydub...")
                segment = AudioSegment(
                    audio_data.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=audio_data.dtype.itemsize,
                    channels=1,
                )
                play(segment)
            else:
                logger.warning("Audio playback not available (sounddevice/pydub not installed)")
                
        except Exception as e:
            logger.error(f"Error playing audio: {e}")

    def save_audio(self, audio_data: np.ndarray, filename: str, sample_rate: Optional[int] = None):
        """
        Save audio data to file.
        
        :param audio_data: Audio data as numpy array
        :param filename: Output filename
        :param sample_rate: Sample rate
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        try:
            if HAS_SOUNDFILE and sf is not None:
                sf.write(filename, audio_data, sample_rate)  # type: ignore
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
            if HAS_SOUNDFILE and sf is not None:
                audio_data, sample_rate = sf.read(filename)  # type: ignore
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

    def play_tts_payload(self, payload: Optional[dict]) -> bool:
        """Play audio payload returned by TTS providers."""
        if not payload:
            return False

        audio_bytes: Optional[bytes] = None
        if isinstance(payload, dict):
            audio_bytes = payload.get('audio_bytes') or payload.get('data')
            if audio_bytes is None and 'audio_url' in payload:
                logger.info("TTS response returned URL; download not implemented")
                return False

        if not audio_bytes:
            logger.debug("No audio bytes present in TTS payload")
            return False

        if HAS_PYDUB and AudioSegment is not None and play is not None:
            try:
                buffer = io.BytesIO(audio_bytes)
                segment = AudioSegment.from_file(buffer)
                play(segment)
                return True
            except Exception as exc:
                logger.error("Failed to play TTS payload via pydub: %s", exc)

        if HAS_SOUNDDEVICE and HAS_SOUNDFILE and sd is not None and sf is not None:
            try:
                buffer = io.BytesIO(audio_bytes)
                audio_array, sample_rate = sf.read(buffer)  # type: ignore
                if isinstance(audio_array, np.ndarray):
                    if audio_array.ndim > 1:
                        audio_array = audio_array[:, 0]
                    self.play_audio(audio_array.astype(np.float32), sample_rate=sample_rate)
                    return True
            except Exception as exc:
                logger.error("Failed to play TTS payload via soundfile: %s", exc)

        logger.warning("Unable to play TTS payload; missing audio backends")
        return False

    def wait_for_push_to_talk(
        self,
        key: str = "space",
        timeout: Optional[float] = None,
        prompt: Optional[str] = None,
    ) -> bool:
        """Block until the push-to-talk key is pressed or timeout occurs."""
        if prompt:
            print(prompt)

        if HAS_KEYBOARD and keyboard is not None:
            start = time.time()
            while True:
                if keyboard.is_pressed(key):  # type: ignore[call-arg]
                    return True
                if timeout is not None and (time.time() - start) > timeout:
                    return False
                time.sleep(0.05)
        else:
            try:
                input("Pressione Enter para falar...")
                return True
            except (EOFError, KeyboardInterrupt):
                return False

        return False

    def capture_with_vad(
        self,
        speech_processor=None,
        audio_config=None,
        max_duration: float = 12.0,
    ) -> Optional[np.ndarray]:
        """Capture audio chunks until silence is detected using simple VAD heuristics."""
        if audio_config is not None:
            max_duration = float(getattr(audio_config, 'max_phrase_duration', max_duration) or max_duration)

        window_s = max(0.2, getattr(audio_config, 'vad_frame_duration_ms', 300) / 1000.0 if audio_config else 0.3)
        silence_limit = max(0.4, getattr(audio_config, 'vad_silence_duration_ms', 600) / 1000.0 if audio_config else 0.8)
        energy_floor = float(getattr(audio_config, 'input_threshold', self.input_threshold) or self.input_threshold)

        chunks: List[np.ndarray] = []
        silence_time = 0.0
        start_time = time.time()

        vad_detector = getattr(speech_processor, "_vad_detector", None) if speech_processor else None

        while time.time() - start_time < max_duration:
            chunk = self.capture_chunk(duration_s=window_s)
            if chunk is None or chunk.size == 0:
                continue

            energy = self.compute_energy(chunk)
            has_speech = energy >= energy_floor

            if vad_detector and hasattr(vad_detector, "is_available") and vad_detector.is_available():
                try:
                    has_speech = has_speech or vad_detector.has_speech(
                        self._to_pcm16(chunk),
                        self.sample_rate,
                    )
                except Exception as exc:  # pragma: no cover - library call
                    logger.debug("VAD detector failed: %s", exc)

            if has_speech:
                chunks.append(chunk)
                silence_time = 0.0
            else:
                if chunks:
                    silence_time += window_s
                    if silence_time >= silence_limit:
                        break

        if not chunks:
            return None

        return np.concatenate(chunks).astype(np.float32)

    @staticmethod
    def compute_energy(audio: np.ndarray) -> float:
        """Compute root-mean-square energy of audio."""
        if audio.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))

    @staticmethod
    def _to_pcm16(audio: np.ndarray) -> bytes:
        """Convert float audio array to PCM16 bytes."""
        scaled = np.clip(audio.astype(np.float32), -1.0, 1.0)
        return (scaled * 32767).astype(np.int16).tobytes()
