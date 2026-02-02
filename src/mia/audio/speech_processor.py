"""Speech processing module for audio transcription and recognition."""

import logging
import os
from typing import Any, Dict, Optional

# Import configuration manager
from ..config_manager import ConfigManager
from .vad_detector import VoiceActivityDetector

# Optional imports with fallbacks
try:
    import whisper  # type: ignore

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

# Import resource manager
from ..resource_manager import (
    ResourceState,
    WhisperModelResource,
    resource_manager,
)

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

        # Use configuration values if not provided (with safe access)
        config = self.config_manager.config
        if config and hasattr(config, "audio"):
            self.model_name = model_name or config.audio.speech_model
            self.sample_rate = config.audio.sample_rate
            self.chunk_size = config.audio.chunk_size
            self.device_id = config.audio.device_id
            self.input_threshold = config.audio.input_threshold
            self._vad_detector = VoiceActivityDetector(
                aggressiveness=config.audio.vad_aggressiveness,
                frame_duration_ms=config.audio.vad_frame_duration_ms,
                min_active_duration_ms=config.audio.vad_silence_duration_ms,
                enabled=config.audio.vad_enabled,
                logger_instance=logger,
            )
        else:
            # Default values if config is not available
            self.model_name = model_name or "base"
            self.sample_rate = 16000
            self.chunk_size = 1024
            self.device_id = None
            self.input_threshold = 0.5
            self._vad_detector = VoiceActivityDetector(enabled=False)

        self.use_whisper = use_whisper
        self.whisper_model = None
        self.recognizer = None

        # Configuration values
        self.microphone = None

        # Initialize resource manager
        self.resource_manager = resource_manager
        if not self.resource_manager._running:
            self.resource_manager.start()

        self._init_whisper()
        self._init_speech_recognition()

    def _init_whisper(self):
        """Initialize Whisper model using resource manager."""
        if not WHISPER_AVAILABLE or not whisper or not self.use_whisper:
            logger.warning("Whisper not available or disabled")
            return

        try:
            # Check if model is already loaded
            resource_id = f"whisper_{self.model_name}"
            if resource_id in self.resource_manager.resources:
                self.whisper_model = self.resource_manager.resources[
                    resource_id
                ]
                logger.info(f"Using existing Whisper model: {self.model_name}")
            else:
                # Create new model resource
                model_resource = WhisperModelResource(self.model_name)
                self.resource_manager.resources[resource_id] = model_resource
                self.whisper_model = model_resource
                logger.info(
                    f"Whisper model resource created: {self.model_name}"
                )

        except Exception as e:
            logger.error(f"Failed to initialize Whisper model resource: {e}")
            self.whisper_model = None

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

    def transcribe(self, audio_input: Any) -> Optional[str]:
        """Transcribe audio from a file path or raw audio bytes."""
        if audio_input is None:
            logger.warning("No audio input provided")
            return None

        if isinstance(audio_input, (str, os.PathLike)):
            return self.transcribe_audio_file(str(audio_input))

        if isinstance(audio_input, (bytes, bytearray)):
            return self.transcribe_audio_data(bytes(audio_input))

        logger.warning("Unsupported audio input type: %s", type(audio_input))
        return None

    def transcribe_audio_file(self, audio_file_path: str) -> Optional[str]:
        """
        Transcribe audio from file.

        :param audio_file_path: Path to audio file
        :return: Transcribed text or None if failed
        """
        # Try Whisper first
        if self.whisper_model and isinstance(
            self.whisper_model, WhisperModelResource
        ):
            try:
                # Ensure model is loaded
                if self.whisper_model.state == ResourceState.CREATED:
                    self.whisper_model.initialize()

                result = self.whisper_model.transcribe(audio_file_path)
                text = (
                    str(result.get("text", "")).strip()
                    if isinstance(result, dict)
                    else ""
                )
                if text:
                    logger.info(f"Whisper transcription: {text[:100]}...")
                    return text
            except Exception as e:
                logger.error(f"Whisper transcription failed: {e}")

        # Fallback to speech recognition
        if self.recognizer and sr is not None:
            try:
                with sr.AudioFile(audio_file_path) as source:  # type: ignore
                    audio = self.recognizer.record(source)
                    text = getattr(self.recognizer, "recognize_google")(audio)  # type: ignore
                    if text:
                        logger.info(
                            f"Speech recognition transcription: {text[:100]}..."
                        )
                        return text
            except Exception as e:
                logger.error(f"Speech recognition transcription failed: {e}")

        logger.error("All transcription methods failed")
        return None

    def transcribe_audio_data(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> Optional[str]:
        """
        Transcribe audio from raw audio data.

        :param audio_data: Raw audio data as bytes
        :param sample_rate: Sample rate of the audio
        :return: Transcribed text or None if failed
        """
        if not audio_data:
            logger.warning("Empty audio data provided")
            return None

        if (
            self._vad_detector
            and self._vad_detector.is_available()
            and NUMPY_AVAILABLE
            and np is not None
        ):
            try:
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                pcm16 = np.clip(audio_array, -1.0, 1.0)
                pcm16 = (pcm16 * 32767).astype(np.int16).tobytes()
                if not self._vad_detector.has_speech(pcm16, sample_rate):
                    logger.info("VAD rejected audio data as silence")
                    return None
            except Exception as exc:
                logger.debug("VAD preprocessing failed: %s", exc)

        # Try Whisper first
        if (
            self.whisper_model
            and isinstance(self.whisper_model, WhisperModelResource)
            and NUMPY_AVAILABLE
            and np
        ):
            try:
                # Ensure model is loaded
                if self.whisper_model.state == ResourceState.CREATED:
                    self.whisper_model.initialize()

                # Convert bytes to numpy array for Whisper
                # These are raw float32 bytes from numpy array, not a file format
                audio_array = np.frombuffer(audio_data, dtype=np.float32)

                # Ensure audio is mono and normalized
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.flatten()

                # Whisper expects audio to be normalized between -1 and 1
                audio_array = np.clip(audio_array, -1.0, 1.0)

                result = self.whisper_model.transcribe(audio_array)
                text = (
                    str(result.get("text", "")).strip()
                    if isinstance(result, dict)
                    else ""
                )
                if text:
                    logger.info(f"Whisper transcription: {text[:100]}...")
                    return text
            except Exception as e:
                logger.error(f"Whisper transcription from data failed: {e}")

        # Fallback to speech recognition
        if self.recognizer and SPEECH_RECOGNITION_AVAILABLE and sr is not None:
            try:
                audio = sr.AudioData(audio_data, sample_rate, 2)  # type: ignore # Assuming 16-bit
                text = getattr(self.recognizer, "recognize_google")(audio)  # type: ignore
                if text:
                    logger.info(
                        f"Speech recognition transcription: {text[:100]}..."
                    )
                    return text
            except Exception as e:
                logger.error(f"Speech recognition from data failed: {e}")

        logger.error("All transcription methods failed for audio data")
        return None

    def listen_microphone(
        self, timeout: float = 5.0, phrase_time_limit: float = 10.0
    ) -> Optional[str]:
        """
        Listen to microphone and transcribe speech.

        :param timeout: Maximum time to wait for speech to start
        :param phrase_time_limit: Maximum time to record a phrase
        :return: Transcribed text or None if failed
        """
        if not self.recognizer or not self.microphone:
            logger.error(
                "Speech recognition not available for microphone input"
            )
            return None

        try:
            logger.info("Listening for speech...")
            with self.microphone as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)

            # Listen for audio
            with self.microphone as source:
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit,
                )

            sample_rate = getattr(audio, "sample_rate", 16000)  # type: ignore

            if self._vad_detector and self._vad_detector.is_available():
                try:
                    raw_pcm = audio.get_raw_data(convert_rate=sample_rate, convert_width=2)  # type: ignore
                    if not self._vad_detector.has_speech(raw_pcm, sample_rate):
                        logger.info(
                            "No speech detected by VAD; skipping transcription"
                        )
                        return None
                except Exception as exc:
                    logger.debug("VAD check failed: %s", exc)

            # Try Whisper first if available
            if self.whisper_model:
                try:
                    # Convert to format suitable for Whisper
                    audio_data = getattr(audio, "get_wav_data")()  # type: ignore
                    return self.transcribe_audio_data(audio_data, sample_rate)
                except Exception as e:
                    logger.error(
                        f"Whisper microphone transcription failed: {e}"
                    )

            # Fallback to Google Speech Recognition
            try:
                text = getattr(self.recognizer, "recognize_google")(audio)  # type: ignore
                if text:
                    logger.info(f"Microphone transcription: {text}")
                    return text
            except Exception as e:
                if (
                    sr
                    and hasattr(sr, "UnknownValueError")
                    and isinstance(e, sr.UnknownValueError)
                ):
                    logger.warning("Could not understand audio")
                elif (
                    sr
                    and hasattr(sr, "RequestError")
                    and isinstance(e, sr.RequestError)
                ):
                    logger.error(f"Speech recognition service error: {e}")
                else:
                    logger.error(f"Speech recognition error: {e}")

        except Exception as e:
            if (
                sr
                and hasattr(sr, "WaitTimeoutError")
                and isinstance(e, sr.WaitTimeoutError)
            ):
                logger.warning("Listening timeout - no speech detected")
            else:
                logger.error(f"Microphone listening failed: {e}")

        return None

    def cleanup(self) -> None:
        """Clean up resources used by the speech processor."""
        try:
            # Note: We don't cleanup the Whisper model here as it might be shared
            # The resource manager will handle cleanup based on usage patterns
            if self.whisper_model and isinstance(
                self.whisper_model, WhisperModelResource
            ):
                # Just update last used time, let resource manager handle cleanup
                self.whisper_model.update_last_used()

            logger.info("Speech processor cleanup completed")
        except Exception as e:
            logger.error(f"Error during speech processor cleanup: {e}")

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
                "soundfile": SOUNDFILE_AVAILABLE,
                "webrtcvad": (
                    self._vad_detector.is_available()
                    if self._vad_detector
                    else False
                ),
            },
        }
