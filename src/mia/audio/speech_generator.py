import base64
import logging
import os
import queue
import threading
import time
from typing import Any, Dict, Optional

import requests

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
    import pyttsx3

    HAS_PYTTSX3 = True
except ImportError:
    pyttsx3 = None
    HAS_PYTTSX3 = False

try:
    import torch

    TORCH_AVAILABLE = True
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    device = "cpu"

try:
    from ..config_manager import ConfigManager
except ImportError:  # pragma: no cover - optional during isolated tests
    ConfigManager = None

try:
    from mia.llm.llm_inference import LLMInference

    LLM_AVAILABLE = True
except ImportError:
    LLMInference = None
    LLM_AVAILABLE = False

logger = logging.getLogger(__name__)


class SpeechGenerator:
    def __init__(
        self,
        device: Optional[str] = None,
        model_id: str = "microsoft/speecht5_tts",
        speaker: int = 7306,
        llama_model_path: Optional[str] = None,
        config_manager: Optional[Any] = None,
        audio_config: Optional[Any] = None,
        default_tts_provider: Optional[str] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the SpeechGenerator with TTS and LLM capabilities.

        :param device: Device for computation (e.g., 'cpu' or 'cuda:0').
        :param model_id: Model identifier for text-to-speech.
        :param speaker: Speaker ID for voice synthesis.
        :param llama_model_path: Path to local LLama model for text generation.
        :param config_manager: Optional configuration manager instance (shared across components).
        :param audio_config: Optional audio configuration object (overrides derived config).
        :param default_tts_provider: Optional override for default TTS provider.
        :param llm_kwargs: Additional keyword arguments forwarded to :class:`LLMInference`.
        """
        self.device = device or globals().get("device", "cpu")
        self.model_id = model_id
        self.speaker = speaker
        self.llama_model_path = llama_model_path
        self._llm_kwargs = llm_kwargs or {}

        self.config_manager = config_manager or (
            ConfigManager() if ConfigManager else None
        )
        self.audio_config = audio_config
        if (
            self.audio_config is None
            and self.config_manager
            and getattr(self.config_manager, "config", None)
        ):
            self.audio_config = getattr(
                self.config_manager.config, "audio", None
            )
        self.tts_provider = default_tts_provider
        if (
            not self.tts_provider
            and self.audio_config
            and hasattr(self.audio_config, "tts_provider")
        ):
            self.tts_provider = self.audio_config.tts_provider
        self.tts_provider = self.tts_provider or "local"
        self.tts_providers: Dict[str, Dict[str, Any]] = {}

        # Initialize components if available
        self.synthesiser = None
        self.embeddings_dataset = None
        self.speaker_embedding = None
        self.llm_inference = None

        self._init_tts()
        self._init_embeddings()
        self._init_llm()
        self._init_api_providers()

        # Initialize TTS playback queue
        self.tts_queue = queue.Queue()
        self.tts_thread = None
        self.tts_thread_running = False
        self._start_tts_queue_processor()

    def _init_tts(self):
        """Initialize TTS pipeline."""
        if HAS_PYTTSX3 and pyttsx3:
            try:
                self.synthesiser = pyttsx3.init()
                # Configure voice settings
                voices = self.synthesiser.getProperty("voices")
                if voices and hasattr(voices, "__iter__"):
                    # Try to set a female voice if available
                    for voice in voices:  # type: ignore
                        if hasattr(voice, "name") and hasattr(voice, "id"):
                            if (
                                "female" in voice.name.lower()
                                or "zira" in voice.name.lower()
                            ):
                                self.synthesiser.setProperty("voice", voice.id)
                                break
                # Set speech rate
                self.synthesiser.setProperty("rate", 180)
                # Set volume
                self.synthesiser.setProperty("volume", 0.8)
                logger.info("Local TTS engine initialized with pyttsx3")
            except Exception as e:
                logger.error(f"Failed to initialize pyttsx3 TTS: {e}")
                self.synthesiser = None
        else:
            logger.warning("pyttsx3 not available - local TTS disabled")
            self.synthesiser = None

    def _init_embeddings(self):
        """Initialize speaker embeddings."""
        if DATASETS_AVAILABLE and load_dataset:
            try:
                self.embeddings_dataset = list(
                    load_dataset(
                        "Matthijs/cmu-arctic-xvectors", split="validation"
                    )
                )
                logger.info("Speaker embeddings dataset loaded")

                # Initialize speaker embedding if torch is available
                if TORCH_AVAILABLE and torch and self.embeddings_dataset:
                    self.speaker_embedding = torch.tensor(
                        self.embeddings_dataset[self.speaker]["xvector"]
                    ).unsqueeze(0)

            except Exception as e:
                logger.error(f"Failed to load embeddings dataset: {e}")
                self.embeddings_dataset = None
        else:
            logger.warning(
                "Datasets not available - speaker embeddings disabled"
            )

    def _init_llm(self):
        """Initialize LLM inference."""
        if not (LLM_AVAILABLE and LLMInference):
            logger.warning("LLM inference not available")
            return

        provider_override = None
        model_override = None
        api_key_override = None
        url_override = None

        if self.audio_config:
            provider_override = (
                getattr(self.audio_config, "llm_provider", None) or None
            )
            model_override = (
                getattr(self.audio_config, "llm_model_id", None) or None
            )
            api_key_override = (
                getattr(self.audio_config, "llm_api_key", None) or None
            )
            url_override = getattr(self.audio_config, "llm_url", None) or None

        llm_params: Dict[str, Any] = dict(self._llm_kwargs)

        if provider_override:
            llm_params.setdefault("provider", provider_override)
        if model_override:
            llm_params.setdefault("model_id", model_override)
        if api_key_override:
            llm_params.setdefault("api_key", api_key_override)
        if url_override:
            llm_params.setdefault("url", url_override)

        # Preserve backwards compatibility with local model usage
        if self.llama_model_path:
            llm_params.setdefault("llama_model_path", self.llama_model_path)

        llm_params.setdefault("config_manager", self.config_manager)

        try:
            self.llm_inference = LLMInference(**llm_params)
            logger.info("LLM inference initialized")
        except Exception as exc:
            logger.error(f"Failed to initialize LLM inference: {exc}")
            self.llm_inference = None

    def _init_api_providers(self) -> None:
        """Configure metadata for external TTS providers."""
        self.tts_providers = {}

        def _resolve_from_config(
            provider_name: str, attribute: str, fallback: Optional[str]
        ) -> Optional[str]:
            if (
                self.audio_config
                and getattr(self.audio_config, "tts_provider", None)
                == provider_name
            ):
                value = getattr(self.audio_config, attribute, None)
                if value:
                    return value
            return fallback

        # NanoChat
        nanochat_url = _resolve_from_config(
            "nanochat",
            "tts_url",
            os.getenv("NANOCHAT_TTS_URL")
            or os.getenv("NANOCHAT_URL")
            or "http://localhost:8081/api/tts",
        )
        self.tts_providers["nanochat"] = {
            "url": nanochat_url,
            "api_key": _resolve_from_config(
                "nanochat",
                "tts_api_key",
                os.getenv("NANOCHAT_TTS_API_KEY")
                or os.getenv("NANOCHAT_API_KEY"),
            ),
            "model_id": _resolve_from_config(
                "nanochat",
                "tts_model_id",
                os.getenv("NANOCHAT_TTS_MODEL") or "nanochat-voice",
            ),
            "format": "mp3",
        }

        # Minimax AI
        minimax_url = _resolve_from_config(
            "minimax",
            "tts_url",
            os.getenv("MINIMAX_TTS_URL")
            or "https://api.minimax.chat/v1/text-to-speech",
        )
        self.tts_providers["minimax"] = {
            "url": minimax_url,
            "api_key": _resolve_from_config(
                "minimax",
                "tts_api_key",
                os.getenv("MINIMAX_TTS_API_KEY")
                or os.getenv("MINIMAX_API_KEY"),
            ),
            "model_id": _resolve_from_config(
                "minimax",
                "tts_model_id",
                os.getenv("MINIMAX_TTS_MODEL") or "minimax-tts",
            ),
            "voice": os.getenv("MINIMAX_TTS_VOICE"),
            "format": os.getenv("MINIMAX_TTS_FORMAT") or "mp3",
        }

        # OpenAI TTS endpoint (optional)
        openai_url = _resolve_from_config(
            "openai",
            "tts_url",
            os.getenv("OPENAI_TTS_URL")
            or "https://api.openai.com/v1/audio/speech",
        )
        self.tts_providers["openai"] = {
            "url": openai_url,
            "api_key": _resolve_from_config(
                "openai", "tts_api_key", os.getenv("OPENAI_API_KEY")
            ),
            "model_id": _resolve_from_config(
                "openai",
                "tts_model_id",
                os.getenv("OPENAI_TTS_MODEL") or "gpt-4o-mini-tts",
            ),
            "voice": os.getenv("OPENAI_TTS_VOICE") or "alloy",
            "format": _resolve_from_config(
                "openai", "tts_format", os.getenv("OPENAI_TTS_FORMAT") or "mp3"
            ),
        }

        # Custom provider supplied via configuration
        if (
            self.audio_config
            and getattr(self.audio_config, "tts_provider", None) == "custom"
        ):
            custom_url = getattr(self.audio_config, "tts_url", None)
            if custom_url:
                self.tts_providers["custom"] = {
                    "url": custom_url,
                    "api_key": getattr(self.audio_config, "tts_api_key", None),
                    "model_id": getattr(
                        self.audio_config, "tts_model_id", None
                    ),
                    "format": getattr(self.audio_config, "tts_format", None),
                }

    def _start_tts_queue_processor(self):
        """Start the TTS queue processing thread."""
        if self.tts_thread is not None:
            return

        self.tts_thread_running = True
        self.tts_thread = threading.Thread(
            target=self._process_tts_queue, daemon=True
        )
        self.tts_thread.start()
        logger.info("TTS queue processor started")

    def _process_tts_queue(self):
        """Process TTS requests from the queue."""
        while self.tts_thread_running:
            try:
                # Get TTS request from queue with timeout
                tts_request = self.tts_queue.get(timeout=1.0)
                if tts_request is None:  # Shutdown signal
                    break

                text, provider, kwargs = tts_request
                self._execute_tts(text, provider, **kwargs)

            except queue.Empty:
                continue
            except Exception as exc:
                logger.error(f"Error processing TTS queue: {exc}")

    def _execute_tts(
        self, text: str, provider: Optional[str] = None, **kwargs
    ):
        """Execute TTS for a single request."""
        try:
            if provider and provider != "local":
                # Use API provider
                payload = self.generate_speech_via_api(
                    text, provider=provider, **kwargs
                )
                if payload:
                    # For API providers, we assume they handle playback
                    logger.info(
                        f"TTS API request completed for provider: {provider}"
                    )
                    return

            # Use local TTS
            if HAS_PYTTSX3 and self.synthesiser:
                self.synthesiser.say(text)
                self.synthesiser.runAndWait()
                logger.info("Local TTS playback completed")
            else:
                logger.warning("No TTS provider available")

        except Exception as exc:
            logger.error(f"TTS execution failed: {exc}")

    def enqueue_speech(
        self, text: str, provider: Optional[str] = None, **kwargs
    ):
        """Enqueue text for speech synthesis and playback."""
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS enqueue")
            return

        self.tts_queue.put((text, provider or self.tts_provider, kwargs))
        logger.debug(f"TTS request enqueued: {len(text)} characters")

    def _get_tts_provider_config(
        self, provider: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if not provider or provider == "local":
            return None

        config = dict(self.tts_providers.get(provider, {}))
        if not config:
            logger.debug(
                "No configuration found for TTS provider %s", provider
            )
            return None

        # Environment overrides take precedence
        prefix = provider.upper()
        config["url"] = os.getenv(f"{prefix}_TTS_URL", config.get("url"))
        config["api_key"] = os.getenv(
            f"{prefix}_TTS_API_KEY", config.get("api_key")
        )
        config["model_id"] = os.getenv(
            f"{prefix}_TTS_MODEL", config.get("model_id")
        )

        # Final fallback to configuration defaults
        if (
            (not config.get("url"))
            and self.audio_config
            and getattr(self.audio_config, "tts_provider", None) == provider
        ):
            config["url"] = getattr(self.audio_config, "tts_url", None)
            config["api_key"] = getattr(self.audio_config, "tts_api_key", None)
            config["model_id"] = getattr(
                self.audio_config, "tts_model_id", None
            )

        return config if config.get("url") else None

    def set_speaker(self, speaker_id):
        """Set the speaker for TTS voice."""
        if not TORCH_AVAILABLE or not torch or not self.embeddings_dataset:
            logger.warning("Cannot set speaker - dependencies not available")
            return False

        try:
            self.speaker = speaker_id
            self.speaker_embedding = torch.tensor(
                self.embeddings_dataset[speaker_id]["xvector"]
            ).unsqueeze(0)
            logger.info(f"Speaker set to {speaker_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to set speaker {speaker_id}: {e}")
            return False

    def set_female_speaker(self):
        """Set a female speaker for TTS voice."""
        if not self.embeddings_dataset:
            logger.warning(
                "Cannot set female speaker - embeddings not available"
            )
            return False

        female_speaker_id = 7306  # Default female speaker
        return self.set_speaker(female_speaker_id)

    def generate_speech(
        self, text: str, provider: Optional[str] = None, **kwargs: Any
    ) -> Optional[Any]:
        """Generate speech from text using local or API-based providers."""
        if not text or not text.strip():
            logger.warning("Empty text provided for speech generation")
            return None

        provider = provider or self.tts_provider

        if provider and provider != "local":
            api_payload = self.generate_speech_via_api(
                text, provider=provider, **kwargs
            )
            if api_payload is not None:
                return api_payload
            logger.debug(
                "Falling back to local TTS after API provider '%s' returned no result",
                provider,
            )

        if not self.synthesiser:
            logger.error("TTS synthesizer not available")
            return None

        try:
            # Use pyttsx3 for local TTS
            if HAS_PYTTSX3 and self.synthesiser:
                self.synthesiser.say(text)
                self.synthesiser.runAndWait()
                return {"provider": "local", "text": text, "status": "played"}
            else:
                logger.error("Local TTS not available")
                return None
        except Exception as exc:
            logger.error(f"Failed to generate speech: {exc}")
            return None

    def generate_speech_via_api(
        self, text: str, provider: Optional[str] = None, **kwargs: Any
    ) -> Optional[Dict[str, Any]]:
        """Call an external TTS API provider."""
        provider = provider or self.tts_provider
        if not provider or provider == "local":
            return None

        config = self._get_tts_provider_config(provider)
        if not config:
            logger.warning(
                "No configuration available for TTS provider %s", provider
            )
            return None

        payload = self._build_tts_payload(provider, text, config, kwargs)
        if payload is None:
            logger.warning(
                "Failed to build payload for TTS provider %s", provider
            )
            return None

        headers = {"Content-Type": "application/json"}
        if config.get("api_key"):
            headers["Authorization"] = f"Bearer {config['api_key']}"

        timeout = kwargs.get("timeout", 60)

        try:
            response = requests.post(
                config["url"], headers=headers, json=payload, timeout=timeout
            )
            response.raise_for_status()
            return self._parse_tts_response(provider, response, config, kwargs)
        except requests.exceptions.Timeout:
            logger.error("%s TTS request timed out", provider)
        except requests.exceptions.RequestException as exc:
            logger.error("%s TTS request failed: %s", provider, exc)
        except Exception as exc:
            logger.error(
                "Failed to process %s TTS response: %s", provider, exc
            )

        return None

    def _build_tts_payload(
        self,
        provider: str,
        text: str,
        config: Dict[str, Any],
        options: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        model_id = options.get("model_id") or config.get("model_id")
        fmt = options.get("format") or config.get("format") or "mp3"
        voice = options.get("voice") or config.get("voice")

        if provider == "nanochat":
            payload = {
                "model": model_id or "nanochat-voice",
                "text": text,
                "format": fmt,
                "stream": options.get("stream", False),
            }
            if voice:
                payload["voice"] = voice
            if options.get("language"):
                payload["language"] = options["language"]
            return payload

        if provider == "minimax":
            payload = {
                "model": model_id or "minimax-tts",
                "text": text,
                "format": fmt,
            }
            if voice:
                payload["voice_id"] = voice
            if options.get("style") or config.get("style"):
                payload["style"] = options.get("style") or config.get("style")
            return payload

        if provider == "openai":
            payload = {
                "model": model_id or "gpt-4o-mini-tts",
                "input": text,
                "voice": voice or "alloy",
                "format": fmt,
            }
            return payload

        if provider == "custom":
            base_payload = dict(config.get("payload_template", {}))
            base_payload.setdefault("text", text)
            if model_id:
                base_payload.setdefault("model", model_id)
            if voice:
                base_payload.setdefault("voice", voice)
            base_payload.setdefault("format", fmt)
            extra_payload = options.get("extra_payload") or {}
            base_payload.update(extra_payload)
            return base_payload

        logger.warning(
            "No payload builder available for provider %s", provider
        )
        return None

    def _parse_tts_response(
        self,
        provider: str,
        response: requests.Response,
        config: Dict[str, Any],
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        content_type = (response.headers.get("Content-Type") or "").lower()

        if "application/json" in content_type:
            data = response.json()
            for key in ("audio", "audio_base64", "audio_data", "data"):
                audio_blob = data.get(key)
                if isinstance(audio_blob, dict):
                    audio_blob = audio_blob.get("audio") or audio_blob.get(
                        "content"
                    )
                if isinstance(audio_blob, str):
                    audio_bytes = self._decode_audio_blob(audio_blob)
                    if audio_bytes:
                        return {
                            "provider": provider,
                            "audio_bytes": audio_bytes,
                            "mime_type": self._infer_mime_type(
                                config, data, options
                            ),
                            "raw": data,
                        }

            audio_url = data.get("audio_url") or data.get("url")
            if audio_url:
                return {
                    "provider": provider,
                    "audio_url": audio_url,
                    "raw": data,
                }

            text_response = data.get("text") or data.get("response")
            if text_response:
                return {
                    "provider": provider,
                    "text": text_response,
                    "raw": data,
                }

            return {"provider": provider, "raw": data}

        # Fallback to treating the response body as binary audio
        return {
            "provider": provider,
            "audio_bytes": response.content,
            "mime_type": response.headers.get(
                "Content-Type", self._infer_mime_type(config, {}, options)
            ),
        }

    @staticmethod
    def _decode_audio_blob(blob: str) -> Optional[bytes]:
        try:
            return base64.b64decode(blob)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _infer_mime_type(
        config: Dict[str, Any], data: Dict[str, Any], options: Dict[str, Any]
    ) -> str:
        fmt = (
            options.get("format")
            or data.get("format")
            or config.get("format")
            or "mp3"
        )
        mapping = {
            "mp3": "audio/mpeg",
            "mpeg": "audio/mpeg",
            "wav": "audio/wav",
            "ogg": "audio/ogg",
            "opus": "audio/opus",
            "flac": "audio/flac",
        }
        return mapping.get(str(fmt).lower(), "audio/mpeg")

    def set_tts_provider(self, provider: str) -> bool:
        if provider == "local":
            self.tts_provider = provider
            return True
        if provider not in self.tts_providers:
            logger.warning("Unknown TTS provider: %s", provider)
            return False
        self.tts_provider = provider
        return True

    def generate_response_and_speech(
        self,
        prompt: str,
        *,
        tts_provider: Optional[str] = None,
        llm_options: Optional[Dict[str, Any]] = None,
        tts_options: Optional[Dict[str, Any]] = None,
    ) -> tuple[Optional[str], Optional[Any]]:
        """Generate text response using an LLM and convert it to speech."""
        if not self.llm_inference:
            logger.error("LLM inference not available")
            return None, None

        llm_kwargs = llm_options or {}
        tts_kwargs = tts_options or {}

        try:
            response = self.llm_inference.generate_response(
                prompt, **llm_kwargs
            )
            if not response:
                return None, None

            speech = self.generate_speech(
                response, provider=tts_provider, **tts_kwargs
            )
            return response, speech
        except Exception as exc:
            logger.error(f"Failed to generate response and speech: {exc}")
            return None, None

    def is_available(self):
        """Check if TTS functionality is available."""
        has_api_provider = (
            self._get_tts_provider_config(self.tts_provider) is not None
        )
        return self.synthesiser is not None or has_api_provider

    def get_status(self):
        """Get status of all components."""
        return {
            "tts_available": self.synthesiser is not None,
            "embeddings_available": self.embeddings_dataset is not None,
            "llm_available": self.llm_inference is not None,
            "torch_available": TORCH_AVAILABLE,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "datasets_available": DATASETS_AVAILABLE,
            "sounddevice_available": SOUNDDEVICE_AVAILABLE,
            "tts_provider": self.tts_provider,
            "available_tts_providers": sorted(list(self.tts_providers.keys())),
        }
