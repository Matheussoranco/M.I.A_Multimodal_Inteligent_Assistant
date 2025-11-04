"""Voice activity detection helper built on top of webrtcvad."""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import webrtcvad  # type: ignore
    _HAS_WEBRTCVAD = True
except ImportError:  # pragma: no cover - optional dependency
    webrtcvad = None  # type: ignore
    _HAS_WEBRTCVAD = False


class VoiceActivityDetector:
    """Detect speech segments using WebRTC VAD if available."""

    SUPPORTED_SAMPLE_RATES = {8000, 16000, 32000, 48000}

    def __init__(
        self,
        aggressiveness: int = 2,
        frame_duration_ms: int = 30,
        min_active_duration_ms: int = 600,
        enabled: bool = True,
        logger_instance: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger_instance or logger
        self.enabled = enabled and _HAS_WEBRTCVAD
        self.frame_duration_ms = frame_duration_ms
        self.min_active_duration_ms = max(frame_duration_ms, min_active_duration_ms)
        self._vad = None
        self._min_voiced_frames = max(1, self.min_active_duration_ms // self.frame_duration_ms)

        if not enabled:
            self.logger.debug("VoiceActivityDetector disabled via configuration")
            return

        if not _HAS_WEBRTCVAD:
            self.logger.info("webrtcvad not installed; VAD disabled")
            return

        try:
            self._vad = webrtcvad.Vad(int(aggressiveness))  # type: ignore[call-arg]
        except Exception as exc:  # pragma: no cover - misconfiguration
            self.logger.warning("Failed to initialize WebRTC VAD: %s", exc)
            self._vad = None
            self.enabled = False

    def is_available(self) -> bool:
        return self.enabled and self._vad is not None

    def has_speech(self, audio_bytes: bytes, sample_rate: int) -> bool:
        if not self.is_available():
            return True  # fail open so audio continues to process

        if sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            self.logger.debug(
                "Sample rate %s unsupported by VAD; skipping detection", sample_rate
            )
            return True

        frame_length = int(sample_rate * (self.frame_duration_ms / 1000.0) * 2)
        if frame_length <= 0:
            return True

        if len(audio_bytes) < frame_length:
            return False

        voiced_frames = 0
        total_frames = 0

        for start in range(0, len(audio_bytes) - frame_length + 1, frame_length):
            frame = audio_bytes[start : start + frame_length]
            total_frames += 1
            try:
                if self._vad.is_speech(frame, sample_rate):  # type: ignore[call-arg]
                    voiced_frames += 1
                    if voiced_frames >= self._min_voiced_frames:
                        return True
            except Exception as exc:  # pragma: no cover - library level error
                self.logger.debug("VAD error on frame %s: %s", total_frames, exc)
                continue

        self.logger.debug(
            "VAD detected no speech (voiced=%s/%s, threshold=%s)",
            voiced_frames,
            total_frames,
            self._min_voiced_frames,
        )
        return False


__all__ = ["VoiceActivityDetector"]
