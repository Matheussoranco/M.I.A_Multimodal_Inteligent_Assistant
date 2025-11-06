"""Lightweight hotword detection helpers for the audio pipeline."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from rapidfuzz import fuzz  # type: ignore
    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - optional dependency
    fuzz = None  # type: ignore
    _HAS_RAPIDFUZZ = False

try:
    from difflib import SequenceMatcher
except ImportError:  # pragma: no cover - python standard library should always exist
    SequenceMatcher = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class HotwordDetection:
    """Result produced when a hotword is detected."""

    confidence: float
    transcript: str
    energy: float


class HotwordDetector:
    """Simple hotword detector leveraging fuzzy text matching and signal energy."""

    def __init__(self, hotword: str = "mia", sensitivity: float = 0.5, energy_floor: float = 0.01) -> None:
        self.hotword = (hotword or "mia").strip().lower()
        self.sensitivity = max(0.1, min(sensitivity, 0.95))
        self.energy_floor = max(1e-4, energy_floor)

        if not self.hotword:
            raise ValueError("Hotword cannot be empty")

        if not _HAS_RAPIDFUZZ and SequenceMatcher is None:
            logger.warning("Fuzzy match libraries unavailable; hotword detection will rely on substring matches.")

    def _score_text(self, transcript: str) -> float:
        cleaned = (transcript or "").strip().lower()
        if not cleaned:
            return 0.0

        if _HAS_RAPIDFUZZ and fuzz is not None:  # pragma: no cover - depends on extra package
            score = fuzz.partial_ratio(self.hotword, cleaned) / 100.0
        elif SequenceMatcher is not None:
            score = max(
                SequenceMatcher(None, self.hotword, token).ratio()
                for token in cleaned.split()
            ) if cleaned.split() else SequenceMatcher(None, self.hotword, cleaned).ratio()
        else:
            score = 1.0 if self.hotword in cleaned else 0.0
        return float(score)

    def _signal_energy(self, audio: Optional[np.ndarray]) -> float:
        if audio is None or audio.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))

    def detect(self, transcript: str, audio: Optional[np.ndarray] = None) -> Optional[HotwordDetection]:
        """Return detection details if the hotword is present."""
        score = self._score_text(transcript)
        energy = self._signal_energy(audio)

        logger.debug("Hotword check score=%.3f energy=%.4f", score, energy)

        if score >= self.sensitivity and (energy is None or energy >= self.energy_floor):
            return HotwordDetection(confidence=score, transcript=transcript, energy=energy)
        return None


__all__ = ["HotwordDetector", "HotwordDetection"]
