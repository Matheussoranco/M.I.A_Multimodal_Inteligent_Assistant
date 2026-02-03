"""M.I.A multimodal processing module"""

from .processor import MultimodalProcessor
from .vision_processor import VisionProcessor
from .vision_resource_manager import *

# OCR is optional (can pull heavy native deps like opencv).
try:
    from .ocr_processor import OCRProcessor, OCRResult, OCRConfig
except Exception:  # pragma: no cover - optional dependency chain
    OCRProcessor = None  # type: ignore[assignment]
    OCRResult = None  # type: ignore[assignment]
    OCRConfig = None  # type: ignore[assignment]

__all__ = [
    "MultimodalProcessor",
    "VisionProcessor",
    "OCRProcessor",
    "OCRResult",
    "OCRConfig",
]
