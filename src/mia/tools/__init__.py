"""M.I.A tools module"""

from .action_executor import ActionExecutor
from .document_generator import *

# Optional: document intelligence pulls in OCR/vision stacks (opencv/transformers)
# which may not be available in minimal installs.
try:
    from .document_intelligence import DocumentIntelligence, DocumentAnalysisResult
except Exception:  # pragma: no cover - optional dependency chain
    DocumentIntelligence = None  # type: ignore[assignment]
    DocumentAnalysisResult = None  # type: ignore[assignment]

__all__ = [
    "ActionExecutor",
    "DocumentIntelligence",
    "DocumentAnalysisResult",
]
