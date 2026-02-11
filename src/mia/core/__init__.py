# Core module init
#
# Lazy-import heavy modules (cognitive_architecture pulls in transformers/CLIP).
# Only re-export the lightweight agent components by default.

from .agent import ChatResponse, ToolCall, ToolCallingAgent  # noqa: F401
from .tool_registry import CORE_TOOLS  # noqa: F401
