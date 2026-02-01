from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Type
import inspect

@dataclass
class SkillManifest:
    """Metadata for a skill."""
    name: str
    version: str
    description: str
    author: str = "Unknown"
    dependencies: List[str] = field(default_factory=list)

@dataclass
class ToolDefinition:
    """Defines a tool exposed by a skill to the LLM."""
    name: str
    description: str
    func: Callable
    arguments_schema: Dict[str, Any]
    return_type: str = "str"
    is_dangerous: bool = False  # Requires confirmation

class BaseSkill(ABC):
    """
    Abstract base class for M.I.A Skills (Plugins).
    Encapsulates logic, tools, and event handlers.
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self.register_tools()

    @property
    @abstractmethod
    def manifest(self) -> SkillManifest:
        """Returns the skill metadata."""
        pass

    def register_tools(self):
        """
        Scans the class for methods decorated with @tool or manually registers them.
        Can be overridden for custom registration logic.
        """
        # Simple auto-discovery for methods starting with 'tool_'
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, "_is_tool"):
                tool_def = getattr(method, "_tool_def", None)
                if tool_def is not None:
                    self._tools[tool_def.name] = tool_def

    def get_tools(self) -> List[ToolDefinition]:
        """Returns list of availables tools for the LLM."""
        return list(self._tools.values())

    async def on_load(self):
        """Lifecycle hook: Called when skill is loaded."""
        pass

    async def on_unload(self):
        """Lifecycle hook: Called when skill is unloaded."""
        pass

    async def on_message(self, message: Any):
        """Optional: Intercept messages before LLM (middleware pattern)."""
        pass

# Decorator for tools
def tool(name: str, description: str, dangerous: bool = False):
    def decorator(func):
        func._is_tool = True
        # Naive schema generation - in production use Pydantic
        sig = inspect.signature(func)
        schema = {k: str(v.annotation) for k, v in sig.parameters.items()}
        
        func._tool_def = ToolDefinition(
            name=name,
            description=description,
            func=func,
            arguments_schema=schema,
            is_dangerous=dangerous
        )
        return func
    return decorator
