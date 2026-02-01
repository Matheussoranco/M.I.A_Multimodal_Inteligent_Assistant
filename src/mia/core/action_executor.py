"""
Compatibility Action Executor for tests and legacy imports.

This module provides a minimal, test-friendly ActionExecutor API that
wraps the main tools action executor while exposing the types used by
unit tests.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ..tools.action_executor import ActionExecutor as ToolsActionExecutor


class ActionType(Enum):
    FILESYSTEM = "filesystem"
    WEB = "web"
    CODE = "code"
    SYSTEM = "system"
    COMMUNICATION = "communication"


class ToolCategory(Enum):
    FILESYSTEM = "filesystem"
    WEB = "web"
    CODE = "code"
    SYSTEM = "system"
    COMMUNICATION = "communication"


@dataclass
class ToolDefinition:
    name: str
    description: str
    category: ToolCategory
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_permissions: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    is_async: bool = False


@dataclass
class ActionResult:
    success: bool
    output: Any
    action_type: ActionType
    execution_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ActionExecutor:
    """Minimal ActionExecutor API used by tests and legacy code."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, security_manager: Any = None) -> None:
        self.config = config or {}
        self.security_manager = security_manager
        self.tools: Dict[str, Callable[..., Any]] = {}
        self.tool_definitions: Dict[str, ToolDefinition] = {}
        self.action_history: List[Dict[str, Any]] = []
        self._tools_executor = ToolsActionExecutor()
        self._initialize_tools()

    def _initialize_tools(self) -> None:
        """Initialize default tool stubs (overridden by tests)."""
        # Keep lightweight by default. Tests patch this method.
        return None

    def list_available_tools(self) -> List[str]:
        return list(self.tools.keys())

    def get_tool_info(self, name: str) -> Optional[ToolDefinition]:
        return self.tool_definitions.get(name)

    def get_tools_by_category(self, category: ToolCategory) -> List[ToolDefinition]:
        return [tool for tool in self.tool_definitions.values() if tool.category == category]

    def _record_action(self, tool_name: str, params: Dict[str, Any], result: ActionResult) -> None:
        self.action_history.append(
            {
                "tool": tool_name,
                "params": params,
                "success": result.success,
                "error": result.error,
                "time": time.time(),
            }
        )

    def get_action_history(self) -> List[Dict[str, Any]]:
        return list(self.action_history)

    def clear_action_history(self) -> None:
        self.action_history.clear()

    def _check_permissions(self, tool_name: str, tool_def: Optional[ToolDefinition]) -> Optional[str]:
        if not self.security_manager:
            return None
        if tool_def and tool_def.required_permissions:
            for perm in tool_def.required_permissions:
                if not self.security_manager.check_permission(perm):
                    return f"Permission denied for tool '{tool_name}'"
        return None

    def execute(self, tool_name: str, params: Dict[str, Any], timeout: Optional[float] = None) -> ActionResult:
        start = time.time()
        tool = self.tools.get(tool_name)
        tool_def = self.tool_definitions.get(tool_name)
        if not tool:
            result = ActionResult(False, None, ActionType.SYSTEM, time.time() - start, error="Tool not found")
            self._record_action(tool_name, params, result)
            return result

        permission_error = self._check_permissions(tool_name, tool_def)
        if permission_error:
            result = ActionResult(False, None, ActionType.SYSTEM, time.time() - start, error=permission_error)
            self._record_action(tool_name, params, result)
            return result

        effective_timeout = timeout if timeout is not None else (tool_def.timeout if tool_def else None)

        try:
            if effective_timeout is not None:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(tool, **params)
                    output = future.result(timeout=effective_timeout)
            else:
                output = tool(**params)
            result = ActionResult(True, output, ActionType.SYSTEM, time.time() - start)
        except TimeoutError:
            result = ActionResult(False, None, ActionType.SYSTEM, time.time() - start, error="Timeout executing tool")
        except Exception as exc:
            result = ActionResult(False, None, ActionType.SYSTEM, time.time() - start, error=str(exc))

        self._record_action(tool_name, params, result)
        return result

    async def async_execute(self, tool_name: str, params: Dict[str, Any]) -> ActionResult:
        start = time.time()
        tool = self.tools.get(tool_name)
        tool_def = self.tool_definitions.get(tool_name)
        if not tool:
            result = ActionResult(False, None, ActionType.SYSTEM, time.time() - start, error="Tool not found")
            self._record_action(tool_name, params, result)
            return result

        permission_error = self._check_permissions(tool_name, tool_def)
        if permission_error:
            result = ActionResult(False, None, ActionType.SYSTEM, time.time() - start, error=permission_error)
            self._record_action(tool_name, params, result)
            return result

        try:
            if tool_def and tool_def.is_async:
                output = await tool(**params)
            else:
                output = await asyncio.to_thread(tool, **params)
            result = ActionResult(True, output, ActionType.SYSTEM, time.time() - start)
        except Exception as exc:
            result = ActionResult(False, None, ActionType.SYSTEM, time.time() - start, error=str(exc))

        self._record_action(tool_name, params, result)
        return result

    # Minimal tool setup methods used by tests
    def _setup_filesystem_tools(self) -> None:
        def read_file(path: str) -> str:
            with open(path, "r", encoding="utf-8") as handle:
                return handle.read()

        self.tools.setdefault("read_file", read_file)
        self.tool_definitions.setdefault(
            "read_file",
            ToolDefinition(
                name="read_file",
                description="Reads a file",
                category=ToolCategory.FILESYSTEM,
                parameters={"path": {"type": "string"}},
            ),
        )

    def _setup_web_tools(self) -> None:
        import requests

        def web_fetch(url: str) -> str:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text

        self.tools.setdefault("web_fetch", web_fetch)
        self.tool_definitions.setdefault(
            "web_fetch",
            ToolDefinition(
                name="web_fetch",
                description="Fetch a webpage",
                category=ToolCategory.WEB,
                parameters={"url": {"type": "string"}},
            ),
        )

    def _setup_code_tools(self) -> None:
        def python_eval(code: str) -> Any:
            return eval(code, {"__builtins__": {}})

        self.tools.setdefault("python_eval", python_eval)
        self.tool_definitions.setdefault(
            "python_eval",
            ToolDefinition(
                name="python_eval",
                description="Evaluate Python expression",
                category=ToolCategory.CODE,
                parameters={"code": {"type": "string"}},
            ),
        )
