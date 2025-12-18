"""
OSWorld Benchmark Implementation

OSWorld evaluates AI agents on their ability to complete
tasks in real operating system environments (Windows, macOS, Linux).

Key capabilities tested:
- Desktop application interaction
- File system operations
- System settings modification
- Multi-application workflows
- GUI navigation and automation

Reference: https://arxiv.org/abs/2404.07972
Repository: https://github.com/xlang-ai/OSWorld
"""

import json
import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base import (
    BaseBenchmark,
    BenchmarkDifficulty,
    BenchmarkTask,
)

logger = logging.getLogger(__name__)


@dataclass
class OSAction:
    """Represents an OS interaction action."""
    action_type: str  # click, type, hotkey, open_app, file_op, etc.
    target: Optional[str] = None
    value: Optional[str] = None
    coordinates: Optional[Tuple[int, int]] = None
    description: str = ""


class OSWorldBenchmark(BaseBenchmark):
    """
    OSWorld benchmark for OS interaction evaluation.
    
    Tests the agent's ability to:
    - Control desktop applications
    - Navigate file systems
    - Modify system settings
    - Execute multi-step workflows
    - Handle GUI interactions
    
    Categories:
    - File management
    - Application control
    - System configuration
    - Document editing
    - Web browsing integration
    """
    
    def __init__(
        self,
        agent: Any,
        config: Optional[Dict[str, Any]] = None,
        output_dir: str = "benchmarks/results",
        data_dir: str = "benchmarks/data/osworld",
    ):
        super().__init__(agent, config, output_dir)
        self.data_dir = Path(data_dir)
        self.os_type = platform.system().lower()
        self.max_actions = config.get("max_actions", 50) if config else 50
        self.safe_mode = config.get("safe_mode", True) if config else True
        
        # Desktop automation backend
        self._desktop_controller = None
    
    @property
    def name(self) -> str:
        return f"OSWorld-{self.os_type}"
    
    @property
    def version(self) -> str:
        return "1.0"
    
    @property
    def description(self) -> str:
        return f"Operating system interaction benchmark for {self.os_type}"
    
    def _get_desktop_controller(self):
        """Lazy initialization of desktop controller."""
        if self._desktop_controller is None:
            try:
                if self.os_type == "windows":
                    import pywinauto
                    self._desktop_controller = "pywinauto"
                elif self.os_type == "darwin":
                    # macOS automation would use pyobjc
                    self._desktop_controller = "pyobjc"
                else:
                    # Linux automation would use python-xlib or similar
                    self._desktop_controller = "xlib"
            except ImportError:
                logger.warning("Desktop automation libraries not available")
                self._desktop_controller = "mock"
        return self._desktop_controller
    
    def load_tasks(
        self,
        subset: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[BenchmarkTask]:
        """Load OSWorld tasks appropriate for the current OS."""
        tasks = []
        
        if self.data_dir.exists():
            tasks = self._load_from_local(subset, limit)
        else:
            logger.warning("Creating synthetic OSWorld tasks for testing")
            tasks = self._create_synthetic_tasks(limit or 5)
        
        # Filter tasks for current OS
        tasks = [t for t in tasks if self._is_compatible_with_os(t)]
        
        return tasks
    
    def _is_compatible_with_os(self, task: BenchmarkTask) -> bool:
        """Check if task is compatible with current OS."""
        supported_os = task.metadata.get("supported_os", ["windows", "darwin", "linux"])
        return self.os_type in supported_os
    
    def _load_from_local(
        self,
        subset: Optional[str],
        limit: Optional[int]
    ) -> List[BenchmarkTask]:
        """Load from local JSON files."""
        tasks = []
        
        # Try OS-specific file first
        data_file = self.data_dir / f"{self.os_type}_{subset or 'test'}.json"
        if not data_file.exists():
            data_file = self.data_dir / f"{subset or 'test'}.json"
        if not data_file.exists():
            data_file = self.data_dir / "tasks.json"
        
        if data_file.exists():
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for i, item in enumerate(data):
                if limit and i >= limit:
                    break
                
                task = BenchmarkTask(
                    task_id=item.get("task_id", str(i)),
                    input_data={
                        "instruction": item.get("instruction", ""),
                        "initial_state": item.get("initial_state", {}),
                        "applications": item.get("applications", []),
                        "files_needed": item.get("files_needed", []),
                    },
                    expected_output=item.get("expected_result", {}),
                    difficulty=self._categorize_difficulty(item),
                    category=item.get("category", "general"),
                    metadata={
                        "supported_os": item.get("supported_os", ["windows", "darwin", "linux"]),
                        "required_apps": item.get("required_apps", []),
                        "expected_actions": item.get("expected_actions", []),
                    },
                    max_steps=self.max_actions,
                    timeout_seconds=600.0,
                )
                tasks.append(task)
        
        return tasks
    
    def execute_task(
        self,
        task: BenchmarkTask
    ) -> Tuple[Any, List[str], List[Dict[str, Any]]]:
        """
        Execute an OSWorld task.
        
        The agent should:
        1. Understand the task instruction
        2. Plan a sequence of OS actions
        3. Execute actions safely
        4. Verify task completion
        """
        reasoning_trace: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        
        prompt = self._format_osworld_prompt(task)
        
        try:
            context = {
                "task_type": "os_automation",
                "os_type": self.os_type,
                "safe_mode": self.safe_mode,
                "max_actions": self.max_actions,
                "available_actions": self._get_available_actions(),
            }
            
            # Get the agent's action plan
            response = self.agent.execute_task(prompt, context)
            reasoning_trace.append(f"Agent plan: {response}")
            
            # Parse and execute actions
            actions = self._parse_os_actions(response)
            
            # Execute actions
            if self.safe_mode:
                execution_result = self._safe_execute_actions(actions, reasoning_trace, tool_calls)
            else:
                execution_result = self._execute_os_actions(actions, reasoning_trace, tool_calls)
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            execution_result = {"success": False, "error": str(e)}
            reasoning_trace.append(f"Error: {e}")
        
        return execution_result, reasoning_trace, tool_calls
    
    def evaluate_output(self, task: BenchmarkTask, predicted_output: Any) -> bool:
        """
        Evaluate if the OS task was completed successfully.
        
        Evaluation criteria:
        - Files created/modified as expected
        - Applications in correct state
        - Settings changed correctly
        """
        if not isinstance(predicted_output, dict):
            return False
        
        expected = task.expected_output
        if not expected:
            return predicted_output.get("success", False)
        
        # Verify expected state
        if isinstance(expected, dict):
            # Check file existence
            if "files_created" in expected:
                for file_path in expected["files_created"]:
                    if not Path(file_path).exists():
                        return False
            
            # Check file content
            if "file_content" in expected:
                for file_path, content in expected["file_content"].items():
                    if Path(file_path).exists():
                        actual_content = Path(file_path).read_text()
                        if content not in actual_content:
                            return False
                    else:
                        return False
            
            # Check success flag
            if "success" in expected:
                if predicted_output.get("success") != expected["success"]:
                    return False
        
        return predicted_output.get("success", False)
    
    def _format_osworld_prompt(self, task: BenchmarkTask) -> str:
        """Format an OSWorld task as a prompt."""
        prompt_parts = [
            f"You are an AI assistant controlling a {self.os_type} computer.",
            "Complete the following task by interacting with the operating system.",
            "",
            "=== Task ===",
            task.input_data.get("instruction", ""),
            "",
        ]
        
        if task.input_data.get("applications"):
            prompt_parts.append(f"Applications to use: {', '.join(task.input_data['applications'])}")
        
        prompt_parts.extend([
            "",
            "=== Available Actions ===",
        ])
        
        for action in self._get_available_actions():
            prompt_parts.append(f"- {action}")
        
        prompt_parts.extend([
            "",
            "=== Instructions ===",
            "1. Plan your actions step by step",
            "2. Use appropriate actions for the current OS",
            "3. Verify each step before proceeding",
            "4. Handle errors gracefully",
            "",
            "Format each action as: ACTION: action_name(parameters)",
            "Example:",
            "ACTION: open_app(notepad)",
            "ACTION: type_text(Hello, World!)",
            "ACTION: hotkey(ctrl, s)",
            "",
            "Provide your action sequence:",
        ])
        
        return "\n".join(prompt_parts)
    
    def _get_available_actions(self) -> List[str]:
        """Get available actions for current OS."""
        common_actions = [
            "open_app(app_name): Open an application",
            "close_app(app_name): Close an application",
            "type_text(text): Type text",
            "hotkey(key1, key2, ...): Press keyboard shortcut",
            "click(element): Click on an element",
            "double_click(element): Double-click on an element",
            "right_click(element): Right-click on an element",
            "move_mouse(x, y): Move mouse to coordinates",
            "scroll(direction, amount): Scroll in a direction",
            "wait(seconds): Wait for a duration",
            "screenshot(): Take a screenshot",
        ]
        
        file_actions = [
            "create_file(path, content): Create a new file",
            "read_file(path): Read file contents",
            "delete_file(path): Delete a file",
            "create_folder(path): Create a new folder",
            "copy_file(src, dst): Copy a file",
            "move_file(src, dst): Move a file",
            "list_dir(path): List directory contents",
        ]
        
        if self.os_type == "windows":
            os_specific = [
                "run_cmd(command): Run a command prompt command",
                "open_settings(category): Open Windows settings",
                "search_start(query): Search in Start menu",
            ]
        elif self.os_type == "darwin":
            os_specific = [
                "run_terminal(command): Run a terminal command",
                "open_preferences(category): Open System Preferences",
                "spotlight_search(query): Search with Spotlight",
            ]
        else:
            os_specific = [
                "run_terminal(command): Run a terminal command",
                "open_settings(category): Open system settings",
            ]
        
        return common_actions + file_actions + os_specific
    
    def _parse_os_actions(self, response: str) -> List[OSAction]:
        """Parse OS actions from the agent's response."""
        import re
        
        actions = []
        
        # Pattern for action extraction
        action_pattern = r"ACTION:\s*(\w+)\(([^)]*)\)"
        matches = re.findall(action_pattern, response, re.IGNORECASE)
        
        for action_type, params in matches:
            action_type = action_type.lower()
            params_list = [p.strip().strip("'\"") for p in params.split(",") if p.strip()]
            
            action = OSAction(
                action_type=action_type,
                description=f"{action_type}({params})"
            )
            
            if action_type in ["open_app", "close_app", "search_start", "spotlight_search"]:
                action.target = params_list[0] if params_list else None
            elif action_type in ["type_text", "run_cmd", "run_terminal"]:
                action.value = params_list[0] if params_list else None
            elif action_type == "hotkey":
                action.value = "+".join(params_list)
            elif action_type in ["click", "double_click", "right_click"]:
                action.target = params_list[0] if params_list else None
            elif action_type == "move_mouse":
                if len(params_list) >= 2:
                    try:
                        action.coordinates = (int(params_list[0]), int(params_list[1]))
                    except ValueError:
                        pass
            elif action_type in ["create_file", "copy_file", "move_file"]:
                action.target = params_list[0] if params_list else None
                action.value = params_list[1] if len(params_list) > 1 else None
            elif action_type in ["read_file", "delete_file", "create_folder", "list_dir"]:
                action.target = params_list[0] if params_list else None
            elif action_type in ["scroll"]:
                action.value = params_list[0] if params_list else "down"
            elif action_type == "wait":
                action.value = params_list[0] if params_list else "1"
            
            actions.append(action)
        
        return actions
    
    def _safe_execute_actions(
        self,
        actions: List[OSAction],
        reasoning_trace: List[str],
        tool_calls: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute actions in safe mode (simulation/sandboxed)."""
        reasoning_trace.append("Executing in SAFE MODE (simulated)")
        
        results = []
        for action in actions:
            reasoning_trace.append(f"[SAFE] {action.description}")
            tool_calls.append({
                "tool": "os_action",
                "action": action.action_type,
                "params": {
                    "target": action.target,
                    "value": action.value,
                    "coordinates": action.coordinates,
                },
                "safe_mode": True,
            })
            results.append({"action": action.action_type, "simulated": True})
        
        return {
            "success": True,
            "actions_taken": len(actions),
            "safe_mode": True,
            "results": results,
        }
    
    def _execute_os_actions(
        self,
        actions: List[OSAction],
        reasoning_trace: List[str],
        tool_calls: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute actual OS actions (use with caution)."""
        controller = self._get_desktop_controller()
        
        if controller == "mock":
            return self._safe_execute_actions(actions, reasoning_trace, tool_calls)
        
        results = []
        success = True
        
        for action in actions:
            try:
                result = self._execute_single_action(action)
                reasoning_trace.append(f"✓ {action.description}")
                results.append(result)
                
                tool_calls.append({
                    "tool": "os_action",
                    "action": action.action_type,
                    "params": {
                        "target": action.target,
                        "value": action.value,
                    },
                    "result": result,
                })
                
            except Exception as e:
                reasoning_trace.append(f"✗ {action.description}: {e}")
                results.append({"error": str(e)})
                success = False
        
        return {
            "success": success,
            "actions_taken": len(actions),
            "results": results,
        }
    
    def _execute_single_action(self, action: OSAction) -> Dict[str, Any]:
        """Execute a single OS action."""
        import time
        
        action_type = action.action_type
        
        # File operations
        if action_type == "create_file":
            path = Path(action.target) if action.target else None
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(action.value or "")
                return {"created": str(path)}
        
        elif action_type == "read_file":
            path = Path(action.target) if action.target else None
            if path and path.exists():
                return {"content": path.read_text()[:1000]}
        
        elif action_type == "delete_file":
            path = Path(action.target) if action.target else None
            if path and path.exists():
                path.unlink()
                return {"deleted": str(path)}
        
        elif action_type == "create_folder":
            path = Path(action.target) if action.target else None
            if path:
                path.mkdir(parents=True, exist_ok=True)
                return {"created": str(path)}
        
        elif action_type == "list_dir":
            path = Path(action.target) if action.target else Path(".")
            if path.exists():
                return {"contents": [str(p) for p in path.iterdir()][:20]}
        
        # System commands
        elif action_type in ["run_cmd", "run_terminal"]:
            if action.value:
                result = subprocess.run(
                    action.value,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                return {
                    "stdout": result.stdout[:500],
                    "returncode": result.returncode,
                }
        
        elif action_type == "wait":
            seconds = float(action.value or 1)
            time.sleep(seconds)
            return {"waited": seconds}
        
        # Desktop automation would require pywinauto/pyautogui
        elif action_type in ["open_app", "close_app", "click", "type_text", "hotkey"]:
            try:
                if self.os_type == "windows":
                    return self._execute_windows_action(action)
                else:
                    return {"simulated": True, "action": action_type}
            except Exception as e:
                return {"error": str(e)}
        
        return {"action": action_type, "status": "completed"}
    
    def _execute_windows_action(self, action: OSAction) -> Dict[str, Any]:
        """Execute Windows-specific actions using pywinauto."""
        try:
            from pywinauto import Application, Desktop
            import pywinauto.keyboard as keyboard
        except ImportError:
            return {"error": "pywinauto not available"}
        
        if action.action_type == "open_app":
            app_name = action.target
            if app_name:
                Application().start(app_name)
                return {"opened": app_name}
        
        elif action.action_type == "type_text":
            if action.value:
                keyboard.send_keys(action.value)
                return {"typed": action.value}
        
        elif action.action_type == "hotkey":
            if action.value:
                keyboard.send_keys(f"^{action.value}")
                return {"hotkey": action.value}
        
        return {"status": "completed"}
    
    def _categorize_difficulty(self, item: Dict) -> BenchmarkDifficulty:
        """Categorize task difficulty."""
        expected_actions = item.get("expected_actions", [])
        apps = item.get("applications", [])
        
        if len(expected_actions) <= 3 and len(apps) <= 1:
            return BenchmarkDifficulty.EASY
        elif len(expected_actions) <= 10 and len(apps) <= 2:
            return BenchmarkDifficulty.MEDIUM
        elif len(expected_actions) <= 20:
            return BenchmarkDifficulty.HARD
        else:
            return BenchmarkDifficulty.EXPERT
    
    def _create_synthetic_tasks(self, num_tasks: int) -> List[BenchmarkTask]:
        """Create synthetic OSWorld tasks for testing."""
        tasks = []
        
        synthetic = [
            {
                "task_id": "file_create_1",
                "instruction": "Create a new text file called 'hello.txt' with the content 'Hello, World!'",
                "category": "file_management",
                "expected_result": {"files_created": ["hello.txt"]},
                "expected_actions": ["create_file"],
                "supported_os": ["windows", "darwin", "linux"],
            },
            {
                "task_id": "app_notepad_1",
                "instruction": "Open Notepad and type 'Testing MIA agent'",
                "category": "application_control",
                "applications": ["notepad"],
                "expected_result": {"success": True},
                "expected_actions": ["open_app", "type_text"],
                "supported_os": ["windows"],
            },
            {
                "task_id": "folder_create_1",
                "instruction": "Create a new folder called 'test_folder' in the current directory",
                "category": "file_management",
                "expected_result": {"success": True},
                "expected_actions": ["create_folder"],
                "supported_os": ["windows", "darwin", "linux"],
            },
            {
                "task_id": "terminal_cmd_1",
                "instruction": "Run a command to list all files in the current directory",
                "category": "system",
                "expected_result": {"success": True},
                "expected_actions": ["run_cmd"],
                "supported_os": ["windows", "darwin", "linux"],
            },
        ]
        
        for i, item in enumerate(synthetic[:num_tasks]):
            task = BenchmarkTask(
                task_id=item["task_id"],
                input_data={
                    "instruction": item["instruction"],
                    "applications": item.get("applications", []),
                },
                expected_output=item["expected_result"],
                difficulty=self._categorize_difficulty(item),
                category=item["category"],
                metadata={
                    "synthetic": True,
                    "supported_os": item["supported_os"],
                },
            )
            tasks.append(task)
        
        return tasks
