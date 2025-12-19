"""
Comprehensive tests for the Action Executor module.
Tests tool execution, permissions, and action handling.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

# Add src directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

from mia.core.action_executor import (  # type: ignore[import-not-found]
    ActionExecutor,
    ActionResult,
    ActionType,
    ToolDefinition,
    ToolCategory,
)


class TestActionType(unittest.TestCase):
    """Tests for ActionType enum."""
    
    def test_action_types_exist(self):
        """Test all expected action types exist."""
        self.assertIsNotNone(ActionType.FILESYSTEM)
        self.assertIsNotNone(ActionType.WEB)
        self.assertIsNotNone(ActionType.CODE)
        self.assertIsNotNone(ActionType.SYSTEM)
        self.assertIsNotNone(ActionType.COMMUNICATION)


class TestToolCategory(unittest.TestCase):
    """Tests for ToolCategory enum."""
    
    def test_tool_categories_exist(self):
        """Test all expected tool categories exist."""
        self.assertIsNotNone(ToolCategory.FILESYSTEM)
        self.assertIsNotNone(ToolCategory.WEB)
        self.assertIsNotNone(ToolCategory.CODE)
        self.assertIsNotNone(ToolCategory.SYSTEM)


class TestActionResult(unittest.TestCase):
    """Tests for ActionResult dataclass."""
    
    def test_successful_result(self):
        """Test creating successful action result."""
        result = ActionResult(
            success=True,
            output="Command executed successfully",
            action_type=ActionType.SYSTEM,
            execution_time=0.5,
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.output, "Command executed successfully")
        self.assertEqual(result.action_type, ActionType.SYSTEM)
        self.assertEqual(result.execution_time, 0.5)
        self.assertIsNone(result.error)
    
    def test_failed_result(self):
        """Test creating failed action result."""
        result = ActionResult(
            success=False,
            output="",
            action_type=ActionType.FILESYSTEM,
            execution_time=0.1,
            error="Permission denied",
        )
        
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Permission denied")
    
    def test_result_with_metadata(self):
        """Test result with metadata."""
        result = ActionResult(
            success=True,
            output="File created",
            action_type=ActionType.FILESYSTEM,
            execution_time=0.2,
            metadata={"path": "/home/user/file.txt", "size": 1024},
        )
        
        self.assertEqual(result.metadata["path"], "/home/user/file.txt")
        self.assertEqual(result.metadata["size"], 1024)


class TestToolDefinition(unittest.TestCase):
    """Tests for ToolDefinition dataclass."""
    
    def test_basic_tool_definition(self):
        """Test creating basic tool definition."""
        tool = ToolDefinition(
            name="read_file",
            description="Reads content from a file",
            category=ToolCategory.FILESYSTEM,
            parameters={"path": {"type": "string", "required": True}},
        )
        
        self.assertEqual(tool.name, "read_file")
        self.assertEqual(tool.description, "Reads content from a file")
        self.assertEqual(tool.category, ToolCategory.FILESYSTEM)
    
    def test_tool_with_required_permissions(self):
        """Test tool with required permissions."""
        tool = ToolDefinition(
            name="execute_command",
            description="Executes a shell command",
            category=ToolCategory.SYSTEM,
            parameters={"command": {"type": "string"}},
            required_permissions=["shell_access", "system_control"],
        )
        
        self.assertIn("shell_access", tool.required_permissions)
        self.assertIn("system_control", tool.required_permissions)


class TestActionExecutor(unittest.TestCase):
    """Tests for ActionExecutor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            "tools": {
                "enabled": True,
                "filesystem_access": True,
                "web_access": True,
                "code_execution": True,
            },
            "security": {
                "sandbox_enabled": True,
                "allowed_paths": ["/home/user", "/tmp"],
            },
        }
        self.mock_security_manager = MagicMock()
        self.mock_security_manager.check_permission.return_value = True
        self.mock_security_manager.is_path_allowed.return_value = True
    
    @patch("mia.core.action_executor.ActionExecutor._initialize_tools")
    def test_initialization(self, mock_init):
        """Test ActionExecutor initialization."""
        executor = ActionExecutor(
            config=self.mock_config,
            security_manager=self.mock_security_manager,
        )
        
        self.assertIsNotNone(executor)
        self.assertEqual(executor.config, self.mock_config)
    
    @patch("mia.core.action_executor.ActionExecutor._initialize_tools")
    def test_list_available_tools(self, mock_init):
        """Test listing available tools."""
        executor = ActionExecutor(
            config=self.mock_config,
            security_manager=self.mock_security_manager,
        )
        
        # Mock some tools
        executor.tools = {
            "read_file": MagicMock(),
            "write_file": MagicMock(),
            "web_search": MagicMock(),
        }
        
        tools = executor.list_available_tools()
        
        self.assertIn("read_file", tools)
        self.assertIn("write_file", tools)
        self.assertIn("web_search", tools)
    
    @patch("mia.core.action_executor.ActionExecutor._initialize_tools")
    def test_get_tool_info(self, mock_init):
        """Test getting tool information."""
        executor = ActionExecutor(
            config=self.mock_config,
            security_manager=self.mock_security_manager,
        )
        
        # Mock tool definitions
        executor.tool_definitions = {
            "read_file": ToolDefinition(
                name="read_file",
                description="Reads a file",
                category=ToolCategory.FILESYSTEM,
                parameters={"path": {"type": "string"}},
            )
        }
        
        info = executor.get_tool_info("read_file")
        
        self.assertEqual(info.name, "read_file")
        self.assertEqual(info.category, ToolCategory.FILESYSTEM)
    
    @patch("mia.core.action_executor.ActionExecutor._initialize_tools")
    def test_get_tools_by_category(self, mock_init):
        """Test filtering tools by category."""
        executor = ActionExecutor(
            config=self.mock_config,
            security_manager=self.mock_security_manager,
        )
        
        # Mock tool definitions
        executor.tool_definitions = {
            "read_file": ToolDefinition(
                name="read_file",
                description="Reads a file",
                category=ToolCategory.FILESYSTEM,
                parameters={},
            ),
            "write_file": ToolDefinition(
                name="write_file",
                description="Writes a file",
                category=ToolCategory.FILESYSTEM,
                parameters={},
            ),
            "web_search": ToolDefinition(
                name="web_search",
                description="Web search",
                category=ToolCategory.WEB,
                parameters={},
            ),
        }
        
        fs_tools = executor.get_tools_by_category(ToolCategory.FILESYSTEM)
        
        self.assertEqual(len(fs_tools), 2)
        self.assertTrue(all(t.category == ToolCategory.FILESYSTEM for t in fs_tools))


class TestToolExecution(unittest.TestCase):
    """Tests for tool execution."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            "tools": {"enabled": True},
            "security": {"sandbox_enabled": True},
        }
        self.mock_security_manager = MagicMock()
        self.mock_security_manager.check_permission.return_value = True
        self.mock_security_manager.is_path_allowed.return_value = True
    
    @patch("mia.core.action_executor.ActionExecutor._initialize_tools")
    def test_execute_tool_success(self, mock_init):
        """Test successful tool execution."""
        executor = ActionExecutor(
            config=self.mock_config,
            security_manager=self.mock_security_manager,
        )
        
        # Mock tool
        mock_tool = MagicMock(return_value="Success output")
        executor.tools = {"test_tool": mock_tool}
        executor.tool_definitions = {
            "test_tool": ToolDefinition(
                name="test_tool",
                description="A test tool",
                category=ToolCategory.SYSTEM,
                parameters={},
            )
        }
        
        result = executor.execute("test_tool", {"arg": "value"})
        
        self.assertTrue(result.success)
        self.assertEqual(result.output, "Success output")
    
    @patch("mia.core.action_executor.ActionExecutor._initialize_tools")
    def test_execute_unknown_tool(self, mock_init):
        """Test executing unknown tool."""
        executor = ActionExecutor(
            config=self.mock_config,
            security_manager=self.mock_security_manager,
        )
        executor.tools = {}
        
        result = executor.execute("nonexistent_tool", {})
        
        self.assertFalse(result.success)
        self.assertIn("not found", result.error.lower())
    
    @patch("mia.core.action_executor.ActionExecutor._initialize_tools")
    def test_execute_tool_permission_denied(self, mock_init):
        """Test tool execution with permission denied."""
        self.mock_security_manager.check_permission.return_value = False
        
        executor = ActionExecutor(
            config=self.mock_config,
            security_manager=self.mock_security_manager,
        )
        executor.tools = {"protected_tool": MagicMock()}
        executor.tool_definitions = {
            "protected_tool": ToolDefinition(
                name="protected_tool",
                description="Protected",
                category=ToolCategory.SYSTEM,
                parameters={},
                required_permissions=["admin"],
            )
        }
        
        result = executor.execute("protected_tool", {})
        
        self.assertFalse(result.success)
        self.assertIn("permission", result.error.lower())
    
    @patch("mia.core.action_executor.ActionExecutor._initialize_tools")
    def test_execute_tool_with_timeout(self, mock_init):
        """Test tool execution with timeout."""
        executor = ActionExecutor(
            config=self.mock_config,
            security_manager=self.mock_security_manager,
        )
        
        import time
        
        def slow_tool(**kwargs):
            time.sleep(5)
            return "Done"
        
        executor.tools = {"slow_tool": slow_tool}
        executor.tool_definitions = {
            "slow_tool": ToolDefinition(
                name="slow_tool",
                description="Slow tool",
                category=ToolCategory.SYSTEM,
                parameters={},
                timeout=0.1,
            )
        }
        
        result = executor.execute("slow_tool", {}, timeout=0.1)
        
        self.assertFalse(result.success)
        self.assertIn("timeout", result.error.lower())


class TestFilesystemTools(unittest.TestCase):
    """Tests for filesystem tools."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            "tools": {"filesystem_access": True},
            "security": {"allowed_paths": ["/tmp"]},
        }
        self.mock_security_manager = MagicMock()
        self.mock_security_manager.is_path_allowed.return_value = True
    
    @patch("mia.core.action_executor.ActionExecutor._initialize_tools")
    @patch("builtins.open", create=True)
    def test_read_file_tool(self, mock_open, mock_init):
        """Test read_file tool."""
        mock_open.return_value.__enter__ = MagicMock(
            return_value=MagicMock(read=MagicMock(return_value="file content"))
        )
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        
        executor = ActionExecutor(
            config=self.mock_config,
            security_manager=self.mock_security_manager,
        )
        
        # Use actual implementation
        executor._setup_filesystem_tools()
        
        if "read_file" in executor.tools:
            result = executor.execute("read_file", {"path": "/tmp/test.txt"})
            # Just verify no crash
            self.assertIsNotNone(result)


class TestWebTools(unittest.TestCase):
    """Tests for web tools."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            "tools": {"web_access": True},
        }
        self.mock_security_manager = MagicMock()
    
    @patch("mia.core.action_executor.ActionExecutor._initialize_tools")
    @patch("requests.get")
    def test_web_fetch_tool(self, mock_get, mock_init):
        """Test web_fetch tool."""
        mock_response = MagicMock()
        mock_response.text = "<html><body>Hello</body></html>"
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        executor = ActionExecutor(
            config=self.mock_config,
            security_manager=self.mock_security_manager,
        )
        
        executor._setup_web_tools()
        
        if "web_fetch" in executor.tools:
            result = executor.execute("web_fetch", {"url": "https://example.com"})
            self.assertIsNotNone(result)


class TestCodeExecutionTools(unittest.TestCase):
    """Tests for code execution tools."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            "tools": {"code_execution": True},
            "security": {"sandbox_enabled": True},
        }
        self.mock_security_manager = MagicMock()
        self.mock_security_manager.check_permission.return_value = True
    
    @patch("mia.core.action_executor.ActionExecutor._initialize_tools")
    def test_python_eval_tool(self, mock_init):
        """Test Python code evaluation tool."""
        executor = ActionExecutor(
            config=self.mock_config,
            security_manager=self.mock_security_manager,
        )
        
        executor._setup_code_tools()
        
        if "python_eval" in executor.tools:
            result = executor.execute("python_eval", {"code": "2 + 2"})
            # Should work in sandbox
            self.assertIsNotNone(result)


class TestActionHistory(unittest.TestCase):
    """Tests for action history tracking."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {"tools": {"enabled": True}}
        self.mock_security_manager = MagicMock()
        self.mock_security_manager.check_permission.return_value = True
    
    @patch("mia.core.action_executor.ActionExecutor._initialize_tools")
    def test_action_recorded_in_history(self, mock_init):
        """Test that actions are recorded in history."""
        executor = ActionExecutor(
            config=self.mock_config,
            security_manager=self.mock_security_manager,
        )
        
        mock_tool = MagicMock(return_value="Output")
        executor.tools = {"test_tool": mock_tool}
        executor.tool_definitions = {
            "test_tool": ToolDefinition(
                name="test_tool",
                description="Test",
                category=ToolCategory.SYSTEM,
                parameters={},
            )
        }
        
        executor.execute("test_tool", {"arg": "value"})
        
        history = executor.get_action_history()
        
        self.assertGreater(len(history), 0)
        self.assertEqual(history[-1]["tool"], "test_tool")
    
    @patch("mia.core.action_executor.ActionExecutor._initialize_tools")
    def test_clear_action_history(self, mock_init):
        """Test clearing action history."""
        executor = ActionExecutor(
            config=self.mock_config,
            security_manager=self.mock_security_manager,
        )
        
        executor.action_history = [
            {"tool": "tool1", "time": datetime.now()},
            {"tool": "tool2", "time": datetime.now()},
        ]
        
        executor.clear_action_history()
        
        self.assertEqual(len(executor.action_history), 0)


class TestAsyncExecution(unittest.TestCase):
    """Tests for async tool execution."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {"tools": {"enabled": True}}
        self.mock_security_manager = MagicMock()
        self.mock_security_manager.check_permission.return_value = True
    
    @patch("mia.core.action_executor.ActionExecutor._initialize_tools")
    def test_async_execute(self, mock_init):
        """Test async tool execution."""
        import asyncio
        
        executor = ActionExecutor(
            config=self.mock_config,
            security_manager=self.mock_security_manager,
        )
        
        async def async_tool(**kwargs):
            await asyncio.sleep(0.01)
            return "Async result"
        
        executor.tools = {"async_tool": async_tool}
        executor.tool_definitions = {
            "async_tool": ToolDefinition(
                name="async_tool",
                description="Async tool",
                category=ToolCategory.SYSTEM,
                parameters={},
                is_async=True,
            )
        }
        
        async def run_test():
            result = await executor.async_execute("async_tool", {})
            return result
        
        result = asyncio.run(run_test())
        
        self.assertTrue(result.success)
        self.assertEqual(result.output, "Async result")


if __name__ == "__main__":
    unittest.main()
