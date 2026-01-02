"""
MCP Client implementation for M.I.A.
Handles connections to MCP servers and tool execution.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    
    class _DummyClientSession:
        def __init__(self, read, write): pass
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc_val, exc_tb): pass
        async def initialize(self): pass
        async def list_tools(self): 
            class Result:
                tools = []
            return Result()
        async def call_tool(self, name, args): pass

    class _DummyStdioServerParameters:
        def __init__(self, command, args, env): pass

    def _dummy_stdio_client(server):
        class Context:
            async def __aenter__(self): return (None, None)
            async def __aexit__(self, *args): pass
        return Context()

    ClientSession = _DummyClientSession # type: ignore
    StdioServerParameters = _DummyStdioServerParameters # type: ignore
    stdio_client = _dummy_stdio_client # type: ignore

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for connecting to an MCP server."""

    def __init__(self, name: str, command: str, args: List[str], env: Optional[Dict[str, str]] = None):
        self.name = name
        self.command = command
        self.args = args
        self.env = env or os.environ.copy()
        self.session: Any = None
        self.tools: List[Any] = []
        self._exit_stack = None

    async def connect(self):
        """Connect to the MCP server."""
        if not HAS_MCP:
            logger.warning("MCP package not installed. Skipping connection.")
            return

        logger.info(f"Connecting to MCP server: {self.name}")
        
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env
        )

        try:
            from contextlib import AsyncExitStack
            self._exit_stack = AsyncExitStack()
            
            # Connect via stdio
            read, write = await self._exit_stack.enter_async_context(stdio_client(server_params)) # type: ignore
            self.session = await self._exit_stack.enter_async_context(ClientSession(read, write)) # type: ignore
            
            # Initialize
            if self.session:
                await self.session.initialize()
            
            # List tools
            if self.session:
                result = await self.session.list_tools()
                self.tools = result.tools
            
            logger.info(f"Connected to {self.name}. Found {len(self.tools)} tools.")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.name}: {e}")
            if self._exit_stack:
                await self._exit_stack.aclose()
            self.session = None

    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self.session = None
            logger.info(f"Disconnected from {self.name}")

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools."""
        if not self.session:
            return []
        # Convert MCP tool objects to dicts if necessary, or return as is
        # The MCP SDK returns objects, we might need to serialize them for the LLM
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
            for tool in self.tools
        ]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the server."""
        if not self.session:
            raise RuntimeError(f"Not connected to server {self.name}")

        logger.info(f"Calling tool {tool_name} on {self.name}")
        result = await self.session.call_tool(tool_name, arguments)
        return result
