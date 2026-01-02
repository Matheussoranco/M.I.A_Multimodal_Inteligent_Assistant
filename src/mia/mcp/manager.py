"""
MCP Manager for M.I.A.
Manages multiple MCP clients and aggregates tools.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from .client import MCPClient

logger = logging.getLogger(__name__)


class MCPManager:
    """Manages multiple MCP clients."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.clients: Dict[str, MCPClient] = {}
        self.tools_map: Dict[str, MCPClient] = {}  # Map tool name to client

    async def initialize(self):
        """Initialize all configured MCP clients."""
        servers = self.config.get("mcp_servers", {})
        
        tasks = []
        for name, server_config in servers.items():
            if not server_config.get("enabled", True):
                continue
                
            client = MCPClient(
                name=name,
                command=server_config.get("command"),
                args=server_config.get("args", []),
                env=server_config.get("env")
            )
            self.clients[name] = client
            tasks.append(client.connect())
            
        if tasks:
            await asyncio.gather(*tasks)
            self._refresh_tool_map()

    def _refresh_tool_map(self):
        """Refresh the mapping of tools to clients."""
        self.tools_map.clear()
        for client_name, client in self.clients.items():
            for tool in client.tools:
                # Handle potential naming conflicts? For now, last one wins or we could namespace
                # Handle both object (MCP) and dict (fallback) access
                tool_name = getattr(tool, "name", None)
                if tool_name is None and isinstance(tool, dict):
                    tool_name = tool.get("name")
                
                if tool_name:
                    self.tools_map[tool_name] = client

    async def shutdown(self):
        """Shutdown all clients."""
        tasks = [client.disconnect() for client in self.clients.values()]
        if tasks:
            await asyncio.gather(*tasks)

    async def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all tools from all clients."""
        all_tools = []
        for client in self.clients.values():
            tools = await client.list_tools()
            all_tools.extend(tools)
        return all_tools

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool on the appropriate client."""
        client = self.tools_map.get(tool_name)
        if not client:
            raise ValueError(f"Tool {tool_name} not found in any MCP server")
            
        return await client.call_tool(tool_name, arguments)

    def get_tool_client(self, tool_name: str) -> Optional[MCPClient]:
        """Get the client for a specific tool."""
        return self.tools_map.get(tool_name)
