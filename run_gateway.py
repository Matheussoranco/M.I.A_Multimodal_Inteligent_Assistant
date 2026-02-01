#!/usr/bin/env python3
"""
M.I.A Gateway Server
Refactored architecture inspired by Moltbot.
Runs M.I.A as an always-on background service with pluggable skills and channels.
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import TYPE_CHECKING

# Add src to path
src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

if TYPE_CHECKING:
    from mia.gateway.manager import GatewayManager  # type: ignore[import-not-found]
    from mia.core.channels import BaseChannel, ChannelMessage, ChannelConfig  # type: ignore[import-not-found]
    from mia.skills.basic_tools import BasicToolsSkill  # type: ignore[import-not-found]
else:
    from mia.gateway.manager import GatewayManager  # type: ignore[import-not-found]
    from mia.core.channels import BaseChannel, ChannelMessage, ChannelConfig  # type: ignore[import-not-found]
    from mia.skills.basic_tools import BasicToolsSkill  # type: ignore[import-not-found]

# --- Simple CLI Channel for testing ---
class ConsoleChannel(BaseChannel):
    """
    A simple channel that reads from stdin and writes to stdout.
    Represents the 'Terminal' integration.
    """
    async def connect(self) -> bool:
        self.is_connected = True
        print("[ConsoleChannel] Connected. Type a message:")
        # In a real async CLI, we'd use aioconsole
        return True

    async def disconnect(self):
        print("[ConsoleChannel] Disconnected.")
        self.is_connected = False

    async def send_message(self, message: str, recipient_id: str, **kwargs) -> str:
        print(f"\n>> M.I.A: {message}\n")
        return "sent"

    async def broadcast(self, message: str, **kwargs):
        print(f"\n>> [BROADCAST] M.I.A: {message}\n")

    async def input_loop(self):
        """Simulates an event loop for CLI input (blocking for demo simplicity)"""
        while self.is_connected:
            try:
                # Use a separate thread or aioconsole in prod for non-blocking
                user_input = await asyncio.to_thread(input, "You: ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                msg = ChannelMessage(
                    id="1",
                    content=user_input,
                    sender_id="user_console",
                    channel_id="console"
                )
                await self._on_message_received(msg)
            except EOFError:
                break


async def main():
    # 1. Initialize Gateway
    gateway = GatewayManager()
    
    # 2. Add Channels (Pluggable I/O)
    console_channel = ConsoleChannel(ChannelConfig(name="CLI"))
    gateway.register_channel("console", console_channel)
    
    # 3. Add Skills (Pluggable Capabilities)
    gateway.register_skill(BasicToolsSkill())
    
    # 4. Start
    await gateway.start()
    
    # 5. Keep alive (simulating daemon mode)
    try:
        await console_channel.input_loop()
    except KeyboardInterrupt:
        pass
    finally:
        await gateway.stop()

if __name__ == "__main__":
    asyncio.run(main())
