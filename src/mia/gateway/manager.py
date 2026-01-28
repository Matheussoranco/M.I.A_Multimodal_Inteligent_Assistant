import asyncio
import logging
from typing import Dict, List, Optional, Any
from mia.core.channels import BaseChannel, ChannelMessage, ChannelConfig
from mia.core.skills import BaseSkill

logger = logging.getLogger(__name__)

class GatewayManager:
    """
    Central hub inspired by Moltbot's Gateway.
    Manages multiple channels (I/O) and routes messages to the Agent Core.
    """

    def __init__(self, agent_engine: Any = None):
        self.channels: Dict[str, BaseChannel] = {}
        self.skills: Dict[str, BaseSkill] = {}
        self.agent_engine = agent_engine  # The "Brain" (Orchestrator)
        self.is_running = False

    def register_channel(self, channel_id: str, channel: BaseChannel):
        """Registers a new communication channel."""
        self.channels[channel_id] = channel
        channel.set_handler(self.handle_incoming_message)
        logger.info(f"Channel registered: {channel_id}")

    def register_skill(self, skill: BaseSkill):
        """Registers a new capability/skill."""
        manifest = skill.manifest
        self.skills[manifest.name] = skill
        logger.info(f"Skill registered: {manifest.name} v{manifest.version}")

    async def start(self):
        """Starts all channels and loops."""
        self.is_running = True
        logger.info("Starting M.I.A Gateway...")
        
        # Start all channels in parallel
        tasks = [channel.connect() for channel in self.channels.values()]
        await asyncio.gather(*tasks)
        
        # Lifecycle hooks for skills
        for skill in self.skills.values():
            await skill.on_load()
            
        logger.info("Gateway fully operational.")

    async def stop(self):
        """Stops all channels."""
        self.is_running = False
        for skill in self.skills.values():
            await skill.on_unload()
            
        tasks = [channel.disconnect() for channel in self.channels.values()]
        await asyncio.gather(*tasks)

    async def handle_incoming_message(self, message: ChannelMessage):
        """
        Main Event Loop:
        1. Receive Message
        2. Check for Skill Middleware interceptions
        3. Send to Agent/LLM
        4. Route response back to Channel
        """
        logger.info(f"Gateway received message from {message.sender_id}: {message.content}")

        # 1. Skill Middleware (Allow skills to react first)
        for skill in self.skills.values():
            await skill.on_message(message)

        # 2. Process with Agent (Architecture Bridge)
        if self.agent_engine:
            # TODO: Adapt ChannelMessage to whatever the Agent expects
            # For now, we simulate a response
            response_text = await self._simulate_agent_processing(message)
        else:
            response_text = "M.I.A Core not connected. Gateway is running in headless mode."

        # 3. Send Response
        target_channel = self.channels.get(message.channel_id)
        if target_channel:
            await target_channel.send_message(response_text, message.sender_id)
        else:
            logger.error(f"Channel {message.channel_id} not found for response.")

    async def _simulate_agent_processing(self, message: ChannelMessage) -> str:
        """Temporary mock for agent processing."""
        # In a real implementation, this calls orchestrator.process(message)
        return f"Echo: {message.content} (Processed via Gateway)"
