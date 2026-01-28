from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Awaitable
from datetime import datetime
import uuid

@dataclass
class ChannelMessage:
    """Standardized message format across all channels."""
    id: str
    content: str
    sender_id: str
    sender_name: Optional[str] = None
    channel_id: str = "default"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    attachments: List[str] = field(default_factory=list)  # Paths to files
    reply_to_id: Optional[str] = None

@dataclass
class ChannelConfig:
    """Configuration for a channel instance."""
    name: str
    enabled: bool = True
    credentials: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)

class BaseChannel(ABC):
    """
    Abstract base class for all communication channels.
    Inspired by Moltbot's ChannelAdapter architecture.
    """
    
    def __init__(self, config: ChannelConfig):
        self.config = config
        self.message_handler: Optional[Callable[[ChannelMessage], Awaitable[None]]] = None
        self.is_connected = False

    def set_handler(self, handler: Callable[[ChannelMessage], Awaitable[None]]):
        """Sets the callback for incoming messages."""
        self.message_handler = handler

    @abstractmethod
    async def connect(self) -> bool:
        """Establishes connection to the channel service."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Closes connection."""
        pass

    @abstractmethod
    async def send_message(self, message: str, recipient_id: str, **kwargs) -> str:
        """Sends a message to a specific recipient."""
        pass

    @abstractmethod
    async def broadcast(self, message: str, **kwargs):
        """Sends a message to all subscribed users/channels."""
        pass

    async def _on_message_received(self, message: ChannelMessage):
        """Internal hook when a hardware/API message is received."""
        if self.message_handler:
            await self.message_handler(message)

class TextChannel(BaseChannel):
    """Helper for text-based channels (CLI, Simple Web, etc)."""
    pass

class VoiceChannel(BaseChannel):
    """Helper for voice-based channels."""
    pass
