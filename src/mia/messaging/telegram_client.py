"""Telegram messaging integration built on Telethon."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable, Coroutine, Optional, Any


class TelegramMessenger:
    """Lightweight wrapper that sends messages through Telethon."""

    def __init__(
        self,
        api_id: Optional[int] = None,
        api_hash: Optional[str] = None,
        bot_token: Optional[str] = None,
        phone_number: Optional[str] = None,
        session_dir: str = "sessions",
        session_name: str = "mia_telegram",
        default_peer: Optional[str] = None,
        parse_mode: Optional[str] = "markdown",
        request_timeout: int = 30,
        enabled: bool = True,
        logger_instance: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger_instance or logging.getLogger(__name__)
        self.enabled = enabled
        self.default_peer = (default_peer or "").strip() or None
        self.parse_mode = (parse_mode or "").strip() or None
        self.request_timeout = request_timeout

        self.api_id = self._to_int(api_id)
        self.api_hash = api_hash.strip() if isinstance(api_hash, str) else api_hash
        self.bot_token = bot_token.strip() if isinstance(bot_token, str) else bot_token
        self.phone_number = phone_number.strip() if isinstance(phone_number, str) else phone_number

        self.session_path = Path(session_dir or "sessions") / (session_name or "mia_telegram")
        self.session_path.parent.mkdir(parents=True, exist_ok=True)

        self._available = self._validate_credentials()
        if not self._available:
            self.logger.debug("TelegramMessenger disabled due to missing credentials")

    @staticmethod
    def _to_int(value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None

    def _validate_credentials(self) -> bool:
        if not self.enabled:
            return False
        if self.bot_token and self.api_id and self.api_hash:
            return True
        if self.phone_number and self.api_id and self.api_hash:
            return True
        return False

    def is_available(self) -> bool:
        return self._available

    def send_message(self, message: str, recipient: Optional[str] = None, silent: bool = False) -> str:
        if not self._available:
            return "Telegram messenger not configured. Provide API credentials and enable it."
        if not message:
            return "No message content provided."
        peer = (recipient or self.default_peer)
        if not peer:
            return "No Telegram recipient configured."
        try:
            return self._run_coroutine(lambda: self._send_async(peer, message, silent))
        except ImportError:
            return "Telethon not installed. Run: pip install telethon"
        except Exception as exc:  # pragma: no cover - runtime errors
            self.logger.error("Failed to send Telegram message: %s", exc)
            return f"Error sending Telegram message: {exc}"

    def _run_coroutine(self, coroutine_factory: Callable[[], Coroutine[Any, Any, str]]):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coroutine_factory())
            finally:
                new_loop.close()
        return asyncio.run(coroutine_factory())

    async def _send_async(self, peer: str, message: str, silent: bool) -> str:
        try:
            from telethon import TelegramClient
            from telethon.errors import RPCError
        except ImportError:
            raise

        client = TelegramClient(
            str(self.session_path),
            self.api_id,
            self.api_hash,
            request_retries=1,
            timeout=self.request_timeout,
        )
        try:
            if self.bot_token:
                await client.start(bot_token=self.bot_token)
            elif self.phone_number:
                await client.start(phone=self.phone_number)
            else:  # pragma: no cover - guarded by _validate_credentials
                return "Telegram credentials missing."

            await client.send_message(peer, message, parse_mode=self.parse_mode, silent=silent)
            self.logger.info("Telegram message sent to %s", peer)
            return f"Telegram message sent to {peer}"
        except RPCError as exc:
            self.logger.error("Telegram RPC error: %s", exc)
            return f"Telegram RPC error: {exc}"
        finally:
            await client.disconnect()


__all__ = ["TelegramMessenger"]
