"""Long term memory store with simple metadata support."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


class LongTermMemory:
    """Maintain a bounded list of past facts or observations."""

    def __init__(self, max_entries: int = 10_000) -> None:
        self.max_entries = max_entries
        self.memory: List[Dict[str, Any]] = []

    def remember(self, fact: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        entry_id = metadata.get("id") if isinstance(metadata, dict) else None
        if not entry_id:
            entry_id = f"ltm_{len(self.memory) + 1}_{datetime.utcnow().timestamp():.0f}"
        entry = {
            "id": entry_id,
            "fact": fact,
            "metadata": dict(metadata or {}),
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.memory.append(entry)
        if len(self.memory) > self.max_entries:
            self.memory.pop(0)
        return entry

    def recall(self, query: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        results = self.memory
        if query:
            query_lower = query.lower()
            results = [
                entry
                for entry in self.memory
                if query_lower in str(entry.get("fact", "")).lower()
            ]
        if limit is not None:
            results = results[-limit:]
        return list(results)

    def clear(self) -> None:
        self.memory.clear()

    def get_status(self) -> Dict[str, Any]:
        return {
            "entries": len(self.memory),
            "max_entries": self.max_entries,
        }
