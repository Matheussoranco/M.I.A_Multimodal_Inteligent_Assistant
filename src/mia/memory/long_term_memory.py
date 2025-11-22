"""Long term memory store with simple metadata support."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


class LongTermMemory:
    """Maintain a bounded list of past facts or observations with JSON persistence."""

    def __init__(self, max_entries: int = 10_000, storage_path: str = "memory/long_term_memory.json") -> None:
        self.max_entries = max_entries
        self.storage_path = storage_path
        self.memory: List[Dict[str, Any]] = []
        self._load_memory()

    def _load_memory(self) -> None:
        """Load memory from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.memory = data
            except Exception as e:
                print(f"Error loading long term memory: {e}")

    def _save_memory(self) -> None:
        """Save memory to disk."""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving long term memory: {e}")

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
        
        self._save_memory()
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
