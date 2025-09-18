"""
Long Term Memory: Stores and retrieves long-term facts and events.
"""
from typing import List, Optional, Any

class LongTermMemory:
    def __init__(self) -> None:
        self.memory: List[Any] = []

    def remember(self, fact: Any) -> None:
        self.memory.append(fact)

    def recall(self, query: Optional[str] = None) -> List[Any]:
        # Simple recall logic
        if query:
            return [m for m in self.memory if query in m]
        return self.memory
