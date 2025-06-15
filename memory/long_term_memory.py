"""
Long Term Memory: Stores and retrieves long-term facts and events.
"""

class LongTermMemory:
    def __init__(self):
        self.memory = []

    def remember(self, fact):
        self.memory.append(fact)

    def recall(self, query=None):
        # Simple recall logic
        if query:
            return [m for m in self.memory if query in m]
        return self.memory
