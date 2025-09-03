import unittest
from mia.memory.knowledge_graph import AgentMemory  # type: ignore

class TestAgentMemory(unittest.TestCase):
    def test_init(self):
        memory = AgentMemory(persist_directory="memory/")
        self.assertIsNotNone(memory.kg)
        self.assertEqual(memory.persist_directory, "memory/")

if __name__ == "__main__":
    unittest.main()
