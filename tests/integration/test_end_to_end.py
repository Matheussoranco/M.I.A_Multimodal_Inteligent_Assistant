import unittest
from src.mia.llm.llm_manager import LLMManager
from src.mia.memory.knowledge_graph import AgentMemory
from src.mia.multimodal.processor import MultimodalProcessor
from unittest.mock import patch, MagicMock

class TestEndToEndFlows(unittest.TestCase):
    def test_llm_memory_multimodal_flow(self):
        # Mock LLMManager to return a canned response
        with patch('src.mia.llm.llm_manager.LLMManager') as MockLLM:
            mock_llm = MockLLM()
            mock_llm.generate.return_value = "This is a test response."
            memory = AgentMemory(persist_directory="memory/")
            processor = MultimodalProcessor()
            # Simulate storing and retrieving a conversation
            memory.kg.add_node('conv1', text='Hello')
            self.assertIn('conv1', memory.kg.nodes)
            # Simulate LLM response
            response = mock_llm.generate('Say hello')
            self.assertEqual(response, "This is a test response.")
            # Simulate multimodal processing (mock image/audio)
            processor._get_dominant_color = MagicMock(return_value='red')
            processor._extract_text = MagicMock(return_value='text')
            with patch('src.mia.multimodal.processor.Image.open') as mock_open:
                mock_img = MagicMock()
                mock_img.size = (100, 100)
                mock_open.return_value = mock_img
                result = processor.process_image('fake_path.jpg')
                self.assertEqual(result['dominant_color'], 'red')
                self.assertEqual(result['text_ocr'], 'text')

if __name__ == "__main__":
    unittest.main()
