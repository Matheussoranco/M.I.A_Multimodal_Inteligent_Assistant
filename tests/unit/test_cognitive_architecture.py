"""
Comprehensive tests for the Cognitive Architecture module.
Tests the ReAct loop, memory management, and embedding generation.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np

# Add src directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)


class TestMIACognitiveCore(unittest.TestCase):
    """Tests for MIACognitiveCore class."""
    
    @patch('mia.core.cognitive_architecture.SpeechProcessor')
    @patch('mia.core.cognitive_architecture.ActionExecutor')
    @patch('mia.core.cognitive_architecture.torch')
    def setUp(self, mock_torch, mock_action_executor, mock_speech_processor):
        """Set up test fixtures."""
        # Mock torch.cuda.is_available
        mock_torch.cuda.is_available.return_value = False
        
        # Mock dependencies
        mock_action_executor.return_value = MagicMock()
        mock_speech_processor.return_value = MagicMock()
        
        # Create mock LLM client
        self.mock_llm = MagicMock()
        self.mock_llm.query.return_value = "Test response"
        self.mock_llm.query_model.return_value = "Test response"
        
        # Import after mocking
        from mia.core.cognitive_architecture import MIACognitiveCore  # type: ignore[import-not-found]
        
        with patch('mia.core.cognitive_architecture.HAS_CLIP', False):
            with patch('mia.core.cognitive_architecture.HAS_EMBEDDING_MANAGER', False):
                self.core = MIACognitiveCore(self.mock_llm, device="cpu")
    
    def test_initialization(self):
        """Test cognitive core initialization."""
        self.assertIsNotNone(self.core)
        self.assertEqual(self.core.device, "cpu")
        self.assertIsNotNone(self.core.action_executor)
        self.assertIsInstance(self.core.working_memory, list)
        self.assertIsInstance(self.core.long_term_memory, dict)
        self.assertIsInstance(self.core.knowledge_graph, dict)
    
    def test_generate_fallback_embedding(self):
        """Test fallback embedding generation."""
        text = "This is a test sentence for embedding generation."
        embedding = self.core._generate_fallback_embedding(text)
        
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 384)  # Expected dimension
        
        # Check values are normalized
        for val in embedding:
            self.assertGreaterEqual(val, -1.0)
            self.assertLessEqual(val, 1.0)
    
    def test_generate_embeddings_empty_text(self):
        """Test embedding generation with empty text."""
        embedding = self.core.generate_embeddings("")
        self.assertEqual(embedding, [])
        
        embedding = self.core.generate_embeddings(None)
        self.assertEqual(embedding, [])
        
        embedding = self.core.generate_embeddings("   ")
        self.assertEqual(embedding, [])
    
    def test_working_memory_update(self):
        """Test working memory updates."""
        self.core._update_working_memory("item1")
        self.core._update_working_memory("item2")
        
        self.assertEqual(len(self.core.working_memory), 2)
        self.assertIn("item1", self.core.working_memory)
        self.assertIn("item2", self.core.working_memory)
    
    def test_working_memory_limit(self):
        """Test working memory size limit."""
        # Fill memory beyond limit
        for i in range(150):
            self.core._update_working_memory(f"item_{i}")
        
        # Should be limited to 100 items
        self.assertEqual(len(self.core.working_memory), 100)
    
    def test_knowledge_graph_update(self):
        """Test knowledge graph updates."""
        self.core._update_knowledge_graph("entity1", "entity2", "related_to")
        
        self.assertIn("entity1", self.core.knowledge_graph)
        self.assertEqual(
            self.core.knowledge_graph["entity1"]["entity2"],
            "related_to"
        )
    
    def test_memory_stats(self):
        """Test memory statistics retrieval."""
        self.core._update_working_memory("test")
        self.core.long_term_memory["key"] = "value"
        self.core._update_knowledge_graph("a", "b", "c")
        
        stats = self.core.get_memory_stats()
        
        self.assertIn("working_memory_size", stats)
        self.assertIn("long_term_memory_size", stats)
        self.assertIn("knowledge_graph_size", stats)
        self.assertEqual(stats["working_memory_size"], 1)
        self.assertEqual(stats["long_term_memory_size"], 1)
        self.assertEqual(stats["knowledge_graph_size"], 1)
    
    def test_reset_memory(self):
        """Test memory reset functionality."""
        self.core._update_working_memory("test")
        self.core.long_term_memory["key"] = "value"
        self.core._update_knowledge_graph("a", "b", "c")
        
        self.core.reset_memory()
        
        self.assertEqual(len(self.core.working_memory), 0)
        self.assertEqual(len(self.core.long_term_memory), 0)
        self.assertEqual(len(self.core.knowledge_graph), 0)
    
    @unittest.skip("_parse_action moved to core.agent.ToolCallingAgent")
    def test_parse_action_valid(self):
        """Test parsing valid action from LLM response."""
        response = '''
        Thought: I need to search the web for this.
        Action: web_search
        Action Input: {"query": "test query"}
        '''
        
        result = self.core._parse_action(response)
        
        self.assertIsNotNone(result)
        tool_name, tool_params = result
        self.assertEqual(tool_name, "web_search")
        self.assertEqual(tool_params["query"], "test query")
    
    @unittest.skip("_parse_action moved to core.agent.ToolCallingAgent")
    def test_parse_action_invalid(self):
        """Test parsing invalid action response."""
        response = "Final Answer: This is the final answer."
        
        result = self.core._parse_action(response)
        self.assertIsNone(result)
    
    @unittest.skip("_parse_action moved to core.agent.ToolCallingAgent")
    def test_parse_action_invalid_json(self):
        """Test parsing action with invalid JSON."""
        response = '''
        Action: test_action
        Action Input: {invalid json}
        '''
        
        result = self.core._parse_action(response)
        self.assertIsNone(result)


class TestSimilarityFunctions(unittest.TestCase):
    """Tests for similarity computation functions."""
    
    @patch('mia.core.cognitive_architecture.SpeechProcessor')
    @patch('mia.core.cognitive_architecture.ActionExecutor')
    @patch('mia.core.cognitive_architecture.torch')
    def setUp(self, mock_torch, mock_action_executor, mock_speech_processor):
        mock_torch.cuda.is_available.return_value = False
        mock_action_executor.return_value = MagicMock()
        mock_speech_processor.return_value = MagicMock()
        
        self.mock_llm = MagicMock()
        
        from mia.core.cognitive_architecture import MIACognitiveCore  # type: ignore[import-not-found]
        
        with patch('mia.core.cognitive_architecture.HAS_CLIP', False):
            with patch('mia.core.cognitive_architecture.HAS_EMBEDDING_MANAGER', False):
                self.core = MIACognitiveCore(self.mock_llm, device="cpu")
    
    def test_compute_similarity_identical(self):
        """Test similarity of identical texts."""
        text = "This is a test sentence."
        similarity = self.core.compute_similarity(text, text)
        
        # Should be very close to 1.0 for identical texts
        self.assertGreater(similarity, 0.99)
    
    def test_compute_similarity_different(self):
        """Test similarity of very different texts."""
        text1 = "The quick brown fox jumps."
        text2 = "Quantum physics explains atoms."
        
        similarity = self.core.compute_similarity(text1, text2)
        
        # Should be lower than identical (but hash-based may vary)
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, -1.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_find_similar_memories_empty(self):
        """Test finding similar memories with empty candidates."""
        results = self.core.find_similar_memories("query", [])
        self.assertEqual(results, [])
    
    def test_find_similar_memories(self):
        """Test finding similar memories."""
        candidates = [
            {"text": "The weather is sunny today."},
            {"text": "Machine learning is fascinating."},
            {"text": "The sun is shining brightly."},
        ]
        
        results = self.core.find_similar_memories(
            "sunny weather",
            candidates,
            top_k=2,
            threshold=0.0,
        )
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)
        
        # Each result should have similarity score
        for result in results:
            self.assertIn("similarity", result)


class TestMultimodalProcessing(unittest.TestCase):
    """Tests for multimodal input processing."""
    
    @patch('mia.core.cognitive_architecture.SpeechProcessor')
    @patch('mia.core.cognitive_architecture.ActionExecutor')
    @patch('mia.core.cognitive_architecture.torch')
    def setUp(self, mock_torch, mock_action_executor, mock_speech_processor):
        mock_torch.cuda.is_available.return_value = False
        mock_action_executor.return_value = MagicMock()
        mock_speech_processor.return_value = MagicMock()
        
        self.mock_llm = MagicMock()
        self.mock_llm.query.return_value = "Processed response"
        
        from mia.core.cognitive_architecture import MIACognitiveCore  # type: ignore[import-not-found]
        
        with patch('mia.core.cognitive_architecture.HAS_CLIP', False):
            with patch('mia.core.cognitive_architecture.HAS_EMBEDDING_MANAGER', False):
                self.core = MIACognitiveCore(self.mock_llm, device="cpu")
    
    def test_process_empty_input(self):
        """Test processing empty input."""
        result = self.core.process_multimodal_input({})
        
        self.assertIn("text", result)
        self.assertEqual(result["text"], "No input provided")
    
    def test_process_text_only_input(self):
        """Test processing text-only input."""
        result = self.core.process_multimodal_input({"text": "Hello, world!"})
        
        self.assertIn("text", result)


if __name__ == "__main__":
    unittest.main()
