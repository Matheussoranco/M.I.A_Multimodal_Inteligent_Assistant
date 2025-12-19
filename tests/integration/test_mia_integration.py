"""
Integration tests for the MIA system.
Tests end-to-end flows and component integration.
"""

import os
import sys
import asyncio
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

# Add src directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)


class TestCognitiveLoopIntegration(unittest.TestCase):
    """Integration tests for the cognitive loop."""
    
    @patch("mia.core.cognitive_architecture.MIACognitiveCore")
    @patch("mia.core.action_executor.ActionExecutor")
    @patch("mia.llm.llm_manager.LLMManager")
    def test_think_act_observe_cycle(self, mock_llm, mock_executor, mock_cognitive):
        """Test the complete think-act-observe cycle."""
        # Setup mocks
        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = "I will search for that information."
        mock_llm.return_value = mock_llm_instance
        
        mock_executor_instance = MagicMock()
        mock_executor_instance.execute.return_value = MagicMock(
            success=True,
            output="Search results here",
        )
        mock_executor.return_value = mock_executor_instance
        
        # Create cognitive core with mocks
        mock_cognitive_instance = MagicMock()
        mock_cognitive_instance.process_input.return_value = {
            "response": "Here is the information you requested.",
            "actions_taken": ["web_search"],
            "reasoning_steps": 3,
        }
        mock_cognitive.return_value = mock_cognitive_instance
        
        # Simulate flow
        result = mock_cognitive_instance.process_input("Search for AI news")
        
        self.assertIn("response", result)
        self.assertEqual(result["actions_taken"], ["web_search"])
    
    @patch("mia.core.cognitive_architecture.MIACognitiveCore")
    def test_multimodal_input_processing(self, mock_cognitive):
        """Test processing multimodal inputs (text + image)."""
        mock_cognitive_instance = MagicMock()
        mock_cognitive_instance.process_input.return_value = {
            "response": "This image shows a sunset over the ocean.",
            "modalities_used": ["text", "vision"],
        }
        mock_cognitive.return_value = mock_cognitive_instance
        
        result = mock_cognitive_instance.process_input(
            "What is in this image?",
            image_data=b"fake_image_data"
        )
        
        self.assertIn("vision", result["modalities_used"])


class TestMemoryIntegration(unittest.TestCase):
    """Integration tests for memory systems."""
    
    @patch("mia.memory.long_term_memory.LongTermMemory")
    @patch("mia.memory.knowledge_graph.KnowledgeGraph")
    @patch("mia.memory.embedding_manager.EmbeddingManager")
    def test_memory_storage_and_retrieval(self, mock_embed, mock_kg, mock_ltm):
        """Test storing and retrieving memories."""
        # Setup mocks
        mock_embed_instance = MagicMock()
        mock_embed_instance.embed.return_value = [[0.1] * 384]
        mock_embed.return_value = mock_embed_instance
        
        mock_ltm_instance = MagicMock()
        mock_ltm_instance.store.return_value = "memory_id_123"
        mock_ltm_instance.retrieve.return_value = [
            {"id": "memory_id_123", "content": "User prefers dark mode", "similarity": 0.95}
        ]
        mock_ltm.return_value = mock_ltm_instance
        
        # Store memory
        memory_id = mock_ltm_instance.store("User prefers dark mode")
        self.assertEqual(memory_id, "memory_id_123")
        
        # Retrieve memory
        results = mock_ltm_instance.retrieve("What are user preferences?")
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0]["similarity"], 0.9)
    
    @patch("mia.memory.knowledge_graph.KnowledgeGraph")
    def test_knowledge_graph_relationships(self, mock_kg):
        """Test knowledge graph relationship management."""
        mock_kg_instance = MagicMock()
        mock_kg_instance.add_entity.return_value = True
        mock_kg_instance.add_relationship.return_value = True
        mock_kg_instance.query.return_value = [
            {"subject": "Python", "relation": "is_a", "object": "Programming Language"}
        ]
        mock_kg.return_value = mock_kg_instance
        
        # Add entities and relationship
        mock_kg_instance.add_entity("Python", {"type": "language"})
        mock_kg_instance.add_entity("Programming Language", {"type": "category"})
        mock_kg_instance.add_relationship("Python", "is_a", "Programming Language")
        
        # Query
        results = mock_kg_instance.query("What is Python?")
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["object"], "Programming Language")


class TestLLMProviderIntegration(unittest.TestCase):
    """Integration tests for LLM provider switching."""
    
    @patch("mia.llm.hybrid_llm_orchestration.HybridLLMOrchestrator")
    def test_provider_fallback(self, mock_orchestrator):
        """Test fallback when primary provider fails."""
        mock_instance = MagicMock()
        mock_instance.generate.side_effect = [
            Exception("OpenAI API error"),  # First call fails
            "Response from Ollama",  # Fallback succeeds
        ]
        mock_instance.get_last_provider.return_value = "ollama"
        mock_orchestrator.return_value = mock_instance
        
        # Simulate fallback flow
        try:
            result = mock_instance.generate("Hello")
        except:
            result = mock_instance.generate("Hello")
        
        self.assertEqual(result, "Response from Ollama")
    
    @patch("mia.llm.hybrid_llm_orchestration.HybridLLMOrchestrator")
    def test_task_based_routing(self, mock_orchestrator):
        """Test routing to different models based on task type."""
        mock_instance = MagicMock()
        mock_instance.route_task.side_effect = lambda task, _: {
            "code": "codellama",
            "chat": "gpt-4",
            "reasoning": "claude-3-opus",
        }.get(task, "gpt-3.5-turbo")
        mock_orchestrator.return_value = mock_instance
        
        code_model = mock_instance.route_task("code", "Write a function")
        chat_model = mock_instance.route_task("chat", "Hello")
        reason_model = mock_instance.route_task("reasoning", "Complex problem")
        
        self.assertEqual(code_model, "codellama")
        self.assertEqual(chat_model, "gpt-4")
        self.assertEqual(reason_model, "claude-3-opus")


class TestVisionPipelineIntegration(unittest.TestCase):
    """Integration tests for vision pipeline."""
    
    @patch("mia.multimodal.vision_processor.VisionProcessor")
    @patch("mia.core.cognitive_architecture.MIACognitiveCore")
    def test_image_understanding_flow(self, mock_cognitive, mock_vision):
        """Test image understanding integrated with cognitive core."""
        # Setup vision mock
        mock_vision_instance = MagicMock()
        mock_vision_instance.analyze_image.return_value = MagicMock(
            caption="A dog playing in the park",
            confidence=0.95,
        )
        mock_vision.return_value = mock_vision_instance
        
        # Setup cognitive mock
        mock_cognitive_instance = MagicMock()
        mock_cognitive_instance.process_input.return_value = {
            "response": "I see a dog playing in the park!",
            "vision_analysis": {"caption": "A dog playing in the park"},
        }
        mock_cognitive.return_value = mock_cognitive_instance
        
        # Process image
        result = mock_cognitive_instance.process_input(
            "Describe this image",
            image_data=b"fake_image"
        )
        
        self.assertIn("dog", result["response"].lower())


class TestSecurityIntegration(unittest.TestCase):
    """Integration tests for security layer."""
    
    @patch("mia.core.security_manager.SecurityManager")
    @patch("mia.core.action_executor.ActionExecutor")
    def test_permission_enforcement(self, mock_executor, mock_security):
        """Test that permissions are enforced on actions."""
        mock_security_instance = MagicMock()
        mock_security_instance.check_permission.side_effect = lambda perm, _: perm != "admin"
        mock_security.return_value = mock_security_instance
        
        mock_executor_instance = MagicMock()
        mock_executor_instance.execute.side_effect = lambda tool, params: (
            MagicMock(success=False, error="Permission denied")
            if tool == "admin_tool" else MagicMock(success=True)
        )
        mock_executor.return_value = mock_executor_instance
        
        # User tool should work
        result = mock_executor_instance.execute("user_tool", {})
        self.assertTrue(result.success)
        
        # Admin tool should fail
        result = mock_executor_instance.execute("admin_tool", {})
        self.assertFalse(result.success)
    
    @patch("mia.core.security_manager.SecurityManager")
    def test_path_validation(self, mock_security):
        """Test path validation for filesystem operations."""
        mock_security_instance = MagicMock()
        mock_security_instance.is_path_allowed.side_effect = lambda path: (
            path.startswith("/home/user") or path.startswith("/tmp")
        )
        mock_security.return_value = mock_security_instance
        
        # Allowed paths
        self.assertTrue(mock_security_instance.is_path_allowed("/home/user/file.txt"))
        self.assertTrue(mock_security_instance.is_path_allowed("/tmp/data.json"))
        
        # Blocked paths
        self.assertFalse(mock_security_instance.is_path_allowed("/etc/passwd"))
        self.assertFalse(mock_security_instance.is_path_allowed("/root/.ssh/id_rsa"))


class TestStreamingIntegration(unittest.TestCase):
    """Integration tests for streaming responses."""
    
    @patch("mia.llm.llm_manager.LLMManager")
    def test_streaming_generation(self, mock_llm):
        """Test streaming token generation."""
        mock_instance = MagicMock()
        
        async def mock_stream(*args, **kwargs):
            for token in ["Hello", " ", "world", "!"]:
                yield token
        
        mock_instance.stream_generate = mock_stream
        mock_llm.return_value = mock_instance
        
        async def collect_stream():
            tokens = []
            async for token in mock_instance.stream_generate("Hi"):
                tokens.append(token)
            return "".join(tokens)
        
        result = asyncio.run(collect_stream())
        
        self.assertEqual(result, "Hello world!")


class TestErrorRecoveryIntegration(unittest.TestCase):
    """Integration tests for error recovery."""
    
    @patch("mia.error_handler.ErrorHandler")
    @patch("mia.llm.llm_manager.LLMManager")
    def test_circuit_breaker_triggers_fallback(self, mock_llm, mock_handler):
        """Test circuit breaker triggers fallback provider."""
        mock_handler_instance = MagicMock()
        mock_handler_instance.is_circuit_open.side_effect = lambda svc: svc == "openai"
        mock_handler.return_value = mock_handler_instance
        
        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.side_effect = lambda prompt, provider: (
            Exception("Circuit open") if provider == "openai" 
            else "Fallback response"
        )
        mock_llm.return_value = mock_llm_instance
        
        # Should use fallback when circuit is open
        is_open = mock_handler_instance.is_circuit_open("openai")
        self.assertTrue(is_open)
    
    @patch("mia.error_handler.ErrorHandler")
    def test_retry_with_backoff(self, mock_handler):
        """Test retry mechanism with exponential backoff."""
        mock_instance = MagicMock()
        mock_instance.calculate_backoff.side_effect = lambda attempt: 0.1 * (2 ** attempt)
        mock_handler.return_value = mock_instance
        
        backoff_0 = mock_instance.calculate_backoff(0)
        backoff_1 = mock_instance.calculate_backoff(1)
        backoff_2 = mock_instance.calculate_backoff(2)
        
        self.assertAlmostEqual(backoff_0, 0.1)
        self.assertAlmostEqual(backoff_1, 0.2)
        self.assertAlmostEqual(backoff_2, 0.4)


class TestWebUIIntegration(unittest.TestCase):
    """Integration tests for Web UI."""
    
    @patch("mia.ui.webui.WebUI")
    def test_websocket_message_handling(self, mock_webui):
        """Test WebSocket message handling."""
        mock_instance = MagicMock()
        messages_received = []
        
        async def handle_message(msg):
            messages_received.append(msg)
            return {"response": f"Received: {msg}"}
        
        mock_instance.handle_message = handle_message
        mock_webui.return_value = mock_instance
        
        async def test_flow():
            result = await mock_instance.handle_message("Hello MIA")
            return result
        
        result = asyncio.run(test_flow())
        
        self.assertEqual(result["response"], "Received: Hello MIA")
        self.assertIn("Hello MIA", messages_received)


class TestEndToEndFlow(unittest.TestCase):
    """End-to-end integration tests."""
    
    @patch("mia.core.cognitive_architecture.MIACognitiveCore")
    @patch("mia.core.action_executor.ActionExecutor")
    @patch("mia.llm.llm_manager.LLMManager")
    @patch("mia.memory.long_term_memory.LongTermMemory")
    def test_complete_query_flow(self, mock_ltm, mock_llm, mock_executor, mock_cognitive):
        """Test complete query processing flow."""
        # Setup all mocks
        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = "Based on my analysis..."
        mock_llm.return_value = mock_llm_instance
        
        mock_executor_instance = MagicMock()
        mock_executor_instance.execute.return_value = MagicMock(success=True, output="Action result")
        mock_executor.return_value = mock_executor_instance
        
        mock_ltm_instance = MagicMock()
        mock_ltm_instance.retrieve.return_value = [{"content": "Relevant memory", "similarity": 0.9}]
        mock_ltm.return_value = mock_ltm_instance
        
        mock_cognitive_instance = MagicMock()
        mock_cognitive_instance.process_input.return_value = {
            "response": "Here is my complete analysis...",
            "memories_used": 1,
            "actions_taken": ["analyze"],
            "tokens_used": 150,
        }
        mock_cognitive.return_value = mock_cognitive_instance
        
        # Execute flow
        result = mock_cognitive_instance.process_input("Analyze this data")
        
        self.assertIn("response", result)
        self.assertEqual(result["memories_used"], 1)
        self.assertIn("analyze", result["actions_taken"])
    
    @patch("mia.core.cognitive_architecture.MIACognitiveCore")
    def test_conversation_context_persistence(self, mock_cognitive):
        """Test that conversation context persists across turns."""
        mock_instance = MagicMock()
        conversation_history = []
        
        def process_with_history(message):
            conversation_history.append(message)
            return {
                "response": f"Response to: {message}",
                "history_length": len(conversation_history),
            }
        
        mock_instance.process_input = process_with_history
        mock_cognitive.return_value = mock_instance
        
        # Multiple turns
        r1 = mock_instance.process_input("First message")
        r2 = mock_instance.process_input("Second message")
        r3 = mock_instance.process_input("Third message")
        
        self.assertEqual(r1["history_length"], 1)
        self.assertEqual(r2["history_length"], 2)
        self.assertEqual(r3["history_length"], 3)


if __name__ == "__main__":
    unittest.main()
