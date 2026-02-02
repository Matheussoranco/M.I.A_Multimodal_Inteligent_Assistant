import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

# Mock imports safely
class SafeMock(MagicMock):
    @property
    def __spec__(self):
        return MagicMock()

sys.modules["llama_cpp"] = SafeMock()

# Try to import psutil, if fails, mock it
try:
    import psutil
except ImportError:
    sys.modules["psutil"] = SafeMock()

from mia.llm.llm_manager import LLMManager  # type: ignore[import-not-found]
from mia.core.cognitive_architecture import MIACognitiveCore  # type: ignore[import-not-found]
from mia.resource_manager import ResourceManager  # type: ignore[import-not-found]

class TestSOTAUpgrade(unittest.TestCase):
    def setUp(self):
        # Reset singletons if necessary or just instantiate
        self.resource_manager = ResourceManager()
        
    def test_gguf_initialization(self):
        """Verify LLMManager can initialize GGUF provider."""
        logger.info("Testing GGUF initialization...")
        with patch("mia.llm.llm_manager.HAS_LLAMA_CPP", True):
            # Patch Llama class
            with patch("mia.llm.llm_manager.Llama", MagicMock()) as mock_llama:
                manager = LLMManager(provider="gguf", model_id="llama-2-7b-chat.Q4_K_M.gguf")
                manager._initialize_llama() # Force init for testing isolation
                
                self.assertEqual(manager.provider, "gguf")
                self.assertTrue(manager._available)
                mock_llama.assert_called()
                logger.info("GGUF initialization successful.")

    def test_gguf_query_dispatch(self):
        """Verify query dispatch routes to _query_llama."""
        logger.info("Testing GGUF query dispatch...")
        with patch("mia.llm.llm_manager.HAS_LLAMA_CPP", True):
            manager = LLMManager(provider="gguf", model_id="test.gguf")
            manager.model = MagicMock()
            # Configure mock to behave like Llama instance
            manager.model.return_value = {"choices": [{"text": "GGUF Response"}]}
            
            # Direct call to _query_llama to verify logic
            response = manager._query_llama("Hello")
            self.assertEqual(response, "GGUF Response")
            
            # Verify dispatch
            with patch.object(manager, '_query_llama', return_value="Dispatched") as mock_method:
                manager.query("Hello")
                mock_method.assert_called()
                logger.info("GGUF query dispatch successful.")

    def test_lazy_loading_vision(self):
        """Verify Vision components are lazy loaded."""
        logger.info("Testing Vision lazy loading...")
        core = MIACognitiveCore(llm_client=MagicMock())
        
        # Initially false
        self.assertFalse(core._vision_initialized)
        
        # Trigger loading via input
        with patch("mia.core.cognitive_architecture.CLIPProcessor") as mock_proc:
             with patch("mia.core.cognitive_architecture.CLIPModel") as mock_model:
                with patch("mia.core.cognitive_architecture.HAS_CLIP", True):
                    core.process_multimodal_input({"image": "test.jpg"})
                    
                    # Should be true now
                    self.assertTrue(core._vision_initialized)
                    logger.info("Vision lazy loading successful.")

    def test_memory_callback_registration(self):
        """Verify LLM unload callback is registered."""
        logger.info("Testing Memory Callback Registration...")
        
        # Mock LLM component
        mock_llm = MagicMock()
        
        # Manually register
        self.resource_manager.register_memory_pressure_callback(mock_llm.unload_model)
        
        # Trigger pressure
        # Use patch on the psutil module imported in resource_manager
        with patch("mia.resource_manager.psutil.virtual_memory") as mock_mem:
            mock_mem.return_value.percent = 95.0
            self.resource_manager._check_memory_usage()
            
            # Should have called unload
            mock_llm.unload_model.assert_called()
            logger.info("Memory callback triggered successfully.")

if __name__ == "__main__":
    unittest.main()
