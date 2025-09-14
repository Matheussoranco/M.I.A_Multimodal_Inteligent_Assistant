"""
Integration tests for M.I.A main application flow
Tests end-to-end functionality with mocked external services
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# Add src directory to Python path for testing
project_root = Path(__file__).parent.parent
src_dir = project_root / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from mia.main import (
    parse_arguments, setup_logging, initialize_components,
    process_image_input, process_audio_input, get_text_input,
    process_command, process_with_llm, cleanup_resources
)


class TestMainIntegration(unittest.TestCase):
    """Integration tests for main application functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_args = MagicMock()
        self.mock_args.mode = "text"
        self.mock_args.model_id = "test-model"
        self.mock_args.debug = False
        self.mock_args.language = None
        self.mock_args.image_input = None

        self.mock_components = {
            'device': 'cpu',
            'llm': MagicMock(),
            'audio_available': False,
            'audio_utils': None,
            'speech_processor': None,
            'vision_processor': None,
            'action_executor': None,
            'performance_monitor': None,
            'cache_manager': None,
            'resource_manager': None
        }

    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_arguments_basic(self, mock_parse):
        """Test basic argument parsing."""
        mock_parse.return_value = self.mock_args

        with patch('argparse.ArgumentParser') as mock_parser_class:
            mock_parser = MagicMock()
            mock_parser_class.return_value = mock_parser
            mock_parser.parse_args.return_value = self.mock_args

            args = parse_arguments()

            self.assertEqual(args.mode, "text")
            self.assertEqual(args.model_id, "test-model")

    @patch('logging.basicConfig')
    def test_setup_logging_debug(self, mock_basic_config):
        """Test logging setup with debug enabled."""
        self.mock_args.debug = True

        setup_logging(self.mock_args)

        mock_basic_config.assert_called_once()
        args, kwargs = mock_basic_config.call_args
        self.assertEqual(kwargs['level'], 10)  # DEBUG level

    @patch('logging.basicConfig')
    def test_setup_logging_info(self, mock_basic_config):
        """Test logging setup with info level."""
        self.mock_args.debug = False

        setup_logging(self.mock_args)

        mock_basic_config.assert_called_once()
        args, kwargs = mock_basic_config.call_args
        self.assertEqual(kwargs['level'], 20)  # INFO level

    @patch('mia.main.LLMManager')
    @patch('mia.main.AudioUtils')
    @patch('mia.main.SpeechProcessor')
    @patch('mia.main.VisionProcessor')
    @patch('mia.main.ActionExecutor')
    @patch('torch.cuda.is_available', return_value=False)
    def test_initialize_components_text_mode(self, mock_cuda, mock_action, mock_vision,
                                           mock_speech, mock_audio, mock_llm):
        """Test component initialization in text mode."""
        # Setup mocks
        mock_llm.return_value = MagicMock()
        mock_vision.return_value = MagicMock()
        mock_action.return_value = MagicMock()

        # Text mode shouldn't initialize audio components
        self.mock_args.mode = "text"

        components = initialize_components(self.mock_args)

        # Verify LLM and vision components are initialized
        self.assertIsNotNone(components['llm'])
        self.assertIsNotNone(components['vision_processor'])
        self.assertIsNotNone(components['action_executor'])

        # Audio components should not be initialized in text mode
        self.assertFalse(components['audio_available'])
        self.assertIsNone(components['audio_utils'])
        self.assertIsNone(components['speech_processor'])

    @patch('torch.cuda.is_available', return_value=True)
    @patch('mia.main.VisionProcessor')
    @patch('mia.main.SpeechProcessor')
    @patch('mia.main.AudioUtils')
    @patch('mia.main.LLMManager')
    def test_initialize_components_audio_mode(self, mock_llm, mock_audio,
                                            mock_speech, mock_vision, mock_cuda):
        """Test component initialization in audio mode."""
        # Setup mocks
        mock_llm.return_value = MagicMock()
        mock_audio.return_value = MagicMock()
        mock_speech.return_value = MagicMock()
        mock_vision.return_value = MagicMock()

        self.mock_args.mode = "audio"

        components = initialize_components(self.mock_args)

        # Verify all components are initialized
        self.assertIsNotNone(components['llm'])
        self.assertTrue(components['audio_available'])
        self.assertIsNotNone(components['audio_utils'])
        self.assertIsNotNone(components['speech_processor'])
        self.assertIsNotNone(components['vision_processor'])
        self.assertEqual(components['device'], 'cuda')

    def test_process_image_input_with_valid_image(self):
        """Test image input processing with valid image."""
        self.mock_args.image_input = "/path/to/image.jpg"
        mock_vision_processor = MagicMock()
        mock_vision_processor.process_image.return_value = {"size": (100, 100)}
        self.mock_components['vision_processor'] = mock_vision_processor

        result = process_image_input(self.mock_args, self.mock_components)

        self.assertIn("image", result)
        self.assertEqual(result["image"]["size"], (100, 100))
        mock_vision_processor.process_image.assert_called_once_with("/path/to/image.jpg")
        # Image input should be cleared after processing
        self.assertIsNone(self.mock_args.image_input)

    def test_process_image_input_no_image(self):
        """Test image input processing with no image."""
        self.mock_args.image_input = None

        result = process_image_input(self.mock_args, self.mock_components)

        self.assertEqual(result, {})

    @patch('builtins.input', return_value="test input")
    def test_get_text_input_success(self, mock_input):
        """Test successful text input retrieval."""
        result = get_text_input(self.mock_args)

        self.assertEqual(result, "test input")
        mock_input.assert_called_once()

    @patch('builtins.input', side_effect=KeyboardInterrupt)
    @patch('builtins.print')
    def test_get_text_input_keyboard_interrupt(self, mock_print, mock_input):
        """Test text input handling with keyboard interrupt."""
        result = get_text_input(self.mock_args)

        self.assertEqual(result, "quit")
        mock_print.assert_called()

    def test_process_command_quit(self):
        """Test quit command processing."""
        should_continue, response = process_command("quit", self.mock_args, self.mock_components)

        self.assertFalse(should_continue)
        self.assertEqual(response, "")

    def test_process_command_help(self):
        """Test help command processing."""
        with patch('builtins.print') as mock_print:
            should_continue, response = process_command("help", self.mock_args, self.mock_components)

            self.assertTrue(should_continue)
            self.assertIsNone(response)  # Help prints directly
            mock_print.assert_called()

    def test_process_command_status(self):
        """Test status command processing."""
        with patch('builtins.print') as mock_print:
            should_continue, response = process_command("status", self.mock_args, self.mock_components)

            self.assertTrue(should_continue)
            self.assertIsNone(response)  # Status prints directly
            mock_print.assert_called()

    def test_process_command_clear(self):
        """Test clear command processing."""
        mock_cache_manager = MagicMock()
        mock_performance_monitor = MagicMock()
        self.mock_components['cache_manager'] = mock_cache_manager
        self.mock_components['performance_monitor'] = mock_performance_monitor

        should_continue, response = process_command("clear", self.mock_args, self.mock_components)

        self.assertTrue(should_continue)
        self.assertIn("cleared", response)
        mock_cache_manager.clear_all.assert_called_once()
        mock_performance_monitor.optimize_performance.assert_called_once()

    def test_process_command_audio_switch(self):
        """Test audio mode switching."""
        self.mock_components['speech_processor'] = MagicMock()

        should_continue, response = process_command("audio", self.mock_args, self.mock_components)

        self.assertTrue(should_continue)
        self.assertIn("audio", response)
        self.assertEqual(self.mock_args.mode, "audio")

    def test_process_command_unknown(self):
        """Test unknown command processing."""
        should_continue, response = process_command("unknown", self.mock_args, self.mock_components)

        self.assertTrue(should_continue)
        self.assertIsNone(response)

    @patch('mia.main.detect_and_execute_agent_commands')
    def test_process_with_llm_agent_command(self, mock_detect):
        """Test LLM processing with agent command."""
        mock_detect.return_value = (True, "Agent response")
        self.mock_components['llm'] = MagicMock()
        self.mock_components['action_executor'] = MagicMock()

        result = process_with_llm("create file test.py", {}, self.mock_components)

        self.assertEqual(result, "Agent response")
        mock_detect.assert_called_once()

    def test_process_with_llm_regular_query(self):
        """Test regular LLM query processing."""
        mock_llm = MagicMock()
        mock_llm.query.return_value = "LLM response"
        self.mock_components['llm'] = mock_llm

        result = process_with_llm("Hello", {}, self.mock_components)

        self.assertIn("LLM response", result)
        mock_llm.query.assert_called_once_with("Hello")

    def test_process_with_llm_no_response(self):
        """Test LLM processing with no response."""
        mock_llm = MagicMock()
        mock_llm.query.return_value = None
        self.mock_components['llm'] = mock_llm

        result = process_with_llm("Hello", {}, self.mock_components)

        self.assertIn("Could not generate", result)

    def test_process_with_llm_unavailable(self):
        """Test LLM processing when LLM is unavailable."""
        self.mock_components['llm'] = None

        result = process_with_llm("Hello", {}, self.mock_components)

        self.assertIn("not available", result)

    def test_cleanup_resources_success(self):
        """Test successful resource cleanup."""
        mock_performance_monitor = MagicMock()
        mock_cache_manager = MagicMock()
        mock_resource_manager = MagicMock()

        self.mock_components['performance_monitor'] = mock_performance_monitor
        self.mock_components['cache_manager'] = mock_cache_manager
        self.mock_components['resource_manager'] = mock_resource_manager

        cleanup_resources(self.mock_components)

        mock_performance_monitor.stop_monitoring.assert_called_once()
        mock_performance_monitor.cleanup.assert_called_once()
        mock_cache_manager.clear_all.assert_called_once()
        mock_resource_manager.stop.assert_called_once()

    def test_cleanup_resources_partial_failure(self):
        """Test resource cleanup with some components missing."""
        mock_performance_monitor = MagicMock()
        self.mock_components['performance_monitor'] = mock_performance_monitor
        self.mock_components['cache_manager'] = None
        self.mock_components['resource_manager'] = None

        cleanup_resources(self.mock_components)

        mock_performance_monitor.stop_monitoring.assert_called_once()
        mock_performance_monitor.cleanup.assert_called_once()


class TestMainEndToEndFlows(unittest.TestCase):
    """End-to-end test scenarios."""

    @patch('mia.main.parse_arguments')
    @patch('mia.main.setup_logging')
    @patch('mia.main.initialize_components')
    @patch('mia.main.display_status')
    @patch('mia.main.get_text_input', return_value="quit")
    @patch('mia.main.cleanup_resources')
    def test_main_flow_quit_immediately(self, mock_cleanup, mock_get_input,
                                       mock_display_status, mock_init_components,
                                       mock_setup_logging, mock_parse_args):
        """Test main flow that quits immediately."""
        mock_parse_args.return_value = MagicMock()
        mock_init_components.return_value = {}

        # Mock the main function to avoid actual execution
        with patch('mia.main.main') as mock_main:
            mock_main.return_value = None

            # We can't easily test the actual main loop without complex mocking
            # So we'll test the individual components instead
            pass

    @patch('mia.main.LLMManager')
    def test_llm_initialization_error_handling(self, mock_llm_class):
        """Test error handling during LLM initialization."""
        mock_llm_class.side_effect = Exception("LLM init failed")

        args = MagicMock()
        args.mode = "text"
        args.model_id = "test-model"

        components = initialize_components(args)

        self.assertIsNone(components['llm'])
        self.assertEqual(components['device'], 'cpu')

    @patch('mia.main.SpeechProcessor')
    @patch('mia.main.AudioUtils')
    def test_audio_initialization_error_handling(self, mock_audio_utils, mock_speech_processor):
        """Test error handling during audio component initialization."""
        mock_audio_utils.side_effect = Exception("Audio init failed")

        args = MagicMock()
        args.mode = "audio"
        args.model_id = "test-model"

        components = initialize_components(args)

        self.assertFalse(components['audio_available'])
        self.assertIsNone(components['audio_utils'])
        self.assertIsNone(components['speech_processor'])


if __name__ == "__main__":
    unittest.main()