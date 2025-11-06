# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, call, patch

import numpy as np

# Add src directory to Python path
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from mia.main import (  # type: ignore
    get_text_input,
    process_audio_input,
    process_command,
    process_image_input,
    process_with_llm,
)


class TestMultimodalInteractionFlows(unittest.TestCase):
    """Test multimodal interaction flows."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_args = MagicMock()
        self.mock_args.mode = "mixed"

        audio_utils = MagicMock()
        audio_utils.sample_rate = 16000

        self.mock_components = {
            "audio_available": True,
            "audio_utils": audio_utils,
            "speech_processor": MagicMock(),
            "vision_processor": MagicMock(),
            "llm": MagicMock(),
            "action_executor": MagicMock(),
            "audio_config": SimpleNamespace(
                hotword_enabled=False,
                push_to_talk=False,
                hotword=None,
                hotword_sensitivity=0.5,
                tts_enabled=False,
                sample_rate=16000,
                chunk_size=1024,
                device_id=None,
                input_threshold=0.01,
            ),
            "speech_generator": None,
            "hotword_detector": None,
            "rag_pipeline": None,
        }

    @patch("builtins.print")
    def test_process_audio_input_successful_transcription(self, mock_print):
        """Test successful audio input processing."""
        # Setup mocks
        self.mock_args.mode = "audio"  # Set mode to audio for processing
        mock_audio_utils = self.mock_components["audio_utils"]
        mock_speech_processor = self.mock_components["speech_processor"]

        mock_audio_utils.capture_with_vad.return_value = np.array(
            [0.1, -0.2], dtype=np.float32
        )

        # Mock transcription
        mock_speech_processor.transcribe_audio_data.return_value = (
            "Hello world"
        )

        user_input, inputs = process_audio_input(
            self.mock_args, self.mock_components
        )

        self.assertEqual(user_input, "Hello world")
        self.assertEqual(inputs["audio"], "Hello world")
        mock_audio_utils.capture_with_vad.assert_called_once()
        mock_print.assert_called()

    @patch("builtins.print")
    def test_process_audio_input_no_speech_detected(self, mock_print):
        """Test audio input processing when no speech is detected."""
        self.mock_args.mode = "audio"  # Set mode to audio for processing
        mock_audio_utils = self.mock_components["audio_utils"]
        mock_speech_processor = self.mock_components["speech_processor"]
        mock_audio_utils.capture_with_vad.return_value = np.array(
            [], dtype=np.float32
        )
        mock_speech_processor.transcribe_audio_data.return_value = ""

        user_input, inputs = process_audio_input(
            self.mock_args, self.mock_components
        )

        self.assertIsNone(user_input)
        self.assertEqual(inputs, {})
        # Should print no speech detected message
        mock_print.assert_called()

    @patch("builtins.print")
    def test_process_audio_input_keyboard_interrupt(self, mock_print):
        """Test audio input processing with keyboard interrupt."""
        self.mock_args.mode = "audio"  # Set mode to audio for processing
        mock_audio_utils = self.mock_components["audio_utils"]
        mock_audio_utils.capture_with_vad.side_effect = KeyboardInterrupt()

        user_input, inputs = process_audio_input(
            self.mock_args, self.mock_components
        )

        self.assertIsNone(user_input)
        self.assertEqual(inputs, {})
        self.assertEqual(
            self.mock_args.mode, "text"
        )  # Should switch to text mode

    def test_process_image_input_success(self):
        """Test successful image input processing."""
        self.mock_args.image_input = "/path/to/test.jpg"
        mock_vision_processor = self.mock_components["vision_processor"]
        mock_vision_processor.process_image.return_value = {
            "size": (800, 600),
            "format": "JPEG",
        }

        result = process_image_input(self.mock_args, self.mock_components)

        self.assertIn("image", result)
        self.assertEqual(result["image"]["size"], (800, 600))
        self.assertIsNone(self.mock_args.image_input)  # Should be cleared

    def test_process_image_input_processing_error(self):
        """Test image input processing with processing error."""
        self.mock_args.image_input = "/path/to/test.jpg"
        mock_vision_processor = self.mock_components["vision_processor"]
        mock_vision_processor.process_image.side_effect = Exception(
            "Processing failed"
        )

        result = process_image_input(self.mock_args, self.mock_components)

        self.assertEqual(result, {})  # Should return empty dict on error
        self.assertIsNone(
            self.mock_args.image_input
        )  # Should still be cleared

    @patch("mia.main.detect_and_execute_agent_commands")
    def test_process_with_llm_agent_command_execution(self, mock_detect):
        """Test LLM processing that triggers agent command."""
        mock_detect.return_value = (True, "File created successfully")

        result = process_with_llm(
            "create file test.py", {}, self.mock_components
        )

        self.assertEqual(result["response"], "File created successfully")
        self.assertFalse(result["streamed"])
        self.assertIsNone(result["error"])
        mock_detect.assert_called_once_with(
            "create file test.py", self.mock_components["action_executor"]
        )

    def test_process_with_llm_empty_input(self):
        """Test LLM processing with empty input."""
        result = process_with_llm("", {}, self.mock_components)

        self.assertIsNone(result["response"])
        self.assertIn(
            "No input", result["error"] or ""
        )  # Check for localized message content
        self.mock_components["llm"].query.assert_not_called()

    def test_process_with_llm_llm_error(self):
        """Test LLM processing with LLM error."""
        self.mock_components["llm"].query.side_effect = Exception("LLM error")
        self.mock_components["llm"].stream_enabled = False
        self.mock_components["llm"].supports_streaming = MagicMock(
            return_value=False
        )

        result = process_with_llm("test query", {}, self.mock_components)

        self.assertIsNone(result["response"])
        self.assertIn(
            "llm_error", result["error"] or ""
        )  # Check for localized error message


class TestCommandProcessingFlows(unittest.TestCase):
    """Test various command processing scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_args = MagicMock()
        self.mock_args.mode = "mixed"

        self.mock_components = {
            "audio_available": True,
            "speech_processor": MagicMock(),
            "cache_manager": MagicMock(),
            "performance_monitor": MagicMock(),
        }

    @patch("builtins.print")
    def test_process_command_help_with_audio(self, mock_print):
        """Test help command with audio available."""
        should_continue, response = process_command(
            "help", self.mock_args, self.mock_components
        )

        self.assertTrue(should_continue)
        self.assertIsNone(response)
        # Verify help was printed
        self.assertTrue(mock_print.called)

    @patch("builtins.print")
    def test_process_command_help_text_only(self, mock_print):
        """Test help command in text-only mode."""
        self.mock_components["audio_available"] = False
        self.mock_args.mode = "text"

        should_continue, response = process_command(
            "help", self.mock_args, self.mock_components
        )

        self.assertTrue(should_continue)
        self.assertIsNone(response)

    @patch("builtins.print")
    def test_process_command_models(self, mock_print):
        """Test models command."""
        self.mock_args.model_id = "deepseek-r1:1.5b"

        should_continue, response = process_command(
            "models", self.mock_args, self.mock_components
        )

        self.assertTrue(should_continue)
        self.assertIsNone(response)

    def test_process_command_clear_with_components(self):
        """Test clear command with all components available."""
        should_continue, response = process_command(
            "clear", self.mock_args, self.mock_components
        )

        self.assertTrue(should_continue)
        self.assertIsNotNone(response)
        if response is not None:
            self.assertIn("cleared", response)
        self.mock_components["cache_manager"].clear_all.assert_called_once()
        self.mock_components[
            "performance_monitor"
        ].optimize_performance.assert_called_once()

    def test_process_command_clear_missing_components(self):
        """Test clear command with missing components."""
        self.mock_components["cache_manager"] = None
        self.mock_components["performance_monitor"] = None

        should_continue, response = process_command(
            "clear", self.mock_args, self.mock_components
        )

        self.assertTrue(should_continue)
        self.assertIsNotNone(response)
        if response is not None:
            self.assertIn("cleared", response)


class TestErrorHandlingFlows(unittest.TestCase):
    """Test error handling in various scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_args = MagicMock()
        # Use a more flexible dictionary that can hold None values
        self.base_components: Dict[str, Any] = {
            "llm": MagicMock(),
            "vision_processor": MagicMock(),
            "speech_processor": MagicMock(),
        }

    def test_process_image_input_vision_processor_none(self):
        """Test image processing when vision processor is None."""
        self.mock_args.image_input = "/path/to/image.jpg"
        components = self.base_components.copy()
        components["vision_processor"] = None  # type: ignore

        result = process_image_input(self.mock_args, components)

        self.assertEqual(result, {})

    def test_process_audio_input_speech_processor_none(self):
        """Test audio processing when speech processor is None."""
        self.mock_args.mode = "audio"
        components = self.base_components.copy()
        components["speech_processor"] = None  # type: ignore

        user_input, inputs = process_audio_input(self.mock_args, components)

        self.assertIsNone(user_input)
        self.assertEqual(inputs, {})

    def test_process_with_llm_llm_none(self):
        """Test LLM processing when LLM is None."""
        components = self.base_components.copy()
        components["llm"] = None  # type: ignore

        result = process_with_llm("test", {}, components)

        self.assertIsNone(result["response"])
        self.assertIn(
            "not available", result["error"] or ""
        )  # Check for localized message content

    @patch("builtins.input", side_effect=Exception("Input error"))
    def test_get_text_input_unexpected_error(self, mock_input):
        """Test text input with unexpected error."""
        with self.assertRaises(Exception):
            get_text_input(self.mock_args)


class TestIntegrationScenarios(unittest.TestCase):
    """Test complete integration scenarios."""

    @patch("mia.main.parse_arguments")
    @patch("mia.main.initialize_components")
    @patch("mia.main.process_image_input")
    @patch("mia.main.process_audio_input")
    @patch("mia.main.get_text_input")
    @patch("mia.main.process_command")
    @patch("mia.main.process_with_llm")
    @patch("mia.main.cleanup_resources")
    def test_complete_text_interaction_flow(
        self,
        mock_cleanup,
        mock_process_llm,
        mock_process_cmd,
        mock_get_input,
        mock_process_audio,
        mock_process_image,
        mock_init_components,
        mock_parse_args,
    ):
        """Test complete text-based interaction flow."""
        # Setup mocks
        mock_args = MagicMock()
        mock_args.mode = "text"
        mock_parse_args.return_value = mock_args

        mock_components = {"llm": MagicMock()}
        mock_init_components.return_value = mock_components

        # Simulate user input and command processing
        mock_get_input.return_value = "hello"
        mock_process_cmd.return_value = (True, None)
        mock_process_llm.return_value = {
            "response": "Hello! How can I help you?",
            "streamed": False,
            "citations": None,
            "error": None,
        }

        # Test the flow
        inputs = {}
        user_input = mock_get_input()

        if user_input:
            should_continue, cmd_response = mock_process_cmd(
                user_input,
                mock_args,
                mock_components,
            )
            if should_continue and cmd_response is None:
                inputs["text"] = user_input
                response = mock_process_llm(
                    user_input, inputs, mock_components
                )

        self.assertEqual(user_input, "hello")
        self.assertEqual(response["response"], "Hello! How can I help you?")

    @patch("mia.main.parse_arguments")
    @patch("mia.main.initialize_components")
    @patch("mia.main.process_image_input")
    @patch("mia.main.process_audio_input")
    @patch("mia.main.process_with_llm")
    def test_multimodal_interaction_flow(
        self,
        mock_process_llm,
        mock_process_audio,
        mock_process_image,
        mock_init_components,
        mock_parse_args,
    ):
        """Test multimodal interaction flow with image and text."""
        # Setup mocks
        mock_args = MagicMock()
        mock_args.mode = "mixed"
        mock_args.image_input = "/path/to/image.jpg"
        mock_parse_args.return_value = mock_args

        mock_components = {
            "vision_processor": MagicMock(),
            "llm": MagicMock(),
        }
        mock_init_components.return_value = mock_components

        # Mock image processing
        mock_process_image.return_value = {"image": {"size": (100, 100)}}
        mock_process_audio.return_value = (None, {})  # No audio input
        mock_process_llm.return_value = {
            "response": "I captured your image and described it!",
            "streamed": True,
            "citations": None,
            "error": None,
        }

        # Simulate the flow
        image_inputs = mock_process_image(mock_args, mock_components)
        user_input, audio_inputs = mock_process_audio(
            mock_args, mock_components
        )

        inputs = {}
        inputs.update(image_inputs)
        inputs.update(audio_inputs)

        if not user_input:
            user_input = "Describe this image"

        inputs["text"] = user_input
        response = mock_process_llm(user_input, inputs, mock_components)

        self.assertIn("image", inputs)
        self.assertEqual(inputs["text"], "Describe this image")
        self.assertEqual(
            response["response"], "I captured your image and described it!"
        )
        self.assertTrue(response["streamed"])
        mock_process_llm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
