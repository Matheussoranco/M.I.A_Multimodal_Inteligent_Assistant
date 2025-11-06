import base64
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src directory to Python path for imports
project_root = Path(__file__).parent.parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


class TestSpeechGenerator(unittest.TestCase):
    """Unit tests for the SpeechGenerator audio module."""

    @patch("mia.audio.speech_generator.TRANSFORMERS_AVAILABLE", new=False)
    @patch("mia.audio.speech_generator.DATASETS_AVAILABLE", new=False)
    @patch("mia.audio.speech_generator.LLMInference")
    @patch("mia.audio.speech_generator.ConfigManager")
    @patch("mia.audio.speech_generator.requests.post")
    def test_generate_speech_via_api_nanochat(
        self, mock_post, mock_config_manager, mock_llm_inference, *_
    ):
        mock_config = MagicMock()
        mock_config.config = None
        mock_config_manager.return_value = mock_config
        mock_llm_inference.return_value = MagicMock()

        from mia.audio.speech_generator import SpeechGenerator  # type: ignore

        generator = SpeechGenerator(default_tts_provider="nanochat")
        generator.tts_providers.setdefault("nanochat", {})[
            "api_key"
        ] = "test-token"

        audio_bytes = b"unit-test-audio"
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {
            "audio": base64.b64encode(audio_bytes).decode("utf-8")
        }
        mock_post.return_value = mock_response

        payload = generator.generate_speech_via_api(
            "Generate voice output", provider="nanochat"
        )

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["audio_bytes"], audio_bytes)
        mock_post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
