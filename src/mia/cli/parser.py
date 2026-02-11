"""CLI argument parsing and logging setup for M.I.A."""

from __future__ import annotations

import argparse
import logging
import sys

from .utils import suppress_warnings
from ..__version__ import __version__


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments and return the resulting namespace."""
    parser = argparse.ArgumentParser(
        description="M.I.A - Multimodal Intelligent Assistant"
    )
    parser.add_argument(
        "--mode",
        choices=["text", "audio", "mixed", "auto", "web"],
        default="mixed",
        help="Interaction mode: text|audio|mixed|auto|web",
    )
    parser.add_argument(
        "--web", action="store_true", help="Start the Ollama-style Web UI"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for web UI (default: 8080)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for web UI (default: 0.0.0.0)",
    )
    parser.add_argument("--text-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--audio-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--language",
        choices=["en", "pt"],
        default=None,
        help="Interface language (en=English, pt=Portuguese)",
    )
    parser.add_argument(
        "--image-input", type=str, default=None, help="Image to process"
    )
    parser.add_argument(
        "--model-id", type=str, default="gpt-oss:latest", help="Model ID"
    )
    parser.add_argument(
        "--profile",
        type=str,
        help="LLM profile name defined in config.yaml",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"M.I.A {__version__}",
        help="Show version information",
    )
    parser.add_argument(
        "--info", action="store_true", help="Show detailed version and system info"
    )
    return parser.parse_args()


def setup_logging(args: argparse.Namespace) -> None:
    """Configure logging levels and suppress noisy ML-framework output."""
    level = logging.DEBUG if getattr(args, "debug", False) else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    for name in (
        "transformers",
        "torch",
        "tensorflow",
        "numba",
        "chromadb",
        "urllib3",
        "requests",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
    suppress_warnings()
