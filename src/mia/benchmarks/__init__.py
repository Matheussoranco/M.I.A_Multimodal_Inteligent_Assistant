"""
M.I.A AGI Benchmarks Module

This module provides state-of-the-art benchmark implementations for evaluating
the M.I.A assistant across multiple AI capability dimensions:

- ARC-AGI: Abstract Reasoning Corpus for measuring fluid intelligence
- GAIA: General AI Assistant benchmark for real-world task completion
- SWE-BENCH: Software Engineering benchmark for code understanding/generation
- GPQA: Graduate-level science questions for expert knowledge reasoning
- WebVoyager: Web navigation and automation benchmark
- OSWorld: Operating system interaction benchmark
- MMMU: Massive Multi-discipline Multimodal Understanding

Each benchmark follows a standardized interface for consistent evaluation.
"""

from .base import BaseBenchmark, BenchmarkResult, BenchmarkMetrics
from .arc_agi import ARCAGIBenchmark
from .gaia import GAIABenchmark
from .swe_bench import SWEBenchBenchmark
from .gpqa import GPQABenchmark
from .webvoyager import WebVoyagerBenchmark
from .osworld import OSWorldBenchmark
from .mmmu import MMMUBenchmark
from .runner import AGIBenchmarkRunner

__all__ = [
    "BaseBenchmark",
    "BenchmarkResult",
    "BenchmarkMetrics",
    "ARCAGIBenchmark",
    "GAIABenchmark",
    "SWEBenchBenchmark",
    "GPQABenchmark",
    "WebVoyagerBenchmark",
    "OSWorldBenchmark",
    "MMMUBenchmark",
    "AGIBenchmarkRunner",
]
