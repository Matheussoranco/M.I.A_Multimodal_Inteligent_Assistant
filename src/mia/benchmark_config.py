"""
Benchmark configuration for M.I.A performance testing.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarks."""

    name: str
    description: str
    iterations: int = 10
    warmup_iterations: int = 3
    timeout_seconds: float = 300.0
    memory_limit_mb: int = 1024
    parameters: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


# Benchmark configurations
BENCHMARK_CONFIGS = {
    "text_processing": BenchmarkConfig(
        name="text_processing",
        description="Benchmark text processing and LLM inference",
        iterations=20,
        warmup_iterations=5,
        parameters={
            "input_sizes": [100, 500, 1000, 2000],
            "models": ["deepseek-r1:1.5b", "llama3:8b"],
        },
    ),
    "audio_processing": BenchmarkConfig(
        name="audio_processing",
        description="Benchmark audio transcription and processing",
        iterations=10,
        warmup_iterations=3,
        parameters={
            "audio_lengths": [5, 15, 30, 60],  # seconds
            "sample_rates": [16000, 22050, 44100],
        },
    ),
    "vision_processing": BenchmarkConfig(
        name="vision_processing",
        description="Benchmark image analysis and processing",
        iterations=15,
        warmup_iterations=4,
        parameters={
            "image_sizes": ["256x256", "512x512", "1024x1024"],
            "formats": ["jpg", "png"],
        },
    ),
    "multimodal_processing": BenchmarkConfig(
        name="multimodal_processing",
        description="Benchmark combined multimodal processing",
        iterations=10,
        warmup_iterations=3,
        parameters={
            "text_length": 500,
            "audio_length": 10,
            "image_size": "512x512",
        },
    ),
    "memory_operations": BenchmarkConfig(
        name="memory_operations",
        description="Benchmark memory and caching operations",
        iterations=50,
        warmup_iterations=10,
        parameters={
            "cache_sizes": [100, 500, 1000, 5000],
            "item_sizes": [100, 1000, 10000],  # bytes
        },
    ),
    "concurrent_requests": BenchmarkConfig(
        name="concurrent_requests",
        description="Benchmark concurrent request handling",
        iterations=5,
        warmup_iterations=2,
        parameters={
            "concurrency_levels": [1, 5, 10, 20],
            "request_types": ["text", "audio", "vision", "multimodal"],
        },
    ),
}


def get_benchmark_config(name: str) -> BenchmarkConfig:
    """Get benchmark configuration by name."""
    if name not in BENCHMARK_CONFIGS:
        raise ValueError(f"Unknown benchmark: {name}")
    return BENCHMARK_CONFIGS[name]


def list_available_benchmarks() -> List[str]:
    """List all available benchmark names."""
    return list(BENCHMARK_CONFIGS.keys())


def create_benchmark_report(
    results: Dict[str, Any], output_dir: str = "benchmarks/results"
) -> str:
    """Create a benchmark report file."""
    os.makedirs(output_dir, exist_ok=True)

    import json
    from datetime import datetime

    report = {"timestamp": datetime.now().isoformat(), "results": results}

    report_path = os.path.join(
        output_dir, f"benchmark_report_{int(datetime.now().timestamp())}.json"
    )

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report_path
