#!/usr/bin/env python3
"""
M.I.A Benchmark Runner
Runs performance benchmarks for various M.I.A components.
"""
import argparse
import logging

# Add src to path
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent))

from mia.benchmark_config import (
    create_benchmark_report,
    get_benchmark_config,
    list_available_benchmarks,
)
from mia.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs performance benchmarks for M.I.A components."""

    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.results = {}

    def run_text_processing_benchmark(self, config) -> Dict[str, Any]:
        """Run text processing benchmark."""
        logger.info("Running text processing benchmark...")

        results = {"benchmark": "text_processing", "iterations": [], "summary": {}}

        # Mock text processing benchmark
        for i in range(config.iterations):
            start_time = time.time()

            # Simulate text processing
            time.sleep(0.01)  # Mock processing time

            end_time = time.time()
            processing_time = end_time - start_time

            results["iterations"].append(
                {"iteration": i + 1, "processing_time": processing_time}
            )

        # Calculate summary
        processing_times = [iter["processing_time"] for iter in results["iterations"]]
        results["summary"] = {
            "total_iterations": len(processing_times),
            "average_time": sum(processing_times) / len(processing_times),
            "min_time": min(processing_times),
            "max_time": max(processing_times),
            "total_time": sum(processing_times),
        }

        return results

    def run_memory_benchmark(self, config) -> Dict[str, Any]:
        """Run memory operations benchmark."""
        logger.info("Running memory benchmark...")

        results = {"benchmark": "memory_operations", "iterations": [], "summary": {}}

        # Mock memory operations
        for i in range(config.iterations):
            start_time = time.time()

            # Simulate memory operations
            test_data = [0] * 1000  # Create some data
            processed_data = [x * 2 for x in test_data]  # Process it
            del processed_data  # Clean up

            end_time = time.time()
            processing_time = end_time - start_time

            results["iterations"].append(
                {"iteration": i + 1, "processing_time": processing_time}
            )

        # Calculate summary
        processing_times = [iter["processing_time"] for iter in results["iterations"]]
        results["summary"] = {
            "total_iterations": len(processing_times),
            "average_time": sum(processing_times) / len(processing_times),
            "min_time": min(processing_times),
            "max_time": max(processing_times),
            "total_time": sum(processing_times),
        }

        return results

    def run_benchmark(self, benchmark_name: str) -> Dict[str, Any]:
        """Run a specific benchmark."""
        config = get_benchmark_config(benchmark_name)

        logger.info(f"Starting benchmark: {benchmark_name}")
        logger.info(f"Description: {config.description}")
        logger.info(f"Iterations: {config.iterations}")

        # Start performance monitoring
        self.performance_monitor.start_monitoring()

        try:
            if benchmark_name == "text_processing":
                result = self.run_text_processing_benchmark(config)
            elif benchmark_name == "memory_operations":
                result = self.run_memory_benchmark(config)
            else:
                # Mock implementation for other benchmarks
                result = {
                    "benchmark": benchmark_name,
                    "status": "mock_implemented",
                    "message": f"Benchmark {benchmark_name} is configured but uses mock implementation",
                }

            self.results[benchmark_name] = result
            logger.info(f"Benchmark {benchmark_name} completed")

        except Exception as e:
            logger.error(f"Error running benchmark {benchmark_name}: {e}")
            self.results[benchmark_name] = {
                "benchmark": benchmark_name,
                "status": "error",
                "error": str(e),
            }

        finally:
            # Stop performance monitoring
            self.performance_monitor.stop_monitoring()

        return self.results.get(benchmark_name, {})

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all available benchmarks."""
        logger.info("Running all benchmarks...")

        available_benchmarks = list_available_benchmarks()

        for benchmark_name in available_benchmarks:
            self.run_benchmark(benchmark_name)

        return self.results

    def generate_report(self, output_dir: str = "benchmarks/results") -> str:
        """Generate benchmark report."""
        logger.info("Generating benchmark report...")

        # Add performance summary
        performance_summary = self.performance_monitor.get_performance_summary()
        self.results["performance_summary"] = performance_summary

        report_path = create_benchmark_report(self.results, output_dir)
        logger.info(f"Benchmark report saved to: {report_path}")

        return report_path


def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(description="M.I.A Benchmark Runner")
    parser.add_argument(
        "--benchmark",
        choices=list_available_benchmarks() + ["all"],
        default="all",
        help="Specific benchmark to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results",
        help="Output directory for benchmark reports",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run benchmarks
    runner = BenchmarkRunner()

    if args.benchmark == "all":
        results = runner.run_all_benchmarks()
    else:
        results = runner.run_benchmark(args.benchmark)

    # Generate report
    report_path = runner.generate_report(args.output_dir)

    print(f"\nBenchmark completed!")
    print(f"Results saved to: {report_path}")

    # Print summary
    if results:
        print("\nBenchmark Summary:")
        for name, result in results.items():
            if name != "performance_summary":
                if "summary" in result:
                    avg_time = result["summary"].get("average_time", "N/A")
                    print(
                        f"  {name}: {result['summary'].get('total_iterations', 0)} iterations, avg {avg_time:.4f}s"
                    )
                else:
                    print(f"  {name}: {result.get('status', 'unknown')}")


if __name__ == "__main__":
    main()
