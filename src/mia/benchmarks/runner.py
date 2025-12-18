"""
AGI Benchmark Runner

Unified runner for executing all AGI benchmarks on the M.I.A agent.
Provides CLI interface and programmatic API for benchmark execution.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .base import BaseBenchmark, BenchmarkResult, BenchmarkSuite
from .arc_agi import ARCAGIBenchmark
from .gaia import GAIABenchmark
from .swe_bench import SWEBenchBenchmark
from .gpqa import GPQABenchmark
from .webvoyager import WebVoyagerBenchmark
from .osworld import OSWorldBenchmark
from .mmmu import MMMUBenchmark

logger = logging.getLogger(__name__)

# Registry of available benchmarks
BENCHMARK_REGISTRY: Dict[str, Type[BaseBenchmark]] = {
    "arc-agi": ARCAGIBenchmark,
    "gaia": GAIABenchmark,
    "swe-bench": SWEBenchBenchmark,
    "gpqa": GPQABenchmark,
    "webvoyager": WebVoyagerBenchmark,
    "osworld": OSWorldBenchmark,
    "mmmu": MMMUBenchmark,
}

# Benchmark categories for grouping
BENCHMARK_CATEGORIES = {
    "reasoning": ["arc-agi", "gpqa"],
    "coding": ["swe-bench"],
    "assistant": ["gaia"],
    "multimodal": ["mmmu"],
    "agentic": ["webvoyager", "osworld"],
}


class AGIBenchmarkRunner:
    """
    Unified runner for AGI benchmarks.
    
    Supports:
    - Running individual benchmarks
    - Running benchmark suites
    - Parallel execution
    - Result aggregation and reporting
    """
    
    def __init__(
        self,
        agent: Any,
        output_dir: str = "benchmarks/results",
        data_dir: str = "benchmarks/data",
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            agent: The AI agent to evaluate
            output_dir: Directory for saving results
            data_dir: Base directory for benchmark data
        """
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.results: Dict[str, BenchmarkResult] = {}
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def list_benchmarks(self) -> List[Dict[str, Any]]:
        """List all available benchmarks."""
        benchmarks = []
        
        for name, benchmark_class in BENCHMARK_REGISTRY.items():
            # Get category
            category = "other"
            for cat, names in BENCHMARK_CATEGORIES.items():
                if name in names:
                    category = cat
                    break
            
            benchmarks.append({
                "name": name,
                "class": benchmark_class.__name__,
                "category": category,
                "description": benchmark_class.__doc__.split("\n")[0] if benchmark_class.__doc__ else "",
            })
        
        return benchmarks
    
    def run_benchmark(
        self,
        benchmark_name: str,
        config: Optional[Dict[str, Any]] = None,
        subset: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Run a single benchmark.
        
        Args:
            benchmark_name: Name of the benchmark to run
            config: Optional configuration overrides
            subset: Dataset subset to use
            limit: Maximum number of tasks
            
        Returns:
            Benchmark results
        """
        if benchmark_name not in BENCHMARK_REGISTRY:
            raise ValueError(f"Unknown benchmark: {benchmark_name}. "
                           f"Available: {list(BENCHMARK_REGISTRY.keys())}")
        
        benchmark_class = BENCHMARK_REGISTRY[benchmark_name]
        
        # Prepare config
        full_config = {
            "output_dir": str(self.output_dir / benchmark_name),
            "data_dir": str(self.data_dir / benchmark_name),
        }
        if config:
            full_config.update(config)
        
        # Create benchmark instance
        benchmark = benchmark_class(
            agent=self.agent,
            config=full_config,
            output_dir=full_config["output_dir"],
        )
        
        logger.info(f"Running {benchmark_name} benchmark...")
        
        # Run benchmark
        result = benchmark.run(subset=subset, limit=limit)
        
        # Store result
        self.results[benchmark_name] = result
        
        # Print summary
        self._print_result_summary(benchmark_name, result)
        
        return result
    
    def run_suite(
        self,
        benchmark_names: Optional[List[str]] = None,
        category: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        limit_per_benchmark: Optional[int] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        Run a suite of benchmarks.
        
        Args:
            benchmark_names: List of benchmark names to run (default: all)
            category: Run only benchmarks in this category
            config: Configuration overrides
            limit_per_benchmark: Limit tasks per benchmark
            
        Returns:
            Dictionary of results
        """
        # Determine which benchmarks to run
        if benchmark_names:
            names_to_run = benchmark_names
        elif category:
            names_to_run = BENCHMARK_CATEGORIES.get(category, [])
        else:
            names_to_run = list(BENCHMARK_REGISTRY.keys())
        
        logger.info(f"Running benchmark suite: {names_to_run}")
        
        results = {}
        for name in names_to_run:
            try:
                result = self.run_benchmark(
                    name,
                    config=config,
                    limit=limit_per_benchmark,
                )
                results[name] = result
            except Exception as e:
                logger.error(f"Failed to run {name}: {e}")
                continue
        
        # Generate suite report
        self._generate_suite_report(results)
        
        return results
    
    def run_all(
        self,
        limit_per_benchmark: Optional[int] = None,
    ) -> Dict[str, BenchmarkResult]:
        """Run all available benchmarks."""
        return self.run_suite(limit_per_benchmark=limit_per_benchmark)
    
    def _print_result_summary(self, name: str, result: BenchmarkResult) -> None:
        """Print a summary of benchmark results."""
        print("\n" + "=" * 60)
        print(f"  {name.upper()} Benchmark Results")
        print("=" * 60)
        print(f"  Status: {result.status.value}")
        print(f"  Total Tasks: {result.metrics.total_tasks}")
        print(f"  Accuracy: {result.metrics.accuracy:.2%}")
        print(f"  Correct: {result.metrics.correct_tasks}")
        print(f"  Failed: {result.metrics.failed_tasks}")
        print(f"  Timeout: {result.metrics.timeout_tasks}")
        print(f"  Average Time: {result.metrics.average_time:.2f}s")
        print(f"  Total Time: {result.metrics.total_time:.2f}s")
        
        if result.metrics.accuracy_by_difficulty:
            print("\n  Accuracy by Difficulty:")
            for diff, acc in result.metrics.accuracy_by_difficulty.items():
                print(f"    {diff}: {acc:.2%}")
        
        print("=" * 60 + "\n")
    
    def _generate_suite_report(self, results: Dict[str, BenchmarkResult]) -> str:
        """Generate a comprehensive suite report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_benchmarks": len(BENCHMARK_REGISTRY),
            "benchmarks_run": len(results),
            "overall_metrics": self._calculate_overall_metrics(results),
            "benchmark_results": {},
        }
        
        for name, result in results.items():
            report["benchmark_results"][name] = {
                "accuracy": result.metrics.accuracy,
                "total_tasks": result.metrics.total_tasks,
                "correct_tasks": result.metrics.correct_tasks,
                "average_time": result.metrics.average_time,
                "status": result.status.value,
            }
        
        # Save report
        report_path = self.output_dir / f"suite_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print overall summary
        self._print_suite_summary(report)
        
        return str(report_path)
    
    def _calculate_overall_metrics(
        self,
        results: Dict[str, BenchmarkResult]
    ) -> Dict[str, Any]:
        """Calculate overall metrics across all benchmarks."""
        if not results:
            return {}
        
        total_tasks = sum(r.metrics.total_tasks for r in results.values())
        total_correct = sum(r.metrics.correct_tasks for r in results.values())
        total_time = sum(r.metrics.total_time for r in results.values())
        
        accuracies = [r.metrics.accuracy for r in results.values()]
        
        return {
            "overall_accuracy": total_correct / total_tasks if total_tasks > 0 else 0,
            "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
            "total_tasks": total_tasks,
            "total_correct": total_correct,
            "total_time": total_time,
        }
    
    def _print_suite_summary(self, report: Dict) -> None:
        """Print suite summary."""
        print("\n" + "=" * 70)
        print("  AGI BENCHMARK SUITE - OVERALL RESULTS")
        print("=" * 70)
        
        metrics = report.get("overall_metrics", {})
        
        print(f"\n  Benchmarks Run: {report['benchmarks_run']}/{report['total_benchmarks']}")
        print(f"  Overall Accuracy: {metrics.get('overall_accuracy', 0):.2%}")
        print(f"  Mean Accuracy: {metrics.get('mean_accuracy', 0):.2%}")
        print(f"  Total Tasks: {metrics.get('total_tasks', 0)}")
        print(f"  Total Time: {metrics.get('total_time', 0):.2f}s")
        
        print("\n  Individual Results:")
        print("-" * 70)
        
        for name, result in report.get("benchmark_results", {}).items():
            acc = result.get("accuracy", 0)
            tasks = result.get("total_tasks", 0)
            status = result.get("status", "unknown")
            
            status_symbol = "✓" if status == "completed" else "✗"
            print(f"    {status_symbol} {name:20} {acc:>8.2%} ({result.get('correct_tasks', 0)}/{tasks})")
        
        print("=" * 70 + "\n")


def create_agent_wrapper(cognitive_core):
    """
    Create an agent wrapper that adapts the cognitive core to the benchmark interface.
    
    The benchmark expects an agent with an execute_task(prompt, context) method.
    """
    class AgentWrapper:
        def __init__(self, core):
            self.core = core
        
        def execute_task(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
            """Execute a task using the cognitive core."""
            return self.core.execute_task(prompt, context or {})
    
    return AgentWrapper(cognitive_core)


def main():
    """CLI entry point for benchmark runner."""
    parser = argparse.ArgumentParser(
        description="M.I.A AGI Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run all benchmarks:
    python -m mia.benchmarks.runner --all

  Run specific benchmark:
    python -m mia.benchmarks.runner --benchmark arc-agi

  Run benchmark category:
    python -m mia.benchmarks.runner --category reasoning

  Run with limit:
    python -m mia.benchmarks.runner --benchmark gaia --limit 10

Available benchmarks:
  - arc-agi: Abstract reasoning
  - gaia: General assistant
  - swe-bench: Software engineering
  - gpqa: Graduate-level science
  - webvoyager: Web navigation
  - osworld: OS interaction
  - mmmu: Multimodal understanding

Categories:
  - reasoning: arc-agi, gpqa
  - coding: swe-bench
  - assistant: gaia
  - multimodal: mmmu
  - agentic: webvoyager, osworld
        """
    )
    
    parser.add_argument(
        "--benchmark",
        choices=list(BENCHMARK_REGISTRY.keys()),
        help="Specific benchmark to run"
    )
    parser.add_argument(
        "--category",
        choices=list(BENCHMARK_CATEGORIES.keys()),
        help="Run all benchmarks in category"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum tasks per benchmark"
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--data-dir",
        default="benchmarks/data",
        help="Data directory for benchmark datasets"
    )
    parser.add_argument(
        "--subset",
        help="Dataset subset to use"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmarks"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # List benchmarks if requested
    if args.list:
        print("\nAvailable Benchmarks:")
        print("-" * 60)
        for name in BENCHMARK_REGISTRY:
            category = "other"
            for cat, names in BENCHMARK_CATEGORIES.items():
                if name in names:
                    category = cat
                    break
            print(f"  {name:20} [{category}]")
        print()
        return 0
    
    # Validate arguments
    if not (args.benchmark or args.category or args.all):
        parser.error("Must specify --benchmark, --category, or --all")
    
    # Initialize agent (mock for CLI testing)
    try:
        from ..core.cognitive_architecture import MIACognitiveCore
        from ..llm.llm_manager import LLMManager
        
        # Try to initialize actual agent
        llm_manager = LLMManager()
        cognitive_core = MIACognitiveCore(llm_manager)
        agent = create_agent_wrapper(cognitive_core)
        
    except Exception as e:
        logger.warning(f"Could not initialize full agent: {e}")
        logger.info("Using mock agent for testing")
        
        # Mock agent for testing
        class MockAgent:
            def execute_task(self, prompt, context=None):
                return "Mock response for testing"
        
        agent = MockAgent()
    
    # Create runner
    runner = AGIBenchmarkRunner(
        agent=agent,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
    )
    
    # Run benchmarks
    try:
        if args.benchmark:
            runner.run_benchmark(
                args.benchmark,
                subset=args.subset,
                limit=args.limit,
            )
        elif args.category:
            runner.run_suite(
                category=args.category,
                limit_per_benchmark=args.limit,
            )
        elif args.all:
            runner.run_all(limit_per_benchmark=args.limit)
        
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark run failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
