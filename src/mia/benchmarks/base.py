"""
Base classes and interfaces for AGI benchmarks.

Provides a standardized interface for all benchmark implementations,
ensuring consistent evaluation methodology and result reporting.
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class BenchmarkDifficulty(Enum):
    """Difficulty levels for benchmark tasks."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"
    UNKNOWN = "unknown"


class BenchmarkStatus(Enum):
    """Status of a benchmark run."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class BenchmarkTask:
    """Represents a single benchmark task/problem."""
    task_id: str
    input_data: Any
    expected_output: Optional[Any] = None
    difficulty: BenchmarkDifficulty = BenchmarkDifficulty.UNKNOWN
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_steps: int = 100
    timeout_seconds: float = 300.0


@dataclass
class TaskResult:
    """Result of executing a single task."""
    task_id: str
    status: BenchmarkStatus
    predicted_output: Any
    is_correct: bool
    execution_time: float
    steps_taken: int
    reasoning_trace: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics for a benchmark run."""
    accuracy: float
    total_tasks: int
    correct_tasks: int
    failed_tasks: int
    timeout_tasks: int
    average_time: float
    median_time: float
    total_time: float
    average_steps: float
    accuracy_by_difficulty: Dict[str, float] = field(default_factory=dict)
    accuracy_by_category: Dict[str, float] = field(default_factory=dict)
    pass_at_k: Dict[int, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "accuracy": self.accuracy,
            "total_tasks": self.total_tasks,
            "correct_tasks": self.correct_tasks,
            "failed_tasks": self.failed_tasks,
            "timeout_tasks": self.timeout_tasks,
            "average_time": self.average_time,
            "median_time": self.median_time,
            "total_time": self.total_time,
            "average_steps": self.average_steps,
            "accuracy_by_difficulty": self.accuracy_by_difficulty,
            "accuracy_by_category": self.accuracy_by_category,
            "pass_at_k": self.pass_at_k,
        }


@dataclass
class BenchmarkResult:
    """Complete result of a benchmark run."""
    benchmark_name: str
    version: str
    timestamp: str
    status: BenchmarkStatus
    metrics: BenchmarkMetrics
    task_results: List[TaskResult]
    config: Dict[str, Any] = field(default_factory=dict)
    system_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "version": self.version,
            "timestamp": self.timestamp,
            "status": self.status.value,
            "metrics": self.metrics.to_dict(),
            "task_results": [
                {
                    "task_id": r.task_id,
                    "status": r.status.value,
                    "is_correct": r.is_correct,
                    "execution_time": r.execution_time,
                    "steps_taken": r.steps_taken,
                    "error_message": r.error_message,
                }
                for r in self.task_results
            ],
            "config": self.config,
            "system_info": self.system_info,
        }
    
    def save(self, output_dir: str) -> str:
        """Save benchmark result to file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"{self.benchmark_name}_{self.timestamp.replace(':', '-')}.json"
        filepath = output_path / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        return str(filepath)


class BaseBenchmark(ABC):
    """
    Abstract base class for all AGI benchmarks.
    
    Implements the Template Method pattern for consistent benchmark execution.
    Subclasses must implement the abstract methods for loading data,
    executing tasks, and evaluating results.
    """
    
    def __init__(
        self,
        agent: Any,
        config: Optional[Dict[str, Any]] = None,
        output_dir: str = "benchmarks/results",
    ):
        """
        Initialize the benchmark.
        
        Args:
            agent: The AI agent to evaluate (must have execute_task method)
            config: Optional configuration overrides
            output_dir: Directory to save results
        """
        self.agent = agent
        self.config = config or {}
        self.output_dir = output_dir
        self.tasks: List[BenchmarkTask] = []
        self._status = BenchmarkStatus.NOT_STARTED
        
        # Validate agent interface
        if not hasattr(agent, "execute_task"):
            raise ValueError("Agent must have 'execute_task' method")
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the benchmark name."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Return the benchmark version."""
        pass
    
    @property
    def description(self) -> str:
        """Return a description of the benchmark."""
        return ""
    
    @abstractmethod
    def load_tasks(self, subset: Optional[str] = None, limit: Optional[int] = None) -> List[BenchmarkTask]:
        """
        Load benchmark tasks from the dataset.
        
        Args:
            subset: Optional subset name (e.g., "train", "test", "validation")
            limit: Optional maximum number of tasks to load
            
        Returns:
            List of benchmark tasks
        """
        pass
    
    @abstractmethod
    def execute_task(self, task: BenchmarkTask) -> Tuple[Any, List[str], List[Dict[str, Any]]]:
        """
        Execute a single benchmark task.
        
        Args:
            task: The task to execute
            
        Returns:
            Tuple of (predicted_output, reasoning_trace, tool_calls)
        """
        pass
    
    @abstractmethod
    def evaluate_output(self, task: BenchmarkTask, predicted_output: Any) -> bool:
        """
        Evaluate if the predicted output is correct.
        
        Args:
            task: The original task
            predicted_output: The agent's output
            
        Returns:
            True if correct, False otherwise
        """
        pass
    
    def preprocess_task(self, task: BenchmarkTask) -> BenchmarkTask:
        """
        Optional preprocessing hook for tasks.
        Override in subclasses for custom preprocessing.
        """
        return task
    
    def postprocess_output(self, output: Any) -> Any:
        """
        Optional postprocessing hook for outputs.
        Override in subclasses for custom postprocessing.
        """
        return output
    
    def run(
        self,
        subset: Optional[str] = None,
        limit: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BenchmarkResult:
        """
        Run the complete benchmark evaluation.
        
        Args:
            subset: Optional subset to run on
            limit: Optional limit on number of tasks
            progress_callback: Optional callback for progress updates
            
        Returns:
            Complete benchmark results
        """
        self._status = BenchmarkStatus.RUNNING
        logger.info(f"Starting {self.name} benchmark (v{self.version})")
        
        # Load tasks
        self.tasks = self.load_tasks(subset=subset, limit=limit)
        logger.info(f"Loaded {len(self.tasks)} tasks")
        
        # Execute tasks
        task_results: List[TaskResult] = []
        start_time = time.time()
        
        for idx, task in enumerate(self.tasks):
            logger.info(f"Running task {idx + 1}/{len(self.tasks)}: {task.task_id}")
            
            if progress_callback:
                progress_callback(idx, len(self.tasks))
            
            result = self._run_single_task(task)
            task_results.append(result)
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self._calculate_metrics(task_results, total_time)
        
        # Build result
        result = BenchmarkResult(
            benchmark_name=self.name,
            version=self.version,
            timestamp=datetime.now().isoformat(),
            status=BenchmarkStatus.COMPLETED,
            metrics=metrics,
            task_results=task_results,
            config=self.config,
            system_info=self._get_system_info(),
        )
        
        # Save result
        result_path = result.save(self.output_dir)
        logger.info(f"Benchmark complete. Results saved to: {result_path}")
        
        self._status = BenchmarkStatus.COMPLETED
        return result
    
    def _run_single_task(self, task: BenchmarkTask) -> TaskResult:
        """Execute a single task with timeout and error handling."""
        task = self.preprocess_task(task)
        
        start_time = time.time()
        status = BenchmarkStatus.COMPLETED
        predicted_output = None
        reasoning_trace: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        error_message = None
        is_correct = False
        steps_taken = 0
        
        try:
            # Execute with timeout
            predicted_output, reasoning_trace, tool_calls = self.execute_task(task)
            predicted_output = self.postprocess_output(predicted_output)
            steps_taken = len(reasoning_trace)
            
            # Evaluate
            is_correct = self.evaluate_output(task, predicted_output)
            
        except TimeoutError:
            status = BenchmarkStatus.TIMEOUT
            error_message = "Task execution timed out"
            logger.warning(f"Task {task.task_id} timed out")
            
        except Exception as e:
            status = BenchmarkStatus.FAILED
            error_message = str(e)
            logger.error(f"Task {task.task_id} failed: {e}")
        
        execution_time = time.time() - start_time
        
        return TaskResult(
            task_id=task.task_id,
            status=status,
            predicted_output=predicted_output,
            is_correct=is_correct,
            execution_time=execution_time,
            steps_taken=steps_taken,
            reasoning_trace=reasoning_trace,
            tool_calls=tool_calls,
            error_message=error_message,
            metadata=task.metadata,
        )
    
    def _calculate_metrics(
        self,
        task_results: List[TaskResult],
        total_time: float
    ) -> BenchmarkMetrics:
        """Calculate aggregate metrics from task results."""
        import statistics
        
        correct = sum(1 for r in task_results if r.is_correct)
        failed = sum(1 for r in task_results if r.status == BenchmarkStatus.FAILED)
        timeout = sum(1 for r in task_results if r.status == BenchmarkStatus.TIMEOUT)
        
        times = [r.execution_time for r in task_results]
        steps = [r.steps_taken for r in task_results]
        
        # Calculate accuracy by difficulty
        accuracy_by_difficulty: Dict[str, float] = {}
        for difficulty in BenchmarkDifficulty:
            tasks = [r for r in task_results 
                     if self._get_task_difficulty(r.task_id) == difficulty]
            if tasks:
                accuracy_by_difficulty[difficulty.value] = sum(
                    1 for r in tasks if r.is_correct
                ) / len(tasks)
        
        # Calculate accuracy by category
        accuracy_by_category: Dict[str, float] = {}
        categories = set(
            r.metadata.get("category", "unknown") 
            for r in task_results
        )
        for category in categories:
            tasks = [r for r in task_results 
                     if r.metadata.get("category") == category]
            if tasks:
                accuracy_by_category[category] = sum(
                    1 for r in tasks if r.is_correct
                ) / len(tasks)
        
        return BenchmarkMetrics(
            accuracy=correct / len(task_results) if task_results else 0.0,
            total_tasks=len(task_results),
            correct_tasks=correct,
            failed_tasks=failed,
            timeout_tasks=timeout,
            average_time=statistics.mean(times) if times else 0.0,
            median_time=statistics.median(times) if times else 0.0,
            total_time=total_time,
            average_steps=statistics.mean(steps) if steps else 0.0,
            accuracy_by_difficulty=accuracy_by_difficulty,
            accuracy_by_category=accuracy_by_category,
        )
    
    def _get_task_difficulty(self, task_id: str) -> BenchmarkDifficulty:
        """Get difficulty for a task by ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task.difficulty
        return BenchmarkDifficulty.UNKNOWN
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information for reproducibility."""
        import platform
        
        info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
        }
        
        try:
            import torch
            info["torch_version"] = torch.__version__
            info["cuda_available"] = str(torch.cuda.is_available())
            if torch.cuda.is_available():
                info["cuda_version"] = str(torch.version.cuda) if torch.version.cuda else "N/A"
                info["gpu_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            pass
        
        return info


class BenchmarkSuite:
    """
    Run multiple benchmarks as a suite.
    
    Provides unified execution and reporting across all benchmarks.
    """
    
    def __init__(
        self,
        agent: Any,
        benchmarks: Optional[List[BaseBenchmark]] = None,
        output_dir: str = "benchmarks/results",
    ):
        self.agent = agent
        self.benchmarks = benchmarks or []
        self.output_dir = output_dir
        self.results: Dict[str, BenchmarkResult] = {}
    
    def add_benchmark(self, benchmark: BaseBenchmark) -> None:
        """Add a benchmark to the suite."""
        self.benchmarks.append(benchmark)
    
    def run_all(
        self,
        limit_per_benchmark: Optional[int] = None
    ) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks in the suite."""
        logger.info(f"Running benchmark suite with {len(self.benchmarks)} benchmarks")
        
        for benchmark in self.benchmarks:
            try:
                result = benchmark.run(limit=limit_per_benchmark)
                self.results[benchmark.name] = result
            except Exception as e:
                logger.error(f"Benchmark {benchmark.name} failed: {e}")
                continue
        
        # Save combined report
        self._save_suite_report()
        
        return self.results
    
    def _save_suite_report(self) -> str:
        """Save a combined report for all benchmarks."""
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_benchmarks": len(self.benchmarks),
            "completed_benchmarks": len(self.results),
            "summary": {},
            "detailed_results": {},
        }
        
        for name, result in self.results.items():
            report["summary"][name] = {
                "accuracy": result.metrics.accuracy,
                "total_tasks": result.metrics.total_tasks,
                "status": result.status.value,
            }
            report["detailed_results"][name] = result.to_dict()
        
        filepath = output_path / f"suite_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        
        return str(filepath)
