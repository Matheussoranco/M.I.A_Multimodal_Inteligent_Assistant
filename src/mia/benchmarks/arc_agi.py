"""
ARC-AGI Benchmark Implementation

The Abstraction and Reasoning Corpus (ARC) measures an AI system's ability
to efficiently acquire new skills, testing fluid intelligence and abstract reasoning.

ARC-AGI-2 (2024) is the current gold standard for measuring general intelligence,
with tasks requiring the model to:
1. Observe input-output examples
2. Identify abstract patterns and transformations
3. Apply the discovered rule to new inputs

Reference: https://arcprize.org/
Dataset: https://github.com/fchollet/ARC-AGI
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import (
    BaseBenchmark,
    BenchmarkDifficulty,
    BenchmarkTask,
)

logger = logging.getLogger(__name__)

# ARC grid constants
MAX_GRID_SIZE = 30
NUM_COLORS = 10  # 0-9 color palette


class ARCAGIBenchmark(BaseBenchmark):
    """
    ARC-AGI Benchmark for abstract reasoning evaluation.
    
    Tests the agent's ability to:
    - Identify patterns from few examples
    - Apply abstract transformations
    - Generalize to novel inputs
    """
    
    def __init__(
        self,
        agent: Any,
        config: Optional[Dict[str, Any]] = None,
        output_dir: str = "benchmarks/results",
        data_dir: str = "benchmarks/data/arc-agi",
    ):
        super().__init__(agent, config, output_dir)
        self.data_dir = Path(data_dir)
        self.num_attempts = config.get("num_attempts", 2) if config else 2
    
    @property
    def name(self) -> str:
        return "ARC-AGI"
    
    @property
    def version(self) -> str:
        return "2.0"
    
    @property
    def description(self) -> str:
        return "Abstract Reasoning Corpus for measuring fluid intelligence"
    
    def load_tasks(
        self,
        subset: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[BenchmarkTask]:
        """
        Load ARC tasks from JSON files.
        
        ARC structure:
        - training/: Training examples
        - evaluation/: Evaluation examples
        """
        tasks = []
        
        # Default to evaluation subset
        subset = subset or "evaluation"
        subset_dir = self.data_dir / subset
        
        if not subset_dir.exists():
            logger.warning(f"ARC data not found at {subset_dir}. Attempting to download...")
            self._download_dataset()
        
        if subset_dir.exists():
            task_files = list(subset_dir.glob("*.json"))
            if limit:
                task_files = task_files[:limit]
            
            for task_file in task_files:
                try:
                    with open(task_file, "r") as f:
                        task_data = json.load(f)
                    
                    task = BenchmarkTask(
                        task_id=task_file.stem,
                        input_data={
                            "train": task_data.get("train", []),
                            "test": task_data.get("test", []),
                        },
                        expected_output=[t.get("output") for t in task_data.get("test", [])],
                        difficulty=self._estimate_difficulty(task_data),
                        category=self._categorize_task(task_data),
                        metadata={"source_file": str(task_file)},
                        max_steps=50,
                        timeout_seconds=120.0,
                    )
                    tasks.append(task)
                except Exception as e:
                    logger.error(f"Failed to load task {task_file}: {e}")
        else:
            # Create synthetic tasks for testing
            logger.warning("Creating synthetic ARC tasks for testing")
            tasks = self._create_synthetic_tasks(limit or 5)
        
        return tasks
    
    def execute_task(
        self,
        task: BenchmarkTask
    ) -> Tuple[Any, List[str], List[Dict[str, Any]]]:
        """
        Execute an ARC task using the agent's reasoning capabilities.
        
        The agent receives:
        - Training examples (input-output pairs)
        - Test inputs
        
        Must return:
        - Predicted output grids for each test input
        """
        reasoning_trace: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        
        # Format prompt for the agent
        prompt = self._format_arc_prompt(task)
        
        # Execute with the agent
        try:
            context = {
                "task_type": "arc_reasoning",
                "num_attempts": self.num_attempts,
                "training_examples": task.input_data["train"],
                "test_inputs": [t.get("input") for t in task.input_data["test"]],
            }
            
            response = self.agent.execute_task(prompt, context)
            reasoning_trace.append(f"Agent response: {response}")
            
            # Parse the response to extract grids
            predicted_outputs = self._parse_arc_response(response, task)
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            predicted_outputs = []
            reasoning_trace.append(f"Error: {e}")
        
        return predicted_outputs, reasoning_trace, tool_calls
    
    def evaluate_output(self, task: BenchmarkTask, predicted_output: Any) -> bool:
        """
        Evaluate if the predicted grids match the expected outputs.
        
        ARC uses exact match evaluation - the entire grid must be correct.
        """
        expected = task.expected_output
        
        if not predicted_output or not expected:
            return False
        
        if len(predicted_output) != len(expected):
            return False
        
        for pred, exp in zip(predicted_output, expected):
            if not self._grids_equal(pred, exp):
                return False
        
        return True
    
    def _format_arc_prompt(self, task: BenchmarkTask) -> str:
        """Format an ARC task as a prompt for the agent."""
        prompt_parts = [
            "You are solving an ARC (Abstract Reasoning Corpus) task.",
            "Observe the training examples to identify the transformation pattern.",
            "Then apply the same pattern to the test input(s).",
            "",
            "=== Training Examples ===",
        ]
        
        for i, example in enumerate(task.input_data["train"], 1):
            prompt_parts.append(f"\nExample {i}:")
            prompt_parts.append(f"Input:\n{self._grid_to_string(example.get('input', []))}")
            prompt_parts.append(f"Output:\n{self._grid_to_string(example.get('output', []))}")
        
        prompt_parts.append("\n=== Test Input(s) ===")
        for i, test in enumerate(task.input_data["test"], 1):
            prompt_parts.append(f"\nTest {i}:")
            prompt_parts.append(f"Input:\n{self._grid_to_string(test.get('input', []))}")
        
        prompt_parts.extend([
            "",
            "Analyze the pattern and provide the output grid(s) for the test input(s).",
            "Format your answer as a JSON array of 2D arrays.",
            "Example format: [[0,1,2],[3,4,5]] for a 2x3 grid.",
        ])
        
        return "\n".join(prompt_parts)
    
    def _grid_to_string(self, grid: List[List[int]]) -> str:
        """Convert a grid to a readable string representation."""
        if not grid:
            return "[]"
        
        # Use symbols for colors
        symbols = "⬛🔴🟢🟡🔵🟣🟠⬜🩵🩷"
        
        lines = []
        for row in grid:
            line = "".join(symbols[min(c, 9)] for c in row)
            lines.append(line)
        
        return "\n".join(lines)
    
    def _parse_arc_response(
        self,
        response: str,
        task: BenchmarkTask
    ) -> List[List[List[int]]]:
        """Parse the agent's response to extract output grids."""
        import re
        
        outputs = []
        
        # Try to find JSON arrays in the response
        json_pattern = r'\[\s*\[[\d,\s\[\]]+\]\s*\]'
        matches = re.findall(json_pattern, response)
        
        for match in matches:
            try:
                grid = json.loads(match)
                if self._is_valid_grid(grid):
                    outputs.append(grid)
            except json.JSONDecodeError:
                continue
        
        # If we couldn't parse enough grids, return empty
        expected_count = len(task.input_data["test"])
        if len(outputs) < expected_count:
            logger.warning(f"Expected {expected_count} grids, found {len(outputs)}")
        
        return outputs[:expected_count]
    
    def _is_valid_grid(self, grid: Any) -> bool:
        """Check if a grid is valid (2D array of integers 0-9)."""
        if not isinstance(grid, list):
            return False
        if not grid:
            return False
        if not all(isinstance(row, list) for row in grid):
            return False
        
        try:
            for row in grid:
                for cell in row:
                    if not isinstance(cell, int) or cell < 0 or cell > 9:
                        return False
            return True
        except:
            return False
    
    def _grids_equal(self, grid1: Any, grid2: Any) -> bool:
        """Check if two grids are exactly equal."""
        if grid1 is None or grid2 is None:
            return False
        
        try:
            return np.array_equal(np.array(grid1), np.array(grid2))
        except:
            return False
    
    def _estimate_difficulty(self, task_data: Dict) -> BenchmarkDifficulty:
        """Estimate task difficulty based on grid size and complexity."""
        train = task_data.get("train", [])
        test = task_data.get("test", [])
        
        if not train or not test:
            return BenchmarkDifficulty.UNKNOWN
        
        # Calculate average grid size
        sizes = []
        for example in train + test:
            for key in ["input", "output"]:
                grid = example.get(key, [])
                if grid:
                    sizes.append(len(grid) * len(grid[0]) if grid else 0)
        
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        num_examples = len(train)
        
        if avg_size < 25 and num_examples >= 3:
            return BenchmarkDifficulty.EASY
        elif avg_size < 100 and num_examples >= 2:
            return BenchmarkDifficulty.MEDIUM
        elif avg_size < 400:
            return BenchmarkDifficulty.HARD
        else:
            return BenchmarkDifficulty.EXPERT
    
    def _categorize_task(self, task_data: Dict) -> str:
        """Categorize the task based on transformation type."""
        # This is a simplified categorization
        # In practice, would use more sophisticated analysis
        return "transformation"
    
    def _download_dataset(self) -> None:
        """Download the ARC dataset from GitHub."""
        logger.info("Downloading ARC-AGI dataset...")
        
        import urllib.request
        import zipfile
        import io
        
        url = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
        
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Download and extract
            with urllib.request.urlopen(url, timeout=60) as response:
                with zipfile.ZipFile(io.BytesIO(response.read())) as zf:
                    for member in zf.namelist():
                        if "evaluation/" in member or "training/" in member:
                            # Extract to data directory
                            target = self.data_dir / member.split("/", 2)[-1]
                            if member.endswith("/"):
                                target.mkdir(parents=True, exist_ok=True)
                            else:
                                target.parent.mkdir(parents=True, exist_ok=True)
                                with open(target, "wb") as f:
                                    f.write(zf.read(member))
            
            logger.info(f"Dataset downloaded to {self.data_dir}")
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
    
    def _create_synthetic_tasks(self, num_tasks: int) -> List[BenchmarkTask]:
        """Create synthetic ARC tasks for testing."""
        tasks = []
        
        # Simple transformation: fill solid color
        for i in range(num_tasks):
            size = np.random.randint(3, 8)
            color = np.random.randint(1, 10)
            
            train = []
            for _ in range(3):
                s = np.random.randint(3, 6)
                input_grid = np.random.randint(0, 10, (s, s)).tolist()
                output_grid = [[color] * s for _ in range(s)]
                train.append({"input": input_grid, "output": output_grid})
            
            test_input = np.random.randint(0, 10, (size, size)).tolist()
            test_output = [[color] * size for _ in range(size)]
            
            task = BenchmarkTask(
                task_id=f"synthetic_{i}",
                input_data={
                    "train": train,
                    "test": [{"input": test_input}],
                },
                expected_output=[test_output],
                difficulty=BenchmarkDifficulty.EASY,
                category="fill",
                metadata={"synthetic": True},
            )
            tasks.append(task)
        
        return tasks
