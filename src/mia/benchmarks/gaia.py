"""
GAIA Benchmark Implementation

GAIA (General AI Assistants) benchmark evaluates AI assistants on
real-world tasks that require multiple capabilities:
- Web browsing and information retrieval
- Multi-step reasoning
- Tool use (calculator, code execution, etc.)
- File handling and document understanding

Reference: https://huggingface.co/datasets/gaia-benchmark/GAIA
Paper: https://arxiv.org/abs/2311.12983
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    BaseBenchmark,
    BenchmarkDifficulty,
    BenchmarkTask,
)

logger = logging.getLogger(__name__)


class GAIABenchmark(BaseBenchmark):
    """
    GAIA Benchmark for general AI assistant evaluation.
    
    Tests the agent's ability to:
    - Answer questions requiring web search
    - Process and understand documents
    - Use tools for calculations and code execution
    - Perform multi-step reasoning
    
    GAIA has three difficulty levels:
    - Level 1: Simple questions, 1-2 steps
    - Level 2: Moderate complexity, 3-5 steps
    - Level 3: Complex questions, 5+ steps, multiple tools
    """
    
    def __init__(
        self,
        agent: Any,
        config: Optional[Dict[str, Any]] = None,
        output_dir: str = "benchmarks/results",
        data_dir: str = "benchmarks/data/gaia",
    ):
        super().__init__(agent, config, output_dir)
        self.data_dir = Path(data_dir)
        self.max_steps = config.get("max_steps", 20) if config else 20
    
    @property
    def name(self) -> str:
        return "GAIA"
    
    @property
    def version(self) -> str:
        return "1.0"
    
    @property
    def description(self) -> str:
        return "General AI Assistant benchmark for real-world task completion"
    
    def load_tasks(
        self,
        subset: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[BenchmarkTask]:
        """
        Load GAIA tasks from the dataset.
        
        Levels:
        - validation: Public validation set with answers
        - test: Hidden test set (requires submission)
        """
        tasks = []
        
        # Default to validation subset
        subset = subset or "validation"
        
        # Try to load from HuggingFace datasets
        try:
            tasks = self._load_from_huggingface(subset, limit)
        except Exception as e:
            logger.warning(f"Could not load from HuggingFace: {e}")
            
            # Try local files
            if self.data_dir.exists():
                tasks = self._load_from_local(subset, limit)
            else:
                logger.warning("Creating synthetic GAIA tasks for testing")
                tasks = self._create_synthetic_tasks(limit or 5)
        
        return tasks
    
    def _load_from_huggingface(
        self,
        subset: str,
        limit: Optional[int]
    ) -> List[BenchmarkTask]:
        """Load GAIA from HuggingFace datasets."""
        from datasets import load_dataset
        
        dataset = load_dataset("gaia-benchmark/GAIA", "2023_all", trust_remote_code=True)
        
        tasks = []
        data = dataset[subset]
        
        if limit and hasattr(data, 'select'):
            data = data.select(range(min(limit, len(data))))  # type: ignore
        
        for item in data:  # type: ignore
            item_dict: Dict[str, Any] = dict(item) if hasattr(item, '__iter__') else item  # type: ignore
            difficulty = self._level_to_difficulty(item_dict.get("Level", 1))
            
            task = BenchmarkTask(
                task_id=item_dict.get("task_id", str(len(tasks))),
                input_data={
                    "question": item_dict.get("Question", ""),
                    "file_name": item_dict.get("file_name", ""),
                    "file_path": item_dict.get("file_path", ""),
                    "annotator_metadata": item_dict.get("Annotator Metadata", {}),
                },
                expected_output=item_dict.get("Final answer", ""),
                difficulty=difficulty,
                category=self._categorize_question(item_dict.get("Question", "")),
                metadata={
                    "level": item_dict.get("Level", 1),
                    "steps": item_dict.get("Steps", ""),
                },
                max_steps=self.max_steps,
                timeout_seconds=300.0 * item_dict.get("Level", 1),
            )
            tasks.append(task)
        
        return tasks
    
    def _load_from_local(
        self,
        subset: str,
        limit: Optional[int]
    ) -> List[BenchmarkTask]:
        """Load from local JSON files."""
        tasks = []
        
        subset_file = self.data_dir / f"{subset}.json"
        if subset_file.exists():
            with open(subset_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for i, item in enumerate(data):
                if limit and i >= limit:
                    break
                
                task = BenchmarkTask(
                    task_id=item.get("task_id", str(i)),
                    input_data={
                        "question": item.get("Question", ""),
                        "file_name": item.get("file_name", ""),
                    },
                    expected_output=item.get("Final answer", ""),
                    difficulty=self._level_to_difficulty(item.get("Level", 1)),
                    category=self._categorize_question(item.get("Question", "")),
                    metadata={"level": item.get("Level", 1)},
                )
                tasks.append(task)
        
        return tasks
    
    def execute_task(
        self,
        task: BenchmarkTask
    ) -> Tuple[Any, List[str], List[Dict[str, Any]]]:
        """
        Execute a GAIA task using the agent.
        
        The agent should:
        1. Understand the question
        2. Determine required tools/steps
        3. Execute the plan
        4. Provide a final answer
        """
        reasoning_trace: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        
        # Format prompt for GAIA
        prompt = self._format_gaia_prompt(task)
        
        try:
            context = {
                "task_type": "gaia_question",
                "max_steps": self.max_steps,
                "available_tools": [
                    "web_search",
                    "calculator",
                    "code_execution",
                    "file_reader",
                ],
            }
            
            # Add file context if available
            if task.input_data.get("file_path"):
                context["attached_file"] = task.input_data["file_path"]
            
            response = self.agent.execute_task(prompt, context)
            reasoning_trace.append(f"Agent response: {response}")
            
            # Extract final answer
            final_answer = self._extract_final_answer(response)
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            final_answer = ""
            reasoning_trace.append(f"Error: {e}")
        
        return final_answer, reasoning_trace, tool_calls
    
    def evaluate_output(self, task: BenchmarkTask, predicted_output: Any) -> bool:
        """
        Evaluate the predicted answer against the expected answer.
        
        GAIA uses exact match after normalization.
        """
        expected = str(task.expected_output).strip().lower()
        predicted = str(predicted_output).strip().lower()
        
        if not expected or not predicted:
            return False
        
        # Normalize both strings
        expected_norm = self._normalize_answer(expected)
        predicted_norm = self._normalize_answer(predicted)
        
        # Exact match
        if expected_norm == predicted_norm:
            return True
        
        # Check if predicted contains expected (for numeric answers)
        if expected_norm in predicted_norm:
            return True
        
        # Try numeric comparison
        try:
            exp_num = float(expected_norm.replace(",", ""))
            pred_num = float(predicted_norm.replace(",", ""))
            if abs(exp_num - pred_num) < 0.01:
                return True
        except ValueError:
            pass
        
        return False
    
    def _format_gaia_prompt(self, task: BenchmarkTask) -> str:
        """Format a GAIA task as a prompt."""
        prompt_parts = [
            "You are a helpful AI assistant. Answer the following question accurately.",
            "Use tools when necessary: web search, calculator, code execution, file reading.",
            "Think step by step and show your reasoning.",
            "",
            f"Question: {task.input_data['question']}",
        ]
        
        if task.input_data.get("file_name"):
            prompt_parts.append(f"\nAttached file: {task.input_data['file_name']}")
        
        prompt_parts.extend([
            "",
            "Provide your final answer in the format: FINAL ANSWER: <answer>",
            "Be precise and concise in your final answer.",
        ])
        
        return "\n".join(prompt_parts)
    
    def _extract_final_answer(self, response: str) -> str:
        """Extract the final answer from the response."""
        # Look for explicit final answer
        patterns = [
            r"FINAL ANSWER:\s*(.+?)(?:\n|$)",
            r"Final Answer:\s*(.+?)(?:\n|$)",
            r"The answer is:\s*(.+?)(?:\n|$)",
            r"Answer:\s*(.+?)(?:\n|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no explicit answer, take the last line
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        if lines:
            return lines[-1]
        
        return response.strip()
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize an answer for comparison."""
        # Remove common prefixes
        prefixes = ["the answer is", "answer:", "final answer:"]
        answer_lower = answer.lower()
        for prefix in prefixes:
            if answer_lower.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # Remove punctuation
        answer = re.sub(r'[^\w\s\d.-]', '', answer)
        
        # Normalize whitespace
        answer = " ".join(answer.split())
        
        return answer.lower()
    
    def _level_to_difficulty(self, level: int) -> BenchmarkDifficulty:
        """Convert GAIA level to difficulty."""
        mapping = {
            1: BenchmarkDifficulty.EASY,
            2: BenchmarkDifficulty.MEDIUM,
            3: BenchmarkDifficulty.HARD,
        }
        return mapping.get(level, BenchmarkDifficulty.UNKNOWN)
    
    def _categorize_question(self, question: str) -> str:
        """Categorize the question type."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["calculate", "compute", "math", "sum", "average"]):
            return "math"
        elif any(word in question_lower for word in ["search", "find", "look up", "who", "when", "where"]):
            return "retrieval"
        elif any(word in question_lower for word in ["code", "program", "function", "python"]):
            return "coding"
        elif any(word in question_lower for word in ["file", "document", "pdf", "image"]):
            return "document"
        else:
            return "reasoning"
    
    def _create_synthetic_tasks(self, num_tasks: int) -> List[BenchmarkTask]:
        """Create synthetic GAIA tasks for testing."""
        tasks = []
        
        synthetic_qa = [
            {
                "question": "What is the capital of France?",
                "answer": "Paris",
                "level": 1,
                "category": "retrieval",
            },
            {
                "question": "Calculate 15% of 240.",
                "answer": "36",
                "level": 1,
                "category": "math",
            },
            {
                "question": "What is the result of 2^10?",
                "answer": "1024",
                "level": 1,
                "category": "math",
            },
            {
                "question": "Who wrote the book '1984'?",
                "answer": "George Orwell",
                "level": 1,
                "category": "retrieval",
            },
            {
                "question": "What is the square root of 144?",
                "answer": "12",
                "level": 1,
                "category": "math",
            },
        ]
        
        for i, qa in enumerate(synthetic_qa[:num_tasks]):
            task = BenchmarkTask(
                task_id=f"synthetic_{i}",
                input_data={"question": qa["question"]},
                expected_output=qa["answer"],
                difficulty=self._level_to_difficulty(qa["level"]),
                category=qa["category"],
                metadata={"synthetic": True, "level": qa["level"]},
            )
            tasks.append(task)
        
        return tasks
