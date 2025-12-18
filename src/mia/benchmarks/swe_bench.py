"""
SWE-Bench Benchmark Implementation

SWE-bench evaluates language models on their ability to resolve
real GitHub issues from popular Python repositories.

The agent must:
1. Understand the issue description
2. Navigate the codebase
3. Identify the relevant files
4. Generate a valid patch

Reference: https://www.swebench.com/
Paper: https://arxiv.org/abs/2310.06770
Dataset: https://huggingface.co/datasets/princeton-nlp/SWE-bench
"""

import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    BaseBenchmark,
    BenchmarkDifficulty,
    BenchmarkTask,
)

logger = logging.getLogger(__name__)


class SWEBenchBenchmark(BaseBenchmark):
    """
    SWE-bench for evaluating software engineering capabilities.
    
    Tests the agent's ability to:
    - Understand code and issue descriptions
    - Navigate large codebases
    - Generate correct patches
    - Fix bugs and implement features
    
    Variants:
    - SWE-bench Full: Complete dataset (~2294 instances)
    - SWE-bench Lite: Filtered subset (~300 instances)
    - SWE-bench Verified: Human-verified subset (~500 instances)
    """
    
    def __init__(
        self,
        agent: Any,
        config: Optional[Dict[str, Any]] = None,
        output_dir: str = "benchmarks/results",
        data_dir: str = "benchmarks/data/swe-bench",
    ):
        super().__init__(agent, config, output_dir)
        self.data_dir = Path(data_dir)
        self.variant = config.get("variant", "lite") if config else "lite"
        self.max_tokens = config.get("max_tokens", 8192) if config else 8192
    
    @property
    def name(self) -> str:
        return f"SWE-bench-{self.variant}"
    
    @property
    def version(self) -> str:
        return "1.0"
    
    @property
    def description(self) -> str:
        return "Software Engineering benchmark for code understanding and generation"
    
    def load_tasks(
        self,
        subset: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[BenchmarkTask]:
        """
        Load SWE-bench tasks.
        
        Args:
            subset: 'test' or 'dev'
            limit: Maximum number of tasks
        """
        tasks = []
        subset = subset or "test"
        
        try:
            tasks = self._load_from_huggingface(subset, limit)
        except Exception as e:
            logger.warning(f"Could not load from HuggingFace: {e}")
            
            if self.data_dir.exists():
                tasks = self._load_from_local(subset, limit)
            else:
                logger.warning("Creating synthetic SWE-bench tasks for testing")
                tasks = self._create_synthetic_tasks(limit or 5)
        
        return tasks
    
    def _load_from_huggingface(
        self,
        subset: str,
        limit: Optional[int]
    ) -> List[BenchmarkTask]:
        """Load from HuggingFace datasets."""
        from datasets import load_dataset
        
        # Select dataset variant
        if self.variant == "lite":
            dataset_name = "princeton-nlp/SWE-bench_Lite"
        elif self.variant == "verified":
            dataset_name = "princeton-nlp/SWE-bench_Verified"
        else:
            dataset_name = "princeton-nlp/SWE-bench"
        
        dataset = load_dataset(dataset_name, split=subset)
        
        tasks = []
        if limit and hasattr(dataset, 'select'):
            dataset = dataset.select(range(min(limit, len(dataset))))  # type: ignore
        
        for item in dataset:  # type: ignore
            item_dict: Dict[str, Any] = dict(item) if hasattr(item, '__iter__') else item  # type: ignore
            difficulty = self._estimate_difficulty(item_dict)
            
            task = BenchmarkTask(
                task_id=item_dict.get("instance_id", str(len(tasks))),
                input_data={
                    "repo": item_dict.get("repo", ""),
                    "base_commit": item_dict.get("base_commit", ""),
                    "problem_statement": item_dict.get("problem_statement", ""),
                    "hints_text": item_dict.get("hints_text", ""),
                    "created_at": item_dict.get("created_at", ""),
                    "version": item_dict.get("version", ""),
                    "environment_setup_commit": item_dict.get("environment_setup_commit", ""),
                },
                expected_output=item_dict.get("patch", ""),
                difficulty=difficulty,
                category=item_dict.get("repo", "").split("/")[-1] if item_dict.get("repo") else "unknown",
                metadata={
                    "repo": item_dict.get("repo", ""),
                    "fail_to_pass": item_dict.get("FAIL_TO_PASS", ""),
                    "pass_to_pass": item_dict.get("PASS_TO_PASS", ""),
                },
                max_steps=100,
                timeout_seconds=600.0,
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
        
        subset_file = self.data_dir / f"{self.variant}_{subset}.json"
        if not subset_file.exists():
            subset_file = self.data_dir / f"{subset}.json"
        
        if subset_file.exists():
            with open(subset_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for i, item in enumerate(data):
                if limit and i >= limit:
                    break
                
                task = BenchmarkTask(
                    task_id=item.get("instance_id", str(i)),
                    input_data={
                        "repo": item.get("repo", ""),
                        "problem_statement": item.get("problem_statement", ""),
                        "base_commit": item.get("base_commit", ""),
                    },
                    expected_output=item.get("patch", ""),
                    difficulty=self._estimate_difficulty(item),
                    category=item.get("repo", "").split("/")[-1] if item.get("repo") else "unknown",
                )
                tasks.append(task)
        
        return tasks
    
    def execute_task(
        self,
        task: BenchmarkTask
    ) -> Tuple[Any, List[str], List[Dict[str, Any]]]:
        """
        Execute a SWE-bench task.
        
        The agent should:
        1. Understand the issue
        2. Explore the codebase
        3. Identify relevant files
        4. Generate a patch
        """
        reasoning_trace: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        
        prompt = self._format_swe_prompt(task)
        
        try:
            context = {
                "task_type": "code_repair",
                "repo": task.input_data.get("repo", ""),
                "available_tools": [
                    "code_search",
                    "file_reader",
                    "code_execution",
                    "git_operations",
                ],
            }
            
            response = self.agent.execute_task(prompt, context)
            reasoning_trace.append(f"Agent response: {response[:1000]}...")
            
            # Extract patch from response
            patch = self._extract_patch(response)
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            patch = ""
            reasoning_trace.append(f"Error: {e}")
        
        return patch, reasoning_trace, tool_calls
    
    def evaluate_output(self, task: BenchmarkTask, predicted_output: Any) -> bool:
        """
        Evaluate the generated patch.
        
        For full evaluation, we would need to:
        1. Apply the patch to the repo
        2. Run the test suite
        3. Check if failing tests now pass
        
        Here we use a simplified evaluation based on patch similarity.
        """
        expected_patch = str(task.expected_output).strip()
        predicted_patch = str(predicted_output).strip()
        
        if not expected_patch or not predicted_patch:
            return False
        
        # Extract modified files from both patches
        expected_files = self._extract_modified_files(expected_patch)
        predicted_files = self._extract_modified_files(predicted_patch)
        
        # Check if the same files are modified
        if not expected_files or not predicted_files:
            return False
        
        file_overlap = len(expected_files & predicted_files) / len(expected_files)
        
        # For a simplified evaluation, require at least 50% file overlap
        # and some code changes in the predicted patch
        if file_overlap >= 0.5 and len(predicted_patch) > 50:
            # Check for key code patterns from expected patch
            expected_lines = set(self._extract_code_lines(expected_patch))
            predicted_lines = set(self._extract_code_lines(predicted_patch))
            
            if expected_lines and predicted_lines:
                code_overlap = len(expected_lines & predicted_lines) / len(expected_lines)
                return code_overlap >= 0.3
        
        return False
    
    def _format_swe_prompt(self, task: BenchmarkTask) -> str:
        """Format a SWE-bench task as a prompt."""
        prompt_parts = [
            "You are a software engineer tasked with fixing a bug in a Python project.",
            "",
            f"Repository: {task.input_data.get('repo', 'unknown')}",
            "",
            "=== Issue Description ===",
            task.input_data.get("problem_statement", "No description provided"),
            "",
        ]
        
        if task.input_data.get("hints_text"):
            prompt_parts.extend([
                "=== Hints ===",
                task.input_data["hints_text"],
                "",
            ])
        
        prompt_parts.extend([
            "=== Instructions ===",
            "1. Analyze the issue description to understand what needs to be fixed",
            "2. Search the codebase to find relevant files",
            "3. Identify the root cause of the bug",
            "4. Generate a patch that fixes the issue",
            "",
            "Provide your patch in unified diff format:",
            "```diff",
            "--- a/path/to/file.py",
            "+++ b/path/to/file.py",
            "@@ -line,count +line,count @@",
            " context line",
            "-removed line",
            "+added line",
            " context line",
            "```",
        ])
        
        return "\n".join(prompt_parts)
    
    def _extract_patch(self, response: str) -> str:
        """Extract a patch from the response."""
        # Look for diff blocks
        diff_pattern = r"```(?:diff)?\n((?:---|\+\+\+|@@|[-+ ].*\n)+)```"
        matches = re.findall(diff_pattern, response, re.MULTILINE)
        
        if matches:
            return "\n".join(matches)
        
        # Look for unified diff format
        lines = response.split("\n")
        diff_lines = []
        in_diff = False
        
        for line in lines:
            if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
                in_diff = True
            if in_diff:
                diff_lines.append(line)
                if line and not (line.startswith(("-", "+", " ", "@", "---", "+++")) or line.strip() == ""):
                    in_diff = False
        
        return "\n".join(diff_lines) if diff_lines else response
    
    def _extract_modified_files(self, patch: str) -> set:
        """Extract the set of files modified by a patch."""
        files = set()
        
        # Look for file paths in diff headers
        pattern = r"(?:---|\+\+\+)\s+[ab]/(.+)"
        matches = re.findall(pattern, patch)
        files.update(matches)
        
        return files
    
    def _extract_code_lines(self, patch: str) -> List[str]:
        """Extract added/removed code lines from a patch."""
        lines = []
        
        for line in patch.split("\n"):
            if line.startswith("+") and not line.startswith("+++"):
                lines.append(line[1:].strip())
            elif line.startswith("-") and not line.startswith("---"):
                lines.append(line[1:].strip())
        
        return [l for l in lines if l and len(l) > 3]
    
    def _estimate_difficulty(self, item: Dict) -> BenchmarkDifficulty:
        """Estimate difficulty based on patch size and test count."""
        patch = item.get("patch", "")
        fail_to_pass = item.get("FAIL_TO_PASS", "")
        
        # Count lines changed
        added = patch.count("\n+") - patch.count("\n+++")
        removed = patch.count("\n-") - patch.count("\n---")
        total_changes = added + removed
        
        # Count affected tests
        test_count = fail_to_pass.count("test") if fail_to_pass else 0
        
        if total_changes < 10 and test_count <= 1:
            return BenchmarkDifficulty.EASY
        elif total_changes < 50 and test_count <= 3:
            return BenchmarkDifficulty.MEDIUM
        elif total_changes < 200:
            return BenchmarkDifficulty.HARD
        else:
            return BenchmarkDifficulty.EXPERT
    
    def _create_synthetic_tasks(self, num_tasks: int) -> List[BenchmarkTask]:
        """Create synthetic SWE-bench tasks for testing."""
        tasks = []
        
        synthetic_issues = [
            {
                "instance_id": "test__fix_typo",
                "repo": "test/repo",
                "problem_statement": "There's a typo in the greeting function. It says 'Helo' instead of 'Hello'.",
                "patch": """--- a/greeting.py
+++ b/greeting.py
@@ -1,3 +1,3 @@
 def greet(name):
-    return f"Helo, {name}!"
+    return f"Hello, {name}!"
""",
            },
            {
                "instance_id": "test__fix_off_by_one",
                "repo": "test/repo",
                "problem_statement": "The range function has an off-by-one error. It should include the last number.",
                "patch": """--- a/utils.py
+++ b/utils.py
@@ -1,3 +1,3 @@
 def get_numbers(n):
-    return list(range(n))
+    return list(range(n + 1))
""",
            },
        ]
        
        for i, issue in enumerate(synthetic_issues[:num_tasks]):
            task = BenchmarkTask(
                task_id=issue["instance_id"],
                input_data={
                    "repo": issue["repo"],
                    "problem_statement": issue["problem_statement"],
                },
                expected_output=issue["patch"],
                difficulty=BenchmarkDifficulty.EASY,
                category="test",
                metadata={"synthetic": True},
            )
            tasks.append(task)
        
        return tasks


def run_swe_bench_evaluation(
    predictions_path: str,
    swe_bench_path: str,
    log_dir: str = "logs/swe-bench",
) -> Dict[str, Any]:
    """
    Run the official SWE-bench evaluation harness.
    
    This requires the swe-bench package to be installed:
    pip install swe-bench
    
    Args:
        predictions_path: Path to predictions JSON file
        swe_bench_path: Path to SWE-bench dataset
        log_dir: Directory for evaluation logs
        
    Returns:
        Evaluation results
    """
    try:
        result = subprocess.run(
            [
                "python", "-m", "swebench.harness.run_evaluation",
                "--predictions_path", predictions_path,
                "--swe_bench_tasks", swe_bench_path,
                "--log_dir", log_dir,
                "--testbed", "testbed",
                "--skip_existing",
                "--timeout", "900",
            ],
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            # Parse results from log directory
            return {"status": "success", "output": result.stdout}
        else:
            return {"status": "error", "error": result.stderr}
            
    except Exception as e:
        return {"status": "error", "error": str(e)}
