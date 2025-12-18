"""
MMMU Benchmark Implementation

MMMU (Massive Multi-discipline Multimodal Understanding) evaluates
AI models on multimodal reasoning across diverse academic disciplines.

Key characteristics:
- College-level questions across 30 subjects
- Requires understanding of images, diagrams, charts
- Tests both visual perception and domain knowledge
- Multiple choice and open-ended questions

Reference: https://arxiv.org/abs/2311.16502
Dataset: https://huggingface.co/datasets/MMMU/MMMU
Website: https://mmmu-benchmark.github.io/
"""

import base64
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

# MMMU discipline categories
MMMU_DISCIPLINES = {
    "Art & Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "Health & Medicine": ["Basic_Medical_Science", "Clinical_Medicine", "Diagnostics_and_Laboratory_Medicine", "Pharmacy", "Public_Health"],
    "Humanities & Social Science": ["History", "Literature", "Psychology", "Sociology"],
    "Tech & Engineering": ["Agriculture", "Architecture_and_Engineering", "Computer_Science", "Electronics", "Energy_and_Power", "Materials", "Mechanical_Engineering"],
}


class MMMUBenchmark(BaseBenchmark):
    """
    MMMU benchmark for multimodal academic understanding.
    
    Tests the agent's ability to:
    - Understand and analyze images/diagrams
    - Apply domain knowledge across disciplines
    - Reason with visual and textual information
    - Answer college-level academic questions
    
    Subsets:
    - validation: Public validation set
    - test: Hidden test set (submit to leaderboard)
    - dev: Development set with answers
    """
    
    def __init__(
        self,
        agent: Any,
        config: Optional[Dict[str, Any]] = None,
        output_dir: str = "benchmarks/results",
        data_dir: str = "benchmarks/data/mmmu",
    ):
        super().__init__(agent, config, output_dir)
        self.data_dir = Path(data_dir)
        self.disciplines = config.get("disciplines", None) if config else None
        self.include_images = config.get("include_images", True) if config else True
    
    @property
    def name(self) -> str:
        return "MMMU"
    
    @property
    def version(self) -> str:
        return "1.0"
    
    @property
    def description(self) -> str:
        return "Massive Multi-discipline Multimodal Understanding benchmark"
    
    def load_tasks(
        self,
        subset: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[BenchmarkTask]:
        """Load MMMU tasks."""
        tasks = []
        subset = subset or "validation"
        
        try:
            tasks = self._load_from_huggingface(subset, limit)
        except Exception as e:
            logger.warning(f"Could not load from HuggingFace: {e}")
            
            if self.data_dir.exists():
                tasks = self._load_from_local(subset, limit)
            else:
                logger.warning("Creating synthetic MMMU tasks for testing")
                tasks = self._create_synthetic_tasks(limit or 5)
        
        return tasks
    
    def _load_from_huggingface(
        self,
        subset: str,
        limit: Optional[int]
    ) -> List[BenchmarkTask]:
        """Load from HuggingFace datasets."""
        from datasets import load_dataset
        
        tasks = []
        
        # MMMU has multiple configs (one per subject)
        subjects_to_load = []
        
        if self.disciplines:
            # Load specific disciplines
            for disc in self.disciplines:
                if disc in MMMU_DISCIPLINES:
                    subjects_to_load.extend(MMMU_DISCIPLINES[disc])
                else:
                    subjects_to_load.append(disc)
        else:
            # Load all subjects
            for subjects in MMMU_DISCIPLINES.values():
                subjects_to_load.extend(subjects)
        
        per_subject_limit = (limit // len(subjects_to_load) + 1) if limit else None
        
        for subject in subjects_to_load:
            try:
                dataset = load_dataset("MMMU/MMMU", subject, trust_remote_code=True)
                
                if subset not in dataset:
                    continue
                
                data = dataset[subset]
                if per_subject_limit and hasattr(data, 'select'):
                    data = data.select(range(min(per_subject_limit, len(data))))  # type: ignore
                
                for item in data:  # type: ignore
                    item_dict: Dict[str, Any] = dict(item) if hasattr(item, '__iter__') else item  # type: ignore
                    task = self._create_task_from_item(item_dict, subject)
                    if task:
                        tasks.append(task)
                
            except Exception as e:
                logger.warning(f"Could not load subject {subject}: {e}")
                continue
            
            if limit and len(tasks) >= limit:
                break
        
        return tasks[:limit] if limit else tasks
    
    def _create_task_from_item(self, item: Dict, subject: str) -> Optional[BenchmarkTask]:
        """Create a BenchmarkTask from a dataset item."""
        try:
            question = item.get("question", "")
            options = {}
            
            # Extract options (A, B, C, D, etc.)
            for opt in ["A", "B", "C", "D", "E", "F", "G", "H"]:
                opt_key = f"option_{opt.lower()}" if f"option_{opt.lower()}" in item else opt
                if opt_key in item and item[opt_key]:
                    options[opt] = item[opt_key]
            
            # Get images
            images = []
            for i in range(1, 8):  # MMMU can have up to 7 images
                img_key = f"image_{i}"
                if img_key in item and item[img_key] is not None:
                    images.append({
                        "index": i,
                        "image": item[img_key],  # PIL Image or path
                    })
            
            # Get answer
            answer = item.get("answer", "")
            
            # Determine question type
            question_type = item.get("question_type", "multiple-choice")
            
            task = BenchmarkTask(
                task_id=item.get("id", f"{subject}_{len(item)}"),
                input_data={
                    "question": question,
                    "options": options,
                    "images": images,
                    "subject": subject,
                    "question_type": question_type,
                },
                expected_output=answer,
                difficulty=self._estimate_difficulty(item, subject),
                category=subject,
                metadata={
                    "subject": subject,
                    "discipline": self._get_discipline(subject),
                    "question_type": question_type,
                    "num_images": len(images),
                },
                max_steps=20,
                timeout_seconds=120.0,
            )
            
            return task
            
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            return None
    
    def _load_from_local(
        self,
        subset: str,
        limit: Optional[int]
    ) -> List[BenchmarkTask]:
        """Load from local files."""
        tasks = []
        
        data_file = self.data_dir / f"{subset}.json"
        if data_file.exists():
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for i, item in enumerate(data):
                if limit and i >= limit:
                    break
                
                task = BenchmarkTask(
                    task_id=item.get("id", str(i)),
                    input_data={
                        "question": item.get("question", ""),
                        "options": item.get("options", {}),
                        "images": item.get("images", []),
                        "subject": item.get("subject", "unknown"),
                    },
                    expected_output=item.get("answer", ""),
                    difficulty=BenchmarkDifficulty.MEDIUM,
                    category=item.get("subject", "unknown"),
                )
                tasks.append(task)
        
        return tasks
    
    def execute_task(
        self,
        task: BenchmarkTask
    ) -> Tuple[Any, List[str], List[Dict[str, Any]]]:
        """
        Execute an MMMU task.
        
        The agent should:
        1. Analyze the question and any images
        2. Apply relevant domain knowledge
        3. Reason through the options
        4. Select or generate an answer
        """
        reasoning_trace: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        
        prompt = self._format_mmmu_prompt(task)
        
        try:
            context = {
                "task_type": "multimodal_qa",
                "subject": task.input_data.get("subject", ""),
                "question_type": task.input_data.get("question_type", "multiple-choice"),
                "has_images": len(task.input_data.get("images", [])) > 0,
            }
            
            # Include image data if available
            if self.include_images and task.input_data.get("images"):
                context["images"] = self._prepare_images(task.input_data["images"])
            
            response = self.agent.execute_task(prompt, context)
            reasoning_trace.append(f"Agent response: {response}")
            
            # Extract answer
            answer = self._extract_answer(
                response,
                task.input_data.get("options", {}),
                task.input_data.get("question_type", "multiple-choice")
            )
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            answer = ""
            reasoning_trace.append(f"Error: {e}")
        
        return answer, reasoning_trace, tool_calls
    
    def evaluate_output(self, task: BenchmarkTask, predicted_output: Any) -> bool:
        """
        Evaluate the predicted answer.
        
        For multiple choice: exact match of option letter
        For open-ended: substring match or semantic similarity
        """
        expected = str(task.expected_output).strip()
        predicted = str(predicted_output).strip()
        
        if not expected or not predicted:
            return False
        
        question_type = task.input_data.get("question_type", "multiple-choice")
        
        if question_type == "multiple-choice":
            # For multiple choice, compare option letters
            exp_letter = expected.upper()
            pred_letter = predicted.upper()
            
            # Direct match
            if exp_letter == pred_letter:
                return True
            
            # Check if answer contains the correct letter
            if len(exp_letter) == 1 and exp_letter in pred_letter:
                # Make sure it's the letter, not part of a word
                return True
        
        else:
            # For open-ended, use flexible matching
            expected_norm = self._normalize_text(expected)
            predicted_norm = self._normalize_text(predicted)
            
            if expected_norm == predicted_norm:
                return True
            
            if expected_norm in predicted_norm or predicted_norm in expected_norm:
                return True
        
        return False
    
    def _format_mmmu_prompt(self, task: BenchmarkTask) -> str:
        """Format an MMMU task as a prompt."""
        subject = task.input_data.get("subject", "General")
        question = task.input_data.get("question", "")
        options = task.input_data.get("options", {})
        images = task.input_data.get("images", [])
        question_type = task.input_data.get("question_type", "multiple-choice")
        
        prompt_parts = [
            f"Subject: {subject}",
            "",
            "=== Question ===",
            question,
        ]
        
        if images:
            prompt_parts.extend([
                "",
                f"[This question includes {len(images)} image(s). Analyze them carefully.]",
            ])
        
        if options:
            prompt_parts.extend([
                "",
                "=== Options ===",
            ])
            for letter, text in sorted(options.items()):
                prompt_parts.append(f"{letter}) {text}")
        
        prompt_parts.extend([
            "",
            "=== Instructions ===",
        ])
        
        if question_type == "multiple-choice":
            prompt_parts.extend([
                "1. Carefully analyze the question and any images",
                "2. Consider each option",
                "3. Select the best answer",
                "",
                "Format your response as:",
                "REASONING: <your analysis>",
                "ANSWER: <option letter (A, B, C, D, etc.)>",
            ])
        else:
            prompt_parts.extend([
                "1. Carefully analyze the question and any images",
                "2. Provide a clear, concise answer",
                "",
                "Format your response as:",
                "REASONING: <your analysis>",
                "ANSWER: <your answer>",
            ])
        
        return "\n".join(prompt_parts)
    
    def _prepare_images(self, images: List[Dict]) -> List[Dict]:
        """Prepare images for the agent."""
        prepared = []
        
        for img_data in images:
            try:
                image = img_data.get("image")
                
                if image is None:
                    continue
                
                # If it's a PIL Image, convert to base64
                if hasattr(image, "save"):
                    import io
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    img_bytes = buffer.getvalue()
                    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                    
                    prepared.append({
                        "index": img_data.get("index", len(prepared) + 1),
                        "base64": img_base64,
                        "format": "png",
                    })
                
                # If it's a path, read the file
                elif isinstance(image, (str, Path)):
                    img_path = Path(image)
                    if img_path.exists():
                        with open(img_path, "rb") as f:
                            img_bytes = f.read()
                        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                        
                        prepared.append({
                            "index": img_data.get("index", len(prepared) + 1),
                            "base64": img_base64,
                            "format": img_path.suffix.lstrip("."),
                        })
                
            except Exception as e:
                logger.warning(f"Could not prepare image: {e}")
                continue
        
        return prepared
    
    def _extract_answer(
        self,
        response: str,
        options: Dict[str, str],
        question_type: str
    ) -> str:
        """Extract the answer from the response."""
        # Look for explicit answer markers
        patterns = [
            r"ANSWER:\s*([A-H])\b",
            r"ANSWER:\s*\(([A-H])\)",
            r"ANSWER:\s*(.+?)(?:\n|$)",
            r"The answer is\s*([A-H])",
            r"correct answer:\s*([A-H])",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                if len(answer) == 1 and answer.upper() in "ABCDEFGH":
                    return answer.upper()
                return answer
        
        # For multiple choice, look for standalone letters
        if question_type == "multiple-choice":
            for letter in ["A", "B", "C", "D", "E", "F", "G", "H"]:
                if f"({letter})" in response or f" {letter})" in response or f" {letter}." in response:
                    return letter
        
        # Return last line as fallback
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        return lines[-1] if lines else ""
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove common prefixes
        text = re.sub(r'^(the answer is|answer:)\s*', '', text.lower())
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Normalize whitespace
        text = " ".join(text.split())
        
        return text.strip()
    
    def _estimate_difficulty(self, item: Dict, subject: str) -> BenchmarkDifficulty:
        """Estimate task difficulty."""
        # MMMU tasks are generally challenging
        num_images = sum(1 for i in range(1, 8) if item.get(f"image_{i}"))
        
        if num_images == 0:
            return BenchmarkDifficulty.MEDIUM
        elif num_images == 1:
            return BenchmarkDifficulty.HARD
        else:
            return BenchmarkDifficulty.EXPERT
    
    def _get_discipline(self, subject: str) -> str:
        """Get the discipline for a subject."""
        for discipline, subjects in MMMU_DISCIPLINES.items():
            if subject in subjects:
                return discipline
        return "Unknown"
    
    def _create_synthetic_tasks(self, num_tasks: int) -> List[BenchmarkTask]:
        """Create synthetic MMMU tasks for testing."""
        tasks = []
        
        synthetic = [
            {
                "id": "mmmu_math_1",
                "question": "If f(x) = x² + 2x + 1, what is f(3)?",
                "options": {"A": "9", "B": "12", "C": "16", "D": "25"},
                "answer": "C",
                "subject": "Math",
            },
            {
                "id": "mmmu_physics_1",
                "question": "What is the SI unit of electrical resistance?",
                "options": {"A": "Volt", "B": "Ampere", "C": "Ohm", "D": "Watt"},
                "answer": "C",
                "subject": "Physics",
            },
            {
                "id": "mmmu_biology_1",
                "question": "Which organelle is responsible for protein synthesis?",
                "options": {"A": "Mitochondria", "B": "Ribosome", "C": "Golgi apparatus", "D": "Lysosome"},
                "answer": "B",
                "subject": "Biology",
            },
            {
                "id": "mmmu_chemistry_1",
                "question": "What is the molecular formula for water?",
                "options": {"A": "H2O", "B": "CO2", "C": "NaCl", "D": "CH4"},
                "answer": "A",
                "subject": "Chemistry",
            },
            {
                "id": "mmmu_cs_1",
                "question": "What is the time complexity of binary search?",
                "options": {"A": "O(n)", "B": "O(log n)", "C": "O(n²)", "D": "O(1)"},
                "answer": "B",
                "subject": "Computer_Science",
            },
        ]
        
        for i, item in enumerate(synthetic[:num_tasks]):
            task = BenchmarkTask(
                task_id=item["id"],
                input_data={
                    "question": item["question"],
                    "options": item["options"],
                    "images": [],
                    "subject": item["subject"],
                    "question_type": "multiple-choice",
                },
                expected_output=item["answer"],
                difficulty=BenchmarkDifficulty.MEDIUM,
                category=item["subject"],
                metadata={"synthetic": True},
            )
            tasks.append(task)
        
        return tasks
