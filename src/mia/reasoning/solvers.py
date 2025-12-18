"""
Specialized Problem Solvers for AGI Benchmarks

Implements domain-specific solving strategies optimized for
different benchmark types:

- Abstract Reasoning (ARC-AGI)
- Scientific Reasoning (GPQA)
- Code Understanding (SWE-BENCH)
- Multimodal Understanding (MMMU)
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SolverResult:
    """Result from a specialized solver."""
    answer: Any
    confidence: float
    reasoning: str
    method: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class AbstractReasoningSolver:
    """
    Solver optimized for abstract visual reasoning tasks (ARC-AGI).
    
    Uses pattern recognition, transformation detection, and 
    systematic hypothesis testing.
    """
    
    def __init__(self, llm: Any):
        self.llm = llm
    
    def solve(
        self,
        input_grid: List[List[int]],
        output_grid: Optional[List[List[int]]] = None,
        examples: Optional[List[Dict]] = None,
    ) -> SolverResult:
        """
        Solve an abstract reasoning task.
        
        Args:
            input_grid: The input grid to transform
            output_grid: Expected output (for training)
            examples: Input-output example pairs
        """
        # Analyze examples to find patterns
        patterns = self._analyze_patterns(examples) if examples else []
        
        # Generate hypothesis about transformation
        hypothesis = self._generate_hypothesis(input_grid, examples)
        
        # Apply transformation
        predicted = self._apply_transformation(input_grid, hypothesis)
        
        # Calculate confidence
        confidence = self._calculate_confidence(predicted, output_grid)
        
        return SolverResult(
            answer=predicted,
            confidence=confidence,
            reasoning=f"Applied transformation: {hypothesis}",
            method="pattern_recognition",
            metadata={"patterns": patterns, "hypothesis": hypothesis},
        )
    
    def _analyze_patterns(self, examples: List[Dict]) -> List[str]:
        """Analyze examples to identify patterns."""
        patterns = []
        
        for ex in examples:
            inp = ex.get("input", [])
            out = ex.get("output", [])
            
            # Check size transformations
            if len(inp) != len(out) or (inp and out and len(inp[0]) != len(out[0])):
                patterns.append("size_change")
            
            # Check rotation/flip
            if self._is_rotation(inp, out):
                patterns.append("rotation")
            
            # Check color mapping
            if self._has_color_mapping(inp, out):
                patterns.append("color_mapping")
            
            # Check symmetry
            if self._is_symmetric(out):
                patterns.append("creates_symmetry")
        
        return list(set(patterns))
    
    def _is_rotation(self, inp: List[List], out: List[List]) -> bool:
        """Check if output is a rotation of input."""
        if not inp or not out:
            return False
        
        # Check 90 degree rotation
        try:
            rotated = list(zip(*inp[::-1]))
            return list(map(list, rotated)) == out
        except:
            return False
    
    def _has_color_mapping(self, inp: List[List], out: List[List]) -> bool:
        """Check if there's a consistent color mapping."""
        if not inp or not out:
            return False
        
        try:
            mapping = {}
            for i, row_in in enumerate(inp):
                for j, val_in in enumerate(row_in):
                    if i < len(out) and j < len(out[i]):
                        val_out = out[i][j]
                        if val_in in mapping:
                            if mapping[val_in] != val_out:
                                return False
                        else:
                            mapping[val_in] = val_out
            return len(mapping) > 0
        except:
            return False
    
    def _is_symmetric(self, grid: List[List]) -> bool:
        """Check if grid has any symmetry."""
        if not grid:
            return False
        
        try:
            # Horizontal symmetry
            h_sym = grid == grid[::-1]
            
            # Vertical symmetry
            v_sym = all(row == row[::-1] for row in grid)
            
            return h_sym or v_sym
        except:
            return False
    
    def _generate_hypothesis(
        self,
        input_grid: List[List],
        examples: Optional[List[Dict]],
    ) -> str:
        """Generate hypothesis about transformation using LLM."""
        prompt = f"""Analyze this abstract reasoning task and describe the transformation rule.

Input grid:
{self._grid_to_string(input_grid)}

"""
        if examples:
            prompt += "Training examples:\n"
            for i, ex in enumerate(examples):
                prompt += f"\nExample {i+1}:\n"
                prompt += f"Input:\n{self._grid_to_string(ex.get('input', []))}\n"
                prompt += f"Output:\n{self._grid_to_string(ex.get('output', []))}\n"
        
        prompt += """
Describe the transformation rule in a concise way. Focus on:
1. Size/shape changes
2. Color/value changes
3. Pattern/structure changes
4. Spatial transformations (rotation, flip, etc.)

Transformation rule:"""
        
        response = self._query_llm(prompt)
        return response.strip()
    
    def _apply_transformation(
        self,
        input_grid: List[List],
        hypothesis: str,
    ) -> List[List]:
        """Apply the hypothesized transformation."""
        prompt = f"""Apply this transformation rule to the input grid.

Input grid:
{self._grid_to_string(input_grid)}

Transformation rule: {hypothesis}

Output the resulting grid in JSON format as a 2D array of integers.
Only output the JSON array, nothing else.

Output grid:"""
        
        response = self._query_llm(prompt)
        
        # Parse JSON from response
        try:
            # Find JSON array in response
            match = re.search(r'\[\s*\[.*?\]\s*\]', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
        
        return input_grid  # Return input if parsing fails
    
    def _calculate_confidence(
        self,
        predicted: List[List],
        expected: Optional[List[List]],
    ) -> float:
        """Calculate confidence in the prediction."""
        if expected is None:
            return 0.5  # No ground truth
        
        if predicted == expected:
            return 1.0
        
        # Calculate similarity
        try:
            correct = 0
            total = 0
            for i, row in enumerate(expected):
                for j, val in enumerate(row):
                    total += 1
                    if i < len(predicted) and j < len(predicted[i]):
                        if predicted[i][j] == val:
                            correct += 1
            return correct / total if total > 0 else 0.0
        except:
            return 0.0
    
    def _grid_to_string(self, grid: List[List]) -> str:
        """Convert grid to readable string."""
        if not grid:
            return "[]"
        return "\n".join(" ".join(str(v) for v in row) for row in grid)
    
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM."""
        if hasattr(self.llm, "query"):
            return self.llm.query(prompt)
        elif hasattr(self.llm, "query_model"):
            return self.llm.query_model(prompt)
        return ""


class ScientificReasoningSolver:
    """
    Solver optimized for graduate-level scientific questions (GPQA).
    
    Uses domain-specific knowledge retrieval, mathematical reasoning,
    and systematic problem decomposition.
    """
    
    DOMAINS = {
        "physics": {
            "keywords": ["force", "energy", "momentum", "wave", "quantum", "relativity"],
            "prompt_prefix": "As a physics expert, ",
        },
        "chemistry": {
            "keywords": ["molecule", "reaction", "bond", "atom", "organic", "inorganic"],
            "prompt_prefix": "As a chemistry expert, ",
        },
        "biology": {
            "keywords": ["cell", "gene", "protein", "evolution", "organism", "DNA"],
            "prompt_prefix": "As a biology expert, ",
        },
        "mathematics": {
            "keywords": ["equation", "theorem", "proof", "integral", "derivative"],
            "prompt_prefix": "As a mathematics expert, ",
        },
    }
    
    def __init__(self, llm: Any):
        self.llm = llm
    
    def solve(
        self,
        question: str,
        choices: Optional[List[str]] = None,
        domain: Optional[str] = None,
    ) -> SolverResult:
        """
        Solve a scientific question.
        
        Args:
            question: The question to answer
            choices: Multiple choice options
            domain: Scientific domain (auto-detected if not provided)
        """
        # Detect domain
        if domain is None:
            domain = self._detect_domain(question)
        
        # Decompose problem
        subproblems = self._decompose_problem(question, domain)
        
        # Solve each subproblem
        partial_solutions = []
        for sub in subproblems:
            solution = self._solve_subproblem(sub, domain)
            partial_solutions.append(solution)
        
        # Synthesize final answer
        answer, reasoning = self._synthesize_answer(
            question, partial_solutions, choices, domain
        )
        
        return SolverResult(
            answer=answer,
            confidence=self._estimate_confidence(question, answer, domain),
            reasoning=reasoning,
            method="scientific_reasoning",
            metadata={
                "domain": domain,
                "subproblems": subproblems,
                "partial_solutions": partial_solutions,
            },
        )
    
    def _detect_domain(self, question: str) -> str:
        """Detect the scientific domain of a question."""
        question_lower = question.lower()
        
        for domain, info in self.DOMAINS.items():
            if any(kw in question_lower for kw in info["keywords"]):
                return domain
        
        return "physics"  # Default
    
    def _decompose_problem(self, question: str, domain: str) -> List[str]:
        """Decompose a complex problem into subproblems."""
        prompt = f"""{self.DOMAINS.get(domain, {}).get('prompt_prefix', '')}
break down this problem into smaller, manageable steps.

Problem: {question}

List the key subproblems or steps needed to solve this (numbered):"""
        
        response = self._query_llm(prompt)
        
        # Parse numbered items
        subproblems = []
        for match in re.finditer(r'\d+\.\s*(.+?)(?=\d+\.|$)', response, re.DOTALL):
            subproblems.append(match.group(1).strip())
        
        return subproblems if subproblems else [question]
    
    def _solve_subproblem(self, subproblem: str, domain: str) -> str:
        """Solve a single subproblem."""
        prompt = f"""{self.DOMAINS.get(domain, {}).get('prompt_prefix', '')}
solve this step:

{subproblem}

Show your work and provide the answer:"""
        
        return self._query_llm(prompt)
    
    def _synthesize_answer(
        self,
        question: str,
        partial_solutions: List[str],
        choices: Optional[List[str]],
        domain: str,
    ) -> Tuple[str, str]:
        """Synthesize final answer from partial solutions."""
        solutions_text = "\n\n".join(
            f"Step {i+1}: {sol}" 
            for i, sol in enumerate(partial_solutions)
        )
        
        if choices:
            choices_text = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
            prompt = f"""Based on the following analysis, select the correct answer.

Question: {question}

Analysis:
{solutions_text}

Options:
{choices_text}

Select the correct option (A, B, C, or D) and explain why:"""
        else:
            prompt = f"""Based on the following analysis, provide the final answer.

Question: {question}

Analysis:
{solutions_text}

Final answer and explanation:"""
        
        response = self._query_llm(prompt)
        
        # Extract answer
        if choices:
            match = re.search(r'\b([A-D])\b', response)
            answer = match.group(1) if match else "A"
        else:
            answer = response.strip()
        
        return answer, response
    
    def _estimate_confidence(
        self,
        question: str,
        answer: str,
        domain: str,
    ) -> float:
        """Estimate confidence in the answer."""
        prompt = f"""Rate your confidence in this answer from 0.0 to 1.0.

Question: {question}
Answer: {answer}
Domain: {domain}

Consider:
- Clarity of the question
- Completeness of reasoning
- Presence of calculations/evidence

Confidence (just the number):"""
        
        response = self._query_llm(prompt)
        
        try:
            match = re.search(r'(\d+\.?\d*)', response)
            if match:
                return min(max(float(match.group(1)), 0.0), 1.0)
        except:
            pass
        
        return 0.6  # Default moderate confidence
    
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM."""
        if hasattr(self.llm, "query"):
            return self.llm.query(prompt)
        elif hasattr(self.llm, "query_model"):
            return self.llm.query_model(prompt)
        return ""


class CodeUnderstandingSolver:
    """
    Solver optimized for code understanding and fixing (SWE-BENCH).
    
    Uses structured code analysis, bug pattern recognition,
    and incremental patch generation.
    """
    
    def __init__(self, llm: Any):
        self.llm = llm
    
    def solve(
        self,
        problem_statement: str,
        codebase: Dict[str, str],
        hints: Optional[List[str]] = None,
    ) -> SolverResult:
        """
        Solve a software engineering task.
        
        Args:
            problem_statement: Description of the issue
            codebase: Dict mapping file paths to contents
            hints: Optional hints about the solution
        """
        # Localize the bug
        relevant_files = self._localize_bug(problem_statement, codebase)
        
        # Understand the context
        context = self._understand_context(relevant_files, codebase)
        
        # Generate fix
        patch = self._generate_patch(
            problem_statement, relevant_files, context, codebase, hints
        )
        
        # Validate patch
        confidence = self._validate_patch(patch, codebase)
        
        return SolverResult(
            answer=patch,
            confidence=confidence,
            reasoning=f"Identified relevant files: {relevant_files}",
            method="code_understanding",
            metadata={
                "relevant_files": relevant_files,
                "context": context,
            },
        )
    
    def _localize_bug(
        self,
        problem_statement: str,
        codebase: Dict[str, str],
    ) -> List[str]:
        """Identify files likely to contain the bug."""
        file_list = list(codebase.keys())
        
        prompt = f"""Given this issue description, identify which files are most likely to need changes.

Issue: {problem_statement}

Available files:
{chr(10).join(f'- {f}' for f in file_list)}

List the most relevant files (up to 5), one per line:"""
        
        response = self._query_llm(prompt)
        
        relevant = []
        for line in response.split('\n'):
            line = line.strip().lstrip('- ')
            if line in codebase:
                relevant.append(line)
        
        return relevant[:5] if relevant else file_list[:2]
    
    def _understand_context(
        self,
        files: List[str],
        codebase: Dict[str, str],
    ) -> str:
        """Build understanding of relevant code."""
        context_parts = []
        
        for f in files:
            if f in codebase:
                content = codebase[f]
                # Truncate if too long
                if len(content) > 2000:
                    content = content[:2000] + "\n... (truncated)"
                context_parts.append(f"### {f}\n```\n{content}\n```")
        
        return "\n\n".join(context_parts)
    
    def _generate_patch(
        self,
        problem_statement: str,
        files: List[str],
        context: str,
        codebase: Dict[str, str],
        hints: Optional[List[str]],
    ) -> str:
        """Generate a patch to fix the issue."""
        hints_text = ""
        if hints:
            hints_text = f"\nHints:\n" + "\n".join(f"- {h}" for h in hints)
        
        prompt = f"""Fix this issue by generating a git diff patch.

Issue: {problem_statement}
{hints_text}

Relevant code:
{context}

Generate a unified diff patch that fixes the issue.
Use the format:
--- a/path/to/file
+++ b/path/to/file
@@ -line,count +line,count @@
 context
-removed
+added
 context

Patch:"""
        
        return self._query_llm(prompt)
    
    def _validate_patch(self, patch: str, codebase: Dict[str, str]) -> float:
        """Validate patch format and content."""
        # Check basic patch format
        if '---' not in patch or '+++' not in patch:
            return 0.3
        
        # Check for meaningful changes
        if '+' not in patch and '-' not in patch:
            return 0.2
        
        # Check file references exist
        file_refs = re.findall(r'--- a/(.+)|--- (.+)', patch)
        for ref in file_refs:
            path = ref[0] or ref[1]
            if path and path not in codebase:
                return 0.4
        
        return 0.7  # Looks valid
    
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM."""
        if hasattr(self.llm, "query"):
            return self.llm.query(prompt)
        elif hasattr(self.llm, "query_model"):
            return self.llm.query_model(prompt)
        return ""


class MultimodalReasoningSolver:
    """
    Solver optimized for multimodal understanding (MMMU).
    
    Combines visual understanding with domain knowledge
    for questions requiring both image and text comprehension.
    """
    
    def __init__(self, llm: Any, vision_processor: Any = None):
        self.llm = llm
        self.vision_processor = vision_processor
    
    def solve(
        self,
        question: str,
        images: Optional[List[Any]] = None,
        choices: Optional[List[str]] = None,
        subject: Optional[str] = None,
    ) -> SolverResult:
        """
        Solve a multimodal reasoning task.
        
        Args:
            question: The question text
            images: Associated images
            choices: Multiple choice options
            subject: Academic subject
        """
        # Process images
        image_descriptions = []
        if images and self.vision_processor:
            for img in images:
                desc = self._describe_image(img)
                image_descriptions.append(desc)
        
        # Generate answer
        answer, reasoning = self._generate_answer(
            question, image_descriptions, choices, subject
        )
        
        return SolverResult(
            answer=answer,
            confidence=0.65,
            reasoning=reasoning,
            method="multimodal_reasoning",
            metadata={
                "image_descriptions": image_descriptions,
                "subject": subject,
            },
        )
    
    def _describe_image(self, image: Any) -> str:
        """Generate description of an image."""
        if self.vision_processor is None:
            return "[Image content not available]"
        
        try:
            if hasattr(self.vision_processor, "describe"):
                return self.vision_processor.describe(image)
            elif hasattr(self.vision_processor, "analyze"):
                return self.vision_processor.analyze(image)
        except Exception as e:
            logger.warning(f"Image processing error: {e}")
        
        return "[Image analysis failed]"
    
    def _generate_answer(
        self,
        question: str,
        image_descriptions: List[str],
        choices: Optional[List[str]],
        subject: Optional[str],
    ) -> Tuple[str, str]:
        """Generate answer using text and image information."""
        prompt = f"""Answer this {'multiple choice ' if choices else ''}question.

"""
        if subject:
            prompt += f"Subject: {subject}\n"
        
        prompt += f"Question: {question}\n"
        
        if image_descriptions:
            prompt += "\nImage content:\n"
            for i, desc in enumerate(image_descriptions):
                prompt += f"Image {i+1}: {desc}\n"
        
        if choices:
            prompt += "\nOptions:\n"
            for i, c in enumerate(choices):
                prompt += f"{chr(65+i)}. {c}\n"
            prompt += "\nSelect the correct answer (A, B, C, or D) and explain your reasoning:"
        else:
            prompt += "\nProvide your answer and reasoning:"
        
        response = self._query_llm(prompt)
        
        # Extract answer for multiple choice
        if choices:
            match = re.search(r'\b([A-D])\b', response)
            answer = match.group(1) if match else "A"
        else:
            answer = response.strip()
        
        return answer, response
    
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM."""
        if hasattr(self.llm, "query"):
            return self.llm.query(prompt)
        elif hasattr(self.llm, "query_model"):
            return self.llm.query_model(prompt)
        return ""
