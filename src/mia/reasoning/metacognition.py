"""
Meta-Cognitive Layer for M.I.A

Implements self-monitoring, strategy selection, and adaptive behavior
to dynamically improve reasoning quality during task execution.

This is critical for state-of-the-art AGI benchmark performance.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TaskDifficulty(Enum):
    """Estimated task difficulty levels."""
    TRIVIAL = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    VERY_HARD = 5


class CognitionMode(Enum):
    """Available cognition modes."""
    FAST = "fast"  # Quick, intuitive reasoning
    SLOW = "slow"  # Deliberate, analytical reasoning
    MIXED = "mixed"  # Adaptive based on subtask


@dataclass
class ExecutionMetrics:
    """Metrics for a single execution."""
    start_time: float = 0.0
    end_time: float = 0.0
    tokens_used: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    errors: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class TaskAnalysis:
    """Analysis of a task's characteristics."""
    difficulty: TaskDifficulty
    domain: str
    requires_tools: bool
    requires_multimodal: bool
    requires_math: bool
    estimated_steps: int
    recommended_strategy: str
    confidence: float


class MetaCognitiveLayer:
    """
    Meta-cognitive layer for self-monitoring and adaptive reasoning.
    
    Capabilities:
    - Task difficulty estimation
    - Strategy recommendation
    - Progress monitoring
    - Error detection and recovery
    - Resource optimization
    """
    
    def __init__(self, llm: Any):
        self.llm = llm
        self.execution_history: List[Dict] = []
        self.current_metrics: Optional[ExecutionMetrics] = None
        self.performance_stats: Dict[str, List[float]] = {}
    
    def analyze_task(self, task: str, context: Optional[Dict] = None) -> TaskAnalysis:
        """
        Analyze a task to determine optimal approach.
        
        Args:
            task: The task description
            context: Additional context
            
        Returns:
            TaskAnalysis with recommendations
        """
        prompt = f"""Analyze this task and provide structured assessment.

Task: {task}

Assess:
1. Difficulty (1-5, where 5 is hardest)
2. Domain (e.g., coding, math, science, general)
3. Requires external tools? (yes/no)
4. Requires image/multimodal understanding? (yes/no)
5. Requires mathematical calculations? (yes/no)
6. Estimated steps to complete (number)
7. Best reasoning strategy (cot/tot/react/reflection)

Provide assessment in JSON format:
{{"difficulty": N, "domain": "...", "requires_tools": bool, "requires_multimodal": bool, "requires_math": bool, "estimated_steps": N, "strategy": "..."}}

Assessment:"""
        
        response = self._query_llm(prompt)
        
        try:
            # Parse JSON from response
            import re
            match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return TaskAnalysis(
                    difficulty=TaskDifficulty(data.get("difficulty", 3)),
                    domain=data.get("domain", "general"),
                    requires_tools=data.get("requires_tools", False),
                    requires_multimodal=data.get("requires_multimodal", False),
                    requires_math=data.get("requires_math", False),
                    estimated_steps=data.get("estimated_steps", 5),
                    recommended_strategy=data.get("strategy", "cot"),
                    confidence=0.7,
                )
        except Exception as e:
            logger.warning(f"Task analysis parsing failed: {e}")
        
        # Default analysis
        return TaskAnalysis(
            difficulty=TaskDifficulty.MEDIUM,
            domain="general",
            requires_tools=False,
            requires_multimodal=False,
            requires_math=False,
            estimated_steps=5,
            recommended_strategy="cot",
            confidence=0.5,
        )
    
    def select_cognition_mode(self, analysis: TaskAnalysis) -> CognitionMode:
        """Select appropriate cognition mode based on task analysis."""
        if analysis.difficulty.value <= 2:
            return CognitionMode.FAST
        elif analysis.difficulty.value >= 4:
            return CognitionMode.SLOW
        else:
            return CognitionMode.MIXED
    
    def estimate_resource_budget(self, analysis: TaskAnalysis) -> Dict[str, int]:
        """Estimate resource budget for a task."""
        base_tokens = 1000
        base_calls = 3
        
        # Scale by difficulty
        difficulty_multiplier = analysis.difficulty.value
        
        return {
            "max_tokens": base_tokens * difficulty_multiplier,
            "max_llm_calls": base_calls * difficulty_multiplier,
            "max_tool_calls": 5 * difficulty_multiplier if analysis.requires_tools else 0,
            "timeout_seconds": 60 * difficulty_multiplier,
        }
    
    def start_execution(self) -> None:
        """Start tracking execution metrics."""
        self.current_metrics = ExecutionMetrics(start_time=time.time())
    
    def record_llm_call(self, tokens: int = 0) -> None:
        """Record an LLM call."""
        if self.current_metrics:
            self.current_metrics.llm_calls += 1
            self.current_metrics.tokens_used += tokens
    
    def record_tool_call(self) -> None:
        """Record a tool call."""
        if self.current_metrics:
            self.current_metrics.tool_calls += 1
    
    def record_error(self, error: str) -> None:
        """Record an error."""
        if self.current_metrics:
            self.current_metrics.errors.append(error)
    
    def end_execution(self, task_id: str, success: bool) -> ExecutionMetrics:
        """End execution and record results."""
        if self.current_metrics:
            self.current_metrics.end_time = time.time()
            
            # Store in history
            self.execution_history.append({
                "task_id": task_id,
                "metrics": self.current_metrics,
                "success": success,
            })
            
            # Update stats
            if task_id not in self.performance_stats:
                self.performance_stats[task_id] = []
            self.performance_stats[task_id].append(
                1.0 if success else 0.0
            )
            
            return self.current_metrics
        
        return ExecutionMetrics()
    
    def should_retry(self, error: str, attempts: int) -> Tuple[bool, str]:
        """
        Determine if a failed attempt should be retried.
        
        Returns:
            (should_retry, modified_strategy)
        """
        if attempts >= 3:
            return False, ""
        
        prompt = f"""An attempt failed with this error. Should we retry with a different approach?

Error: {error}
Previous attempts: {attempts}

If yes, suggest a different strategy. If no, explain why.
Format: {{"retry": bool, "strategy": "...", "reason": "..."}}

Decision:"""
        
        response = self._query_llm(prompt)
        
        try:
            import re
            match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return data.get("retry", False), data.get("strategy", "")
        except:
            pass
        
        # Default: retry with reflection
        return True, "reflection"
    
    def evaluate_progress(
        self,
        task: str,
        steps_completed: List[str],
        current_state: str,
    ) -> Dict[str, Any]:
        """Evaluate progress on a multi-step task."""
        prompt = f"""Evaluate progress on this task.

Task: {task}

Steps completed:
{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(steps_completed))}

Current state: {current_state}

Assess:
1. Progress (0-100%)
2. On track? (yes/no)
3. Blockers/issues
4. Suggested next step

Format: {{"progress": N, "on_track": bool, "issues": [...], "next_step": "..."}}

Assessment:"""
        
        response = self._query_llm(prompt)
        
        try:
            import re
            match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
        
        return {
            "progress": 50,
            "on_track": True,
            "issues": [],
            "next_step": "Continue with current approach",
        }
    
    def detect_stuck(
        self,
        recent_outputs: List[str],
        similarity_threshold: float = 0.8,
    ) -> bool:
        """Detect if reasoning is stuck in a loop."""
        if len(recent_outputs) < 3:
            return False
        
        # Check if recent outputs are too similar
        last_three = recent_outputs[-3:]
        
        # Simple similarity check
        for i in range(len(last_three) - 1):
            for j in range(i + 1, len(last_three)):
                similarity = self._calculate_similarity(last_three[i], last_three[j])
                if similarity > similarity_threshold:
                    return True
        
        return False
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def generate_recovery_strategy(self, stuck_context: str) -> str:
        """Generate a strategy to recover from being stuck."""
        prompt = f"""The reasoning process is stuck. Generate a recovery strategy.

Context: {stuck_context}

Suggest:
1. What might be causing the stuckness
2. Alternative approaches to try
3. How to break out of the current pattern

Recovery strategy:"""
        
        return self._query_llm(prompt)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance across tasks."""
        total = len(self.execution_history)
        successful = sum(1 for e in self.execution_history if e["success"])
        
        avg_duration = 0.0
        avg_calls = 0.0
        
        if total > 0:
            durations = [e["metrics"].duration for e in self.execution_history]
            avg_duration = sum(durations) / len(durations)
            
            calls = [e["metrics"].llm_calls for e in self.execution_history]
            avg_calls = sum(calls) / len(calls)
        
        return {
            "total_tasks": total,
            "successful_tasks": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "average_duration": avg_duration,
            "average_llm_calls": avg_calls,
        }
    
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM."""
        if hasattr(self.llm, "query"):
            return self.llm.query(prompt)
        elif hasattr(self.llm, "query_model"):
            return self.llm.query_model(prompt)
        return ""


class SelfReflection:
    """
    Self-reflection capabilities for improving reasoning quality.
    
    Implements reflexion-style self-improvement through verbal feedback.
    """
    
    def __init__(self, llm: Any):
        self.llm = llm
        self.reflection_history: List[Dict] = []
    
    def reflect_on_attempt(
        self,
        task: str,
        attempt: str,
        result: Any,
        success: bool,
    ) -> str:
        """Generate reflection on an attempt."""
        prompt = f"""Reflect on this problem-solving attempt.

Task: {task}
Attempt: {attempt}
Result: {result}
Success: {success}

Analyze:
1. What went well?
2. What went wrong?
3. What should be done differently?
4. Key lessons learned

Reflection:"""
        
        reflection = self._query_llm(prompt)
        
        self.reflection_history.append({
            "task": task,
            "attempt": attempt,
            "success": success,
            "reflection": reflection,
        })
        
        return reflection
    
    def apply_lessons(
        self,
        new_task: str,
        similar_reflections: Optional[List[str]] = None,
    ) -> str:
        """Apply lessons from past reflections to a new task."""
        if similar_reflections is None:
            # Find relevant past reflections
            similar_reflections = self._find_similar_reflections(new_task)
        
        if not similar_reflections:
            return ""
        
        prompt = f"""Apply lessons from past attempts to this new task.

New task: {new_task}

Past reflections:
{chr(10).join(similar_reflections)}

Guidance for approaching this task based on past lessons:"""
        
        return self._query_llm(prompt)
    
    def _find_similar_reflections(self, task: str, top_k: int = 3) -> List[str]:
        """Find reflections from similar past tasks."""
        if not self.reflection_history:
            return []
        
        # Simple keyword matching
        task_words = set(task.lower().split())
        
        scored = []
        for entry in self.reflection_history:
            entry_words = set(entry["task"].lower().split())
            overlap = len(task_words & entry_words) / max(len(task_words | entry_words), 1)
            scored.append((overlap, entry["reflection"]))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        
        return [r for _, r in scored[:top_k] if _ > 0.1]
    
    def generate_self_critique(self, reasoning: str) -> str:
        """Generate critique of own reasoning."""
        prompt = f"""Critically evaluate this reasoning. Be harsh but constructive.

Reasoning: {reasoning}

Critique (identify flaws, gaps, and questionable assumptions):"""
        
        return self._query_llm(prompt)
    
    def improve_based_on_critique(self, reasoning: str, critique: str) -> str:
        """Improve reasoning based on self-critique."""
        prompt = f"""Improve this reasoning based on the critique.

Original reasoning: {reasoning}

Critique: {critique}

Improved reasoning:"""
        
        return self._query_llm(prompt)
    
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM."""
        if hasattr(self.llm, "query"):
            return self.llm.query(prompt)
        elif hasattr(self.llm, "query_model"):
            return self.llm.query_model(prompt)
        return ""


class StrategyOptimizer:
    """
    Optimizes reasoning strategies based on performance data.
    
    Uses reinforcement learning principles to improve strategy selection.
    """
    
    def __init__(self):
        self.strategy_rewards: Dict[str, List[float]] = {}
        self.task_strategy_map: Dict[str, str] = {}
    
    def record_outcome(
        self,
        task_type: str,
        strategy: str,
        reward: float,
    ) -> None:
        """Record outcome of using a strategy."""
        key = f"{task_type}:{strategy}"
        if key not in self.strategy_rewards:
            self.strategy_rewards[key] = []
        self.strategy_rewards[key].append(reward)
    
    def get_best_strategy(
        self,
        task_type: str,
        available_strategies: List[str],
        exploration_rate: float = 0.1,
    ) -> str:
        """Get best strategy for a task type using epsilon-greedy."""
        import random
        
        # Exploration
        if random.random() < exploration_rate:
            return random.choice(available_strategies)
        
        # Exploitation - pick strategy with highest average reward
        best_strategy = available_strategies[0]
        best_reward = -float("inf")
        
        for strategy in available_strategies:
            key = f"{task_type}:{strategy}"
            rewards = self.strategy_rewards.get(key, [])
            
            if rewards:
                avg_reward = sum(rewards) / len(rewards)
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_strategy = strategy
        
        return best_strategy
    
    def get_strategy_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all strategies."""
        stats = {}
        
        for key, rewards in self.strategy_rewards.items():
            if rewards:
                stats[key] = {
                    "mean": sum(rewards) / len(rewards),
                    "count": len(rewards),
                    "min": min(rewards),
                    "max": max(rewards),
                }
        
        return stats
