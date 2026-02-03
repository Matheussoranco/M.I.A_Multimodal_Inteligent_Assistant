"""
Advanced Reasoning Module for M.I.A

Implements state-of-the-art reasoning techniques to improve performance
on AGI benchmarks:

- Chain-of-Thought (CoT) prompting
- Tree-of-Thought (ToT) exploration
- Self-consistency sampling
- Reflection and self-correction
- Tool-augmented reasoning
- Multi-step planning

These techniques are essential for achieving strong performance on
benchmarks like ARC-AGI, GAIA, and GPQA.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ReasoningStrategy(Enum):
    """Available reasoning strategies."""
    DIRECT = "direct"  # Direct answer
    CHAIN_OF_THOUGHT = "cot"  # Step-by-step reasoning
    TREE_OF_THOUGHT = "tot"  # Explore multiple paths
    SELF_CONSISTENCY = "sc"  # Multiple samples + voting
    REFLECTION = "reflection"  # Self-correction
    REACT = "react"  # Reasoning + Acting
    PLAN_AND_SOLVE = "plan_solve"  # Explicit planning


@dataclass
class ThoughtNode:
    """A node in the thought tree."""
    thought: str
    value: float = 0.0
    children: List["ThoughtNode"] = field(default_factory=list)
    is_terminal: bool = False
    action: Optional[str] = None
    observation: Optional[str] = None


@dataclass
class ReasoningTrace:
    """Complete trace of a reasoning process."""
    strategy: ReasoningStrategy
    steps: List[Dict[str, Any]]
    final_answer: Any
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseReasoner(ABC):
    """Abstract base class for reasoning strategies."""
    
    def __init__(self, llm: Any, max_steps: int = 10):
        self.llm = llm
        self.max_steps = max_steps
    
    @abstractmethod
    def reason(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningTrace:
        """Execute reasoning and return trace."""
        pass
    
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM."""
        if hasattr(self.llm, "query"):
            return self.llm.query(prompt)
        elif hasattr(self.llm, "query_model"):
            return self.llm.query_model(prompt)
        else:
            raise ValueError("LLM must have query or query_model method")


class ChainOfThoughtReasoner(BaseReasoner):
    """
    Chain-of-Thought (CoT) reasoning.
    
    Encourages step-by-step reasoning before arriving at an answer.
    Effective for math, logic, and multi-step problems.
    
    Reference: Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
    """
    
    def reason(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningTrace:
        """Execute Chain-of-Thought reasoning."""
        steps = []
        
        # Build CoT prompt
        prompt = self._build_cot_prompt(query, context)
        
        # Get response
        response = self._query_llm(prompt)
        steps.append({"type": "cot_response", "content": response})
        
        # Extract reasoning and answer
        reasoning, answer = self._parse_cot_response(response)
        steps.append({"type": "parsed_reasoning", "content": reasoning})
        steps.append({"type": "parsed_answer", "content": answer})
        
        return ReasoningTrace(
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            steps=steps,
            final_answer=answer,
            confidence=0.7,  # CoT typically has moderate confidence
        )
    
    def _build_cot_prompt(self, query: str, context: Optional[Dict]) -> str:
        """Build a Chain-of-Thought prompt."""
        prompt = f"""Let's solve this step by step.

Problem: {query}

Think through this carefully:
1. First, identify what we need to find
2. Break down the problem into smaller steps
3. Work through each step
4. Verify the answer

Let's begin:

Step 1:"""
        
        if context:
            prompt = f"Context: {json.dumps(context)}\n\n" + prompt
        
        return prompt
    
    def _parse_cot_response(self, response: str) -> Tuple[str, str]:
        """Parse reasoning and final answer from response."""
        # Look for explicit answer markers
        answer_patterns = [
            r"(?:Therefore|Thus|So|Hence),?\s*the\s*(?:final\s*)?answer\s*is[:\s]*(.+?)(?:\.|$)",
            r"Final\s*Answer[:\s]*(.+?)(?:\n|$)",
            r"Answer[:\s]*(.+?)(?:\n|$)",
        ]
        
        answer = None
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                break
        
        if not answer:
            # Take the last non-empty line
            lines = [l.strip() for l in response.split("\n") if l.strip()]
            answer = lines[-1] if lines else response
        
        return response, answer


class TreeOfThoughtReasoner(BaseReasoner):
    """
    Tree-of-Thought (ToT) reasoning.
    
    Explores multiple reasoning paths and evaluates them to find the best solution.
    Effective for complex problems with multiple potential approaches.
    
    Reference: Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
    """
    
    def __init__(
        self,
        llm: Any,
        max_steps: int = 10,
        branch_factor: int = 3,
        max_depth: int = 5,
    ):
        super().__init__(llm, max_steps)
        self.branch_factor = branch_factor
        self.max_depth = max_depth
    
    def reason(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningTrace:
        """Execute Tree-of-Thought reasoning."""
        steps = []
        
        # Initialize root node
        root = ThoughtNode(thought=f"Problem: {query}")
        steps.append({"type": "init", "thought": root.thought})
        
        # Build thought tree using BFS
        best_leaf = self._build_and_search_tree(root, query, context, steps)
        
        # Extract answer from best path
        answer = self._extract_answer_from_node(best_leaf)
        
        return ReasoningTrace(
            strategy=ReasoningStrategy.TREE_OF_THOUGHT,
            steps=steps,
            final_answer=answer,
            confidence=best_leaf.value if best_leaf else 0.0,
            metadata={"tree_depth": self._get_tree_depth(root)},
        )
    
    def _build_and_search_tree(
        self,
        root: ThoughtNode,
        query: str,
        context: Optional[Dict],
        steps: List,
    ) -> Optional[ThoughtNode]:
        """Build thought tree and find best leaf."""
        frontier = [root]
        best_leaf = None
        best_value = -float("inf")
        
        for depth in range(self.max_depth):
            if not frontier:
                break
            
            next_frontier = []
            
            for node in frontier:
                # Generate children thoughts
                children = self._generate_thoughts(node, query, context)
                steps.append({
                    "type": "expand",
                    "depth": depth,
                    "parent": node.thought[:50],
                    "children_count": len(children),
                })
                
                # Evaluate children
                for child in children:
                    child.value = self._evaluate_thought(child, query)
                    
                    if child.is_terminal:
                        if child.value > best_value:
                            best_value = child.value
                            best_leaf = child
                    else:
                        next_frontier.append(child)
                
                node.children = children
            
            # Keep top k nodes for next level
            next_frontier.sort(key=lambda n: n.value, reverse=True)
            frontier = next_frontier[:self.branch_factor]
        
        return best_leaf
    
    def _generate_thoughts(
        self,
        node: ThoughtNode,
        query: str,
        context: Optional[Dict],
    ) -> List[ThoughtNode]:
        """Generate child thoughts from a node."""
        prompt = f"""Given this problem-solving state, suggest {self.branch_factor} different next steps or approaches.

Problem: {query}
Current thinking: {node.thought}

For each approach, explain:
1. What the next step is
2. Why it might work
3. What we expect to find

Provide {self.branch_factor} different approaches, numbered 1-{self.branch_factor}:"""
        
        response = self._query_llm(prompt)
        
        # Parse response into separate thoughts
        thoughts = []
        for i in range(1, self.branch_factor + 1):
            pattern = rf"{i}\.\s*(.+?)(?=\d\.|$)"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                thought_text = match.group(1).strip()
                is_terminal = self._is_terminal_thought(thought_text)
                thoughts.append(ThoughtNode(
                    thought=thought_text,
                    is_terminal=is_terminal,
                ))
        
        # If parsing failed, create single thought from whole response
        if not thoughts:
            thoughts = [ThoughtNode(thought=response)]
        
        return thoughts
    
    def _evaluate_thought(self, node: ThoughtNode, query: str) -> float:
        """Evaluate the quality/promise of a thought."""
        prompt = f"""Rate how promising this reasoning step is for solving the problem.

Problem: {query}
Reasoning step: {node.thought}

Rate from 0.0 (not helpful) to 1.0 (very promising).
Consider: correctness, progress toward solution, logical coherence.

Rating (just the number):"""
        
        response = self._query_llm(prompt)
        
        try:
            # Extract number from response
            match = re.search(r"(\d+\.?\d*)", response)
            if match:
                value = float(match.group(1))
                return min(max(value, 0.0), 1.0)
        except ValueError:
            pass
        
        return 0.5  # Default to medium confidence
    
    def _is_terminal_thought(self, thought: str) -> bool:
        """Check if thought represents a final answer."""
        terminal_indicators = [
            "therefore the answer is",
            "final answer",
            "the solution is",
            "we conclude that",
            "this gives us",
        ]
        thought_lower = thought.lower()
        return any(ind in thought_lower for ind in terminal_indicators)
    
    def _extract_answer_from_node(self, node: Optional[ThoughtNode]) -> str:
        """Extract final answer from a thought node."""
        if not node:
            return ""
        return node.thought
    
    def _get_tree_depth(self, root: ThoughtNode) -> int:
        """Get the maximum depth of the thought tree."""
        if not root.children:
            return 0
        return 1 + max(self._get_tree_depth(child) for child in root.children)


class SelfConsistencyReasoner(BaseReasoner):
    """
    Self-Consistency reasoning.
    
    Samples multiple reasoning paths and takes majority vote for the answer.
    Improves reliability through ensemble reasoning.
    
    Reference: Wang et al., "Self-Consistency Improves Chain of Thought Reasoning"
    """
    
    def __init__(
        self,
        llm: Any,
        max_steps: int = 10,
        num_samples: int = 5,
        temperature: float = 0.7,
    ):
        super().__init__(llm, max_steps)
        self.num_samples = num_samples
        self.temperature = temperature
        self.cot_reasoner = ChainOfThoughtReasoner(llm, max_steps)
    
    def reason(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningTrace:
        """Execute Self-Consistency reasoning."""
        steps = []
        answers = []
        
        # Generate multiple reasoning paths
        for i in range(self.num_samples):
            trace = self.cot_reasoner.reason(query, context)
            answers.append(trace.final_answer)
            steps.append({
                "type": "sample",
                "index": i,
                "answer": trace.final_answer,
                "reasoning": trace.steps[0]["content"][:200] + "...",
            })
        
        # Majority voting
        answer_counts = {}
        for ans in answers:
            ans_normalized = str(ans).strip().lower()
            answer_counts[ans_normalized] = answer_counts.get(ans_normalized, 0) + 1
        
        best_answer = max(answer_counts.keys(), key=lambda k: answer_counts[k])
        confidence = answer_counts[best_answer] / self.num_samples
        
        steps.append({
            "type": "vote",
            "answer_distribution": answer_counts,
            "selected": best_answer,
            "confidence": confidence,
        })
        
        return ReasoningTrace(
            strategy=ReasoningStrategy.SELF_CONSISTENCY,
            steps=steps,
            final_answer=best_answer,
            confidence=confidence,
            metadata={"num_samples": self.num_samples},
        )


class ReflectionReasoner(BaseReasoner):
    """
    Reflection reasoning with self-correction.
    
    The model critiques its own output and iteratively improves it.
    Effective for catching errors and refining answers.
    
    Reference: Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning"
    """
    
    def __init__(
        self,
        llm: Any,
        max_steps: int = 10,
        max_reflections: int = 3,
    ):
        super().__init__(llm, max_steps)
        self.max_reflections = max_reflections
    
    def reason(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningTrace:
        """Execute Reflection reasoning."""
        steps = []
        
        # Initial attempt
        current_answer = self._initial_attempt(query, context)
        steps.append({"type": "initial", "answer": current_answer})
        
        # Reflection loop
        for i in range(self.max_reflections):
            # Critique current answer
            critique = self._critique_answer(query, current_answer)
            steps.append({"type": "critique", "iteration": i, "critique": critique})
            
            # Check if answer is satisfactory
            if self._is_satisfactory(critique):
                steps.append({"type": "accepted", "iteration": i})
                break
            
            # Refine answer based on critique
            refined_answer = self._refine_answer(query, current_answer, critique)
            steps.append({"type": "refined", "iteration": i, "answer": refined_answer})
            
            current_answer = refined_answer
        
        return ReasoningTrace(
            strategy=ReasoningStrategy.REFLECTION,
            steps=steps,
            final_answer=current_answer,
            confidence=0.8,  # Higher confidence after reflection
            metadata={"reflections": len(steps) // 2},
        )
    
    def _initial_attempt(self, query: str, context: Optional[Dict]) -> str:
        """Generate initial answer."""
        prompt = f"""Solve this problem:

{query}

Provide your answer:"""
        
        if context:
            prompt = f"Context: {json.dumps(context)}\n\n" + prompt
        
        return self._query_llm(prompt)
    
    def _critique_answer(self, query: str, answer: str) -> str:
        """Critique the current answer."""
        prompt = f"""Critically evaluate this answer to the problem.

Problem: {query}

Answer: {answer}

Identify:
1. Any errors or mistakes
2. Missing considerations
3. Logical flaws
4. Areas for improvement

Be specific and constructive. If the answer is correct, say "SATISFACTORY".

Critique:"""
        
        return self._query_llm(prompt)
    
    def _is_satisfactory(self, critique: str) -> bool:
        """Check if the critique indicates satisfactory answer."""
        satisfactory_indicators = [
            "satisfactory",
            "correct",
            "no errors",
            "well done",
            "accurate",
        ]
        critique_lower = critique.lower()
        return any(ind in critique_lower for ind in satisfactory_indicators)
    
    def _refine_answer(self, query: str, answer: str, critique: str) -> str:
        """Refine answer based on critique."""
        prompt = f"""Improve this answer based on the critique.

Problem: {query}

Previous answer: {answer}

Critique: {critique}

Provide an improved answer that addresses the issues raised:"""
        
        return self._query_llm(prompt)


class ReActReasoner(BaseReasoner):
    """
    ReAct (Reasoning + Acting) reasoning.
    
    Interleaves reasoning with tool/action use for grounded problem-solving.
    Essential for agentic tasks like GAIA and WebVoyager.
    
    Reference: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models"
    """
    
    def __init__(
        self,
        llm: Any,
        max_steps: int = 10,
        tools: Optional[Dict[str, Callable]] = None,
    ):
        super().__init__(llm, max_steps)
        self.tools = tools or {}
    
    def reason(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningTrace:
        """Execute ReAct reasoning."""
        steps = []
        history = []
        
        for step_num in range(self.max_steps):
            # Generate thought and action
            response = self._generate_thought_action(query, history, context)
            
            # Parse response
            thought, action, action_input = self._parse_react_response(response)
            
            steps.append({
                "type": "thought",
                "step": step_num,
                "thought": thought,
                "action": action,
                "action_input": action_input,
            })
            
            # Check for final answer
            if action and action.lower() == "finish":
                return ReasoningTrace(
                    strategy=ReasoningStrategy.REACT,
                    steps=steps,
                    final_answer=action_input,
                    confidence=0.85,
                    metadata={"steps_taken": step_num + 1},
                )
            
            # Execute action
            observation = self._execute_action(action, action_input)
            
            steps.append({
                "type": "observation",
                "step": step_num,
                "observation": observation,
            })
            
            # Update history
            history.append({
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": observation,
            })
        
        # Max steps reached
        return ReasoningTrace(
            strategy=ReasoningStrategy.REACT,
            steps=steps,
            final_answer="Max steps reached without conclusion",
            confidence=0.3,
            metadata={"steps_taken": self.max_steps, "completed": False},
        )
    
    def _generate_thought_action(
        self,
        query: str,
        history: List[Dict],
        context: Optional[Dict],
    ) -> str:
        """Generate next thought and action."""
        tools_desc = "\n".join([
            f"- {name}: {func.__doc__ or 'No description'}"
            for name, func in self.tools.items()
        ])
        
        if not tools_desc:
            tools_desc = "- search: Search for information\n- calculate: Perform calculations\n- finish: Provide final answer"
        
        prompt = f"""You are solving a problem using available tools.

Problem: {query}

Available tools:
{tools_desc}
- finish: Use when you have the final answer

Format:
Thought: <your reasoning>
Action: <tool name>
Action Input: <input to the tool>

"""
        
        if history:
            prompt += "Previous steps:\n"
            for h in history:
                prompt += f"Thought: {h['thought']}\n"
                prompt += f"Action: {h['action']}\n"
                prompt += f"Action Input: {h['action_input']}\n"
                prompt += f"Observation: {h['observation']}\n\n"
        
        prompt += "What is your next step?\n"
        
        return self._query_llm(prompt)
    
    def _parse_react_response(self, response: str) -> Tuple[str, str, str]:
        """Parse thought, action, and action input from response."""
        thought = ""
        action = ""
        action_input = ""
        
        # Extract thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        # Extract action
        action_match = re.search(r"Action:\s*(\w+)", response)
        if action_match:
            action = action_match.group(1).strip()
        
        # Extract action input
        input_match = re.search(r"Action Input:\s*(.+?)(?=\n|$)", response, re.DOTALL)
        if input_match:
            action_input = input_match.group(1).strip()
        
        return thought, action, action_input
    
    def _execute_action(self, action: str, action_input: str) -> str:
        """Execute an action and return observation."""
        if not action:
            return "No action specified"
        
        action_lower = action.lower()
        
        if action_lower in self.tools:
            try:
                return str(self.tools[action_lower](action_input))
            except Exception as e:
                return f"Error executing {action}: {e}"
        
        # Built-in mock actions for testing
        if action_lower == "search":
            return f"Search results for '{action_input}': [Mock search result]"
        elif action_lower == "calculate":
            try:
                from mia.utils.safe_arithmetic import safe_eval_arithmetic

                result = safe_eval_arithmetic(action_input)
                return f"Result: {result}"
            except:
                return "Calculation error"
        
        return f"Unknown action: {action}"


class AdvancedReasoningEngine:
    """
    Unified interface for advanced reasoning strategies.
    
    Automatically selects the best reasoning strategy based on
    the problem type or allows manual strategy selection.
    """
    
    def __init__(self, llm: Any):
        self.llm = llm
        self.reasoners = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: ChainOfThoughtReasoner(llm),
            ReasoningStrategy.TREE_OF_THOUGHT: TreeOfThoughtReasoner(llm),
            ReasoningStrategy.SELF_CONSISTENCY: SelfConsistencyReasoner(llm),
            ReasoningStrategy.REFLECTION: ReflectionReasoner(llm),
            ReasoningStrategy.REACT: ReActReasoner(llm),
        }
    
    def reason(
        self,
        query: str,
        strategy: Optional[ReasoningStrategy] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningTrace:
        """
        Execute reasoning with specified or auto-selected strategy.
        
        Args:
            query: The problem to solve
            strategy: Optional strategy to use
            context: Additional context
            
        Returns:
            Complete reasoning trace
        """
        if strategy is None:
            strategy = self._select_strategy(query, context)
        
        reasoner = self.reasoners.get(strategy)
        if reasoner is None:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        logger.info(f"Using {strategy.value} reasoning strategy")
        
        return reasoner.reason(query, context)
    
    def _select_strategy(
        self,
        query: str,
        context: Optional[Dict],
    ) -> ReasoningStrategy:
        """Auto-select the best strategy for a problem."""
        query_lower = query.lower()
        task_type = context.get("task_type", "") if context else ""
        
        # Math/logic problems -> CoT or Self-Consistency
        if any(word in query_lower for word in ["calculate", "compute", "solve", "equation", "math"]):
            return ReasoningStrategy.SELF_CONSISTENCY
        
        # Complex reasoning -> ToT
        if any(word in query_lower for word in ["complex", "multiple", "approach", "strategy"]):
            return ReasoningStrategy.TREE_OF_THOUGHT
        
        # Tasks requiring tools -> ReAct
        if any(word in task_type for word in ["web", "search", "tool", "action", "agent"]):
            return ReasoningStrategy.REACT
        
        # Default to Chain-of-Thought
        return ReasoningStrategy.CHAIN_OF_THOUGHT
    
    def set_tools(self, tools: Dict[str, Callable]) -> None:
        """Set tools for ReAct reasoning."""
        if ReasoningStrategy.REACT in self.reasoners:
            self.reasoners[ReasoningStrategy.REACT].tools = tools
