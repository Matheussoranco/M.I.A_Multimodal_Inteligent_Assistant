"""
WebVoyager Benchmark Implementation

WebVoyager evaluates AI agents on their ability to complete
real-world web navigation tasks autonomously.

Key capabilities tested:
- Understanding natural language instructions
- Navigating complex web interfaces
- Filling forms and interacting with elements
- Multi-step task completion
- Error recovery and adaptation

Reference: https://arxiv.org/abs/2401.13919
Repository: https://github.com/MinorJerry/WebVoyager
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base import (
    BaseBenchmark,
    BenchmarkDifficulty,
    BenchmarkTask,
)

logger = logging.getLogger(__name__)


@dataclass
class WebAction:
    """Represents a web interaction action."""
    action_type: str  # click, type, scroll, navigate, wait, etc.
    selector: Optional[str] = None
    value: Optional[str] = None
    description: str = ""


class WebVoyagerBenchmark(BaseBenchmark):
    """
    WebVoyager benchmark for web navigation evaluation.
    
    Tests the agent's ability to:
    - Navigate websites to complete tasks
    - Interact with web elements (buttons, forms, links)
    - Handle dynamic content and JavaScript
    - Complete multi-step workflows
    - Extract information from web pages
    
    Categories:
    - Information seeking
    - Form filling
    - E-commerce tasks
    - Social media interactions
    - Multi-site navigation
    """
    
    def __init__(
        self,
        agent: Any,
        config: Optional[Dict[str, Any]] = None,
        output_dir: str = "benchmarks/results",
        data_dir: str = "benchmarks/data/webvoyager",
    ):
        super().__init__(agent, config, output_dir)
        self.data_dir = Path(data_dir)
        self.headless = config.get("headless", True) if config else True
        self.screenshot_dir = Path(config.get("screenshot_dir", "benchmarks/screenshots")) if config else Path("benchmarks/screenshots")
        self.max_actions = config.get("max_actions", 30) if config else 30
        
        # Web agent for actual browser automation
        self._web_agent = None
    
    @property
    def name(self) -> str:
        return "WebVoyager"
    
    @property
    def version(self) -> str:
        return "1.0"
    
    @property
    def description(self) -> str:
        return "Web navigation benchmark for autonomous browsing tasks"
    
    def _get_web_agent(self):
        """Lazy initialization of web agent."""
        if self._web_agent is None:
            try:
                from ..web.web_agent import WebAgent
                self._web_agent = WebAgent(
                    headless_default=self.headless,
                    screenshot_dir=str(self.screenshot_dir),
                )
            except ImportError:
                logger.warning("WebAgent not available, using mock implementation")
                self._web_agent = None
        return self._web_agent
    
    def load_tasks(
        self,
        subset: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[BenchmarkTask]:
        """Load WebVoyager tasks."""
        tasks = []
        
        if self.data_dir.exists():
            tasks = self._load_from_local(subset, limit)
        else:
            logger.warning("Creating synthetic WebVoyager tasks for testing")
            tasks = self._create_synthetic_tasks(limit or 5)
        
        return tasks
    
    def _load_from_local(
        self,
        subset: Optional[str],
        limit: Optional[int]
    ) -> List[BenchmarkTask]:
        """Load from local JSON files."""
        tasks = []
        
        data_file = self.data_dir / f"{subset or 'test'}.json"
        if not data_file.exists():
            data_file = self.data_dir / "tasks.json"
        
        if data_file.exists():
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for i, item in enumerate(data):
                if limit and i >= limit:
                    break
                
                task = BenchmarkTask(
                    task_id=item.get("task_id", str(i)),
                    input_data={
                        "instruction": item.get("instruction", ""),
                        "start_url": item.get("start_url", ""),
                        "website": item.get("website", ""),
                        "intent": item.get("intent", ""),
                    },
                    expected_output=item.get("expected_result", {}),
                    difficulty=self._categorize_difficulty(item),
                    category=item.get("category", "navigation"),
                    metadata={
                        "website": item.get("website", ""),
                        "expected_actions": item.get("expected_actions", []),
                    },
                    max_steps=self.max_actions,
                    timeout_seconds=300.0,
                )
                tasks.append(task)
        
        return tasks
    
    def execute_task(
        self,
        task: BenchmarkTask
    ) -> Tuple[Any, List[str], List[Dict[str, Any]]]:
        """
        Execute a WebVoyager task.
        
        The agent should:
        1. Understand the natural language instruction
        2. Plan a sequence of web actions
        3. Execute actions in the browser
        4. Verify task completion
        """
        reasoning_trace: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        
        instruction = task.input_data.get("instruction", "")
        start_url = task.input_data.get("start_url", "")
        
        prompt = self._format_webvoyager_prompt(task)
        
        try:
            context = {
                "task_type": "web_navigation",
                "start_url": start_url,
                "max_actions": self.max_actions,
                "available_actions": [
                    "navigate(url)",
                    "click(selector)",
                    "type(selector, text)",
                    "scroll(direction)",
                    "wait(seconds)",
                    "extract(selector)",
                    "screenshot()",
                ],
            }
            
            # Get the agent's action plan
            response = self.agent.execute_task(prompt, context)
            reasoning_trace.append(f"Agent plan: {response}")
            
            # Parse and execute actions
            actions = self._parse_web_actions(response)
            
            # Execute actions using the web agent
            execution_result = self._execute_web_actions(
                start_url,
                actions,
                reasoning_trace,
                tool_calls
            )
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            execution_result = {"success": False, "error": str(e)}
            reasoning_trace.append(f"Error: {e}")
        
        return execution_result, reasoning_trace, tool_calls
    
    def evaluate_output(self, task: BenchmarkTask, predicted_output: Any) -> bool:
        """
        Evaluate if the web navigation task was completed successfully.
        
        Evaluation criteria:
        - Task completion (reached target state)
        - Extracted correct information
        - Performed required actions
        """
        if not isinstance(predicted_output, dict):
            return False
        
        expected = task.expected_output
        if not expected:
            # If no expected output, check for success flag
            return predicted_output.get("success", False)
        
        # Check if required elements are present in the result
        if isinstance(expected, dict):
            for key, value in expected.items():
                if key not in predicted_output:
                    return False
                if value and str(value).lower() not in str(predicted_output[key]).lower():
                    return False
        
        return predicted_output.get("success", False)
    
    def _format_webvoyager_prompt(self, task: BenchmarkTask) -> str:
        """Format a WebVoyager task as a prompt."""
        prompt_parts = [
            "You are a web navigation agent. Complete the following task by interacting with a web browser.",
            "",
            "=== Task ===",
            task.input_data.get("instruction", ""),
            "",
            f"Starting URL: {task.input_data.get('start_url', 'about:blank')}",
            "",
            "=== Available Actions ===",
            "- navigate(url): Go to a specific URL",
            "- click(selector): Click on an element (use CSS selectors)",
            "- type(selector, text): Type text into an input field",
            "- scroll(direction): Scroll the page (up/down)",
            "- wait(seconds): Wait for page to load",
            "- extract(selector): Extract text from an element",
            "- screenshot(): Take a screenshot of current page",
            "",
            "=== Instructions ===",
            "1. Plan your actions step by step",
            "2. Use CSS selectors to identify elements",
            "3. Handle errors and retry if needed",
            "4. Verify task completion before finishing",
            "",
            "Format each action as: ACTION: action_name(parameters)",
            "Example:",
            "ACTION: navigate(https://example.com)",
            "ACTION: click(#search-button)",
            "ACTION: type(#search-input, search query)",
            "",
            "Provide your action sequence:",
        ]
        
        return "\n".join(prompt_parts)
    
    def _parse_web_actions(self, response: str) -> List[WebAction]:
        """Parse web actions from the agent's response."""
        actions = []
        
        # Pattern for action extraction
        action_pattern = r"ACTION:\s*(\w+)\(([^)]*)\)"
        matches = re.findall(action_pattern, response, re.IGNORECASE)
        
        for action_type, params in matches:
            action_type = action_type.lower()
            params = params.strip()
            
            if action_type == "navigate":
                actions.append(WebAction(
                    action_type="open",
                    value=params.strip("'\""),
                    description=f"Navigate to {params}"
                ))
            elif action_type == "click":
                actions.append(WebAction(
                    action_type="click",
                    selector=params.strip("'\""),
                    description=f"Click on {params}"
                ))
            elif action_type == "type":
                # Parse selector and text
                parts = params.split(",", 1)
                if len(parts) == 2:
                    actions.append(WebAction(
                        action_type="type",
                        selector=parts[0].strip().strip("'\""),
                        value=parts[1].strip().strip("'\""),
                        description=f"Type in {parts[0]}"
                    ))
            elif action_type == "scroll":
                actions.append(WebAction(
                    action_type="scroll",
                    value=params.strip("'\"") if params else "down",
                    description=f"Scroll {params or 'down'}"
                ))
            elif action_type == "wait":
                try:
                    seconds = float(params) if params else 2
                except ValueError:
                    seconds = 2
                actions.append(WebAction(
                    action_type="wait",
                    value=str(seconds),
                    description=f"Wait {seconds} seconds"
                ))
            elif action_type == "extract":
                actions.append(WebAction(
                    action_type="scrape",
                    selector=params.strip("'\""),
                    description=f"Extract from {params}"
                ))
            elif action_type == "screenshot":
                actions.append(WebAction(
                    action_type="screenshot",
                    description="Take screenshot"
                ))
        
        return actions
    
    def _execute_web_actions(
        self,
        start_url: str,
        actions: List[WebAction],
        reasoning_trace: List[str],
        tool_calls: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute web actions using the web agent."""
        web_agent = self._get_web_agent()
        
        if web_agent is None:
            # Mock execution for testing
            return self._mock_execute_actions(start_url, actions, reasoning_trace)
        
        # Build plan for web agent
        plan = []
        
        if start_url:
            plan.append({"action": "open", "url": start_url})
        
        for action in actions:
            step: Dict[str, Any] = {"action": action.action_type}
            
            if action.action_type == "open":
                step["url"] = action.value or ""
            elif action.action_type in ["click", "type", "scrape"]:
                step["css"] = action.selector or ""
                if action.value:
                    step["text"] = action.value
            elif action.action_type == "scroll":
                step["direction"] = action.value or "down"
            elif action.action_type == "wait":
                step["seconds"] = float(action.value or 2)
            
            plan.append(step)
            tool_calls.append({
                "tool": "web_action",
                "action": action.action_type,
                "params": step,
            })
        
        try:
            results = web_agent.run_plan(plan, headless=self.headless)
            
            success = all(r.success for r in results)
            extracted_data = {}
            
            for result in results:
                reasoning_trace.append(f"{result.action}: {'✓' if result.success else '✗'} - {result.message}")
                if result.payload:
                    extracted_data.update(result.payload)
            
            return {
                "success": success,
                "actions_taken": len(results),
                "extracted_data": extracted_data,
            }
            
        except Exception as e:
            logger.error(f"Web execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _mock_execute_actions(
        self,
        start_url: str,
        actions: List[WebAction],
        reasoning_trace: List[str],
    ) -> Dict[str, Any]:
        """Mock execution for testing without browser."""
        reasoning_trace.append(f"Mock execution starting at: {start_url}")
        
        for i, action in enumerate(actions):
            reasoning_trace.append(f"Step {i+1}: {action.description}")
        
        # Simulate success for testing
        return {
            "success": len(actions) > 0,
            "actions_taken": len(actions),
            "mock": True,
        }
    
    def _categorize_difficulty(self, item: Dict) -> BenchmarkDifficulty:
        """Categorize task difficulty based on complexity."""
        expected_actions = item.get("expected_actions", [])
        num_actions = len(expected_actions)
        
        if num_actions <= 3:
            return BenchmarkDifficulty.EASY
        elif num_actions <= 7:
            return BenchmarkDifficulty.MEDIUM
        elif num_actions <= 15:
            return BenchmarkDifficulty.HARD
        else:
            return BenchmarkDifficulty.EXPERT
    
    def _create_synthetic_tasks(self, num_tasks: int) -> List[BenchmarkTask]:
        """Create synthetic WebVoyager tasks for testing."""
        tasks = []
        
        synthetic = [
            {
                "task_id": "web_search_1",
                "instruction": "Go to Google and search for 'Python programming'",
                "start_url": "https://www.google.com",
                "website": "google.com",
                "category": "search",
                "expected_result": {"search_performed": True},
                "expected_actions": ["type", "click"],
            },
            {
                "task_id": "web_nav_1",
                "instruction": "Navigate to Wikipedia and find the article about Artificial Intelligence",
                "start_url": "https://www.wikipedia.org",
                "website": "wikipedia.org",
                "category": "navigation",
                "expected_result": {"page_found": True},
                "expected_actions": ["click", "type", "click"],
            },
            {
                "task_id": "form_fill_1",
                "instruction": "Fill out a contact form with name 'Test User' and email 'test@example.com'",
                "start_url": "https://example.com/contact",
                "website": "example.com",
                "category": "form",
                "expected_result": {"form_submitted": True},
                "expected_actions": ["type", "type", "click"],
            },
        ]
        
        for i, item in enumerate(synthetic[:num_tasks]):
            task = BenchmarkTask(
                task_id=item["task_id"],
                input_data={
                    "instruction": item["instruction"],
                    "start_url": item["start_url"],
                    "website": item["website"],
                },
                expected_output=item["expected_result"],
                difficulty=self._categorize_difficulty(item),
                category=item["category"],
                metadata={"synthetic": True},
            )
            tasks.append(task)
        
        return tasks
