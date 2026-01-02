"""
State-of-the-Art Multi-Agent Orchestration System.

This module implements a sophisticated multi-agent system inspired by
CrewAI and AutoGen patterns for complex task decomposition and execution.

Features:
- Agent specialization and role definition
- Task dependency management
- Inter-agent communication
- Consensus mechanisms
- Work delegation strategies
- Agent memory and learning
"""

import os
import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    TYPE_CHECKING,
)
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Predefined agent roles."""
    COORDINATOR = "coordinator"  # Orchestrates other agents
    RESEARCHER = "researcher"  # Gathers information
    ANALYST = "analyst"  # Analyzes data
    CODER = "coder"  # Writes and reviews code
    CRITIC = "critic"  # Reviews and critiques work
    EXECUTOR = "executor"  # Executes actions
    PLANNER = "planner"  # Creates plans
    WRITER = "writer"  # Generates text content
    CUSTOM = "custom"  # User-defined role


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    WAITING = "waiting"  # Waiting for dependencies
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageType(Enum):
    """Types of inter-agent messages."""
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    FEEDBACK = "feedback"
    CONSENSUS_VOTE = "consensus_vote"


@dataclass
class AgentMessage:
    """Message between agents."""
    id: str
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    message_type: MessageType
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class AgentTask:
    """A task to be executed by an agent."""
    id: str
    name: str
    description: str
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    result: Any = None
    error: Optional[str] = None
    priority: int = 5  # 1-10, higher is more urgent
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    id: str
    name: str
    role: AgentRole
    description: str
    capabilities: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.7
    max_iterations: int = 10
    tools: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for agents."""
    
    def __init__(
        self,
        config: AgentConfig,
        llm_manager=None,
        action_executor=None,
        memory=None,
    ):
        self.config = config
        self.id = config.id
        self.name = config.name
        self.role = config.role
        self.llm_manager = llm_manager
        self.action_executor = action_executor
        self.memory = memory
        
        self.message_inbox: List[AgentMessage] = []
        self.message_outbox: List[AgentMessage] = []
        self.task_history: List[AgentTask] = []
        self.current_task: Optional[AgentTask] = None
        
        logger.info(f"Agent {self.name} ({self.role.value}) initialized")
    
    @abstractmethod
    async def execute_task(self, task: AgentTask) -> Any:
        """Execute a task and return result."""
        pass
    
    async def receive_message(self, message: AgentMessage):
        """Handle incoming message."""
        self.message_inbox.append(message)
        logger.debug(f"Agent {self.name} received message from {message.sender_id}")
        
        if message.message_type == MessageType.TASK_ASSIGNMENT:
            await self._handle_task_assignment(message)
        elif message.message_type == MessageType.REQUEST:
            await self._handle_request(message)
        elif message.message_type == MessageType.FEEDBACK:
            await self._handle_feedback(message)
    
    def send_message(
        self,
        recipient_id: Optional[str],
        message_type: MessageType,
        content: Any,
        metadata: Optional[Dict] = None,
    ) -> AgentMessage:
        """Send a message to another agent or broadcast."""
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender_id=self.id,
            recipient_id=recipient_id,
            message_type=message_type,
            content=content,
            metadata=metadata or {},
        )
        self.message_outbox.append(message)
        return message
    
    async def _handle_task_assignment(self, message: AgentMessage):
        """Handle task assignment message."""
        task = message.content
        if isinstance(task, dict):
            task = AgentTask(**task)
        
        logger.info(f"Agent {self.name} received task: {task.name}")
        
        try:
            result = await self.execute_task(task)
            
            # Send result back
            self.send_message(
                recipient_id=message.sender_id,
                message_type=MessageType.TASK_RESULT,
                content={
                    'task_id': task.id,
                    'result': result,
                    'status': TaskStatus.COMPLETED.value,
                },
            )
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            self.send_message(
                recipient_id=message.sender_id,
                message_type=MessageType.TASK_RESULT,
                content={
                    'task_id': task.id,
                    'error': str(e),
                    'status': TaskStatus.FAILED.value,
                },
            )
    
    async def _handle_request(self, message: AgentMessage):
        """Handle request message."""
        request = message.content
        
        if self.llm_manager:
            try:
                response = self.llm_manager.generate(
                    prompt=f"As {self.name} ({self.role.value}), respond to: {request}",
                    system_prompt=self.config.system_prompt,
                )
                
                self.send_message(
                    recipient_id=message.sender_id,
                    message_type=MessageType.RESPONSE,
                    content=response,
                )
            except Exception as e:
                logger.error(f"Request handling failed: {e}")
    
    async def _handle_feedback(self, message: AgentMessage):
        """Handle feedback message."""
        feedback = message.content
        
        # Store feedback in memory for learning
        if self.memory:
            try:
                self.memory.store({
                    'type': 'feedback',
                    'content': feedback,
                    'from': message.sender_id,
                    'timestamp': message.timestamp.isoformat(),
                })
            except Exception as e:
                logger.warning(f"Failed to store feedback: {e}")
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities."""
        return self.config.capabilities.copy()
    
    def can_handle(self, task: AgentTask) -> bool:
        """Check if agent can handle a task."""
        # Check if task requirements match capabilities
        task_reqs = task.metadata.get('required_capabilities', [])
        if not task_reqs:
            return True
        
        return all(req in self.config.capabilities for req in task_reqs)


class LLMAgent(BaseAgent):
    """Agent that uses LLM for task execution."""
    
    async def execute_task(self, task: AgentTask) -> Any:
        """Execute task using LLM."""
        self.current_task = task
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        try:
            if not self.llm_manager:
                raise ValueError("LLM manager required for LLMAgent")
            
            # Build prompt based on task
            prompt = self._build_task_prompt(task)
            
            # Execute with LLM
            response = self.llm_manager.generate(
                prompt=prompt,
                system_prompt=self.config.system_prompt,
                temperature=self.config.temperature,
            )
            
            # Process response
            result = self._process_response(response, task)
            
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now()
            self.task_history.append(task)
            
            return result
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self.task_history.append(task)
            raise
        finally:
            self.current_task = None
    
    def _build_task_prompt(self, task: AgentTask) -> str:
        """Build prompt for task execution."""
        context = task.metadata.get('context', '')
        
        prompt = f"""Task: {task.name}

Description: {task.description}

{f'Context: {context}' if context else ''}

Please complete this task thoroughly and provide a detailed response."""
        
        return prompt
    
    def _process_response(self, response: str, task: AgentTask) -> Any:
        """Process LLM response."""
        # Default: return raw response
        return response


class ToolAgent(BaseAgent):
    """Agent that executes tools/actions."""
    
    async def execute_task(self, task: AgentTask) -> Any:
        """Execute task using tools."""
        self.current_task = task
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        try:
            if not self.action_executor:
                raise ValueError("Action executor required for ToolAgent")
            
            # Get tool and parameters from task
            tool_name = task.metadata.get('tool')
            tool_params = task.metadata.get('parameters', {})
            
            if not tool_name:
                raise ValueError("No tool specified in task metadata")
            
            # Execute tool
            result = await self.action_executor.execute_async(tool_name, tool_params)
            
            if result.success:
                task.status = TaskStatus.COMPLETED
                task.result = result.output
            else:
                task.status = TaskStatus.FAILED
                task.error = result.error
            
            task.completed_at = datetime.now()
            self.task_history.append(task)
            
            return task.result
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self.task_history.append(task)
            raise
        finally:
            self.current_task = None


class HybridAgent(LLMAgent, ToolAgent):
    """Agent that combines LLM reasoning with tool execution."""
    
    async def execute_task(self, task: AgentTask) -> Any:
        """Execute task using LLM reasoning and tools."""
        self.current_task = task
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        try:
            iterations = 0
            max_iterations = self.config.max_iterations
            
            while iterations < max_iterations:
                iterations += 1
                
                # Use LLM to decide next action
                decision = await self._decide_next_action(task)
                
                if decision['action'] == 'complete':
                    task.status = TaskStatus.COMPLETED
                    task.result = decision.get('result')
                    break
                elif decision['action'] == 'tool':
                    # Execute tool
                    tool_result = await self._execute_tool(
                        decision['tool'],
                        decision.get('parameters', {}),
                    )
                    
                    # Update task context with tool result
                    task.metadata.setdefault('tool_results', []).append({
                        'tool': decision['tool'],
                        'result': tool_result,
                    })
                elif decision['action'] == 'delegate':
                    # Request delegation to another agent
                    self.send_message(
                        recipient_id=None,  # Broadcast
                        message_type=MessageType.REQUEST,
                        content={
                            'request': 'delegate',
                            'task': task,
                            'reason': decision.get('reason'),
                        },
                    )
                    task.status = TaskStatus.WAITING
                    break
            
            if task.status != TaskStatus.COMPLETED:
                task.status = TaskStatus.FAILED
                task.error = "Max iterations reached"
            
            task.completed_at = datetime.now()
            self.task_history.append(task)
            
            return task.result
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self.task_history.append(task)
            raise
        finally:
            self.current_task = None
    
    async def _decide_next_action(self, task: AgentTask) -> Dict:
        """Use LLM to decide next action."""
        if not self.llm_manager:
            return {'action': 'complete', 'result': None}
        
        # Build context from previous tool results
        context = ""
        if 'tool_results' in task.metadata:
            context = "Previous tool results:\n"
            for tr in task.metadata['tool_results']:
                context += f"- {tr['tool']}: {tr['result']}\n"
        
        available_tools = self.config.tools
        
        prompt = f"""Task: {task.name}
Description: {task.description}

{context}

Available tools: {', '.join(available_tools)}

Decide the next action. Respond in JSON format:
{{"action": "tool" | "complete" | "delegate", "tool": "tool_name", "parameters": {{}}, "result": "...", "reason": "..."}}
"""
        
        response = self.llm_manager.generate(prompt, system_prompt=self.config.system_prompt)
        
        try:
            import json
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                decision = json.loads(response[json_start:json_end])
                return decision
        except:
            pass
        
        return {'action': 'complete', 'result': response}
    
    async def _execute_tool(self, tool_name: str, parameters: Dict) -> Any:
        """Execute a tool."""
        if self.action_executor:
            result = self.action_executor.execute(tool_name, parameters)
            return result.output if result.success else f"Error: {result.error}"
        return None


class AgentOrchestrator:
    """
    Orchestrates multiple agents to complete complex tasks.
    
    Implements:
    - Task decomposition
    - Agent selection and assignment
    - Dependency management
    - Result aggregation
    - Consensus mechanisms
    """
    
    def __init__(
        self,
        llm_manager=None,
        action_executor=None,
        memory=None,
    ):
        self.llm_manager = llm_manager
        self.action_executor = action_executor
        self.memory = memory
        
        self.agents: Dict[str, BaseAgent] = {}
        self.tasks: Dict[str, AgentTask] = {}
        self.task_queue: List[AgentTask] = []
        self.completed_tasks: List[AgentTask] = []
        
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        
        logger.info("AgentOrchestrator initialized")
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator."""
        self.agents[agent.id] = agent
        logger.info(f"Registered agent: {agent.name} ({agent.role.value})")
    
    def create_agent(
        self,
        name: str,
        role: AgentRole,
        description: str,
        agent_type: str = "llm",
        **kwargs,
    ) -> BaseAgent:
        """Create and register a new agent."""
        config = AgentConfig(
            id=str(uuid.uuid4()),
            name=name,
            role=role,
            description=description,
            **kwargs,
        )
        
        agent_classes = {
            "llm": LLMAgent,
            "tool": ToolAgent,
            "hybrid": HybridAgent,
        }
        
        agent_class = agent_classes.get(agent_type, LLMAgent)
        agent = agent_class(
            config=config,
            llm_manager=self.llm_manager,
            action_executor=self.action_executor,
            memory=self.memory,
        )
        
        self.register_agent(agent)
        return agent
    
    def create_default_crew(self):
        """Create a default set of specialized agents."""
        # Coordinator agent
        self.create_agent(
            name="Coordinator",
            role=AgentRole.COORDINATOR,
            description="Orchestrates and coordinates other agents",
            system_prompt="You are a coordinator. Break down complex tasks and delegate to appropriate specialists.",
            capabilities=["planning", "delegation", "monitoring"],
        )
        
        # Researcher agent
        self.create_agent(
            name="Researcher",
            role=AgentRole.RESEARCHER,
            description="Gathers and synthesizes information",
            agent_type="hybrid",
            system_prompt="You are a researcher. Find and synthesize relevant information thoroughly.",
            capabilities=["web_search", "document_analysis", "fact_checking"],
            tools=["web_search", "read_file", "web_fetch"],
        )
        
        # Analyst agent
        self.create_agent(
            name="Analyst",
            role=AgentRole.ANALYST,
            description="Analyzes data and provides insights",
            system_prompt="You are an analyst. Analyze data thoroughly and provide actionable insights.",
            capabilities=["data_analysis", "pattern_recognition", "reporting"],
        )
        
        # Coder agent
        self.create_agent(
            name="Coder",
            role=AgentRole.CODER,
            description="Writes and reviews code",
            agent_type="hybrid",
            system_prompt="You are an expert programmer. Write clean, efficient, and well-documented code.",
            capabilities=["coding", "code_review", "debugging"],
            tools=["python_eval", "read_file", "write_file"],
        )
        
        # Critic agent
        self.create_agent(
            name="Critic",
            role=AgentRole.CRITIC,
            description="Reviews and critiques work for quality",
            system_prompt="You are a critic. Review work thoroughly and provide constructive feedback.",
            capabilities=["review", "quality_assurance", "feedback"],
        )
        
        logger.info(f"Created default crew with {len(self.agents)} agents")
    
    async def decompose_task(self, task: AgentTask) -> List[AgentTask]:
        """Decompose a complex task into subtasks."""
        if not self.llm_manager:
            return [task]
        
        prompt = f"""Decompose this task into smaller, actionable subtasks.

Task: {task.name}
Description: {task.description}

Available agents and their capabilities:
{self._get_agent_descriptions()}

Return a JSON array of subtasks:
[{{"name": "subtask name", "description": "what to do", "dependencies": ["other_subtask_id"], "required_capabilities": ["capability1"]}}]
"""
        
        try:
            response = self.llm_manager.generate(prompt)
            
            import json
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                subtasks_data = json.loads(response[json_start:json_end])
                
                subtasks = []
                id_map = {}
                
                for i, st_data in enumerate(subtasks_data):
                    subtask = AgentTask(
                        id=str(uuid.uuid4()),
                        name=st_data.get('name', f"Subtask {i+1}"),
                        description=st_data.get('description', ''),
                        metadata={
                            'parent_task': task.id,
                            'required_capabilities': st_data.get('required_capabilities', []),
                        },
                    )
                    id_map[st_data.get('name', f"Subtask {i+1}")] = subtask.id
                    subtasks.append(subtask)
                
                # Map dependencies
                for st_data, subtask in zip(subtasks_data, subtasks):
                    deps = st_data.get('dependencies', [])
                    subtask.dependencies = [id_map.get(d) for d in deps if d in id_map]
                
                return subtasks
        except Exception as e:
            logger.warning(f"Task decomposition failed: {e}")
        
        return [task]
    
    def _get_agent_descriptions(self) -> str:
        """Get descriptions of all agents."""
        lines = []
        for agent in self.agents.values():
            caps = ', '.join(agent.get_capabilities())
            lines.append(f"- {agent.name} ({agent.role.value}): {agent.config.description}. Capabilities: {caps}")
        return '\n'.join(lines)
    
    def select_agent(self, task: AgentTask) -> Optional[BaseAgent]:
        """Select the best agent for a task."""
        candidates = []
        
        for agent in self.agents.values():
            if agent.can_handle(task):
                # Score based on role match and availability
                score = 0
                
                # Role-based scoring
                role_scores = {
                    AgentRole.RESEARCHER: ['research', 'search', 'find'],
                    AgentRole.ANALYST: ['analyze', 'analyse', 'insight'],
                    AgentRole.CODER: ['code', 'program', 'implement', 'develop'],
                    AgentRole.WRITER: ['write', 'draft', 'compose'],
                    AgentRole.CRITIC: ['review', 'critique', 'evaluate'],
                }
                
                task_lower = task.name.lower() + ' ' + task.description.lower()
                
                if agent.role in role_scores:
                    for keyword in role_scores[agent.role]:
                        if keyword in task_lower:
                            score += 10
                
                # Availability bonus (not currently executing)
                if agent.current_task is None:
                    score += 5
                
                # Capability match
                required_caps = task.metadata.get('required_capabilities', [])
                for cap in required_caps:
                    if cap in agent.get_capabilities():
                        score += 3
                
                candidates.append((agent, score))
        
        if candidates:
            # Select highest scoring agent
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None
    
    async def execute_task(self, task: AgentTask) -> Any:
        """Execute a task through the agent system."""
        self.tasks[task.id] = task
        
        # Check if decomposition is needed
        if task.metadata.get('decompose', True):
            subtasks = await self.decompose_task(task)
            
            if len(subtasks) > 1:
                # Execute subtasks
                results = await self._execute_subtasks(subtasks)
                
                # Aggregate results
                task.result = await self._aggregate_results(task, results)
                task.status = TaskStatus.COMPLETED
                return task.result
        
        # Direct execution
        agent = self.select_agent(task)
        
        if not agent:
            task.status = TaskStatus.FAILED
            task.error = "No suitable agent found"
            return None
        
        task.assigned_agent = agent.id
        
        # Send task assignment
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender_id="orchestrator",
            recipient_id=agent.id,
            message_type=MessageType.TASK_ASSIGNMENT,
            content=task,
        )
        
        await agent.receive_message(message)
        
        # Wait for result
        result = await self._wait_for_result(agent, task)
        
        return result
    
    async def _execute_subtasks(self, subtasks: List[AgentTask]) -> List[Any]:
        """Execute subtasks respecting dependencies."""
        results = {}
        pending = set(st.id for st in subtasks)
        
        while pending:
            # Find tasks with satisfied dependencies
            ready = []
            for subtask in subtasks:
                if subtask.id in pending:
                    deps_satisfied = all(
                        d in results for d in subtask.dependencies
                    )
                    if deps_satisfied:
                        ready.append(subtask)
            
            if not ready:
                # Deadlock - break out
                logger.warning("Dependency deadlock detected")
                break
            
            # Execute ready tasks in parallel
            async_tasks = []
            for subtask in ready:
                # Add dependency results to context
                subtask.metadata['dependency_results'] = {
                    d: results[d] for d in subtask.dependencies if d in results
                }
                async_tasks.append(self._execute_single_task(subtask))
            
            completed = await asyncio.gather(*async_tasks, return_exceptions=True)
            
            for subtask, result in zip(ready, completed):
                pending.remove(subtask.id)
                if isinstance(result, Exception):
                    results[subtask.id] = f"Error: {result}"
                else:
                    results[subtask.id] = result
        
        return [results.get(st.id) for st in subtasks]
    
    async def _execute_single_task(self, task: AgentTask) -> Any:
        """Execute a single task."""
        agent = self.select_agent(task)
        
        if not agent:
            raise ValueError(f"No suitable agent for task: {task.name}")
        
        task.assigned_agent = agent.id
        result = await agent.execute_task(task)
        
        return result
    
    async def _aggregate_results(self, parent_task: AgentTask, results: List[Any]) -> Any:
        """Aggregate subtask results."""
        if not self.llm_manager:
            return results
        
        prompt = f"""Aggregate these subtask results into a final response for the main task.

Main Task: {parent_task.name}
Description: {parent_task.description}

Subtask Results:
{chr(10).join(f'- Result {i+1}: {r}' for i, r in enumerate(results))}

Provide a comprehensive aggregated response:"""
        
        try:
            response = self.llm_manager.generate(prompt)
            return response
        except Exception as e:
            logger.warning(f"Result aggregation failed: {e}")
            return results
    
    async def _wait_for_result(self, agent: BaseAgent, task: AgentTask, timeout: float = 300) -> Any:
        """Wait for agent to complete task."""
        start = datetime.now()
        
        while (datetime.now() - start).total_seconds() < timeout:
            # Check agent's outbox for result
            for message in agent.message_outbox:
                if message.message_type == MessageType.TASK_RESULT:
                    content = message.content
                    if content.get('task_id') == task.id:
                        agent.message_outbox.remove(message)
                        
                        if content.get('status') == TaskStatus.COMPLETED.value:
                            return content.get('result')
                        else:
                            raise Exception(content.get('error', 'Task failed'))
            
            await asyncio.sleep(0.1)
        
        raise TimeoutError(f"Task {task.id} timed out")
    
    async def run_consensus(
        self,
        question: str,
        voting_agents: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> Tuple[Any, float]:
        """Run consensus voting among agents."""
        voters = voting_agents or list(self.agents.keys())
        votes: Dict[str, List[str]] = defaultdict(list)
        
        for agent_id in voters:
            if agent_id not in self.agents:
                continue
            
            agent = self.agents[agent_id]
            
            if agent.llm_manager:
                try:
                    response = agent.llm_manager.generate(
                        prompt=f"Question: {question}\nProvide your answer (be concise):",
                        system_prompt=agent.config.system_prompt,
                    )
                    
                    # Normalize response
                    answer = response.strip().lower()
                    votes[answer].append(agent_id)
                except Exception as e:
                    logger.warning(f"Agent {agent.name} failed to vote: {e}")
        
        if not votes:
            return None, 0.0
        
        # Find majority
        total_votes = sum(len(v) for v in votes.values())
        winner = max(votes.items(), key=lambda x: len(x[1]))
        
        confidence = len(winner[1]) / total_votes
        
        if confidence >= threshold:
            return winner[0], confidence
        else:
            return None, confidence
    
    def get_agent_status(self) -> Dict[str, Dict]:
        """Get status of all agents."""
        status = {}
        for agent_id, agent in self.agents.items():
            status[agent_id] = {
                'name': agent.name,
                'role': agent.role.value,
                'busy': agent.current_task is not None,
                'tasks_completed': len([t for t in agent.task_history if t.status == TaskStatus.COMPLETED]),
                'tasks_failed': len([t for t in agent.task_history if t.status == TaskStatus.FAILED]),
            }
        return status
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            'total_agents': len(self.agents),
            'total_tasks': len(self.tasks),
            'completed_tasks': len(self.completed_tasks),
            'pending_tasks': len(self.task_queue),
            'agents_by_role': {
                role.value: len([a for a in self.agents.values() if a.role == role])
                for role in AgentRole
            },
        }


# Factory functions for common crew configurations

def create_research_crew(llm_manager=None, action_executor=None) -> AgentOrchestrator:
    """Create a crew optimized for research tasks."""
    orchestrator = AgentOrchestrator(llm_manager, action_executor)
    
    orchestrator.create_agent(
        name="Lead Researcher",
        role=AgentRole.RESEARCHER,
        description="Leads research efforts and synthesizes findings",
        capabilities=["web_search", "document_analysis", "synthesis"],
    )
    
    orchestrator.create_agent(
        name="Fact Checker",
        role=AgentRole.ANALYST,
        description="Verifies facts and checks sources",
        capabilities=["fact_checking", "source_verification"],
    )
    
    orchestrator.create_agent(
        name="Report Writer",
        role=AgentRole.WRITER,
        description="Writes comprehensive research reports",
        capabilities=["writing", "formatting", "summarization"],
    )
    
    return orchestrator


def create_development_crew(llm_manager=None, action_executor=None) -> AgentOrchestrator:
    """Create a crew optimized for software development."""
    orchestrator = AgentOrchestrator(llm_manager, action_executor)
    
    orchestrator.create_agent(
        name="Architect",
        role=AgentRole.PLANNER,
        description="Designs system architecture and technical solutions",
        capabilities=["architecture", "design", "planning"],
    )
    
    orchestrator.create_agent(
        name="Developer",
        role=AgentRole.CODER,
        description="Implements features and writes code",
        agent_type="hybrid",
        capabilities=["coding", "implementation", "debugging"],
        tools=["python_eval", "read_file", "write_file"],
    )
    
    orchestrator.create_agent(
        name="Code Reviewer",
        role=AgentRole.CRITIC,
        description="Reviews code for quality and best practices",
        capabilities=["code_review", "quality_assurance", "best_practices"],
    )
    
    orchestrator.create_agent(
        name="Tester",
        role=AgentRole.EXECUTOR,
        description="Tests code and validates functionality",
        agent_type="tool",
        capabilities=["testing", "validation", "automation"],
        tools=["python_eval", "run_tests"],
    )
    
    return orchestrator
