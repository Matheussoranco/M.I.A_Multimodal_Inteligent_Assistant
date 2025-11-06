"""
Workflow Automation Composer
"""

import logging
import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union, Set
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue

from ..providers import provider_registry

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_APPROVAL = "waiting_approval"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    WAITING = "waiting"


class ApprovalStatus(Enum):
    """Approval status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class WorkflowTask:
    """Represents a task in a workflow."""
    id: str
    name: str
    description: str = ""
    task_type: str = "action"
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # Task IDs this task depends on
    conditions: List[Dict[str, Any]] = field(default_factory=list)  # Execution conditions
    timeout: Optional[int] = None  # Timeout in seconds
    retry_count: int = 0
    retry_delay: int = 1  # Delay between retries in seconds
    requires_approval: bool = False
    approvers: List[str] = field(default_factory=list)
    parallel_group: Optional[str] = None  # Group for parallel execution
    on_success: List[str] = field(default_factory=list)  # Next tasks on success
    on_failure: List[str] = field(default_factory=list)  # Next tasks on failure
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskExecution:
    """Represents the execution state of a task."""
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    attempts: int = 0
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: Optional[str] = None
    approval_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """Represents the execution of a workflow."""
    workflow_id: str
    execution_id: str
    status: WorkflowStatus = WorkflowStatus.CREATED
    tasks: Dict[str, TaskExecution] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowDefinition:
    """Complete workflow definition."""
    id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    tasks: Dict[str, WorkflowTask] = field(default_factory=dict)
    entry_points: List[str] = field(default_factory=list)  # Starting task IDs
    global_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class TaskExecutor:
    """
    Base class for task executors.

    Each executor handles a specific type of task.
    """

    def __init__(self, task_type: str, config: Optional[Dict[str, Any]] = None):
        self.task_type = task_type
        self.config = config or {}

    async def execute(self, task: WorkflowTask, context: Dict[str, Any]) -> Any:
        """Execute the task and return result."""
        raise NotImplementedError("Subclasses must implement execute method")

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate task configuration."""
        return True


class ActionExecutor(TaskExecutor):
    """Executor for action tasks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("action", config)

    async def execute(self, task: WorkflowTask, context: Dict[str, Any]) -> Any:
        """Execute an action task."""
        action = task.config.get("action")
        if not action:
            raise ValueError("Action not specified in task config")

        # Simple action execution - in real implementation would have action registry
        if action == "log":
            message = task.config.get("message", "Action executed")
            logger.info(f"Workflow action: {message}")
            return {"status": "logged", "message": message}

        elif action == "set_variable":
            var_name = task.config.get("variable")
            var_value = task.config.get("value")
            if var_name:
                context[var_name] = var_value
                return {"status": "set", "variable": var_name, "value": var_value}

        elif action == "delay":
            delay = task.config.get("seconds", 1)
            await asyncio.sleep(delay)
            return {"status": "delayed", "seconds": delay}

        else:
            raise ValueError(f"Unknown action: {action}")


class HttpExecutor(TaskExecutor):
    """Executor for HTTP request tasks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("http", config)

    async def execute(self, task: WorkflowTask, context: Dict[str, Any]) -> Any:
        """Execute an HTTP request task."""
        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp required for HTTP tasks")

        url = task.config.get("url")
        method = task.config.get("method", "GET")
        headers = task.config.get("headers", {})
        data = task.config.get("data")
        timeout = task.config.get("timeout", 30)

        if not url:
            raise ValueError("URL not specified in HTTP task config")

        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, headers=headers, data=data, timeout=timeout) as response:
                result = {
                    "status_code": response.status,
                    "headers": dict(response.headers),
                    "url": str(response.url)
                }

                if response.content_type and 'json' in response.content_type:
                    result["json"] = await response.json()
                else:
                    result["text"] = await response.text()

                return result


class ApprovalExecutor(TaskExecutor):
    """Executor for approval tasks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("approval", config)

    async def execute(self, task: WorkflowTask, context: Dict[str, Any]) -> Any:
        """Execute an approval task."""
        # Approval tasks are handled specially by the workflow engine
        # This executor just validates the approval configuration
        approvers = task.config.get("approvers", [])
        timeout = task.config.get("timeout", 3600)  # 1 hour default

        return {
            "status": "approval_required",
            "approvers": approvers,
            "timeout": timeout,
            "message": task.config.get("message", "Approval required")
        }


class WorkflowAutomationComposer:
    """
    Workflow automation system.
    """

    def __init__(
        self,
        config_manager=None,
        *,
        max_concurrent_executions: int = 10,
        default_timeout: int = 3600,
        enable_parallel_execution: bool = True
    ):
        self.config_manager = config_manager
        self.max_concurrent_executions = max_concurrent_executions
        self.default_timeout = default_timeout
        self.enable_parallel_execution = enable_parallel_execution

        # Workflow storage
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}

        # Task executors
        self.executors: Dict[str, TaskExecutor] = {}
        self._register_default_executors()

        # Execution management
        self.execution_lock = threading.Lock()
        self.execution_semaphore = asyncio.Semaphore(max_concurrent_executions)
        self.active_executions: Set[str] = set()

        # Approval management
        self.approval_queue: asyncio.Queue = asyncio.Queue()
        self.pending_approvals: Dict[str, Dict[str, Any]] = {}

        # Monitoring
        self.execution_history: List[WorkflowExecution] = []
        self.metrics: Dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0
        }

        logger.info("Workflow Automation Composer initialized")

    def _register_default_executors(self):
        """Register default task executors."""
        self.executors["action"] = ActionExecutor()
        self.executors["http"] = HttpExecutor()
        self.executors["approval"] = ApprovalExecutor()

    def register_executor(self, task_type: str, executor: TaskExecutor):
        """Register a custom task executor."""
        self.executors[task_type] = executor
        logger.info(f"Registered executor for task type: {task_type}")

    def create_workflow(
        self,
        name: str,
        description: str = "",
        tasks: Optional[Dict[str, WorkflowTask]] = None,
        entry_points: Optional[List[str]] = None
    ) -> str:
        """
        Create a new workflow definition.

        Args:
            name: Workflow name
            description: Workflow description
            tasks: Dictionary of task definitions
            entry_points: List of starting task IDs

        Returns:
            Workflow ID
        """
        workflow_id = str(uuid.uuid4())

        workflow = WorkflowDefinition(
            id=workflow_id,
            name=name,
            description=description,
            tasks=tasks or {},
            entry_points=entry_points or []
        )

        self.workflows[workflow_id] = workflow
        logger.info(f"Created workflow: {name} ({workflow_id})")

        return workflow_id

    def update_workflow(self, workflow_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing workflow.

        Args:
            updates: Dictionary of updates to apply

        Returns:
            True if update successful
        """
        if workflow_id not in self.workflows:
            return False

        workflow = self.workflows[workflow_id]

        for key, value in updates.items():
            if hasattr(workflow, key):
                setattr(workflow, key, value)

        workflow.updated_at = datetime.now()
        logger.info(f"Updated workflow: {workflow_id}")

        return True

    def delete_workflow(self, workflow_id: str) -> bool:
        """
        Delete a workflow.

        Returns:
            True if deletion successful
        """
        if workflow_id in self.workflows:
            del self.workflows[workflow_id]
            logger.info(f"Deleted workflow: {workflow_id}")
            return True
        return False

    async def execute_workflow(
        self,
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None
    ) -> str:
        """
        Execute a workflow.

        Args:
            workflow_id: ID of workflow to execute
            context: Initial execution context
            execution_id: Optional custom execution ID

        Returns:
            Execution ID
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")

        async with self.execution_semaphore:
            exec_id = execution_id or str(uuid.uuid4())

            workflow = self.workflows[workflow_id]
            execution = WorkflowExecution(
                workflow_id=workflow_id,
                execution_id=exec_id,
                context=context or {},
                start_time=datetime.now()
            )

            # Initialize task executions
            for task_id, task in workflow.tasks.items():
                execution.tasks[task_id] = TaskExecution(task_id=task_id)

            self.executions[exec_id] = execution
            self.active_executions.add(exec_id)

            try:
                # Start execution
                execution.status = WorkflowStatus.RUNNING
                await self._execute_workflow_tasks(execution, workflow)

                # Mark as completed
                execution.status = WorkflowStatus.COMPLETED
                execution.end_time = datetime.now()

                self.metrics["total_executions"] += 1
                self.metrics["successful_executions"] += 1
                self._update_execution_metrics(execution)

            except Exception as exc:
                execution.status = WorkflowStatus.FAILED
                execution.end_time = datetime.now()
                logger.error(f"Workflow execution failed: {exec_id} - {exc}")

                self.metrics["total_executions"] += 1
                self.metrics["failed_executions"] += 1

            finally:
                self.active_executions.discard(exec_id)
                self.execution_history.append(execution)

            return exec_id

    def _update_execution_metrics(self, execution: WorkflowExecution):
        """Update execution time metrics."""
        if execution.start_time and execution.end_time:
            execution_time = (execution.end_time - execution.start_time).total_seconds()
            total_time = self.metrics["average_execution_time"] * (self.metrics["total_executions"] - 1)
            self.metrics["average_execution_time"] = (total_time + execution_time) / self.metrics["total_executions"]

    async def _execute_workflow_tasks(self, execution: WorkflowExecution, workflow: WorkflowDefinition):
        """Execute workflow tasks with dependency management."""
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(workflow)
        completed_tasks: Set[str] = set()
        running_tasks: Set[str] = set()

        # Start with entry points
        pending_tasks = set(workflow.entry_points)

        while pending_tasks or running_tasks:
            # Check for tasks that can be started
            startable_tasks = []
            for task_id in pending_tasks:
                if self._can_start_task(task_id, workflow, completed_tasks, execution):
                    startable_tasks.append(task_id)

            # Start tasks
            if self.enable_parallel_execution:
                # Start all startable tasks in parallel
                tasks_to_start = []
                for task_id in startable_tasks:
                    task = workflow.tasks[task_id]
                    if task.parallel_group:
                        # Group parallel tasks
                        group_tasks = [t for t in startable_tasks if workflow.tasks[t].parallel_group == task.parallel_group]
                        if task_id == min(group_tasks):  # Start only one from group
                            tasks_to_start.extend(group_tasks)
                    else:
                        tasks_to_start.append(task_id)

                # Remove duplicates
                tasks_to_start = list(set(tasks_to_start))

                # Start tasks
                for task_id in tasks_to_start:
                    pending_tasks.discard(task_id)
                    running_tasks.add(task_id)
                    asyncio.create_task(self._execute_task(task_id, workflow, execution, completed_tasks, running_tasks))
            else:
                # Sequential execution
                if startable_tasks:
                    task_id = startable_tasks[0]
                    pending_tasks.discard(task_id)
                    running_tasks.add(task_id)
                    await self._execute_task(task_id, workflow, execution, completed_tasks, running_tasks)

            # Wait a bit before checking again
            await asyncio.sleep(0.1)

    def _build_dependency_graph(self, workflow: WorkflowDefinition) -> Dict[str, Set[str]]:
        """Build dependency graph for tasks."""
        graph = {}
        for task_id, task in workflow.tasks.items():
            graph[task_id] = set(task.dependencies)
        return graph

    def _can_start_task(self, task_id: str, workflow: WorkflowDefinition, completed_tasks: Set[str], execution: WorkflowExecution) -> bool:
        """Check if a task can be started."""
        task = workflow.tasks[task_id]
        task_exec = execution.tasks[task_id]

        # Check dependencies
        for dep_id in task.dependencies:
            if dep_id not in completed_tasks:
                return False

        # Check conditions
        for condition in task.conditions:
            if not self._evaluate_condition(condition, execution.context):
                return False

        # Check approval if required
        if task.requires_approval and task_exec.approval_status != ApprovalStatus.APPROVED:
            return False

        return True

    async def _execute_task(
        self,
        task_id: str,
        workflow: WorkflowDefinition,
        execution: WorkflowExecution,
        completed_tasks: Set[str],
        running_tasks: Set[str]
    ):
        """Execute a single task."""
        task = workflow.tasks[task_id]
        task_exec = execution.tasks[task_id]

        try:
            task_exec.status = TaskStatus.RUNNING
            task_exec.start_time = datetime.now()

            # Get executor
            executor = self.executors.get(task.task_type)
            if not executor:
                raise ValueError(f"No executor found for task type: {task.task_type}")

            # Execute with timeout
            if task.timeout:
                result = await asyncio.wait_for(
                    executor.execute(task, execution.context),
                    timeout=task.timeout
                )
            else:
                result = await executor.execute(task, execution.context)

            # Handle approval tasks specially
            if task.task_type == "approval":
                task_exec.status = TaskStatus.WAITING
                task_exec.result = result
                # Wait for approval
                await self._wait_for_approval(task_id, execution, task)
            else:
                task_exec.status = TaskStatus.COMPLETED
                task_exec.result = result

        except asyncio.TimeoutError:
            task_exec.status = TaskStatus.FAILED
            task_exec.error = f"Task timeout after {task.timeout}s"
        except Exception as exc:
            task_exec.status = TaskStatus.FAILED
            task_exec.error = str(exc)
        finally:
            task_exec.end_time = datetime.now()
            running_tasks.discard(task_id)

            if task_exec.status == TaskStatus.COMPLETED:
                completed_tasks.add(task_id)
                # Trigger next tasks
                await self._trigger_next_tasks(task, workflow, execution, completed_tasks, running_tasks)
            elif task_exec.status == TaskStatus.FAILED:
                # Handle failure
                await self._handle_task_failure(task, workflow, execution, completed_tasks, running_tasks)

    async def _wait_for_approval(self, task_id: str, execution: WorkflowExecution, task: WorkflowTask):
        """Wait for approval on a task."""
        # Create approval request
        approval_request = {
            "execution_id": execution.execution_id,
            "task_id": task_id,
            "approvers": task.approvers,
            "message": task.config.get("message", "Approval required"),
            "timeout": task.config.get("timeout", 3600),
            "created_at": datetime.now()
        }

        self.pending_approvals[f"{execution.execution_id}_{task_id}"] = approval_request

        # Wait for approval (simplified - in real implementation would have proper approval workflow)
        await asyncio.sleep(1)  # Placeholder

        # For demo purposes, auto-approve
        await self.approve_task(execution.execution_id, task_id, "auto_approver")

    async def _trigger_next_tasks(
        self,
        task: WorkflowTask,
        workflow: WorkflowDefinition,
        execution: WorkflowExecution,
        completed_tasks: Set[str],
        running_tasks: Set[str]
    ):
        """Trigger next tasks based on task outcome."""
        next_tasks = task.on_success if execution.tasks[task.id].status == TaskStatus.COMPLETED else task.on_failure

        for next_task_id in next_tasks:
            if next_task_id in workflow.tasks and next_task_id not in completed_tasks and next_task_id not in running_tasks:
                if self._can_start_task(next_task_id, workflow, completed_tasks, execution):
                    running_tasks.add(next_task_id)
                    asyncio.create_task(self._execute_task(next_task_id, workflow, execution, completed_tasks, running_tasks))

    async def _handle_task_failure(
        self,
        task: WorkflowTask,
        workflow: WorkflowDefinition,
        execution: WorkflowExecution,
        completed_tasks: Set[str],
        running_tasks: Set[str]
    ):
        """Handle task failure."""
        task_exec = execution.tasks[task.id]

        # Check retry logic
        if task_exec.attempts < task.retry_count:
            task_exec.attempts += 1
            logger.info(f"Retrying task {task.id} (attempt {task_exec.attempts})")
            await asyncio.sleep(task.retry_delay)
            running_tasks.add(task.id)
            asyncio.create_task(self._execute_task(task.id, workflow, execution, completed_tasks, running_tasks))
        else:
            # Mark workflow as failed if critical task
            if task.config.get("critical", False):
                execution.status = WorkflowStatus.FAILED

    def _evaluate_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate a task condition."""
        condition_type = condition.get("type")
        if condition_type == "variable_exists":
            var_name = condition.get("variable")
            return var_name is not None and var_name in context
        elif condition_type == "variable_equals":
            var_name = condition.get("variable")
            if var_name is None:
                return False
            return context.get(var_name) == condition.get("value")
        elif condition_type == "expression":
            # Simple expression evaluation (placeholder)
            return True
        else:
            return True

    async def approve_task(self, execution_id: str, task_id: str, approver: str) -> bool:
        """Approve a task requiring approval."""
        approval_key = f"{execution_id}_{task_id}"

        if approval_key not in self.pending_approvals:
            return False

        execution = self.executions.get(execution_id)
        if not execution:
            return False

        task_exec = execution.tasks.get(task_id)
        if not task_exec:
            return False

        task_exec.approval_status = ApprovalStatus.APPROVED
        task_exec.approved_by = approver
        task_exec.approval_time = datetime.now()

        # Remove from pending approvals
        del self.pending_approvals[approval_key]

        # Resume task execution
        task_exec.status = TaskStatus.COMPLETED

        logger.info(f"Task {task_id} approved by {approver}")
        return True

    def reject_task(self, execution_id: str, task_id: str, approver: str, reason: str = "") -> bool:
        """Reject a task requiring approval."""
        approval_key = f"{execution_id}_{task_id}"

        if approval_key not in self.pending_approvals:
            return False

        execution = self.executions.get(execution_id)
        if not execution:
            return False

        task_exec = execution.tasks.get(task_id)
        if not task_exec:
            return False

        task_exec.approval_status = ApprovalStatus.REJECTED
        task_exec.approved_by = approver
        task_exec.approval_time = datetime.now()
        task_exec.error = f"Rejected by {approver}: {reason}"

        # Remove from pending approvals
        del self.pending_approvals[approval_key]

        # Mark task as failed
        task_exec.status = TaskStatus.FAILED

        logger.info(f"Task {task_id} rejected by {approver}")
        return True

    def pause_execution(self, execution_id: str) -> bool:
        """Pause a running workflow execution."""
        execution = self.executions.get(execution_id)
        if execution and execution.status == WorkflowStatus.RUNNING:
            execution.status = WorkflowStatus.PAUSED
            logger.info(f"Paused execution: {execution_id}")
            return True
        return False

    def resume_execution(self, execution_id: str) -> bool:
        """Resume a paused workflow execution."""
        execution = self.executions.get(execution_id)
        if execution and execution.status == WorkflowStatus.PAUSED:
            execution.status = WorkflowStatus.RUNNING
            logger.info(f"Resumed execution: {execution_id}")
            return True
        return False

    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a workflow execution."""
        execution = self.executions.get(execution_id)
        if execution and execution.status in [WorkflowStatus.RUNNING, WorkflowStatus.PAUSED, WorkflowStatus.WAITING_APPROVAL]:
            execution.status = WorkflowStatus.CANCELLED
            execution.end_time = datetime.now()
            logger.info(f"Cancelled execution: {execution_id}")
            return True
        return False

    def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get the status of a workflow execution."""
        return self.executions.get(execution_id)

    def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get workflow execution metrics."""
        return self.metrics.copy()

    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get list of pending approvals."""
        return list(self.pending_approvals.values())

    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows."""
        return [
            {
                "id": wf.id,
                "name": wf.name,
                "description": wf.description,
                "version": wf.version,
                "tasks_count": len(wf.tasks),
                "created_at": wf.created_at.isoformat(),
                "updated_at": wf.updated_at.isoformat()
            }
            for wf in self.workflows.values()
        ]

    def list_executions(self, workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List workflow executions."""
        executions = self.executions.values()
        if workflow_id:
            executions = [e for e in executions if e.workflow_id == workflow_id]

        return [
            {
                "execution_id": e.execution_id,
                "workflow_id": e.workflow_id,
                "status": e.status.value,
                "start_time": e.start_time.isoformat() if e.start_time else None,
                "end_time": e.end_time.isoformat() if e.end_time else None,
                "tasks_completed": sum(1 for t in e.tasks.values() if t.status == TaskStatus.COMPLETED),
                "tasks_total": len(e.tasks)
            }
            for e in executions
        ]


# Register with provider registry
def create_workflow_automation_composer(config_manager=None, **kwargs):
    """Factory function for WorkflowAutomationComposer."""
    return WorkflowAutomationComposer(config_manager=config_manager, **kwargs)


provider_registry.register_lazy(
    'adaptive_intelligence', 'workflow_composer',
    'mia.adaptive_intelligence.workflow_automation_composer', 'create_workflow_automation_composer',
    default=True
)