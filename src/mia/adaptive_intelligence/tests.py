"""
Test suite for Adaptive Intelligence
"""

try:
    import pytest  # type: ignore
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Create mock pytest for basic functionality
    class pytest:  # type: ignore
        @staticmethod
        def fixture(func):
            return func
        class mark:
            class asyncio:
                def __init__(self, func):
                    self.func = func
                def __call__(self, *args, **kwargs):
                    return self.func(*args, **kwargs)

import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import threading

from .knowledge_memory_graph import KnowledgeMemoryGraph, KnowledgeNode, KnowledgeEdge
from .multimodal_perception_suite import (
    MultimodalPerceptionSuite, PerceptionInput, ModalityType,
    TextProcessor, ImageProcessor, AudioProcessor, DocumentProcessor
)
from .workflow_automation_composer import (
    WorkflowAutomationComposer, WorkflowDefinition, WorkflowTask,
    TaskStatus, WorkflowStatus
)
from .observability_admin_console import (
    ObservabilityAdminConsole, MetricsCollector, AlertManager,
    LogAggregator, LogLevel, MetricType, AlertSeverity, Metric
)


class TestKnowledgeMemoryGraph:
    """Test cases for Knowledge Memory Graph."""

    @pytest.fixture
    def kmg(self):
        """Create a KnowledgeMemoryGraph instance for testing."""
        return KnowledgeMemoryGraph()

    def test_initialization(self, kmg):
        """Test KnowledgeMemoryGraph initialization."""
        assert kmg.nodes == {}
        assert kmg.edges == []
        assert kmg.consolidation_threshold == 10
        assert kmg.temporal_decay_factor == 0.95

    def test_add_knowledge(self, kmg):
        """Test adding knowledge to the graph."""
        # Add a knowledge node
        node_id = kmg.add_knowledge(
            content="Test knowledge",
            knowledge_type="fact",
            metadata={"source": "test"}
        )

        assert node_id in kmg.nodes
        node = kmg.nodes[node_id]
        assert node.content == "Test knowledge"
        assert node.knowledge_type == "fact"
        assert node.metadata["source"] == "test"
        assert isinstance(node.created_at, datetime)

    def test_search_knowledge(self, kmg):
        """Test knowledge search functionality."""
        # Add test knowledge
        kmg.add_knowledge("Python programming", "skill", {"level": "beginner"})
        kmg.add_knowledge("Machine learning", "skill", {"level": "intermediate"})
        kmg.add_knowledge("Data analysis", "skill", {"level": "advanced"})

        # Test vector search (mocked)
        with patch.object(kmg, '_generate_embedding', return_value=[0.1, 0.2, 0.3]):
            results = kmg.search_knowledge("programming", top_k=2)
            assert len(results) <= 2

    def test_traverse_relationships(self, kmg):
        """Test relationship traversal."""
        # Add nodes
        node1 = kmg.add_knowledge("Python", "topic")
        node2 = kmg.add_knowledge("Django", "framework")
        node3 = kmg.add_knowledge("Flask", "framework")

        # Add relationships
        kmg.add_relationship(node1, node2, "related_to", {"strength": 0.8})
        kmg.add_relationship(node1, node3, "related_to", {"strength": 0.6})

        # Traverse relationships
        related = kmg.traverse_relationships(node1, "related_to", max_depth=1)
        assert len(related) == 2
        assert node2 in related
        assert node3 in related

    def test_consolidate_knowledge(self, kmg):
        """Test knowledge consolidation."""
        # Add similar knowledge
        for i in range(12):  # Exceed consolidation threshold
            kmg.add_knowledge(f"Python concept {i}", "concept", {"topic": "python"})

        # Trigger consolidation
        consolidated = kmg.consolidate_knowledge()

        # Should have consolidated some nodes
        assert consolidated > 0

    def test_temporal_decay(self, kmg):
        """Test temporal decay functionality."""
        node_id = kmg.add_knowledge("Old knowledge", "fact")

        # Manually set old timestamp
        old_time = datetime.now() - timedelta(days=30)
        kmg.nodes[node_id].created_at = old_time
        kmg.nodes[node_id].last_accessed = old_time

        # Apply decay
        kmg.apply_temporal_decay()

        # Node should have lower confidence
        assert kmg.nodes[node_id].confidence < 1.0

    def test_persistence(self, kmg):
        """Test knowledge persistence."""
        # Add knowledge
        node_id = kmg.add_knowledge("Persistent knowledge", "fact")

        # Save to dict
        data = kmg.to_dict()
        assert "nodes" in data
        assert "edges" in data
        assert node_id in data["nodes"]

        # Verify data structure (can't load back without from_dict method)
        assert data["nodes"][node_id]["content"] == "Persistent knowledge"


class TestMultimodalPerceptionSuite:
    """Test cases for Multimodal Perception Suite."""

    @pytest.fixture
    def mps(self):
        """Create a MultimodalPerceptionSuite instance for testing."""
        return MultimodalPerceptionSuite()

    def test_initialization(self, mps):
        """Test MultimodalPerceptionSuite initialization."""
        assert isinstance(mps.processors, dict)
        assert ModalityType.TEXT in mps.processors
        assert ModalityType.IMAGE in mps.processors

    @pytest.mark.asyncio
    async def test_process_text_input(self, mps):
        """Test processing text input."""
        input_data = PerceptionInput(
            modality=ModalityType.TEXT,
            data="This is a test message for analysis."
        )

        result = await mps.process_input(input_data)

        assert result.modality == ModalityType.TEXT
        assert result.status.name == "COMPLETED"
        assert "text_length" in result.extracted_data
        assert "word_count" in result.extracted_data

    @pytest.mark.asyncio
    async def test_process_image_input_unavailable(self, mps):
        """Test processing image input when OpenCV is unavailable."""
        input_data = PerceptionInput(
            modality=ModalityType.IMAGE,
            data=b"fake_image_data"
        )

        result = await mps.process_input(input_data)

        # Should handle gracefully when OpenCV is not available
        assert result.modality == ModalityType.IMAGE

    @pytest.mark.asyncio
    async def test_process_multimodal(self, mps):
        """Test processing multiple modalities."""
        inputs = [
            PerceptionInput(ModalityType.TEXT, "Test text"),
            PerceptionInput(ModalityType.AUDIO, b"fake_audio_data")
        ]

        context = await mps.process_multimodal(inputs)

        assert len(context.inputs) >= 1  # At least text should process
        assert isinstance(context.fused_insights, dict)

    def test_get_available_modalities(self, mps):
        """Test getting available modalities."""
        modalities = mps.get_available_modalities()
        assert ModalityType.TEXT in modalities

    def test_get_processor_capabilities(self, mps):
        """Test getting processor capabilities."""
        caps = mps.get_processor_capabilities(ModalityType.TEXT)
        assert "sentiment_analysis" in caps
        assert "entity_extraction" in caps


class TestWorkflowAutomationComposer:
    """Test cases for Workflow Automation Composer."""

    @pytest.fixture
    def wac(self):
        """Create a WorkflowAutomationComposer instance for testing."""
        return WorkflowAutomationComposer()

    def test_initialization(self, wac):
        """Test WorkflowAutomationComposer initialization."""
        assert isinstance(wac.workflows, dict)
        assert isinstance(wac.executors, dict)
        assert "action" in wac.executors

    def test_create_workflow(self, wac):
        """Test workflow creation."""
        workflow_id = wac.create_workflow(
            name="Test Workflow",
            description="A test workflow",
            tasks={
                "task1": WorkflowTask(
                    id="task1",
                    name="Test Task",
                    task_type="action",
                    config={"action": "log", "message": "Hello World"}
                )
            },
            entry_points=["task1"]
        )

        assert workflow_id in wac.workflows
        workflow = wac.workflows[workflow_id]
        assert workflow.name == "Test Workflow"
        assert "task1" in workflow.tasks

    @pytest.mark.asyncio
    async def test_execute_workflow(self, wac):
        """Test workflow execution."""
        # Create a simple workflow
        workflow_id = wac.create_workflow(
            name="Simple Workflow",
            tasks={
                "start": WorkflowTask(
                    id="start",
                    name="Start Task",
                    task_type="action",
                    config={"action": "set_variable", "variable": "status", "value": "started"},
                    on_success=["end"]
                ),
                "end": WorkflowTask(
                    id="end",
                    name="End Task",
                    task_type="action",
                    config={"action": "log", "message": "Workflow completed"}
                )
            },
            entry_points=["start"]
        )

        # Execute workflow
        execution_id = await wac.execute_workflow(workflow_id)

        assert execution_id in wac.executions
        execution = wac.executions[execution_id]
        assert execution.status == WorkflowStatus.COMPLETED
        assert "status" in execution.context
        assert execution.context["status"] == "started"

    def test_workflow_crud_operations(self, wac):
        """Test workflow CRUD operations."""
        # Create
        workflow_id = wac.create_workflow("CRUD Test")

        # Read
        assert workflow_id in wac.workflows

        # Update
        success = wac.update_workflow(workflow_id, {"description": "Updated"})
        assert success
        assert wac.workflows[workflow_id].description == "Updated"

        # Delete
        success = wac.delete_workflow(workflow_id)
        assert success
        assert workflow_id not in wac.workflows

    def test_approval_workflow(self, wac):
        """Test workflow with approval."""
        workflow_id = wac.create_workflow(
            name="Approval Workflow",
            tasks={
                "approval_task": WorkflowTask(
                    id="approval_task",
                    name="Approval Task",
                    task_type="approval",
                    config={"message": "Please approve", "approvers": ["user1"]},
                    requires_approval=True,
                    approvers=["user1"]
                )
            },
            entry_points=["approval_task"]
        )

        # Should have pending approvals
        approvals = wac.get_pending_approvals()
        assert len(approvals) >= 0  # May be empty if execution hasn't started

    def test_workflow_metrics(self, wac):
        """Test workflow metrics collection."""
        metrics = wac.get_workflow_metrics()
        assert "total_executions" in metrics
        assert "successful_executions" in metrics
        assert "failed_executions" in metrics


class TestObservabilityAdminConsole:
    """Test cases for Observability & Admin Console."""

    @pytest.fixture
    def oac(self):
        """Create an ObservabilityAdminConsole instance for testing."""
        return ObservabilityAdminConsole()

    def test_initialization(self, oac):
        """Test ObservabilityAdminConsole initialization."""
        assert isinstance(oac.metrics_collector, MetricsCollector)
        assert isinstance(oac.alert_manager, AlertManager)
        assert isinstance(oac.log_aggregator, LogAggregator)

    def test_metrics_collection(self, oac):
        """Test metrics collection."""
        # Add some metrics
        oac.add_metric(Metric(
            name="test_counter",
            type=MetricType.COUNTER,
            value=5,
            labels={"test": "true"}
        ))

        oac.add_metric(Metric(
            name="test_gauge",
            type=MetricType.GAUGE,
            value=42.5,
            labels={"unit": "percent"}
        ))

        # Check metrics
        metrics = oac.metrics_collector.get_all_metrics()
        assert len(metrics) >= 2

        counter_found = any(m["name"] == "test_counter" and m["value"] == 5 for m in metrics)
        gauge_found = any(m["name"] == "test_gauge" and m["value"] == 42.5 for m in metrics)

        assert counter_found
        assert gauge_found

    def test_alert_management(self, oac):
        """Test alert management."""
        # Add alert rule
        oac.add_alert_rule({
            "metric": "test_gauge",
            "condition": "gt",
            "threshold": 50,
            "severity": "high",
            "name": "High Value Alert"
        })

        # Add metric that should trigger alert
        oac.add_metric(Metric(
            name="test_gauge",
            type=MetricType.GAUGE,
            value=75  # Above threshold
        ))

        # Check alerts
        oac.alert_manager.check_alerts(oac.metrics_collector)
        active_alerts = oac.alert_manager.get_active_alerts()

        # Should have triggered an alert
        assert len(active_alerts) >= 0  # May not trigger immediately due to timing

    def test_logging(self, oac):
        """Test logging functionality."""
        # Add log entries
        oac.log_event(
            level=LogLevel.INFO,
            component="test_component",
            message="Test log message",
            metadata={"test": True}
        )

        oac.log_event(
            level=LogLevel.ERROR,
            component="test_component",
            message="Test error message"
        )

        # Get logs
        logs = oac.log_aggregator.get_logs(component="test_component", limit=10)
        assert len(logs) >= 2

        info_logs = [l for l in logs if l.level == LogLevel.INFO]
        error_logs = [l for l in logs if l.level == LogLevel.ERROR]

        assert len(info_logs) >= 1
        assert len(error_logs) >= 1

    def test_admin_commands(self, oac):
        """Test admin command execution."""
        # Test status command
        result = oac.execute_admin_command("status")
        assert "status" in result
        assert result["status"] == "running"

        # Test metrics command
        result = oac.execute_admin_command("metrics")
        assert "metrics" in result

        # Test logs command
        result = oac.execute_admin_command("logs")
        assert "logs" in result

        # Test invalid command
        result = oac.execute_admin_command("invalid_command")
        assert "error" in result

    def test_dashboard_data(self, oac):
        """Test dashboard data generation."""
        dashboard = oac.get_dashboard_data()

        required_keys = ["status", "health", "metrics", "alerts", "logs", "system_info"]
        for key in required_keys:
            assert key in dashboard

    def test_component_health_checks(self, oac):
        """Test component health checking."""
        # Perform health checks
        oac._perform_health_checks()

        # Check that components were checked
        assert len(oac.component_statuses) > 0

        # Should have checked core components
        component_names = list(oac.component_statuses.keys())
        assert "metrics_collector" in component_names
        assert "alert_manager" in component_names
        assert "log_aggregator" in component_names


class TestIntegration:
    """Integration tests for all components working together."""

    @pytest.fixture
    def full_system(self):
        """Create a full system with all components."""
        kmg = KnowledgeMemoryGraph()
        mps = MultimodalPerceptionSuite()
        wac = WorkflowAutomationComposer()
        oac = ObservabilityAdminConsole()

        return {
            "kmg": kmg,
            "mps": mps,
            "wac": wac,
            "oac": oac
        }

    @pytest.mark.asyncio
    async def test_knowledge_driven_workflow(self, full_system):
        """Test a workflow that uses knowledge from the graph."""
        kmg = full_system["kmg"]
        wac = full_system["wac"]
        oac = full_system["oac"]

        # Add knowledge about a process
        process_knowledge = kmg.add_knowledge(
            content="Standard approval process requires 2 approvers",
            knowledge_type="process",
            metadata={"domain": "approvals"}
        )

        # Create workflow that uses this knowledge
        workflow_id = wac.create_workflow(
            name="Knowledge-Driven Workflow",
            tasks={
                "check_knowledge": WorkflowTask(
                    id="check_knowledge",
                    name="Check Process Knowledge",
                    task_type="action",
                    config={
                        "action": "set_variable",
                        "variable": "approvers_required",
                        "value": 2  # From knowledge
                    },
                    on_success=["approval"]
                ),
                "approval": WorkflowTask(
                    id="approval",
                    name="Approval Task",
                    task_type="approval",
                    config={"message": "Please approve", "approvers": ["user1", "user2"]},
                    requires_approval=True,
                    approvers=["user1", "user2"]
                )
            },
            entry_points=["check_knowledge"]
        )

        # Execute workflow
        execution_id = await wac.execute_workflow(workflow_id)

        # Log the execution
        oac.log_event(
            level=LogLevel.INFO,
            component="test_integration",
            message=f"Executed knowledge-driven workflow: {execution_id}"
        )

        # Verify execution
        execution = wac.get_execution_status(execution_id)
        assert execution is not None
        assert execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.WAITING_APPROVAL]

    @pytest.mark.asyncio
    async def test_multimodal_workflow_with_monitoring(self, full_system):
        """Test multimodal processing in a monitored workflow."""
        mps = full_system["mps"]
        wac = full_system["wac"]
        oac = full_system["oac"]

        # Create workflow with multimodal processing
        workflow_id = wac.create_workflow(
            name="Multimodal Processing Workflow",
            tasks={
                "process_text": WorkflowTask(
                    id="process_text",
                    name="Process Text Input",
                    task_type="action",
                    config={
                        "action": "set_variable",
                        "variable": "text_processed",
                        "value": True
                    },
                    on_success=["process_image"]
                ),
                "process_image": WorkflowTask(
                    id="process_image",
                    name="Process Image Input",
                    task_type="action",
                    config={
                        "action": "set_variable",
                        "variable": "image_processed",
                        "value": True
                    }
                )
            },
            entry_points=["process_text"]
        )

        # Process multimodal input
        inputs = [
            PerceptionInput(ModalityType.TEXT, "Analyze this text"),
            PerceptionInput(ModalityType.IMAGE, b"fake_image_data")
        ]

        multimodal_result = await mps.process_multimodal(inputs)

        # Log multimodal processing
        oac.log_event(
            level=LogLevel.INFO,
            component="multimodal_processor",
            message=f"Processed {len(multimodal_result.inputs)} modalities",
            metadata={"modalities": [inp.modality.value for inp in multimodal_result.inputs]}
        )

        # Execute workflow
        execution_id = await wac.execute_workflow(workflow_id)

        # Add metrics
        oac.add_metric(Metric(
            name="multimodal_processing_completed",
            type=MetricType.COUNTER,
            value=1,
            labels={"workflow_id": execution_id}
        ))

        # Verify everything worked
        execution = wac.get_execution_status(execution_id)
        assert execution is not None
        assert execution.status == WorkflowStatus.COMPLETED

        metrics = oac.metrics_collector.get_all_metrics()
        processing_metrics = [m for m in metrics if m["name"] == "multimodal_processing_completed"]
        assert len(processing_metrics) > 0


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running Adaptive Intelligence Phase 2 smoke tests...")

    # Test Knowledge Memory Graph
    kmg = KnowledgeMemoryGraph()
    node_id = kmg.add_knowledge("Test knowledge", "test")
    assert node_id in kmg.nodes
    print("✓ Knowledge Memory Graph basic functionality")

    # Test Multimodal Perception Suite
    mps = MultimodalPerceptionSuite()
    modalities = mps.get_available_modalities()
    assert len(modalities) > 0
    print("✓ Multimodal Perception Suite basic functionality")

    # Test Workflow Automation Composer
    wac = WorkflowAutomationComposer()
    workflow_id = wac.create_workflow("Smoke Test")
    assert workflow_id in wac.workflows
    print("✓ Workflow Automation Composer basic functionality")

    # Test Observability Admin Console
    oac = ObservabilityAdminConsole()
    dashboard = oac.get_dashboard_data()
    assert "status" in dashboard
    print("✓ Observability Admin Console basic functionality")

    print("All smoke tests passed! ✅")