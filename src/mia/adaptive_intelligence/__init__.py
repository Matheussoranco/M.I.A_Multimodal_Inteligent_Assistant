"""
Adaptive Intelligence module for M.I.A.
"""

from .dialog_manager import DialogManager, create_dialog_manager
from .modality_manager import ModalityManager, create_modality_manager
from .context_infuser import ContextInfuser, create_context_infuser
from .feedback_loop import FeedbackLoop, create_feedback_loop
from .knowledge_memory_graph import KnowledgeMemoryGraph, create_knowledge_memory_graph
from .multimodal_perception_suite import MultimodalPerceptionSuite, create_multimodal_perception_suite
from .workflow_automation_composer import WorkflowAutomationComposer, create_workflow_automation_composer
from .observability_admin_console import ObservabilityAdminConsole, create_observability_admin_console
from .hybrid_llm_orchestration import HybridLLMOrchestrator, create_hybrid_llm_orchestrator
from .episodic_memory_personalization import EpisodicMemoryPersonalizationSystem, create_episodic_memory_system
from .proactive_automation_engine import ProactiveAutomationEngine, create_proactive_automation_engine
from .multi_device_synchronization import MultiDeviceSynchronizationSystem, create_multi_device_sync_system

__all__ = [
    # Dialog Management
    'DialogManager',
    'create_dialog_manager',

    # Modality Management
    'ModalityManager',
    'create_modality_manager',

    # Context Infusion
    'ContextInfuser',
    'create_context_infuser',

    # Feedback Loop
    'FeedbackLoop',
    'create_feedback_loop',

    # Knowledge Memory Graph
    'KnowledgeMemoryGraph',
    'create_knowledge_memory_graph',

    # Multimodal Perception Suite
    'MultimodalPerceptionSuite',
    'create_multimodal_perception_suite',

    # Workflow Automation Composer
    'WorkflowAutomationComposer',
    'create_workflow_automation_composer',

    # Observability & Admin Console
    'ObservabilityAdminConsole',
    'create_observability_admin_console',

    # Hybrid LLM Orchestration
    'HybridLLMOrchestrator',
    'create_hybrid_llm_orchestrator',

    # Episodic Memory & Personalization
    'EpisodicMemoryPersonalizationSystem',
    'create_episodic_memory_system',

    # Proactive Automation Engine
    'ProactiveAutomationEngine',
    'create_proactive_automation_engine',

    # Multi-Device Synchronization
    'MultiDeviceSynchronizationSystem',
    'create_multi_device_sync_system',
]

__version__ = "2.0.0"
__author__ = "M.I.A. Development Team"
__description__ = "Advanced AI Foundations for Multimodal Intelligent Assistant"