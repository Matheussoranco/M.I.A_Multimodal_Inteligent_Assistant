"""Register default providers used across the application."""

from __future__ import annotations

from .registry import provider_registry


def register_default_providers() -> None:
    """Register built-in providers with lazy import factories."""
    provider_registry.register_lazy(
        "llm", "default", "mia.llm.llm_manager", "LLMManager", default=True
    )

    provider_registry.register_lazy(
        "audio", "utils", "mia.audio.audio_utils", "AudioUtils", default=True
    )
    provider_registry.register_lazy(
        "audio", "processor", "mia.audio.speech_processor", "SpeechProcessor"
    )
    provider_registry.register_lazy(
        "audio", "generator", "mia.audio.speech_generator", "SpeechGenerator"
    )

    provider_registry.register_lazy(
        "vision",
        "default",
        "mia.multimodal.vision_processor",
        "VisionProcessor",
        default=True,
    )
    provider_registry.register_lazy(
        "multimodal",
        "default",
        "mia.multimodal.processor",
        "MultimodalProcessor",
        default=True,
    )

    provider_registry.register_lazy(
        "memory",
        "knowledge",
        "mia.memory.knowledge_graph",
        "AgentMemory",
        default=True,
    )
    provider_registry.register_lazy(
        "memory", "long_term", "mia.memory.long_term_memory", "LongTermMemory"
    )
    provider_registry.register_lazy(
        "memory", "manager", "mia.memory.memory_manager", "UnifiedMemory"
    )
    provider_registry.register_lazy(
        "rag",
        "pipeline",
        "mia.memory.rag_pipeline",
        "RAGPipeline",
        default=True,
    )

    provider_registry.register_lazy(
        "web", "agent", "mia.web.web_agent", "WebAgent", default=True
    )

    provider_registry.register_lazy(
        "langchain",
        "verifier",
        "mia.langchain.langchain_verifier",
        "LangChainVerifier",
    )
    provider_registry.register_lazy(
        "system", "control", "mia.system.system_control", "SystemControl"
    )
    provider_registry.register_lazy(
        "desktop", "automation", "mia.system.desktop_automation", "DesktopAutomation", default=True
    )
    provider_registry.register_lazy(
        "security",
        "default",
        "mia.security.security_manager",
        "SecurityManager",
    )
    provider_registry.register_lazy(
        "plugins", "default", "mia.plugins.plugin_manager", "PluginManager"
    )

    provider_registry.register_lazy(
        "actions",
        "default",
        "mia.tools.action_executor",
        "ActionExecutor",
        default=True,
    )
    provider_registry.register_lazy(
        "documents",
        "generator",
        "mia.tools.document_generator",
        "DocumentGenerator",
        default=True,
    )
    provider_registry.register_lazy(
        "sandbox",
        "wasi",
        "mia.sandbox.wasi_runner",
        "WasiSandbox",
        default=True,
    )
    provider_registry.register_lazy(
        "messaging",
        "telegram",
        "mia.messaging.telegram_client",
        "TelegramMessenger",
        default=True,
    )


__all__ = ["register_default_providers"]
