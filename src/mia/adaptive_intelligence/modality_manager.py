"""
Modality Manager
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..providers import provider_registry

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Enumeration of supported modalities."""

    TEXT = "text"
    VOICE = "voice"
    VISION = "vision"
    MULTIMODAL = "multimodal"


class ModalityState(Enum):
    """Enumeration of modality states."""

    AVAILABLE = "available"
    ACTIVE = "active"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    RECOVERING = "recovering"


@dataclass
class ModalityCapability:
    """Represents a modality capability with health monitoring."""

    modality: ModalityType
    state: ModalityState = ModalityState.AVAILABLE
    confidence: float = 1.0
    last_check: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    success_count: int = 0
    average_response_time: float = 0.0
    capabilities: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModalitySwitch:
    """Represents a modality switching decision."""

    from_modality: ModalityType
    to_modality: ModalityType
    reason: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModalityContext:
    """Context for modality operations."""

    current_modality: ModalityType
    available_modalities: List[ModalityType]
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    system_capabilities: Dict[str, Any] = field(default_factory=dict)
    environmental_factors: Dict[str, Any] = field(default_factory=dict)
    session_history: List[ModalitySwitch] = field(default_factory=list)


class ModalityManager:
    """
    Intelligent modality manager with dynamic switching capabilities.

    Features:
    - Real-time modality health monitoring
    - Intelligent switching based on context and preferences
    - Graceful degradation and recovery
    - User preference learning
    - Environmental adaptation
    """

    def __init__(
        self,
        config_manager=None,
        *,
        health_check_interval: int = 30,
        max_error_threshold: int = 3,
        recovery_timeout: int = 60,
        switching_enabled: bool = True,
    ):
        self.config_manager = config_manager
        self.health_check_interval = health_check_interval
        self.max_error_threshold = max_error_threshold
        self.recovery_timeout = recovery_timeout
        self.switching_enabled = switching_enabled

        # Modality capabilities tracking
        self.capabilities: Dict[ModalityType, ModalityCapability] = {}
        self.capability_lock = threading.RLock()

        # Switching logic
        self.switch_policies: Dict[str, Callable] = {}
        self.modality_history: List[ModalitySwitch] = []
        self.user_patterns: Dict[str, Dict[str, Any]] = {}

        # Health monitoring
        self.health_thread: Optional[threading.Thread] = None
        self.monitoring_active = False

        # Initialize capabilities
        self._initialize_capabilities()

        # Load default switching policies
        self._load_default_policies()

        # Start health monitoring
        self.start_health_monitoring()

        logger.info(
            "Modality Manager initialized with %d capabilities",
            len(self.capabilities),
        )

    def _initialize_capabilities(self):
        """Initialize modality capabilities."""
        capabilities = {
            ModalityType.TEXT: {
                "input_methods": ["keyboard", "pasted_text"],
                "output_methods": ["text_display", "file_output"],
                "reliability": 0.99,
                "speed": "fast",
                "accessibility": "high",
            },
            ModalityType.VOICE: {
                "input_methods": ["microphone", "audio_file"],
                "output_methods": ["tts_synthesis", "audio_playback"],
                "reliability": 0.85,
                "speed": "medium",
                "accessibility": "medium",
                "requires_hardware": True,
            },
            ModalityType.VISION: {
                "input_methods": ["camera", "image_file", "screen_capture"],
                "output_methods": ["image_display", "video_playback"],
                "reliability": 0.90,
                "speed": "medium",
                "accessibility": "medium",
                "requires_hardware": True,
            },
            ModalityType.MULTIMODAL: {
                "input_methods": ["combined_input"],
                "output_methods": ["multimodal_output"],
                "reliability": 0.80,
                "speed": "slow",
                "accessibility": "low",
                "requires_multiple": True,
            },
        }

        for modality, caps in capabilities.items():
            self.capabilities[modality] = ModalityCapability(
                modality=modality, capabilities=caps
            )

    def _load_default_policies(self):
        """Load default modality switching policies."""
        self.switch_policies = {
            "voice_to_text_fallback": self._policy_voice_to_text_fallback,
            "vision_to_text_fallback": self._policy_vision_to_text_fallback,
            "text_to_voice_enhancement": self._policy_text_to_voice_enhancement,
            "multimodal_optimization": self._policy_multimodal_optimization,
            "accessibility_adaptation": self._policy_accessibility_adaptation,
            "performance_optimization": self._policy_performance_optimization,
            "user_preference_switching": self._policy_user_preference_switching,
        }

    def start_health_monitoring(self):
        """Start background health monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.health_thread = threading.Thread(
            target=self._health_monitoring_loop,
            daemon=True,
            name="ModalityHealthMonitor",
        )
        self.health_thread.start()
        logger.info("Started modality health monitoring")

    def stop_health_monitoring(self):
        """Stop background health monitoring."""
        self.monitoring_active = False
        if self.health_thread:
            self.health_thread.join(timeout=5.0)
        logger.info("Stopped modality health monitoring")

    def _health_monitoring_loop(self):
        """Background health monitoring loop."""
        while self.monitoring_active:
            try:
                self._perform_health_checks()
                time.sleep(self.health_check_interval)
            except Exception as exc:
                logger.error("Health monitoring error: %s", exc)
                time.sleep(5)  # Brief pause on error

    def _perform_health_checks(self):
        """Perform health checks on all modalities."""
        with self.capability_lock:
            for modality, capability in self.capabilities.items():
                try:
                    health_score = self._check_modality_health(modality)
                    capability.confidence = health_score
                    capability.last_check = datetime.now()

                    # Update state based on health
                    if health_score >= 0.8:
                        if capability.error_count > 0:
                            capability.error_count = max(
                                0, capability.error_count - 1
                            )
                        capability.state = ModalityState.AVAILABLE
                    elif health_score >= 0.5:
                        capability.state = ModalityState.DEGRADED
                    else:
                        capability.error_count += 1
                        if capability.error_count >= self.max_error_threshold:
                            capability.state = ModalityState.UNAVAILABLE
                            self._attempt_recovery(modality)

                except Exception as exc:
                    logger.debug(
                        "Health check failed for %s: %s", modality.value, exc
                    )
                    capability.error_count += 1
                    if capability.error_count >= self.max_error_threshold:
                        capability.state = ModalityState.UNAVAILABLE

    def _check_modality_health(self, modality: ModalityType) -> float:
        """Check health of a specific modality."""
        # Simplified health checks - in real implementation would test actual components
        try:
            if modality == ModalityType.TEXT:
                # Text is always available
                return 1.0

            elif modality == ModalityType.VOICE:
                # Check if audio components are available
                try:
                    import sounddevice  # type: ignore

                    devices = sounddevice.query_devices()
                    return 0.9 if devices else 0.1
                except (ImportError, Exception):
                    return 0.0

            elif modality == ModalityType.VISION:
                # Check if vision components are available
                try:
                    import cv2  # type: ignore

                    return 0.9
                except (ImportError, Exception):
                    return 0.0

            elif modality == ModalityType.MULTIMODAL:
                # Check combination of modalities
                voice_health = self._check_modality_health(ModalityType.VOICE)
                vision_health = self._check_modality_health(
                    ModalityType.VISION
                )
                return min(voice_health, vision_health) * 0.8

        except Exception as exc:
            logger.debug("Health check error for %s: %s", modality.value, exc)
            return 0.0

        return 0.5  # Default moderate health

    def _attempt_recovery(self, modality: ModalityType):
        """Attempt to recover a failed modality."""
        capability = self.capabilities[modality]
        if capability.state != ModalityState.UNAVAILABLE:
            return

        capability.state = ModalityState.RECOVERING
        logger.info("Attempting recovery for modality %s", modality.value)

        # Simple recovery logic - in real implementation would restart services, etc.
        try:
            # Wait a bit then recheck
            time.sleep(2)
            health = self._check_modality_health(modality)
            if health >= 0.5:
                capability.state = ModalityState.AVAILABLE
                capability.error_count = 0
                logger.info(
                    "Successfully recovered modality %s", modality.value
                )
            else:
                capability.state = ModalityState.UNAVAILABLE
                logger.warning("Failed to recover modality %s", modality.value)
        except Exception as exc:
            logger.error("Recovery failed for %s: %s", modality.value, exc)
            capability.state = ModalityState.UNAVAILABLE

    def should_switch_modality(
        self,
        current_modality: ModalityType,
        context: ModalityContext,
        user_input: Optional[str] = None,
    ) -> Optional[Tuple[ModalityType, str, float]]:
        """
        Determine if modality switching is recommended.

        Returns:
            Tuple of (new_modality, reason, confidence) or None if no switch needed
        """
        if not self.switching_enabled:
            return None

        # Evaluate all switching policies
        best_switch = None
        best_confidence = 0.0

        for policy_name, policy_func in self.switch_policies.items():
            try:
                switch = policy_func(current_modality, context, user_input)
                if switch and switch.confidence > best_confidence:
                    best_switch = switch
                    best_confidence = switch.confidence
            except Exception as exc:
                logger.debug("Policy %s failed: %s", policy_name, exc)

        return (
            (
                best_switch.to_modality,
                best_switch.reason,
                best_switch.confidence,
            )
            if best_switch
            else None
        )

    def _policy_voice_to_text_fallback(
        self,
        current: ModalityType,
        context: ModalityContext,
        user_input: Optional[str],
    ) -> Optional[ModalitySwitch]:
        """Policy: Switch from voice to text if voice is degraded."""
        if current != ModalityType.VOICE:
            return None

        voice_cap = self.capabilities.get(ModalityType.VOICE)
        if voice_cap and voice_cap.state in [
            ModalityState.DEGRADED,
            ModalityState.UNAVAILABLE,
        ]:
            text_cap = self.capabilities.get(ModalityType.TEXT)
            if text_cap and text_cap.state == ModalityState.AVAILABLE:
                return ModalitySwitch(
                    from_modality=current,
                    to_modality=ModalityType.TEXT,
                    reason="Voice input degraded, switching to text fallback",
                    confidence=0.8,
                )

        return None

    def _policy_vision_to_text_fallback(
        self,
        current: ModalityType,
        context: ModalityContext,
        user_input: Optional[str],
    ) -> Optional[ModalitySwitch]:
        """Policy: Switch from vision to text if vision is degraded."""
        if current != ModalityType.VISION:
            return None

        vision_cap = self.capabilities.get(ModalityType.VISION)
        if vision_cap and vision_cap.state in [
            ModalityState.DEGRADED,
            ModalityState.UNAVAILABLE,
        ]:
            text_cap = self.capabilities.get(ModalityType.TEXT)
            if text_cap and text_cap.state == ModalityState.AVAILABLE:
                return ModalitySwitch(
                    from_modality=current,
                    to_modality=ModalityType.TEXT,
                    reason="Vision input degraded, switching to text fallback",
                    confidence=0.7,
                )

        return None

    def _policy_text_to_voice_enhancement(
        self,
        current: ModalityType,
        context: ModalityContext,
        user_input: Optional[str],
    ) -> Optional[ModalitySwitch]:
        """Policy: Switch from text to voice for better UX."""
        if current != ModalityType.TEXT:
            return None

        # Check user preferences
        if context.user_preferences.get("preferred_output") == "voice":
            voice_cap = self.capabilities.get(ModalityType.VOICE)
            if voice_cap and voice_cap.state == ModalityState.AVAILABLE:
                return ModalitySwitch(
                    from_modality=current,
                    to_modality=ModalityType.VOICE,
                    reason="User prefers voice output",
                    confidence=0.6,
                )

        return None

    def _policy_multimodal_optimization(
        self,
        current: ModalityType,
        context: ModalityContext,
        user_input: Optional[str],
    ) -> Optional[ModalitySwitch]:
        """Policy: Switch to multimodal when multiple inputs are available."""
        if current == ModalityType.MULTIMODAL:
            return None

        # Check if multiple modalities are available and would be beneficial
        available_modalities = [
            mod
            for mod in context.available_modalities
            if self.capabilities[mod].state == ModalityState.AVAILABLE
        ]

        if len(available_modalities) >= 2:
            multimodal_cap = self.capabilities.get(ModalityType.MULTIMODAL)
            if (
                multimodal_cap
                and multimodal_cap.state == ModalityState.AVAILABLE
            ):
                return ModalitySwitch(
                    from_modality=current,
                    to_modality=ModalityType.MULTIMODAL,
                    reason="Multiple modalities available, switching to multimodal",
                    confidence=0.5,
                )

        return None

    def _policy_accessibility_adaptation(
        self,
        current: ModalityType,
        context: ModalityContext,
        user_input: Optional[str],
    ) -> Optional[ModalitySwitch]:
        """Policy: Switch modalities based on accessibility needs."""
        # Check for accessibility preferences
        accessibility_needs = context.user_preferences.get("accessibility", {})

        if (
            accessibility_needs.get("voice_preferred")
            and current != ModalityType.VOICE
        ):
            voice_cap = self.capabilities.get(ModalityType.VOICE)
            if voice_cap and voice_cap.state == ModalityState.AVAILABLE:
                return ModalitySwitch(
                    from_modality=current,
                    to_modality=ModalityType.VOICE,
                    reason="Accessibility preference for voice",
                    confidence=0.9,
                )

        elif (
            accessibility_needs.get("text_preferred")
            and current != ModalityType.TEXT
        ):
            text_cap = self.capabilities.get(ModalityType.TEXT)
            if text_cap and text_cap.state == ModalityState.AVAILABLE:
                return ModalitySwitch(
                    from_modality=current,
                    to_modality=ModalityType.TEXT,
                    reason="Accessibility preference for text",
                    confidence=0.9,
                )

        return None

    def _policy_performance_optimization(
        self,
        current: ModalityType,
        context: ModalityContext,
        user_input: Optional[str],
    ) -> Optional[ModalitySwitch]:
        """Policy: Switch to faster/better performing modality."""
        current_cap = self.capabilities.get(current)
        if not current_cap:
            return None

        # Find better performing modality
        best_modality = current
        best_score = self._calculate_performance_score(current_cap)

        for modality, capability in self.capabilities.items():
            if (
                modality == current
                or capability.state != ModalityState.AVAILABLE
            ):
                continue

            score = self._calculate_performance_score(capability)
            if score > best_score * 1.2:  # 20% improvement threshold
                best_modality = modality
                best_score = score

        if best_modality != current:
            return ModalitySwitch(
                from_modality=current,
                to_modality=best_modality,
                reason="Performance optimization",
                confidence=0.4,
            )

        return None

    def _policy_user_preference_switching(
        self,
        current: ModalityType,
        context: ModalityContext,
        user_input: Optional[str],
    ) -> Optional[ModalitySwitch]:
        """Policy: Switch based on learned user preferences."""
        user_id = context.user_preferences.get("user_id")
        if not user_id:
            return None

        # Get user pattern
        pattern = self.user_patterns.get(user_id, {})
        preferred_modality = pattern.get("preferred_modality")

        if preferred_modality and preferred_modality != current:
            pref_cap = self.capabilities.get(ModalityType(preferred_modality))
            if pref_cap and pref_cap.state == ModalityState.AVAILABLE:
                return ModalitySwitch(
                    from_modality=current,
                    to_modality=ModalityType(preferred_modality),
                    reason="User preference",
                    confidence=0.7,
                )

        return None

    def _calculate_performance_score(
        self, capability: ModalityCapability
    ) -> float:
        """Calculate performance score for a modality capability."""
        base_score = capability.confidence

        # Factor in reliability
        reliability = capability.capabilities.get("reliability", 0.5)
        base_score *= reliability

        # Factor in speed
        speed_map = {"fast": 1.0, "medium": 0.8, "slow": 0.6}
        speed = capability.capabilities.get("speed", "medium")
        base_score *= speed_map.get(speed, 0.8)

        return base_score

    def record_modality_switch(
        self, switch: ModalitySwitch, user_id: Optional[str] = None
    ):
        """Record a modality switch for learning."""
        self.modality_history.append(switch)

        # Keep history manageable
        if len(self.modality_history) > 1000:
            self.modality_history = self.modality_history[-1000:]

        # Update user patterns
        if user_id:
            if user_id not in self.user_patterns:
                self.user_patterns[user_id] = {}

            pattern = self.user_patterns[user_id]
            if "modality_switches" not in pattern:
                pattern["modality_switches"] = []

            pattern["modality_switches"].append(
                {
                    "from": switch.from_modality.value,
                    "to": switch.to_modality.value,
                    "reason": switch.reason,
                    "timestamp": switch.timestamp.isoformat(),
                }
            )

            # Update preferred modality based on frequency
            switches = pattern["modality_switches"]
            if len(switches) >= 5:
                to_modalities = [
                    s["to"] for s in switches[-10:]
                ]  # Last 10 switches
                if to_modalities:
                    preferred = max(
                        set(to_modalities), key=to_modalities.count
                    )
                    pattern["preferred_modality"] = preferred

    def get_modality_status(self) -> Dict[str, Any]:
        """Get current status of all modalities."""
        with self.capability_lock:
            status = {}
            for modality, capability in self.capabilities.items():
                status[modality.value] = {
                    "state": capability.state.value,
                    "confidence": capability.confidence,
                    "last_check": capability.last_check.isoformat(),
                    "error_count": capability.error_count,
                    "success_count": capability.success_count,
                    "capabilities": capability.capabilities,
                }

            return status

    def force_modality_switch(
        self,
        from_modality: ModalityType,
        to_modality: ModalityType,
        reason: str = "Manual override",
    ) -> bool:
        """Force a modality switch (admin function)."""
        if to_modality not in self.capabilities:
            return False

        capability = self.capabilities[to_modality]
        if capability.state not in [
            ModalityState.AVAILABLE,
            ModalityState.ACTIVE,
        ]:
            return False

        switch = ModalitySwitch(
            from_modality=from_modality,
            to_modality=to_modality,
            reason=reason,
            confidence=1.0,
        )

        self.record_modality_switch(switch)
        logger.info(
            "Forced modality switch: %s -> %s (%s)",
            from_modality.value,
            to_modality.value,
            reason,
        )
        return True

    def get_user_patterns(self, user_id: str) -> Dict[str, Any]:
        """Get learned patterns for a user."""
        return self.user_patterns.get(user_id, {})

    def reset_user_patterns(self, user_id: str):
        """Reset learned patterns for a user."""
        if user_id in self.user_patterns:
            del self.user_patterns[user_id]
            logger.info("Reset user patterns for %s", user_id)

    def export_configuration(self) -> Dict[str, Any]:
        """Export modality manager configuration and learned patterns."""
        return {
            "capabilities": {
                mod.value: {
                    "state": cap.state.value,
                    "confidence": cap.confidence,
                    "capabilities": cap.capabilities,
                }
                for mod, cap in self.capabilities.items()
            },
            "user_patterns": dict(self.user_patterns),
            "modality_history": [
                {
                    "from_modality": s.from_modality.value,
                    "to_modality": s.to_modality.value,
                    "reason": s.reason,
                    "confidence": s.confidence,
                    "timestamp": s.timestamp.isoformat(),
                }
                for s in self.modality_history[-100:]  # Last 100 switches
            ],
            "switch_policies": list(self.switch_policies.keys()),
        }


# Register with provider registry
def create_modality_manager(config_manager=None, **kwargs):
    """Factory function for ModalityManager."""
    return ModalityManager(config_manager=config_manager, **kwargs)


provider_registry.register_lazy(
    "modality",
    "manager",
    "mia.adaptive_intelligence.modality_manager",
    "create_modality_manager",
    default=True,
)
