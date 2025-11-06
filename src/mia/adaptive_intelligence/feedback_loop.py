"""
Feedback Loop
"""

import asyncio
import json
import logging
import statistics
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..providers import provider_registry

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEvent:
    """Represents a single feedback event."""

    id: str
    event_type: str
    user_id: Optional[str]
    conversation_id: Optional[str]
    timestamp: datetime = field(default_factory=datetime.now)
    rating: Optional[float] = None
    feedback_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    context_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "timestamp": self.timestamp.isoformat(),
            "rating": self.rating,
            "feedback_text": self.feedback_text,
            "metadata": self.metadata,
            "context_snapshot": self.context_snapshot,
        }


@dataclass
class FeedbackPattern:
    """Represents a learned feedback pattern."""

    pattern_id: str
    trigger_conditions: Dict[str, Any]
    feedback_type: str
    frequency: int
    avg_rating: float
    common_feedback: List[str]
    suggested_actions: List[str]
    confidence: float
    last_updated: datetime = field(default_factory=datetime.now)

    def update(self, rating: Optional[float], feedback_text: Optional[str]):
        """Update pattern with new feedback."""
        self.frequency += 1

        if rating is not None:
            # Update average rating
            self.avg_rating = (
                self.avg_rating * (self.frequency - 1) + rating
            ) / self.frequency

        if feedback_text:
            self.common_feedback.append(feedback_text)
            # Keep only top 10 most common
            if len(self.common_feedback) > 10:
                self.common_feedback = self.common_feedback[-10:]

        self.last_updated = datetime.now()


@dataclass
class PerformanceMetrics:
    """Performance metrics for the system."""

    response_time_avg: float = 0.0
    response_time_p95: float = 0.0
    accuracy_score: float = 0.0
    user_satisfaction: float = 0.0
    error_rate: float = 0.0
    modality_success_rate: Dict[str, float] = field(default_factory=dict)
    context_relevance_score: float = 0.0
    conversation_coherence: float = 0.0

    def update_from_feedback(self, feedback_events: List[FeedbackEvent]):
        """Update metrics from feedback events."""
        if not feedback_events:
            return

        ratings = [e.rating for e in feedback_events if e.rating is not None]
        if ratings:
            self.user_satisfaction = statistics.mean(ratings)

        # Calculate other metrics based on feedback metadata
        response_times = []
        accuracies = []
        errors = 0

        for event in feedback_events:
            if "response_time" in event.metadata:
                response_times.append(event.metadata["response_time"])
            if "accuracy" in event.metadata:
                accuracies.append(event.metadata["accuracy"])
            if event.event_type == "error":
                errors += 1

        if response_times:
            self.response_time_avg = statistics.mean(response_times)
            self.response_time_p95 = statistics.quantiles(response_times, n=20)[
                18
            ]  # 95th percentile

        if accuracies:
            self.accuracy_score = statistics.mean(accuracies)

        total_events = len(feedback_events)
        self.error_rate = errors / total_events if total_events > 0 else 0


class FeedbackLoop:
    """
    Continuous learning feedback loop system.

    Features:
    - Multi-channel feedback capture
    - Real-time feedback analysis
    - Pattern recognition and learning
    - Performance metrics tracking
    - Adaptive system improvements
    - A/B testing support
    - User behavior modeling
    """

    def __init__(
        self,
        config_manager=None,
        *,
        feedback_retention_days: int = 90,
        pattern_min_frequency: int = 5,
        learning_enabled: bool = True,
        real_time_analysis: bool = True,
    ):
        self.config_manager = config_manager
        self.feedback_retention_days = feedback_retention_days
        self.pattern_min_frequency = pattern_min_frequency
        self.learning_enabled = learning_enabled
        self.real_time_analysis = real_time_analysis

        # Feedback storage
        self.feedback_events: List[FeedbackEvent] = []
        self.feedback_lock = threading.RLock()

        # Learning components
        self.feedback_patterns: Dict[str, FeedbackPattern] = {}
        self.performance_metrics = PerformanceMetrics()
        self.user_behavior_models: Dict[str, Dict[str, Any]] = {}

        # Analysis components
        self.analysis_queue = asyncio.Queue()
        self.analysis_task: Optional[asyncio.Task] = None
        self.analysis_lock = threading.Lock()

        # Feedback channels
        self.feedback_channels: Dict[str, Callable] = {}
        self.channel_configs: Dict[str, Dict[str, Any]] = {}

        # Learning triggers
        self.learning_triggers: Dict[str, Callable] = {}

        # Initialize default channels
        self._initialize_default_channels()

        # Start analysis if enabled
        if self.real_time_analysis:
            self.start_real_time_analysis()

        logger.info(
            "Feedback Loop initialized with %d channels", len(self.feedback_channels)
        )

    def _initialize_default_channels(self):
        """Initialize default feedback channels."""
        self.feedback_channels = {
            "explicit_rating": self._capture_explicit_rating,
            "implicit_behavior": self._capture_implicit_behavior,
            "error_reporting": self._capture_error_feedback,
            "performance_metrics": self._capture_performance_feedback,
            "user_correction": self._capture_user_correction,
        }

        # Default channel configurations
        self.channel_configs = {
            "explicit_rating": {"enabled": True, "auto_capture": False},
            "implicit_behavior": {"enabled": True, "auto_capture": True},
            "error_reporting": {"enabled": True, "auto_capture": True},
            "performance_metrics": {"enabled": True, "auto_capture": True},
            "user_correction": {"enabled": True, "auto_capture": True},
        }

    def start_real_time_analysis(self):
        """Start real-time feedback analysis."""
        if self.analysis_task and not self.analysis_task.done():
            return

        self.analysis_task = asyncio.create_task(self._analysis_loop())
        logger.info("Started real-time feedback analysis")

    def stop_real_time_analysis(self):
        """Stop real-time feedback analysis."""
        if self.analysis_task:
            self.analysis_task.cancel()
            try:
                # Wait for task to finish
                pass
            except asyncio.TimeoutError:
                pass
            logger.info("Stopped real-time feedback analysis")

    async def _analysis_loop(self):
        """Background analysis loop."""
        while True:
            try:
                # Wait for feedback events
                await asyncio.sleep(1)  # Check every second

                # Process queued feedback
                feedback_batch = []
                while not self.analysis_queue.empty():
                    feedback_batch.append(self.analysis_queue.get_nowait())

                if feedback_batch:
                    await self._analyze_feedback_batch(feedback_batch)

                # Periodic pattern analysis
                await self._periodic_pattern_analysis()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Analysis loop error: %s", exc)
                await asyncio.sleep(5)

    async def _analyze_feedback_batch(self, feedback_batch: List[FeedbackEvent]):
        """Analyze a batch of feedback events."""
        try:
            # Update performance metrics
            self.performance_metrics.update_from_feedback(feedback_batch)

            # Extract patterns
            new_patterns = self._extract_feedback_patterns(feedback_batch)
            for pattern in new_patterns:
                self.feedback_patterns[pattern.pattern_id] = pattern

            # Trigger learning actions
            await self._trigger_learning_actions(feedback_batch)

            # Update user behavior models
            self._update_user_behavior_models(feedback_batch)

            logger.debug("Analyzed batch of %d feedback events", len(feedback_batch))

        except Exception as exc:
            logger.error("Feedback batch analysis error: %s", exc)

    def capture_feedback(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        rating: Optional[float] = None,
        feedback_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        context_snapshot: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Capture feedback from various sources.

        Returns:
            Feedback event ID
        """
        import uuid

        event_id = str(uuid.uuid4())

        feedback_event = FeedbackEvent(
            id=event_id,
            event_type=event_type,
            user_id=user_id,
            conversation_id=conversation_id,
            rating=rating,
            feedback_text=feedback_text,
            metadata=metadata or {},
            context_snapshot=context_snapshot or {},
        )

        with self.feedback_lock:
            self.feedback_events.append(feedback_event)

            # Keep only recent feedback
            cutoff_date = datetime.now() - timedelta(days=self.feedback_retention_days)
            self.feedback_events = [
                event for event in self.feedback_events if event.timestamp > cutoff_date
            ]

        # Queue for analysis
        if self.real_time_analysis:
            try:
                self.analysis_queue.put_nowait(feedback_event)
            except asyncio.QueueFull:
                logger.warning("Feedback analysis queue full, dropping event")

        # Route to appropriate channel
        channel_func = self.feedback_channels.get(event_type)
        if channel_func:
            try:
                channel_func(feedback_event)
            except Exception as exc:
                logger.debug("Channel processing error for %s: %s", event_type, exc)

        logger.debug("Captured feedback event: %s (%s)", event_id, event_type)
        return event_id

    def _capture_explicit_rating(self, event: FeedbackEvent):
        """Capture explicit user ratings."""
        # Store rating for analysis
        pass  # Handled by general analysis

    def _capture_implicit_behavior(self, event: FeedbackEvent):
        """Capture implicit user behavior feedback."""
        # Analyze user actions, dwell time, etc.
        pass

    def _capture_error_feedback(self, event: FeedbackEvent):
        """Capture error-related feedback."""
        # Log errors for system improvement
        if event.feedback_text:
            logger.warning("User reported error: %s", event.feedback_text)

    def _capture_performance_feedback(self, event: FeedbackEvent):
        """Capture performance-related feedback."""
        # Update performance metrics
        pass

    def _capture_user_correction(self, event: FeedbackEvent):
        """Capture user corrections to AI responses."""
        # Learn from user corrections
        pass

    def _extract_feedback_patterns(
        self, feedback_events: List[FeedbackEvent]
    ) -> List[FeedbackPattern]:
        """Extract patterns from feedback events."""
        patterns = []

        # Group by trigger conditions
        pattern_groups = defaultdict(list)

        for event in feedback_events:
            # Create pattern key based on event characteristics
            key_parts = [
                event.event_type,
                str(event.metadata.get("modality", "")),
                str(event.metadata.get("intent", "")),
                str(event.metadata.get("error_type", "")),
            ]
            pattern_key = "|".join(key_parts)

            pattern_groups[pattern_key].append(event)

        # Create patterns for frequent groups
        for pattern_key, events in pattern_groups.items():
            if len(events) >= self.pattern_min_frequency:
                # Calculate pattern statistics
                ratings = [e.rating for e in events if e.rating is not None]
                avg_rating = statistics.mean(ratings) if ratings else 0.0

                feedback_texts = [e.feedback_text for e in events if e.feedback_text]
                common_feedback = feedback_texts[-5:] if feedback_texts else []

                # Generate pattern ID
                pattern_id = f"pattern_{hash(pattern_key) % 10000}"

                pattern = FeedbackPattern(
                    pattern_id=pattern_id,
                    trigger_conditions={"pattern_key": pattern_key},
                    feedback_type=events[0].event_type,
                    frequency=len(events),
                    avg_rating=avg_rating,
                    common_feedback=common_feedback,
                    suggested_actions=self._generate_suggested_actions(events),
                    confidence=min(
                        len(events) / 10, 1.0
                    ),  # Simple confidence calculation
                )

                patterns.append(pattern)

        return patterns

    def _generate_suggested_actions(self, events: List[FeedbackEvent]) -> List[str]:
        """Generate suggested actions based on feedback patterns."""
        actions = []

        # Analyze common issues
        error_types = Counter(e.metadata.get("error_type", "") for e in events)
        most_common_error = error_types.most_common(1)

        if most_common_error and most_common_error[0][1] > len(events) * 0.5:
            error_type = most_common_error[0][0]
            if "timeout" in error_type.lower():
                actions.append("Increase response timeout limits")
            elif "accuracy" in error_type.lower():
                actions.append("Improve response accuracy with additional training")
            elif "relevance" in error_type.lower():
                actions.append("Enhance context relevance filtering")

        # Check ratings
        low_ratings = [e for e in events if e.rating and e.rating < 3.0]
        if len(low_ratings) > len(events) * 0.3:
            actions.append("Review and improve response quality for this pattern")

        # Check feedback text for common themes
        feedback_texts = [e.feedback_text for e in events if e.feedback_text]
        if feedback_texts:
            common_words = Counter()
            for text in feedback_texts:
                words = text.lower().split()
                common_words.update(words)

            top_words = [
                word for word, count in common_words.most_common(5) if count > 1
            ]
            if "slow" in top_words:
                actions.append("Optimize response generation speed")
            if "confusing" in top_words or "unclear" in top_words:
                actions.append("Improve response clarity and structure")

        return actions[:3]  # Limit to top 3 actions

    async def _trigger_learning_actions(self, feedback_events: List[FeedbackEvent]):
        """Trigger learning actions based on feedback."""
        if not self.learning_enabled:
            return

        # Check for patterns that require immediate action
        urgent_patterns = [
            pattern
            for pattern in self.feedback_patterns.values()
            if pattern.avg_rating < 2.0
            and pattern.frequency >= self.pattern_min_frequency
        ]

        for pattern in urgent_patterns:
            # Trigger learning action
            trigger_func = self.learning_triggers.get(pattern.feedback_type)
            if trigger_func:
                try:
                    await trigger_func(pattern)
                except Exception as exc:
                    logger.error(
                        "Learning trigger error for %s: %s", pattern.pattern_id, exc
                    )

    async def _periodic_pattern_analysis(self):
        """Periodic analysis of feedback patterns."""
        # Clean up old patterns
        cutoff_date = datetime.now() - timedelta(days=self.feedback_retention_days)
        expired_patterns = [
            pid
            for pid, pattern in self.feedback_patterns.items()
            if pattern.last_updated < cutoff_date
        ]

        for pid in expired_patterns:
            del self.feedback_patterns[pid]

        # Update pattern confidence based on age and frequency
        for pattern in self.feedback_patterns.values():
            age_days = (datetime.now() - pattern.last_updated).days
            # Reduce confidence for old patterns
            age_penalty = (
                max(0, age_days - 30) * 0.01
            )  # 1% penalty per day after 30 days
            pattern.confidence = max(0, pattern.confidence - age_penalty)

    def _update_user_behavior_models(self, feedback_events: List[FeedbackEvent]):
        """Update user behavior models based on feedback."""
        user_events = defaultdict(list)

        for event in feedback_events:
            if event.user_id:
                user_events[event.user_id].append(event)

        for user_id, events in user_events.items():
            if user_id not in self.user_behavior_models:
                self.user_behavior_models[user_id] = {
                    "feedback_count": 0,
                    "avg_rating": 0.0,
                    "preferred_modalities": {},
                    "common_feedback_themes": [],
                    "last_updated": datetime.now(),
                }

            model = self.user_behavior_models[user_id]

            # Update statistics
            model["feedback_count"] += len(events)

            ratings = [e.rating for e in events if e.rating is not None]
            if ratings:
                current_avg = model["avg_rating"]
                new_avg = statistics.mean(ratings)
                model["avg_rating"] = (
                    current_avg + new_avg
                ) / 2  # Simple moving average

            # Update modality preferences
            for event in events:
                modality = event.metadata.get("modality", "unknown")
                if modality not in model["preferred_modalities"]:
                    model["preferred_modalities"][modality] = 0
                model["preferred_modalities"][modality] += 1

            model["last_updated"] = datetime.now()

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        with self.feedback_lock:
            total_events = len(self.feedback_events)
            if total_events == 0:
                return {"total_events": 0}

            ratings = [e.rating for e in self.feedback_events if e.rating is not None]
            avg_rating = statistics.mean(ratings) if ratings else 0.0

            event_types = Counter(e.event_type for e in self.feedback_events)

            return {
                "total_events": total_events,
                "avg_rating": avg_rating,
                "event_type_distribution": dict(event_types),
                "patterns_discovered": len(self.feedback_patterns),
                "user_behavior_models": len(self.user_behavior_models),
                "retention_days": self.feedback_retention_days,
            }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            "response_time_avg": self.performance_metrics.response_time_avg,
            "response_time_p95": self.performance_metrics.response_time_p95,
            "accuracy_score": self.performance_metrics.accuracy_score,
            "user_satisfaction": self.performance_metrics.user_satisfaction,
            "error_rate": self.performance_metrics.error_rate,
            "modality_success_rate": self.performance_metrics.modality_success_rate,
            "context_relevance_score": self.performance_metrics.context_relevance_score,
            "conversation_coherence": self.performance_metrics.conversation_coherence,
        }

    def get_feedback_patterns(
        self, min_frequency: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get discovered feedback patterns."""
        patterns = list(self.feedback_patterns.values())

        if min_frequency:
            patterns = [p for p in patterns if p.frequency >= min_frequency]

        return [self._pattern_to_dict(p) for p in patterns]

    def _pattern_to_dict(self, pattern: FeedbackPattern) -> Dict[str, Any]:
        """Convert pattern to dictionary."""
        return {
            "pattern_id": pattern.pattern_id,
            "trigger_conditions": pattern.trigger_conditions,
            "feedback_type": pattern.feedback_type,
            "frequency": pattern.frequency,
            "avg_rating": pattern.avg_rating,
            "common_feedback": pattern.common_feedback,
            "suggested_actions": pattern.suggested_actions,
            "confidence": pattern.confidence,
            "last_updated": pattern.last_updated.isoformat(),
        }

    def export_feedback_data(self) -> Dict[str, Any]:
        """Export all feedback data for analysis."""
        with self.feedback_lock:
            return {
                "feedback_events": [
                    e.to_dict() for e in self.feedback_events[-1000:]
                ],  # Last 1000 events
                "feedback_patterns": [
                    self._pattern_to_dict(p) for p in self.feedback_patterns.values()
                ],
                "performance_metrics": self.get_performance_metrics(),
                "user_behavior_models": dict(self.user_behavior_models),
                "export_timestamp": datetime.now().isoformat(),
            }

    def add_learning_trigger(self, feedback_type: str, trigger_func: Callable):
        """Add a learning trigger for a feedback type."""
        self.learning_triggers[feedback_type] = trigger_func
        logger.info("Added learning trigger for feedback type: %s", feedback_type)

    def configure_channel(self, channel_name: str, config: Dict[str, Any]):
        """Configure a feedback channel."""
        self.channel_configs[channel_name] = config
        logger.info("Configured channel %s: %s", channel_name, config)


# Register with provider registry
def create_feedback_loop(config_manager=None, **kwargs):
    """Factory function for FeedbackLoop."""
    return FeedbackLoop(config_manager=config_manager, **kwargs)


provider_registry.register_lazy(
    "feedback",
    "loop",
    "mia.adaptive_intelligence.feedback_loop",
    "create_feedback_loop",
    default=True,
)
