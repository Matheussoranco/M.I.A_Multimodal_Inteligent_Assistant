"""
Context Infusion Pipeline
"""

import hashlib
import json
import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from ..providers import provider_registry

logger = logging.getLogger(__name__)


@dataclass
class ContextElement:
    """Represents a single context element."""

    id: str
    content: Any
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    relevance_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if context element has expired."""
        return self.expires_at is not None and datetime.now() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "relevance_score": self.relevance_score,
            "metadata": self.metadata,
            "expires_at": (
                self.expires_at.isoformat() if self.expires_at else None
            ),
        }


@dataclass
class UserProfile:
    """User profile with preferences and history."""

    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    behavior_patterns: Dict[str, Any] = field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    context_weights: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)

    def update_preference(self, key: str, value: Any):
        """Update a user preference."""
        self.preferences[key] = value
        self.last_updated = datetime.now()

    def add_interaction(self, interaction: Dict[str, Any]):
        """Add an interaction to history."""
        self.interaction_history.append(interaction)
        # Keep only last 1000 interactions
        if len(self.interaction_history) > 1000:
            self.interaction_history = self.interaction_history[-1000:]
        self.last_updated = datetime.now()


@dataclass
class ConversationContext:
    """Conversation-specific context."""

    conversation_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    topics: Set[str] = field(default_factory=set)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    sentiment_trend: List[float] = field(default_factory=list)
    context_elements: List[ContextElement] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    def add_message(self, message: Dict[str, Any]):
        """Add a message to the conversation."""
        self.messages.append(message)
        self.last_activity = datetime.now()

        # Keep only last 100 messages for context
        if len(self.messages) > 100:
            self.messages = self.messages[-100:]

    def add_topic(self, topic: str):
        """Add a topic to the conversation."""
        self.topics.add(topic)

    def add_entity(self, entity_type: str, entity_value: str):
        """Add an entity to the conversation."""
        if entity_type not in self.entities:
            self.entities[entity_type] = []
        if entity_value not in self.entities[entity_type]:
            self.entities[entity_type].append(entity_value)


@dataclass
class EnvironmentalContext:
    """Environmental factors affecting context."""

    location: Optional[Dict[str, Any]] = None
    time_of_day: str = ""
    day_of_week: str = ""
    season: str = ""
    device_info: Dict[str, Any] = field(default_factory=dict)
    network_info: Dict[str, Any] = field(default_factory=dict)
    ambient_conditions: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class ContextInfuser:
    """
    Intelligent context infusion pipeline.

    Features:
    - Multi-source context aggregation
    - Relevance-based filtering and ranking
    - User profile integration
    - Conversation history analysis
    - Environmental context awareness
    - Knowledge graph integration
    - Adaptive context weighting
    """

    def __init__(
        self,
        config_manager=None,
        *,
        max_context_elements: int = 50,
        context_ttl_hours: int = 24,
        relevance_threshold: float = 0.3,
        enable_learning: bool = True,
    ):
        self.config_manager = config_manager
        self.max_context_elements = max_context_elements
        self.context_ttl_hours = context_ttl_hours
        self.relevance_threshold = relevance_threshold
        self.enable_learning = enable_learning

        # Context storage
        self.global_context: Dict[str, ContextElement] = {}
        self.user_profiles: Dict[str, UserProfile] = {}
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        self.environmental_context = EnvironmentalContext()

        # Context processing
        self.context_lock = threading.RLock()
        self.context_queue = deque(maxlen=1000)
        self.relevance_cache: Dict[str, float] = {}

        # Learning components
        self.context_patterns: Dict[str, Dict[str, Any]] = {}
        self.adaptive_weights: Dict[str, float] = {
            "conversation_history": 0.4,
            "user_profile": 0.3,
            "environmental": 0.1,
            "knowledge_graph": 0.2,
        }

        # Initialize context sources
        self.context_sources = {
            "conversation": self._extract_conversation_context,
            "user_profile": self._extract_user_profile_context,
            "environmental": self._extract_environmental_context,
            "knowledge": self._extract_knowledge_context,
        }

        logger.info(
            "Context Infuser initialized with %d context sources",
            len(self.context_sources),
        )

    def infuse_context(
        self,
        query: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        modality_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Infuse context into a query for enhanced AI processing.

        Args:
            query: The input query
            user_id: Optional user identifier
            conversation_id: Optional conversation identifier
            modality_context: Optional modality-specific context

        Returns:
            Enhanced context dictionary
        """
        with self.context_lock:
            # Clean expired context
            self._clean_expired_context()

            # Gather context from all sources
            context_components = {}

            for source_name, extractor_func in self.context_sources.items():
                try:
                    context_components[source_name] = extractor_func(
                        query, user_id, conversation_id, modality_context
                    )
                except Exception as exc:
                    logger.debug(
                        "Context extraction failed for %s: %s",
                        source_name,
                        exc,
                    )
                    context_components[source_name] = {}

            # Fuse contexts with adaptive weighting
            fused_context = self._fuse_contexts(context_components, query)

            # Apply relevance filtering
            relevant_context = self._filter_relevant_context(
                fused_context, query
            )

            # Update learning if enabled
            if self.enable_learning and user_id:
                self._update_learning_patterns(
                    query, relevant_context, user_id
                )

            # Record context usage
            self._record_context_usage(query, relevant_context)

            return {
                "original_query": query,
                "enhanced_query": self._enhance_query_with_context(
                    query, relevant_context
                ),
                "context_components": context_components,
                "fused_context": fused_context,
                "relevant_context": relevant_context,
                "context_metadata": {
                    "sources_used": list(context_components.keys()),
                    "total_elements": len(
                        relevant_context.get("elements", [])
                    ),
                    "relevance_threshold": self.relevance_threshold,
                    "timestamp": datetime.now().isoformat(),
                },
            }

    def _extract_conversation_context(
        self,
        query: str,
        user_id: Optional[str],
        conversation_id: Optional[str],
        modality_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Extract context from conversation history."""
        if not conversation_id:
            return {}

        conv_ctx = self.conversation_contexts.get(conversation_id)
        if not conv_ctx:
            return {}

        # Get recent messages
        recent_messages = conv_ctx.messages[-10:]  # Last 10 messages

        # Extract topics and entities
        topics = list(conv_ctx.topics)
        entities = dict(conv_ctx.entities)

        # Calculate conversation flow
        conversation_flow = self._analyze_conversation_flow(recent_messages)

        return {
            "recent_messages": recent_messages,
            "topics": topics,
            "entities": entities,
            "conversation_flow": conversation_flow,
            "message_count": len(conv_ctx.messages),
            "last_activity": conv_ctx.last_activity.isoformat(),
        }

    def _extract_user_profile_context(
        self,
        query: str,
        user_id: Optional[str],
        conversation_id: Optional[str],
        modality_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Extract context from user profile."""
        if not user_id:
            return {}

        profile = self.user_profiles.get(user_id)
        if not profile:
            return {}

        # Get relevant preferences
        relevant_prefs = self._get_relevant_preferences(
            profile.preferences, query
        )

        # Get behavior patterns
        patterns = profile.behavior_patterns

        # Get recent interactions
        recent_interactions = profile.interaction_history[
            -5:
        ]  # Last 5 interactions

        return {
            "preferences": relevant_prefs,
            "behavior_patterns": patterns,
            "recent_interactions": recent_interactions,
            "context_weights": profile.context_weights,
            "profile_age_days": (datetime.now() - profile.last_updated).days,
        }

    def _extract_environmental_context(
        self,
        query: str,
        user_id: Optional[str],
        conversation_id: Optional[str],
        modality_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Extract environmental context."""
        env_ctx = self.environmental_context

        # Update current environmental factors
        self._update_environmental_factors()

        return {
            "location": env_ctx.location,
            "time_of_day": env_ctx.time_of_day,
            "day_of_week": env_ctx.day_of_week,
            "season": env_ctx.season,
            "device_info": env_ctx.device_info,
            "network_info": env_ctx.network_info,
            "ambient_conditions": env_ctx.ambient_conditions,
        }

    def _extract_knowledge_context(
        self,
        query: str,
        user_id: Optional[str],
        conversation_id: Optional[str],
        modality_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Extract context from knowledge graph."""
        # This would integrate with the knowledge graph system
        # For now, return basic structure
        return {
            "related_concepts": [],
            "knowledge_links": [],
            "domain_context": {},
            "confidence_scores": {},
        }

    def _fuse_contexts(
        self, context_components: Dict[str, Any], query: str
    ) -> Dict[str, Any]:
        """Fuse multiple context sources with adaptive weighting."""
        fused_elements = []

        for source_name, context_data in context_components.items():
            weight = self.adaptive_weights.get(source_name, 0.25)

            # Convert context data to ContextElements
            elements = self._context_data_to_elements(
                context_data, source_name, weight
            )
            fused_elements.extend(elements)

        # Sort by relevance and weight
        fused_elements.sort(key=lambda x: x.relevance_score, reverse=True)

        # Limit to max elements
        fused_elements = fused_elements[: self.max_context_elements]

        return {
            "elements": fused_elements,
            "source_weights": self.adaptive_weights.copy(),
            "total_sources": len(context_components),
        }

    def _context_data_to_elements(
        self, context_data: Dict[str, Any], source: str, base_weight: float
    ) -> List[ContextElement]:
        """Convert context data to ContextElement objects."""
        elements = []

        for key, value in context_data.items():
            if value is None or (
                isinstance(value, (list, dict)) and not value
            ):
                continue

            # Create element ID
            content_str = json.dumps(value, sort_keys=True, default=str)
            element_id = hashlib.md5(
                f"{source}:{key}:{content_str}".encode()
            ).hexdigest()[:16]

            # Calculate relevance score
            relevance = self._calculate_relevance_score(value, base_weight)

            element = ContextElement(
                id=element_id,
                content=value,
                source=source,
                relevance_score=relevance,
                metadata={"key": key, "data_type": type(value).__name__},
                expires_at=datetime.now()
                + timedelta(hours=self.context_ttl_hours),
            )

            elements.append(element)

        return elements

    def _calculate_relevance_score(
        self, content: Any, base_weight: float
    ) -> float:
        """Calculate relevance score for context content."""
        # Simple relevance calculation - in real implementation would use ML models
        score = base_weight

        # Boost score based on content characteristics
        if isinstance(content, dict):
            score *= 1 + len(content) * 0.1  # More fields = more relevant
        elif isinstance(content, list):
            score *= 1 + len(content) * 0.05  # More items = more relevant
        elif isinstance(content, str) and len(content) > 10:
            score *= 1.2  # Longer strings are more relevant

        return min(score, 1.0)  # Cap at 1.0

    def _filter_relevant_context(
        self, fused_context: Dict[str, Any], query: str
    ) -> Dict[str, Any]:
        """Filter context elements based on relevance to query."""
        elements = fused_context.get("elements", [])

        # Filter by relevance threshold
        relevant_elements = [
            elem
            for elem in elements
            if elem.relevance_score >= self.relevance_threshold
        ]

        # Sort by relevance
        relevant_elements.sort(key=lambda x: x.relevance_score, reverse=True)

        return {
            "elements": relevant_elements,
            "filtered_count": len(relevant_elements),
            "total_count": len(elements),
            "avg_relevance": (
                sum(e.relevance_score for e in relevant_elements)
                / len(relevant_elements)
                if relevant_elements
                else 0
            ),
        }

    def _enhance_query_with_context(
        self, query: str, relevant_context: Dict[str, Any]
    ) -> str:
        """Enhance the original query with relevant context."""
        elements = relevant_context.get("elements", [])

        if not elements:
            return query

        # Build context string
        context_parts = []

        for element in elements[:5]:  # Use top 5 most relevant
            if isinstance(element.content, str):
                context_parts.append(
                    f"{element.metadata.get('key', 'context')}: {element.content}"
                )
            elif isinstance(element.content, (list, dict)):
                context_str = json.dumps(element.content, indent=2)
                context_parts.append(
                    f"{element.metadata.get('key', 'context')}:\n{context_str}"
                )

        if context_parts:
            enhanced_query = f"Context:\n" + "\n\n".join(context_parts)
            enhanced_query += f"\n\nOriginal Query: {query}"
            return enhanced_query

        return query

    def _analyze_conversation_flow(
        self, messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze the flow of conversation."""
        if not messages:
            return {}

        # Simple flow analysis
        user_messages = [m for m in messages if m.get("role") == "user"]
        assistant_messages = [
            m for m in messages if m.get("role") == "assistant"
        ]

        return {
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "avg_user_message_length": (
                sum(len(str(m.get("content", ""))) for m in user_messages)
                / len(user_messages)
                if user_messages
                else 0
            ),
            "conversation_depth": len(
                set(str(m.get("content", ""))[:50] for m in messages)
            ),  # Unique message starts
        }

    def _get_relevant_preferences(
        self, preferences: Dict[str, Any], query: str
    ) -> Dict[str, Any]:
        """Get preferences relevant to the current query."""
        # Simple keyword matching - in real implementation would use semantic matching
        relevant_prefs = {}

        query_lower = query.lower()
        for key, value in preferences.items():
            key_lower = key.lower()
            if any(word in query_lower for word in key_lower.split("_")):
                relevant_prefs[key] = value

        return relevant_prefs

    def _update_environmental_factors(self):
        """Update current environmental factors."""
        now = datetime.now()

        # Update time-based factors
        hour = now.hour
        if 6 <= hour < 12:
            self.environmental_context.time_of_day = "morning"
        elif 12 <= hour < 18:
            self.environmental_context.time_of_day = "afternoon"
        elif 18 <= hour < 22:
            self.environmental_context.time_of_day = "evening"
        else:
            self.environmental_context.time_of_day = "night"

        self.environmental_context.day_of_week = now.strftime("%A")

        # Determine season (simplified)
        month = now.month
        if month in [12, 1, 2]:
            self.environmental_context.season = "winter"
        elif month in [3, 4, 5]:
            self.environmental_context.season = "spring"
        elif month in [6, 7, 8]:
            self.environmental_context.season = "summer"
        else:
            self.environmental_context.season = "fall"

        self.environmental_context.timestamp = now

    def _update_learning_patterns(
        self, query: str, context: Dict[str, Any], user_id: str
    ):
        """Update learning patterns based on context usage."""
        if not self.enable_learning:
            return

        # Simple learning - track which context sources are most useful
        elements = context.get("elements", [])
        source_usage = defaultdict(int)

        for element in elements:
            source_usage[element.source] += 1

        # Update adaptive weights based on usage
        total_usage = sum(source_usage.values())
        if total_usage > 0:
            for source, usage in source_usage.items():
                # Gradually adjust weights based on usage
                current_weight = self.adaptive_weights.get(source, 0.25)
                usage_ratio = usage / total_usage
                new_weight = (
                    current_weight * 0.9 + usage_ratio * 0.1
                )  # Smooth adjustment
                self.adaptive_weights[source] = new_weight

    def _record_context_usage(self, query: str, context: Dict[str, Any]):
        """Record context usage for analytics."""
        usage_record = {
            "query_hash": hashlib.md5(query.encode()).hexdigest()[:16],
            "timestamp": datetime.now().isoformat(),
            "context_elements_used": len(context.get("elements", [])),
            "sources_used": list(
                set(e.source for e in context.get("elements", []))
            ),
        }

        self.context_queue.append(usage_record)

    def _clean_expired_context(self):
        """Clean expired context elements."""
        current_time = datetime.now()
        expired_ids = []

        for element_id, element in self.global_context.items():
            if element.is_expired():
                expired_ids.append(element_id)

        for expired_id in expired_ids:
            del self.global_context[expired_id]

        if expired_ids:
            logger.debug(
                "Cleaned %d expired context elements", len(expired_ids)
            )

    def add_user_profile(
        self,
        user_id: str,
        initial_preferences: Optional[Dict[str, Any]] = None,
    ):
        """Add or update a user profile."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)

        profile = self.user_profiles[user_id]
        if initial_preferences:
            profile.preferences.update(initial_preferences)

        logger.info("Added/updated profile for user %s", user_id)

    def update_conversation_context(
        self, conversation_id: str, message: Dict[str, Any]
    ):
        """Update conversation context with a new message."""
        if conversation_id not in self.conversation_contexts:
            self.conversation_contexts[conversation_id] = ConversationContext(
                conversation_id=conversation_id
            )

        ctx = self.conversation_contexts[conversation_id]
        ctx.add_message(message)

        # Extract topics and entities from message (simplified)
        content = message.get("content", "")
        if isinstance(content, str):
            # Simple topic extraction
            words = content.lower().split()
            if len(words) > 3:
                topic = " ".join(words[:3])
                ctx.add_topic(topic)

    def get_context_stats(self) -> Dict[str, Any]:
        """Get context infusion statistics."""
        with self.context_lock:
            return {
                "global_context_elements": len(self.global_context),
                "user_profiles": len(self.user_profiles),
                "conversation_contexts": len(self.conversation_contexts),
                "context_queue_size": len(self.context_queue),
                "adaptive_weights": self.adaptive_weights.copy(),
                "relevance_cache_size": len(self.relevance_cache),
            }

    def export_context_data(self) -> Dict[str, Any]:
        """Export all context data for backup/analysis."""
        with self.context_lock:
            return {
                "user_profiles": {
                    uid: {
                        "preferences": profile.preferences,
                        "behavior_patterns": profile.behavior_patterns,
                        "context_weights": profile.context_weights,
                        "last_updated": profile.last_updated.isoformat(),
                    }
                    for uid, profile in self.user_profiles.items()
                },
                "conversation_contexts": {
                    cid: {
                        "topics": list(ctx.topics),
                        "entities": ctx.entities,
                        "message_count": len(ctx.messages),
                        "created_at": ctx.created_at.isoformat(),
                        "last_activity": ctx.last_activity.isoformat(),
                    }
                    for cid, ctx in self.conversation_contexts.items()
                },
                "adaptive_weights": self.adaptive_weights.copy(),
                "context_patterns": self.context_patterns.copy(),
                "export_timestamp": datetime.now().isoformat(),
            }


# Register with provider registry
def create_context_infuser(config_manager=None, **kwargs):
    """Factory function for ContextInfuser."""
    return ContextInfuser(config_manager=config_manager, **kwargs)


provider_registry.register_lazy(
    "context",
    "infuser",
    "mia.adaptive_intelligence.context_infuser",
    "create_context_infuser",
    default=True,
)
