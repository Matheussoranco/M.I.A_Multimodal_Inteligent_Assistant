import logging
import asyncio
import json
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
import statistics
import uuid

from ..providers import provider_registry

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of episodic memories."""
    INTERACTION = "interaction"
    DECISION = "decision"
    FEEDBACK = "feedback"
    OUTCOME = "outcome"
    PREFERENCE = "preference"
    CONTEXT = "context"
    LEARNING = "learning"


class PrivacyLevel(Enum):
    """Privacy levels for memory data."""
    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    PRIVATE = "private"


class RetentionPolicy(Enum):
    """Data retention policies."""
    FOREVER = "forever"
    YEARS_7 = "7_years"
    YEARS_3 = "3_years"
    YEARS_1 = "1_year"
    MONTHS_6 = "6_months"
    MONTHS_3 = "3_months"
    DAYS_30 = "30_days"
    SESSION = "session"


@dataclass
class EpisodicMemory:
    """Represents a single episodic memory."""
    id: str
    user_id: str
    session_id: Optional[str]
    memory_type: MemoryType
    content: Any
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5  # 0.0 to 1.0
    confidence: float = 0.8  # 0.0 to 1.0
    privacy_level: PrivacyLevel = PrivacyLevel.INTERNAL
    retention_policy: RetentionPolicy = RetentionPolicy.YEARS_1
    tags: Set[str] = field(default_factory=set)
    related_memories: Set[str] = field(default_factory=set)  # Memory IDs
    embeddings: Optional[List[float]] = None  # Vector representation


@dataclass
class UserProfile:
    """User profile with preferences and patterns."""
    user_id: str
    demographics: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    behavior_patterns: Dict[str, Any] = field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    personalization_vector: Optional[List[float]] = None
    trust_score: float = 0.5  # 0.0 to 1.0
    engagement_level: float = 0.5  # 0.0 to 1.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_interaction: Optional[datetime] = None


@dataclass
class PersonalizationContext:
    """Context for personalization decisions."""
    user_id: str
    current_session: str
    recent_interactions: List[Dict[str, Any]] = field(default_factory=list)
    active_goals: List[str] = field(default_factory=list)
    emotional_state: Optional[str] = None
    cognitive_load: float = 0.5  # 0.0 to 1.0
    time_context: Dict[str, Any] = field(default_factory=dict)
    environmental_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PersonalizationRule:
    """Rule for personalization decisions."""
    id: str
    name: str
    condition: Dict[str, Any]
    action: Dict[str, Any]
    priority: int = 5  # 1-10, higher = more important
    confidence_threshold: float = 0.7
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class EpisodicMemoryStore:
    """
    Storage system for episodic memories.

    Provides efficient storage, retrieval, and management of episodic data.
    """

    def __init__(self, max_memory_size: int = 100000):
        self.memories: Dict[str, EpisodicMemory] = {}
        self.user_index: Dict[str, Set[str]] = {}  # user_id -> memory_ids
        self.session_index: Dict[str, Set[str]] = {}  # session_id -> memory_ids
        self.type_index: Dict[MemoryType, Set[str]] = {}  # type -> memory_ids
        self.tag_index: Dict[str, Set[str]] = {}  # tag -> memory_ids

        self.max_memory_size = max_memory_size
        self._lock = threading.RLock()

        # Cleanup scheduler
        self.cleanup_task: Optional[asyncio.Task] = None

    def store_memory(self, memory: EpisodicMemory) -> str:
        """Store a new episodic memory."""
        with self._lock:
            memory_id = memory.id or str(uuid.uuid4())
            memory.id = memory_id

            # Check size limits
            if len(self.memories) >= self.max_memory_size:
                self._cleanup_old_memories()

            self.memories[memory_id] = memory

            # Update indices
            self._update_indices(memory, add=True)

            logger.debug(f"Stored episodic memory: {memory_id} for user {memory.user_id}")
            return memory_id

    def retrieve_memories(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[List[str]] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        limit: int = 50,
        min_importance: float = 0.0
    ) -> List[EpisodicMemory]:
        """Retrieve memories based on criteria."""
        with self._lock:
            candidate_ids = set(self.memories.keys())

            # Apply filters
            if user_id:
                candidate_ids &= self.user_index.get(user_id, set())

            if session_id:
                candidate_ids &= self.session_index.get(session_id, set())

            if memory_type:
                candidate_ids &= self.type_index.get(memory_type, set())

            if tags:
                for tag in tags:
                    candidate_ids &= self.tag_index.get(tag, set())

            # Filter by time range and importance
            filtered_memories = []
            for memory_id in candidate_ids:
                memory = self.memories[memory_id]

                if memory.importance < min_importance:
                    continue

                if time_range:
                    start_time, end_time = time_range
                    if not (start_time <= memory.timestamp <= end_time):
                        continue

                filtered_memories.append(memory)

            # Sort by importance and recency
            filtered_memories.sort(
                key=lambda m: (m.importance, m.timestamp),
                reverse=True
            )

            return filtered_memories[:limit]

    def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing memory."""
        with self._lock:
            if memory_id not in self.memories:
                return False

            memory = self.memories[memory_id]
            old_memory = memory  # Keep copy for index updates

            # Apply updates
            for key, value in updates.items():
                if hasattr(memory, key):
                    setattr(memory, key, value)

            memory.timestamp = datetime.now()  # Update timestamp

            # Update indices if needed
            self._update_indices(old_memory, add=False)
            self._update_indices(memory, add=True)

            logger.debug(f"Updated episodic memory: {memory_id}")
            return True

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        with self._lock:
            if memory_id not in self.memories:
                return False

            memory = self.memories[memory_id]

            # Remove from indices
            self._update_indices(memory, add=False)

            del self.memories[memory_id]

            logger.debug(f"Deleted episodic memory: {memory_id}")
            return True

    def get_memory_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get memory statistics."""
        with self._lock:
            if user_id:
                user_memories = self.user_index.get(user_id, set())
                memories = [self.memories[mid] for mid in user_memories if mid in self.memories]
            else:
                memories = list(self.memories.values())

            if not memories:
                return {"total_memories": 0}

            # Calculate statistics
            type_counts = {}
            for memory in memories:
                type_counts[memory.memory_type.value] = type_counts.get(memory.memory_type.value, 0) + 1

            avg_importance = statistics.mean(m.importance for m in memories)
            avg_confidence = statistics.mean(m.confidence for m in memories)

            oldest_memory = min(memories, key=lambda m: m.timestamp)
            newest_memory = max(memories, key=lambda m: m.timestamp)

            return {
                "total_memories": len(memories),
                "type_distribution": type_counts,
                "avg_importance": avg_importance,
                "avg_confidence": avg_confidence,
                "date_range": {
                    "oldest": oldest_memory.timestamp.isoformat(),
                    "newest": newest_memory.timestamp.isoformat()
                },
                "storage_utilization": len(self.memories) / self.max_memory_size
            }

    def _update_indices(self, memory: EpisodicMemory, add: bool = True):
        """Update search indices for a memory."""
        memory_id = memory.id

        # User index
        if memory.user_id not in self.user_index:
            self.user_index[memory.user_id] = set()
        if add:
            self.user_index[memory.user_id].add(memory_id)
        else:
            self.user_index[memory.user_id].discard(memory_id)

        # Session index
        if memory.session_id:
            if memory.session_id not in self.session_index:
                self.session_index[memory.session_id] = set()
            if add:
                self.session_index[memory.session_id].add(memory_id)
            else:
                self.session_index[memory.session_id].discard(memory_id)

        # Type index
        if memory.memory_type not in self.type_index:
            self.type_index[memory.memory_type] = set()
        if add:
            self.type_index[memory.memory_type].add(memory_id)
        else:
            self.type_index[memory.memory_type].discard(memory_id)

        # Tag index
        for tag in memory.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            if add:
                self.tag_index[tag].add(memory_id)
            else:
                self.tag_index[tag].discard(memory_id)

    def _cleanup_old_memories(self):
        """Clean up old memories based on retention policies."""
        current_time = datetime.now()
        memories_to_delete = []

        for memory_id, memory in self.memories.items():
            if self._should_delete_memory(memory, current_time):
                memories_to_delete.append(memory_id)

        # Delete old memories
        for memory_id in memories_to_delete:
            self.delete_memory(memory_id)

        if memories_to_delete:
            logger.info(f"Cleaned up {len(memories_to_delete)} old memories")

        # If still over limit, delete lowest importance memories
        if len(self.memories) >= self.max_memory_size:
            sorted_memories = sorted(
                self.memories.items(),
                key=lambda x: (x[1].importance, x[1].timestamp)
            )

            to_delete = len(self.memories) - self.max_memory_size + 1000  # Keep some buffer
            for memory_id, _ in sorted_memories[:to_delete]:
                self.delete_memory(memory_id)

    def _should_delete_memory(self, memory: EpisodicMemory, current_time: datetime) -> bool:
        """Check if a memory should be deleted based on retention policy."""
        age = current_time - memory.timestamp

        policy = memory.retention_policy

        if policy == RetentionPolicy.FOREVER:
            return False
        elif policy == RetentionPolicy.SESSION:
            return age > timedelta(hours=24)  # Session memories last 24 hours
        elif policy == RetentionPolicy.DAYS_30:
            return age > timedelta(days=30)
        elif policy == RetentionPolicy.MONTHS_3:
            return age > timedelta(days=90)
        elif policy == RetentionPolicy.MONTHS_6:
            return age > timedelta(days=180)
        elif policy == RetentionPolicy.YEARS_1:
            return age > timedelta(days=365)
        elif policy == RetentionPolicy.YEARS_3:
            return age > timedelta(days=1095)
        elif policy == RetentionPolicy.YEARS_7:
            return age > timedelta(days=2555)

        return False


class UserProfileManager:
    """
    Manages user profiles and personalization data.

    Tracks user preferences, behavior patterns, and personalization vectors.
    """

    def __init__(self, memory_store: EpisodicMemoryStore):
        self.memory_store = memory_store
        self.profiles: Dict[str, UserProfile] = {}
        self._lock = threading.RLock()

    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Get or create a user profile."""
        with self._lock:
            if user_id not in self.profiles:
                self.profiles[user_id] = UserProfile(user_id=user_id)
                logger.info(f"Created new user profile: {user_id}")

            return self.profiles[user_id]

    def update_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update a user profile."""
        with self._lock:
            if user_id not in self.profiles:
                return False

            profile = self.profiles[user_id]

            # Apply updates
            for key, value in updates.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)

            profile.updated_at = datetime.now()

            # Update personalization vector if preferences changed
            if 'preferences' in updates:
                self._update_personalization_vector(profile)

            logger.debug(f"Updated profile for user: {user_id}")
            return True

    def record_interaction(self, user_id: str, interaction: Dict[str, Any]):
        """Record a user interaction."""
        with self._lock:
            profile = self.get_or_create_profile(user_id)

            # Add to interaction history
            interaction_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": interaction.get("type", "unknown"),
                "content": interaction.get("content", ""),
                "outcome": interaction.get("outcome"),
                "metadata": interaction.get("metadata", {})
            }

            profile.interaction_history.append(interaction_entry)

            # Keep only recent interactions (last 1000)
            if len(profile.interaction_history) > 1000:
                profile.interaction_history = profile.interaction_history[-1000:]

            # Update engagement metrics
            self._update_engagement_metrics(profile, interaction)

            profile.last_interaction = datetime.now()
            profile.updated_at = datetime.now()

    def get_personalization_context(self, user_id: str, session_id: str) -> PersonalizationContext:
        """Get personalization context for a user."""
        profile = self.get_or_create_profile(user_id)

        # Get recent interactions (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_interactions = [
            interaction for interaction in profile.interaction_history
            if datetime.fromisoformat(interaction["timestamp"]) > recent_cutoff
        ]

        # Infer emotional state from recent interactions
        emotional_state = self._infer_emotional_state(recent_interactions)

        # Estimate cognitive load
        cognitive_load = self._estimate_cognitive_load(recent_interactions)

        # Get time context
        time_context = self._get_time_context()

        # Get environmental context (simplified)
        environmental_context = {
            "timezone": "UTC",  # Would be detected
            "device_type": "unknown"  # Would be detected
        }

        return PersonalizationContext(
            user_id=user_id,
            current_session=session_id,
            recent_interactions=recent_interactions[-10:],  # Last 10 interactions
            active_goals=profile.preferences.get("active_goals", []),
            emotional_state=emotional_state,
            cognitive_load=cognitive_load,
            time_context=time_context,
            environmental_context=environmental_context
        )

    def _update_engagement_metrics(self, profile: UserProfile, interaction: Dict[str, Any]):
        """Update engagement metrics based on interaction."""
        # Simple engagement calculation
        interaction_type = interaction.get("type", "")
        outcome = interaction.get("outcome", "")

        engagement_boost = 0.0

        if interaction_type == "positive_feedback":
            engagement_boost = 0.1
        elif interaction_type == "negative_feedback":
            engagement_boost = -0.05
        elif outcome == "success":
            engagement_boost = 0.05
        elif outcome == "failure":
            engagement_boost = -0.02

        # Update engagement level (moving average)
        alpha = 0.1  # Learning rate
        profile.engagement_level = (1 - alpha) * profile.engagement_level + alpha * max(0, min(1, profile.engagement_level + engagement_boost))

    def _update_personalization_vector(self, profile: UserProfile):
        """Update the personalization vector based on preferences and history."""
        # Simplified vector generation based on preferences
        # In a real implementation, this would use embeddings
        preferences = profile.preferences

        vector = []

        # Communication style preferences
        comm_style = preferences.get("communication_style", "balanced")
        if comm_style == "formal":
            vector.extend([0.8, 0.2, 0.5])
        elif comm_style == "casual":
            vector.extend([0.2, 0.8, 0.5])
        else:
            vector.extend([0.5, 0.5, 0.5])

        # Content preferences
        content_prefs = preferences.get("content_preferences", {})
        vector.extend([
            content_prefs.get("technical", 0.5),
            content_prefs.get("creative", 0.5),
            content_prefs.get("practical", 0.5)
        ])

        # Learning style
        learning_style = preferences.get("learning_style", "visual")
        if learning_style == "visual":
            vector.extend([0.8, 0.4, 0.6])
        elif learning_style == "auditory":
            vector.extend([0.4, 0.8, 0.6])
        else:
            vector.extend([0.6, 0.6, 0.6])

        profile.personalization_vector = vector

    def _infer_emotional_state(self, recent_interactions: List[Dict[str, Any]]) -> Optional[str]:
        """Infer emotional state from recent interactions."""
        if not recent_interactions:
            return None

        # Simple sentiment analysis based on outcomes
        positive_count = 0
        negative_count = 0

        for interaction in recent_interactions[-5:]:  # Last 5 interactions
            outcome = interaction.get("outcome", "")
            if outcome in ["success", "positive_feedback"]:
                positive_count += 1
            elif outcome in ["failure", "negative_feedback"]:
                negative_count += 1

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "frustrated"
        else:
            return "neutral"

    def _estimate_cognitive_load(self, recent_interactions: List[Dict[str, Any]]) -> float:
        """Estimate cognitive load from interaction patterns."""
        if not recent_interactions:
            return 0.5

        # Estimate based on interaction frequency and complexity
        recent_count = len([i for i in recent_interactions
                           if datetime.fromisoformat(i["timestamp"]) > datetime.now() - timedelta(minutes=30)])

        # Higher frequency = higher cognitive load
        frequency_load = min(1.0, recent_count / 10.0)

        # Complexity based on interaction types
        complex_types = ["analysis", "problem_solving", "learning"]
        complex_count = sum(1 for i in recent_interactions[-10:]
                           if i.get("type") in complex_types)

        complexity_load = min(1.0, complex_count / 5.0)

        return (frequency_load + complexity_load) / 2.0

    def _get_time_context(self) -> Dict[str, Any]:
        """Get current time context."""
        now = datetime.now()

        return {
            "hour": now.hour,
            "day_of_week": now.weekday(),
            "is_weekend": now.weekday() >= 5,
            "season": self._get_season(now),
            "time_of_day": self._get_time_of_day(now.hour)
        }

    def _get_season(self, dt: datetime) -> str:
        """Get season from datetime."""
        month = dt.month
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall"

    def _get_time_of_day(self, hour: int) -> str:
        """Get time of day from hour."""
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "night"


class PersonalizationEngine:
    """
    Engine for generating personalized responses and recommendations.

    Uses user profiles, episodic memories, and context to personalize interactions.
    """

    def __init__(self, memory_store: EpisodicMemoryStore, profile_manager: UserProfileManager):
        self.memory_store = memory_store
        self.profile_manager = profile_manager
        self.personalization_rules: Dict[str, PersonalizationRule] = {}
        self._initialize_default_rules()

    def _initialize_default_rules(self):
        """Initialize default personalization rules."""
        rules = [
            PersonalizationRule(
                id="time_based_greeting",
                name="Time-based greeting",
                condition={"time_of_day": ["morning", "afternoon", "evening"]},
                action={"modify_response": {"add_greeting": True}},
                priority=8
            ),
            PersonalizationRule(
                id="emotional_support",
                name="Emotional support for frustrated users",
                condition={"emotional_state": "frustrated"},
                action={"modify_response": {"add_encouragement": True, "be_more_helpful": True}},
                priority=9
            ),
            PersonalizationRule(
                id="cognitive_load_adaptation",
                name="Adapt to high cognitive load",
                condition={"cognitive_load": {"gt": 0.7}},
                action={"modify_response": {"simplify_explanation": True, "break_down_steps": True}},
                priority=7
            ),
            PersonalizationRule(
                id="learning_style_adaptation",
                name="Adapt to learning style",
                condition={"learning_style": ["visual", "auditory", "kinesthetic"]},
                action={"modify_response": {"use_learning_style": True}},
                priority=6
            ),
            PersonalizationRule(
                id="previous_success_boost",
                name="Boost confidence for previously successful topics",
                condition={"previous_success": True},
                action={"modify_response": {"add_confidence_booster": True}},
                priority=5
            )
        ]

        for rule in rules:
            self.personalization_rules[rule.id] = rule

    def personalize_response(
        self,
        user_id: str,
        base_response: str,
        context: PersonalizationContext,
        request_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Personalize a response based on user context and history.

        Returns modified response with personalization metadata.
        """
        personalization_applied = []
        modifications = {}

        # Get user profile
        profile = self.profile_manager.get_or_create_profile(user_id)

        # Apply personalization rules
        applicable_rules = self._evaluate_rules(context, profile, request_metadata)

        for rule in applicable_rules:
            modifications.update(rule.action.get("modify_response", {}))
            personalization_applied.append({
                "rule_id": rule.id,
                "rule_name": rule.name,
                "priority": rule.priority
            })

        # Apply modifications to response
        personalized_response = self._apply_modifications(base_response, modifications, context)

        # Get relevant memories for context
        relevant_memories = self._get_relevant_memories(user_id, request_metadata)

        return {
            "personalized_response": personalized_response,
            "personalization_applied": personalization_applied,
            "relevant_memories": [self._memory_to_dict(mem) for mem in relevant_memories[:3]],  # Top 3
            "context_insights": self._generate_context_insights(context, profile),
            "confidence_score": self._calculate_personalization_confidence(personalization_applied)
        }

    def _evaluate_rules(
        self,
        context: PersonalizationContext,
        profile: UserProfile,
        request_metadata: Dict[str, Any]
    ) -> List[PersonalizationRule]:
        """Evaluate which personalization rules apply."""
        applicable_rules = []

        for rule in self.personalization_rules.values():
            if not rule.enabled:
                continue

            if self._rule_matches_condition(rule.condition, context, profile, request_metadata):
                applicable_rules.append(rule)

        # Sort by priority (highest first)
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)

        return applicable_rules

    def _rule_matches_condition(
        self,
        condition: Dict[str, Any],
        context: PersonalizationContext,
        profile: UserProfile,
        request_metadata: Dict[str, Any]
    ) -> bool:
        """Check if a rule condition matches the current context."""
        for key, expected_value in condition.items():
            actual_value = None

            # Get actual value from context/profile/metadata
            if hasattr(context, key):
                actual_value = getattr(context, key)
            elif key in profile.preferences:
                actual_value = profile.preferences[key]
            elif key in request_metadata:
                actual_value = request_metadata[key]
            elif key in context.time_context:
                actual_value = context.time_context[key]
            elif key == "emotional_state":
                actual_value = context.emotional_state
            elif key == "cognitive_load":
                actual_value = context.cognitive_load

            # Check condition
            if isinstance(expected_value, dict):
                # Range conditions like {"gt": 0.7}
                if "gt" in expected_value and actual_value is not None:
                    if not (isinstance(actual_value, (int, float)) and actual_value > expected_value["gt"]):
                        return False
                elif "lt" in expected_value and actual_value is not None:
                    if not (isinstance(actual_value, (int, float)) and actual_value < expected_value["lt"]):
                        return False
            elif isinstance(expected_value, list):
                # List membership
                if actual_value not in expected_value:
                    return False
            else:
                # Exact match
                if actual_value != expected_value:
                    return False

        return True

    def _apply_modifications(
        self,
        base_response: str,
        modifications: Dict[str, Any],
        context: PersonalizationContext
    ) -> str:
        """Apply personalization modifications to the response."""
        response = base_response

        if modifications.get("add_greeting"):
            time_of_day = context.time_context.get("time_of_day", "day")
            greeting = f"Good {time_of_day}! "
            response = greeting + response

        if modifications.get("add_encouragement"):
            encouragement = " I understand this might be frustrating. Let me help you through this step by step. "
            response = encouragement + response

        if modifications.get("simplify_explanation"):
            # Add simplified language markers (would be more sophisticated)
            response = response.replace("Furthermore", "Also").replace("Additionally", "Plus")

        if modifications.get("break_down_steps"):
            # Add step breakdown if response contains numbered steps
            if "1." in response and "2." in response:
                response = "Let me break this down for you:\n\n" + response

        if modifications.get("add_confidence_booster"):
            confidence_booster = " Based on your previous successes in similar areas, I'm confident we can solve this together. "
            response = confidence_booster + response

        return response

    def _get_relevant_memories(self, user_id: str, request_metadata: Dict[str, Any]) -> List[EpisodicMemory]:
        """Get memories relevant to the current request."""
        # Extract keywords/topics from request
        content = request_metadata.get("content", "")
        keywords = self._extract_keywords(content)

        # Search for relevant memories
        relevant_memories = []

        for keyword in keywords:
            memories = self.memory_store.retrieve_memories(
                user_id=user_id,
                tags=[keyword],
                limit=5,
                min_importance=0.3
            )
            relevant_memories.extend(memories)

        # Remove duplicates and sort by relevance
        seen_ids = set()
        unique_memories = []
        for memory in relevant_memories:
            if memory.id not in seen_ids:
                seen_ids.add(memory.id)
                unique_memories.append(memory)

        # Sort by recency and importance
        unique_memories.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)

        return unique_memories[:5]  # Top 5

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content."""
        # Simple keyword extraction (would use NLP in production)
        words = content.lower().split()
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}

        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        return list(set(keywords))[:5]  # Top 5 unique keywords

    def _generate_context_insights(self, context: PersonalizationContext, profile: UserProfile) -> Dict[str, Any]:
        """Generate insights about the current context."""
        insights = {}

        # Time-based insights
        time_of_day = context.time_context.get("time_of_day")
        if time_of_day == "morning":
            insights["time_insight"] = "User might prefer quick, actionable responses in the morning"
        elif time_of_day == "evening":
            insights["time_insight"] = "User might be more receptive to detailed explanations in the evening"

        # Engagement insights
        if profile.engagement_level > 0.8:
            insights["engagement_insight"] = "Highly engaged user - provide comprehensive responses"
        elif profile.engagement_level < 0.3:
            insights["engagement_insight"] = "Low engagement - keep responses concise and focused"

        # Cognitive load insights
        if context.cognitive_load > 0.7:
            insights["cognitive_insight"] = "High cognitive load - simplify explanations and break down complex topics"

        # Emotional insights
        if context.emotional_state == "frustrated":
            insights["emotional_insight"] = "User appears frustrated - provide extra support and encouragement"

        return insights

    def _calculate_personalization_confidence(self, applied_rules: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for personalization."""
        if not applied_rules:
            return 0.5

        # Average priority as confidence proxy
        avg_priority = statistics.mean(rule["priority"] for rule in applied_rules)
        confidence = min(1.0, avg_priority / 10.0)

        return confidence

    def _memory_to_dict(self, memory: EpisodicMemory) -> Dict[str, Any]:
        """Convert memory to dictionary for API responses."""
        return {
            "id": memory.id,
            "type": memory.memory_type.value,
            "content": str(memory.content)[:200] + "..." if len(str(memory.content)) > 200 else str(memory.content),
            "importance": memory.importance,
            "timestamp": memory.timestamp.isoformat(),
            "tags": list(memory.tags)
        }

    def add_personalization_rule(self, rule: PersonalizationRule):
        """Add a new personalization rule."""
        self.personalization_rules[rule.id] = rule
        logger.info(f"Added personalization rule: {rule.name}")

    def remove_personalization_rule(self, rule_id: str) -> bool:
        """Remove a personalization rule."""
        if rule_id in self.personalization_rules:
            del self.personalization_rules[rule_id]
            logger.info(f"Removed personalization rule: {rule_id}")
            return True
        return False


class PrivacyController:
    """
    Controls privacy and data retention for episodic memories.

    Ensures compliance with privacy regulations and user preferences.
    """

    def __init__(self, memory_store: EpisodicMemoryStore, profile_manager: UserProfileManager):
        self.memory_store = memory_store
        self.profile_manager = profile_manager
        self.consent_records: Dict[str, Dict[str, Any]] = {}  # user_id -> consent data
        self.privacy_policies: Dict[str, Dict[str, Any]] = {}
        self._initialize_default_policies()

    def _initialize_default_policies(self):
        """Initialize default privacy policies."""
        self.privacy_policies = {
            "interaction_data": {
                "retention_period": timedelta(days=365),
                "consent_required": False,
                "anonymization_required": False
            },
            "feedback_data": {
                "retention_period": timedelta(days=730),
                "consent_required": False,
                "anonymization_required": False
            },
            "personalization_data": {
                "retention_period": timedelta(days=1095),
                "consent_required": True,
                "anonymization_required": True
            },
            "sensitive_data": {
                "retention_period": timedelta(days=2555),
                "consent_required": True,
                "anonymization_required": True
            }
        }

    def check_privacy_compliance(self, memory: EpisodicMemory) -> bool:
        """Check if a memory complies with privacy policies."""
        user_id = memory.user_id

        # Check user consent
        if not self._has_user_consent(user_id, memory.privacy_level):
            return False

        # Check retention policy compliance
        policy = self.privacy_policies.get(memory.memory_type.value, self.privacy_policies["interaction_data"])
        max_age = policy["retention_period"]

        if datetime.now() - memory.timestamp > max_age:
            return False

        return True

    def _has_user_consent(self, user_id: str, privacy_level: PrivacyLevel) -> bool:
        """Check if user has consented to data collection."""
        consent = self.consent_records.get(user_id, {})

        # Different consent levels required for different privacy levels
        if privacy_level == PrivacyLevel.PRIVATE:
            return consent.get("private_data", False)
        elif privacy_level == PrivacyLevel.SENSITIVE:
            return consent.get("sensitive_data", False)
        else:
            return consent.get("general_data", True)  # Default to allowed for internal/public

    def record_user_consent(self, user_id: str, consent_data: Dict[str, Any]):
        """Record user consent preferences."""
        self.consent_records[user_id] = {
            "timestamp": datetime.now(),
            "consent_data": consent_data
        }
        logger.info(f"Recorded consent for user: {user_id}")

    def anonymize_memory(self, memory: EpisodicMemory) -> EpisodicMemory:
        """Anonymize a memory for privacy compliance."""
        # Create anonymized copy
        anonymized = EpisodicMemory(
            id=memory.id,
            user_id=hashlib.sha256(memory.user_id.encode()).hexdigest()[:16],  # Hash user ID
            session_id=None,  # Remove session info
            memory_type=memory.memory_type,
            content=self._anonymize_content(memory.content),
            context={},  # Remove context
            metadata={},  # Remove metadata
            timestamp=memory.timestamp,
            importance=memory.importance,
            confidence=memory.confidence,
            privacy_level=PrivacyLevel.INTERNAL,
            retention_policy=memory.retention_policy,
            tags=set()  # Remove tags
        )

        return anonymized

    def _anonymize_content(self, content: Any) -> Any:
        """Anonymize content data."""
        if isinstance(content, str):
            # Simple anonymization - replace potential PII
            anonymized = content
            # Replace email-like patterns
            import re
            anonymized = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', anonymized)
            # Replace phone-like patterns
            anonymized = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', anonymized)
            return anonymized
        else:
            return "[ANONYMIZED_DATA]"

    def get_privacy_summary(self, user_id: str) -> Dict[str, Any]:
        """Get privacy summary for a user."""
        consent = self.consent_records.get(user_id, {})

        memory_stats = self.memory_store.get_memory_stats(user_id)

        return {
            "user_id": user_id,
            "consent_status": consent.get("consent_data", {}),
            "memory_statistics": memory_stats,
            "privacy_compliant": self._check_user_compliance(user_id)
        }

    def _check_user_compliance(self, user_id: str) -> bool:
        """Check if user's data is privacy compliant."""
        user_memories = self.memory_store.retrieve_memories(user_id=user_id, limit=1000)

        for memory in user_memories:
            if not self.check_privacy_compliance(memory):
                return False

        return True

    def purge_user_data(self, user_id: str) -> int:
        """Purge all data for a user."""
        memories_deleted = 0

        # Delete all memories for user
        user_memories = self.memory_store.retrieve_memories(user_id=user_id, limit=10000)
        for memory in user_memories:
            if self.memory_store.delete_memory(memory.id):
                memories_deleted += 1

        # Delete profile
        if user_id in self.profile_manager.profiles:
            del self.profile_manager.profiles[user_id]

        # Delete consent records
        if user_id in self.consent_records:
            del self.consent_records[user_id]

        logger.info(f"Purged {memories_deleted} memories for user: {user_id}")
        return memories_deleted


class EpisodicMemoryPersonalizationSystem:
    """
    Episodic memory and personalization system.
    """

    def __init__(self, config_manager=None):
        self.config_manager = config_manager

        # Core components
        self.memory_store = EpisodicMemoryStore()
        self.profile_manager = UserProfileManager(self.memory_store)
        self.personalization_engine = PersonalizationEngine(self.memory_store, self.profile_manager)
        self.privacy_controller = PrivacyController(self.memory_store, self.profile_manager)

        # Background processing
        self.cleanup_task: Optional[asyncio.Task] = None
        self.learning_task: Optional[asyncio.Task] = None

        logger.info("Episodic Memory & Personalization System initialized")

    async def start(self):
        """Start the system."""
        # Start background cleanup
        self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
        # Start learning updates
        self.learning_task = asyncio.create_task(self._periodic_learning())

        logger.info("Episodic Memory & Personalization System started")

    async def stop(self):
        """Stop the system."""
        tasks_to_cancel = []

        if self.cleanup_task:
            self.cleanup_task.cancel()
            tasks_to_cancel.append(self.cleanup_task)

        if self.learning_task:
            self.learning_task.cancel()
            tasks_to_cancel.append(self.learning_task)

        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        logger.info("Episodic Memory & Personalization System stopped")

    async def _periodic_cleanup(self):
        """Periodic cleanup of old memories."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                # Cleanup is handled automatically by memory store
                logger.debug("Performed periodic memory cleanup")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Cleanup error: {exc}")

    async def _periodic_learning(self):
        """Periodic learning updates."""
        while True:
            try:
                await asyncio.sleep(7200)  # Run every 2 hours
                await self._update_personalization_models()
                logger.debug("Performed periodic learning update")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Learning error: {exc}")

    async def _update_personalization_models(self):
        """Update personalization models based on new data."""
        # This would update ML models for personalization
        # For now, just update profile vectors
        for profile in self.profile_manager.profiles.values():
            self.profile_manager._update_personalization_vector(profile)

    def store_episodic_memory(
        self,
        user_id: str,
        memory_type: MemoryType,
        content: Any,
        **kwargs
    ) -> str:
        """Store an episodic memory."""
        memory = EpisodicMemory(
            id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=kwargs.get("session_id"),
            memory_type=memory_type,
            content=content,
            context=kwargs.get("context", {}),
            metadata=kwargs.get("metadata", {}),
            importance=kwargs.get("importance", 0.5),
            confidence=kwargs.get("confidence", 0.8),
            privacy_level=kwargs.get("privacy_level", PrivacyLevel.INTERNAL),
            retention_policy=kwargs.get("retention_policy", RetentionPolicy.YEARS_1),
            tags=set(kwargs.get("tags", []))
        )

        # Check privacy compliance
        if not self.privacy_controller.check_privacy_compliance(memory):
            logger.warning(f"Memory rejected due to privacy compliance: {memory.id}")
            return ""

        return self.memory_store.store_memory(memory)

    def get_personalized_response(
        self,
        user_id: str,
        base_response: str,
        session_id: str,
        request_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get a personalized response."""
        context = self.profile_manager.get_personalization_context(user_id, session_id)

        return self.personalization_engine.personalize_response(
            user_id, base_response, context, request_metadata
        )

    def record_user_interaction(
        self,
        user_id: str,
        interaction_type: str,
        content: str,
        outcome: Optional[str] = None,
        **kwargs
    ):
        """Record a user interaction."""
        interaction = {
            "type": interaction_type,
            "content": content,
            "outcome": outcome,
            "metadata": kwargs
        }

        self.profile_manager.record_interaction(user_id, interaction)

        # Also store as episodic memory
        self.store_episodic_memory(
            user_id=user_id,
            memory_type=MemoryType.INTERACTION,
            content=f"{interaction_type}: {content}",
            context={"outcome": outcome},
            metadata=kwargs,
            importance=0.6 if outcome == "success" else 0.4,
            tags=[interaction_type, outcome] if outcome else [interaction_type]
        )

    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive insights about a user."""
        profile = self.profile_manager.get_or_create_profile(user_id)
        memory_stats = self.memory_store.get_memory_stats(user_id)
        privacy_summary = self.privacy_controller.get_privacy_summary(user_id)

        # Get recent memories
        recent_memories = self.memory_store.retrieve_memories(
            user_id=user_id,
            limit=10,
            time_range=(datetime.now() - timedelta(days=7), datetime.now())
        )

        return {
            "profile": {
                "user_id": profile.user_id,
                "engagement_level": profile.engagement_level,
                "trust_score": profile.trust_score,
                "last_interaction": profile.last_interaction.isoformat() if profile.last_interaction else None,
                "preferences": profile.preferences
            },
            "memory_stats": memory_stats,
            "privacy_summary": privacy_summary,
            "recent_memories": [self._memory_to_summary(mem) for mem in recent_memories],
            "personalization_context": self.profile_manager.get_personalization_context(user_id, "insights")
        }

    def _memory_to_summary(self, memory: EpisodicMemory) -> Dict[str, Any]:
        """Convert memory to summary format."""
        return {
            "id": memory.id,
            "type": memory.memory_type.value,
            "importance": memory.importance,
            "timestamp": memory.timestamp.isoformat(),
            "tags": list(memory.tags),
            "content_preview": str(memory.content)[:100] + "..." if len(str(memory.content)) > 100 else str(memory.content)
        }

    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Update user preferences."""
        return self.profile_manager.update_profile(user_id, {"preferences": preferences})

    def set_user_consent(self, user_id: str, consent_data: Dict[str, Any]):
        """Set user privacy consent."""
        self.privacy_controller.record_user_consent(user_id, consent_data)

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics."""
        memory_stats = self.memory_store.get_memory_stats()
        profile_count = len(self.profile_manager.profiles)

        return {
            "total_users": profile_count,
            "total_memories": memory_stats["total_memories"],
            "memory_utilization": memory_stats["storage_utilization"],
            "avg_memories_per_user": memory_stats["total_memories"] / max(profile_count, 1),
            "privacy_compliant_users": sum(
                1 for uid in self.profile_manager.profiles.keys()
                if self.privacy_controller._check_user_compliance(uid)
            )
        }


# Register with provider registry
def create_episodic_memory_system(config_manager=None, **kwargs):
    """Factory function for EpisodicMemoryPersonalizationSystem."""
    return EpisodicMemoryPersonalizationSystem(config_manager=config_manager)


provider_registry.register_lazy(
    'adaptive_intelligence', 'episodic_memory',
    'mia.adaptive_intelligence.episodic_memory_personalization', 'create_episodic_memory_system',
    default=True
)