"""
Advanced Dialog Manager
"""

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from ..memory.rag_pipeline import RAGPipeline
from ..providers import provider_registry

logger = logging.getLogger(__name__)


class DialogState(Enum):
    """Enumeration of possible dialog states."""

    INITIAL = "initial"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    WAITING_CONFIRMATION = "waiting_confirmation"
    ERROR_RECOVERY = "error_recovery"
    COMPLETED = "completed"


class IntentType(Enum):
    """Enumeration of recognized intent types."""

    QUESTION = "question"
    COMMAND = "command"
    STATEMENT = "statement"
    CLARIFICATION = "clarification"
    CONFIRMATION = "confirmation"
    CORRECTION = "correction"
    GREETING = "greeting"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"


class Modality(Enum):
    """Enumeration of interaction modalities."""

    TEXT = "text"
    VOICE = "voice"
    VISION = "vision"
    MULTIMODAL = "multimodal"


@dataclass
class ConversationContext:
    """Represents the current conversation context."""

    session_id: str
    user_id: str
    current_state: DialogState = DialogState.INITIAL
    modality: Modality = Modality.TEXT
    intent: IntentType = IntentType.UNKNOWN
    confidence: float = 0.0
    entities: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    turn_count: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DialogRule:
    """Represents a dialog management rule."""

    name: str
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    priority: int = 0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntentPattern:
    """Represents an intent recognition pattern."""

    intent: IntentType
    patterns: List[str]
    keywords: Set[str]
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    confidence_threshold: float = 0.7
    examples: List[str] = field(default_factory=list)


class DialogManager:
    """
    Advanced dialog manager with configurable state tracking and intent routing.

    Features:
    - Conversation state management
    - Intent recognition and routing
    - Context-aware responses
    - Adaptive behavior learning
    - Multi-turn conversation handling
    """

    def __init__(
        self,
        config_manager=None,
        rag_pipeline: Optional[RAGPipeline] = None,
        intent_classifier=None,
        rules_file: Optional[str] = None,
        *,
        max_history_length: int = 50,
        session_timeout_minutes: int = 30,
        learning_enabled: bool = True,
    ):
        self.config_manager = config_manager
        self.rag_pipeline = rag_pipeline
        self.intent_classifier = intent_classifier
        self.max_history_length = max_history_length
        self.session_timeout_minutes = session_timeout_minutes
        self.learning_enabled = learning_enabled

        # Core components
        self.intent_patterns: Dict[IntentType, IntentPattern] = {}
        self.dialog_rules: List[DialogRule] = []
        self.active_sessions: Dict[str, ConversationContext] = {}
        self.session_lock = threading.RLock()

        # Learning components
        self.intent_history: List[Dict[str, Any]] = []
        self.rule_performance: Dict[str, Dict[str, Any]] = {}
        self.user_patterns: Dict[str, Dict[str, Any]] = {}

        # Load default intent patterns
        self._load_default_intent_patterns()

        # Load dialog rules
        if rules_file:
            self.load_rules_from_file(rules_file)
        else:
            self._load_default_rules()

        # Initialize intent classifier if not provided
        if not self.intent_classifier:
            self.intent_classifier = self._create_default_intent_classifier()

        logger.info(
            "Dialog Manager initialized with %d rules and %d intent patterns",
            len(self.dialog_rules),
            len(self.intent_patterns),
        )

    def _load_default_intent_patterns(self):
        """Load default intent recognition patterns."""
        patterns = {
            IntentType.QUESTION: IntentPattern(
                intent=IntentType.QUESTION,
                patterns=[
                    r"(?i)^(what|who|where|when|why|how|which|whose|whom)",
                    r"(?i)\?$",
                    r"(?i)(can you|could you|would you|do you know)",
                ],
                keywords={
                    "what",
                    "who",
                    "where",
                    "when",
                    "why",
                    "how",
                    "which",
                    "whose",
                    "whom",
                    "can",
                    "could",
                    "would",
                    "know",
                    "tell",
                    "explain",
                },
                examples=[
                    "What is the weather?",
                    "How do I do this?",
                    "Can you help me?",
                ],
            ),
            IntentType.COMMAND: IntentPattern(
                intent=IntentType.COMMAND,
                patterns=[
                    r"(?i)^(please |can you |could you |would you )?(open|close|start|stop|create|delete|send|show|display)",
                    r"(?i)(do this|do that|make it|run|execute|perform)",
                ],
                keywords={
                    "open",
                    "close",
                    "start",
                    "stop",
                    "create",
                    "delete",
                    "send",
                    "show",
                    "display",
                    "run",
                    "execute",
                    "perform",
                    "do",
                    "make",
                },
                examples=["Open the browser", "Create a new file", "Send an email"],
            ),
            IntentType.CLARIFICATION: IntentPattern(
                intent=IntentType.CLARIFICATION,
                patterns=[
                    r"(?i)(what do you mean|clarify|explain|elaborate|i don't understand)",
                    r"(?i)(can you rephrase|say again|repeat that)",
                ],
                keywords={
                    "clarify",
                    "explain",
                    "elaborate",
                    "understand",
                    "rephrase",
                    "repeat",
                },
                examples=[
                    "What do you mean?",
                    "Can you clarify that?",
                    "I don't understand",
                ],
            ),
            IntentType.CONFIRMATION: IntentPattern(
                intent=IntentType.CONFIRMATION,
                patterns=[
                    r"(?i)^(yes|no|correct|wrong|right|sure|okay|ok|confirm|deny)",
                    r"(?i)(that's right|that's wrong|you're right|you're wrong)",
                ],
                keywords={
                    "yes",
                    "no",
                    "correct",
                    "wrong",
                    "right",
                    "sure",
                    "okay",
                    "ok",
                    "confirm",
                    "deny",
                    "that's",
                },
                examples=["Yes, that's correct", "No, that's wrong", "Sure, go ahead"],
            ),
            IntentType.GREETING: IntentPattern(
                intent=IntentType.GREETING,
                patterns=[
                    r"(?i)^(hi|hello|hey|greetings|good morning|good afternoon|good evening)",
                    r"(?i)(how are you|how's it going|what's up)",
                ],
                keywords={
                    "hi",
                    "hello",
                    "hey",
                    "greetings",
                    "morning",
                    "afternoon",
                    "evening",
                },
                examples=["Hello!", "Good morning", "How are you?"],
            ),
            IntentType.GOODBYE: IntentPattern(
                intent=IntentType.GOODBYE,
                patterns=[
                    r"(?i)^(bye|goodbye|see you|farewell|take care|good night)",
                    r"(?i)(i'm leaving|got to go|have to go)",
                ],
                keywords={
                    "bye",
                    "goodbye",
                    "see",
                    "farewell",
                    "care",
                    "night",
                    "leaving",
                    "go",
                },
                examples=["Goodbye!", "See you later", "Take care"],
            ),
        }

        self.intent_patterns.update(patterns)

    def _load_default_rules(self):
        """Load default dialog management rules."""
        rules = [
            DialogRule(
                name="greeting_response",
                conditions={
                    "intent": IntentType.GREETING,
                    "state": DialogState.INITIAL,
                },
                actions=[
                    {"type": "set_state", "state": DialogState.LISTENING},
                    {"type": "respond", "template": "greeting_response"},
                    {"type": "update_context", "key": "greeted", "value": True},
                ],
                priority=10,
            ),
            DialogRule(
                name="question_routing",
                conditions={
                    "intent": IntentType.QUESTION,
                    "state": [DialogState.LISTENING, DialogState.RESPONDING],
                },
                actions=[
                    {"type": "set_state", "state": DialogState.PROCESSING},
                    {"type": "route_to_llm", "use_rag": True},
                    {"type": "set_state", "state": DialogState.RESPONDING},
                ],
                priority=5,
            ),
            DialogRule(
                name="command_execution",
                conditions={
                    "intent": IntentType.COMMAND,
                    "state": DialogState.LISTENING,
                },
                actions=[
                    {"type": "set_state", "state": DialogState.PROCESSING},
                    {"type": "route_to_action_executor"},
                    {"type": "set_state", "state": DialogState.RESPONDING},
                ],
                priority=8,
            ),
            DialogRule(
                name="clarification_request",
                conditions={
                    "intent": IntentType.CLARIFICATION,
                    "state": DialogState.RESPONDING,
                },
                actions=[
                    {"type": "set_state", "state": DialogState.ERROR_RECOVERY},
                    {"type": "respond", "template": "clarification_response"},
                    {"type": "request_rephrase"},
                ],
                priority=9,
            ),
            DialogRule(
                name="confirmation_handling",
                conditions={
                    "intent": IntentType.CONFIRMATION,
                    "state": DialogState.WAITING_CONFIRMATION,
                },
                actions=[
                    {"type": "process_confirmation"},
                    {"type": "set_state", "state": DialogState.LISTENING},
                ],
                priority=7,
            ),
            DialogRule(
                name="session_timeout",
                conditions={
                    "inactive_minutes": lambda ctx: (
                        datetime.now() - ctx.last_activity
                    ).total_seconds()
                    / 60
                    > self.session_timeout_minutes
                },
                actions=[
                    {"type": "set_state", "state": DialogState.COMPLETED},
                    {"type": "cleanup_session"},
                ],
                priority=1,
            ),
        ]

        self.dialog_rules.extend(rules)

    def _create_default_intent_classifier(self):
        """Create a default intent classifier using pattern matching."""

        def classify_intent(
            text: str, context: ConversationContext
        ) -> tuple[IntentType, float]:
            """Classify intent using pattern matching and context."""
            text_lower = text.lower()
            best_match = IntentType.UNKNOWN
            best_confidence = 0.0

            for intent, pattern in self.intent_patterns.items():
                confidence = 0.0
                matches = 0

                # Check keywords
                keyword_matches = sum(
                    1 for keyword in pattern.keywords if keyword in text_lower
                )
                if keyword_matches > 0:
                    confidence += min(keyword_matches / len(pattern.keywords), 0.5)

                # Check patterns (simplified regex matching)
                for pattern_str in pattern.patterns:
                    if pattern_str.startswith("(?i)") and pattern_str.endswith(")"):
                        # Simple pattern matching for common cases
                        inner_pattern = pattern_str[4:-1]  # Remove (?i) and )
                        if inner_pattern in text_lower:
                            confidence += 0.3
                            matches += 1

                # Context-based adjustments
                if context.intent == intent and context.confidence > 0.8:
                    confidence += 0.2  # Continuity bonus

                if (
                    confidence > best_confidence
                    and confidence >= pattern.confidence_threshold
                ):
                    best_match = intent
                    best_confidence = confidence

            return best_match, min(best_confidence, 1.0)

        return classify_intent

    def start_conversation(
        self, user_id: str, modality: Modality = Modality.TEXT
    ) -> str:
        """Start a new conversation session."""
        session_id = str(uuid.uuid4())

        context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            modality=modality,
            current_state=DialogState.INITIAL,
        )

        # Load user profile if available
        if self.config_manager:
            try:
                # Load user preferences and history
                context.user_profile = getattr(
                    self.config_manager, "get_user_profile", lambda uid: {}
                )(user_id)
                context.preferences = context.user_profile.get("preferences", {})
            except Exception as exc:
                logger.debug("Failed to load user profile: %s", exc)

        with self.session_lock:
            self.active_sessions[session_id] = context

        logger.info("Started conversation session %s for user %s", session_id, user_id)
        return session_id

    def process_input(
        self,
        session_id: str,
        user_input: str,
        modality: Optional[Modality] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process user input and return dialog response."""
        with self.session_lock:
            if session_id not in self.active_sessions:
                raise ValueError(f"Invalid session ID: {session_id}")

            context = self.active_sessions[session_id]

        # Update context
        context.last_activity = datetime.now()
        context.turn_count += 1

        if modality:
            context.modality = modality

        if metadata:
            context.metadata.update(metadata)

        # Add to conversation history
        context.conversation_history.append(
            {
                "turn": context.turn_count,
                "timestamp": context.last_activity.isoformat(),
                "user_input": user_input,
                "modality": modality.value if modality else context.modality.value,
            }
        )

        # Trim history if too long
        if len(context.conversation_history) > self.max_history_length:
            context.conversation_history = context.conversation_history[
                -self.max_history_length :
            ]

        # Classify intent
        if self.intent_classifier is None:
            intent, confidence = IntentType.UNKNOWN, 0.0
        else:
            intent, confidence = self.intent_classifier(user_input, context)
        context.intent = intent
        context.confidence = confidence

        # Extract entities (simplified)
        context.entities = self._extract_entities(user_input, intent)

        # Apply dialog rules
        response = self._apply_dialog_rules(context, user_input)

        # Update conversation history with response
        if response:
            context.conversation_history[-1]["response"] = response.get("text", "")
            context.conversation_history[-1]["actions"] = response.get("actions", [])

        # Learning: record intent classification
        if self.learning_enabled:
            self._record_intent_classification(user_input, intent, confidence, context)

        return response

    def _extract_entities(self, text: str, intent: IntentType) -> Dict[str, Any]:
        """Extract entities from user input (simplified implementation)."""
        entities = {}

        # Simple entity extraction based on intent
        if intent == IntentType.COMMAND:
            # Look for action targets
            words = text.lower().split()
            if "open" in words or "start" in words:
                entities["action"] = "open"
            elif "close" in words or "stop" in words:
                entities["action"] = "close"
            elif "create" in words:
                entities["action"] = "create"
            elif "delete" in words:
                entities["action"] = "delete"

        elif intent == IntentType.QUESTION:
            # Look for question types
            if text.lower().startswith("what"):
                entities["question_type"] = "definition"
            elif text.lower().startswith("how"):
                entities["question_type"] = "process"
            elif text.lower().startswith("why"):
                entities["question_type"] = "reason"

        return entities

    def _apply_dialog_rules(
        self, context: ConversationContext, user_input: str
    ) -> Dict[str, Any]:
        """Apply dialog rules based on current context."""
        applicable_rules = []

        for rule in self.dialog_rules:
            if not rule.enabled:
                continue

            if self._rule_matches(rule, context, user_input):
                applicable_rules.append(rule)

        # Sort by priority (highest first)
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)

        response = {
            "text": "",
            "actions": [],
            "state": context.current_state.value,
            "intent": context.intent.value,
            "confidence": context.confidence,
        }

        # Execute highest priority rule
        if applicable_rules:
            rule = applicable_rules[0]
            response["rule_applied"] = rule.name

            for action in rule.actions:
                self._execute_action(action, context, response, user_input)

            # Learning: record rule performance
            if self.learning_enabled:
                self._record_rule_performance(rule.name, True, context)

        return response

    def _rule_matches(
        self, rule: DialogRule, context: ConversationContext, user_input: str
    ) -> bool:
        """Check if a dialog rule matches the current context."""
        conditions = rule.conditions

        for key, value in conditions.items():
            if key == "intent":
                if isinstance(value, list):
                    if context.intent not in value:
                        return False
                elif context.intent != value:
                    return False

            elif key == "state":
                if isinstance(value, list):
                    if context.current_state not in value:
                        return False
                elif context.current_state != value:
                    return False

            elif key == "modality":
                if context.modality != value:
                    return False

            elif key == "confidence_threshold":
                if context.confidence < value:
                    return False

            elif key == "inactive_minutes":
                if callable(value):
                    if not value(context):
                        return False

            elif key in context.entities:
                if context.entities[key] != value:
                    return False

            elif key in context.metadata:
                if context.metadata[key] != value:
                    return False

        return True

    def _execute_action(
        self,
        action: Dict[str, Any],
        context: ConversationContext,
        response: Dict[str, Any],
        user_input: str,
    ):
        """Execute a dialog action."""
        action_type = action.get("type")

        if action_type == "set_state":
            new_state = DialogState(action["state"])
            context.current_state = new_state
            response["state"] = new_state.value

        elif action_type == "respond":
            template = action.get("template", "")
            response["text"] = self._generate_response(template, context, user_input)

        elif action_type == "route_to_llm":
            use_rag = action.get("use_rag", False)
            llm_response = self._route_to_llm(user_input, context, use_rag)
            response["text"] = llm_response
            response["actions"].append({"type": "llm_query", "use_rag": use_rag})

        elif action_type == "route_to_action_executor":
            action_result = self._route_to_action_executor(user_input, context)
            response["text"] = action_result.get("message", "Action executed")
            response["actions"].append(
                {"type": "action_execution", "result": action_result}
            )

        elif action_type == "update_context":
            key, value = action["key"], action["value"]
            context.metadata[key] = value

        elif action_type == "request_rephrase":
            response["text"] = (
                "I didn't understand that clearly. Could you please rephrase your request?"
            )
            response["actions"].append({"type": "clarification_requested"})

        elif action_type == "process_confirmation":
            confirmed = self._process_confirmation(user_input, context)
            response["confirmation_result"] = confirmed

        elif action_type == "cleanup_session":
            self._cleanup_session(context.session_id)

    def _generate_response(
        self, template: str, context: ConversationContext, user_input: str
    ) -> str:
        """Generate response text based on template."""
        responses = {
            "greeting_response": [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?",
                "Greetings! How may I assist you?",
            ],
            "clarification_response": [
                "I need a bit more clarification on that.",
                "Could you provide more details?",
                "I'm not sure I understood correctly.",
            ],
            "confirmation_request": [
                "Are you sure you want to proceed?",
                "Should I go ahead with this?",
                "Please confirm your request.",
            ],
        }

        if template in responses:
            # Simple template selection based on user preferences
            preference = context.preferences.get("response_style", "neutral")
            options = responses[template]

            # Could implement more sophisticated selection based on context
            return options[0]

        return "I'm processing your request..."

    def _route_to_llm(
        self, user_input: str, context: ConversationContext, use_rag: bool = False
    ) -> str:
        """Route request to LLM with optional RAG augmentation."""
        try:
            # Build enhanced prompt with context
            enhanced_prompt = self._build_enhanced_prompt(user_input, context, use_rag)

            # Get LLM response
            llm_manager = provider_registry.create("llm")
            response = llm_manager.query(enhanced_prompt)

            return response

        except Exception as exc:
            logger.error("LLM routing failed: %s", exc)
            return "I'm having trouble processing your request right now."

    def _route_to_action_executor(
        self, user_input: str, context: ConversationContext
    ) -> Dict[str, Any]:
        """Route command to action executor."""
        try:
            action_executor = provider_registry.create("actions", "default")
            # Parse command from user input (simplified)
            command = self._parse_command(user_input, context)
            result = action_executor.execute(
                command["action"], command.get("params", {})
            )
            return {"success": True, "message": str(result), "command": command}
        except Exception as exc:
            logger.error("Action execution failed: %s", exc)
            return {"success": False, "message": f"Failed to execute action: {exc}"}

    def _parse_command(
        self, user_input: str, context: ConversationContext
    ) -> Dict[str, Any]:
        """Parse command from user input (simplified implementation)."""
        # This would be more sophisticated in a real implementation
        text_lower = user_input.lower()

        if "open" in text_lower and "browser" in text_lower:
            return {"action": "web_search", "params": {"query": "open browser"}}
        elif "create" in text_lower and "file" in text_lower:
            return {"action": "create_file", "params": {"filename": "new_file.txt"}}
        elif "send" in text_lower and "email" in text_lower:
            return {
                "action": "send_email",
                "params": {"to": "example@example.com", "subject": "Test"},
            }

        return {"action": "unknown", "params": {}}

    def _build_enhanced_prompt(
        self, user_input: str, context: ConversationContext, use_rag: bool = False
    ) -> str:
        """Build enhanced prompt with context and RAG if available."""
        prompt_parts = []

        # Add user context
        if context.user_profile:
            prompt_parts.append(
                f"User Profile: {json.dumps(context.user_profile, indent=2)}"
            )

        # Add conversation history (recent turns)
        recent_history = context.conversation_history[-3:]  # Last 3 turns
        if recent_history:
            history_text = "\n".join(
                [
                    f"Turn {h['turn']}: User: {h['user_input']}\nAssistant: {h.get('response', '')}"
                    for h in recent_history
                ]
            )
            prompt_parts.append(f"Recent Conversation:\n{history_text}")

        # Add RAG context if requested
        if use_rag and self.rag_pipeline:
            try:
                context_chunks = self.rag_pipeline.query(user_input, top_k=3)
                if context_chunks:
                    context_text = "\n".join(
                        [
                            f"[{i+1}] {chunk.text}"
                            for i, chunk in enumerate(context_chunks)
                        ]
                    )
                    prompt_parts.append(f"Relevant Context:\n{context_text}")
            except Exception as exc:
                logger.debug("RAG context building failed: %s", exc)

        # Add current user input
        prompt_parts.append(f"Current Request: {user_input}")

        # Add response guidelines
        prompt_parts.append(
            """
Please respond naturally and helpfully. Consider the user's preferences and conversation history.
If you need clarification, ask specific questions. Be concise but informative."""
        )

        return "\n\n".join(prompt_parts)

    def _process_confirmation(
        self, user_input: str, context: ConversationContext
    ) -> bool:
        """Process user confirmation response."""
        text_lower = user_input.lower()
        positive_words = {
            "yes",
            "sure",
            "okay",
            "ok",
            "correct",
            "right",
            "confirm",
            "yes please",
        }
        negative_words = {"no", "nope", "cancel", "stop", "wrong", "incorrect", "deny"}

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return True
        elif negative_count > positive_count:
            return False
        else:
            # Ambiguous - could ask for clarification
            return False

    def _record_intent_classification(
        self,
        user_input: str,
        intent: IntentType,
        confidence: float,
        context: ConversationContext,
    ):
        """Record intent classification for learning."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "intent": intent.value,
            "confidence": confidence,
            "context": {
                "modality": context.modality.value,
                "turn_count": context.turn_count,
                "previous_intent": (
                    context.intent.value if context.turn_count > 1 else None
                ),
            },
        }

        self.intent_history.append(record)

        # Keep only recent history
        if len(self.intent_history) > 1000:
            self.intent_history = self.intent_history[-1000:]

    def _record_rule_performance(
        self, rule_name: str, success: bool, context: ConversationContext
    ):
        """Record rule performance for learning."""
        if rule_name not in self.rule_performance:
            self.rule_performance[rule_name] = {
                "total_uses": 0,
                "successful_uses": 0,
                "avg_confidence": 0.0,
                "context_patterns": [],
            }

        perf = self.rule_performance[rule_name]
        perf["total_uses"] += 1
        if success:
            perf["successful_uses"] += 1

        # Update average confidence
        perf["avg_confidence"] = (
            (perf["avg_confidence"] * (perf["total_uses"] - 1)) + context.confidence
        ) / perf["total_uses"]

    def end_conversation(self, session_id: str):
        """End a conversation session."""
        with self.session_lock:
            if session_id in self.active_sessions:
                context = self.active_sessions[session_id]
                context.current_state = DialogState.COMPLETED
                logger.info("Ended conversation session %s", session_id)

    def _cleanup_session(self, session_id: str):
        """Clean up expired or completed session."""
        with self.session_lock:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                logger.debug("Cleaned up session %s", session_id)

    def get_session_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation context for a session."""
        with self.session_lock:
            return self.active_sessions.get(session_id)

    def get_active_sessions(self) -> Dict[str, ConversationContext]:
        """Get all active conversation sessions."""
        with self.session_lock:
            return dict(self.active_sessions)

    def load_rules_from_file(self, file_path: str):
        """Load dialog rules from JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                rules_data = json.load(f)

            for rule_data in rules_data.get("rules", []):
                rule = DialogRule(
                    name=rule_data["name"],
                    conditions=rule_data["conditions"],
                    actions=rule_data["actions"],
                    priority=rule_data.get("priority", 0),
                    enabled=rule_data.get("enabled", True),
                    metadata=rule_data.get("metadata", {}),
                )
                self.dialog_rules.append(rule)

            logger.info("Loaded %d rules from %s", len(self.dialog_rules), file_path)

        except Exception as exc:
            logger.error("Failed to load rules from %s: %s", file_path, exc)

    def save_rules_to_file(self, file_path: str):
        """Save dialog rules to JSON file."""
        try:
            rules_data = {
                "rules": [
                    {
                        "name": rule.name,
                        "conditions": rule.conditions,
                        "actions": rule.actions,
                        "priority": rule.priority,
                        "enabled": rule.enabled,
                        "metadata": rule.metadata,
                    }
                    for rule in self.dialog_rules
                ]
            }

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(rules_data, f, indent=2, ensure_ascii=False)

            logger.info("Saved %d rules to %s", len(self.dialog_rules), file_path)

        except Exception as exc:
            logger.error("Failed to save rules to %s: %s", file_path, exc)

    def get_statistics(self) -> Dict[str, Any]:
        """Get dialog manager statistics."""
        with self.session_lock:
            active_count = len(self.active_sessions)

        return {
            "active_sessions": active_count,
            "total_rules": len(self.dialog_rules),
            "intent_patterns": len(self.intent_patterns),
            "intent_history_size": len(self.intent_history),
            "rule_performance": dict(self.rule_performance),
        }


# Register with provider registry
def create_dialog_manager(config_manager=None, **kwargs):
    """Factory function for DialogManager."""
    return DialogManager(config_manager=config_manager, **kwargs)


provider_registry.register_lazy(
    "dialog",
    "manager",
    "mia.adaptive_intelligence.dialog_manager",
    "create_dialog_manager",
    default=True,
)
