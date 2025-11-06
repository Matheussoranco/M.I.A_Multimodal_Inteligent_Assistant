import asyncio
import hashlib
import json
import logging
import statistics
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from ..providers import provider_registry

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of specialized models."""

    REASONING = "reasoning"
    CODE = "code"
    VISION = "vision"
    CREATIVE = "creative"
    CONVERSATIONAL = "conversational"
    MATHEMATICAL = "mathematical"
    SCIENTIFIC = "scientific"
    GENERAL = "general"


class RoutingStrategy(Enum):
    """Routing strategies for model selection."""

    PERFORMANCE = "performance"
    COST = "cost"
    LATENCY = "latency"
    ACCURACY = "accuracy"
    BALANCED = "balanced"
    SPECIALIST = "specialist"
    FALLBACK = "fallback"


class ModelCapability(Enum):
    """Model capabilities."""

    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    IMAGE_ANALYSIS = "image_analysis"
    MATHEMATICAL = "mathematical"
    SCIENTIFIC = "scientific"
    CREATIVE_WRITING = "creative_writing"
    CONVERSATION = "conversation"
    REASONING = "reasoning"
    ANALYSIS = "analysis"


@dataclass
class ModelSpec:
    """Specification for a model."""

    id: str
    name: str
    provider: str
    model_type: ModelType
    capabilities: Set[ModelCapability] = field(default_factory=set)
    context_window: int = 4096
    max_tokens: int = 1024
    cost_per_token: float = 0.0
    avg_latency_ms: float = 1000
    accuracy_score: float = 0.8
    is_available: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class RoutingDecision:
    """Decision made by the router."""

    request_id: str
    selected_model: str
    strategy: RoutingStrategy
    confidence: float
    reasoning: str
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RequestContext:
    """Context for a request."""

    id: str
    content: str
    modality: str = "text"
    domain: Optional[str] = None
    complexity: float = 0.5
    urgency: str = "normal"
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPerformance:
    """Performance metrics for a model."""

    model_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    avg_cost: float = 0.0
    avg_accuracy: float = 0.0
    last_used: Optional[datetime] = None
    error_rate: float = 0.0
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class OrchestrationResult:
    """Result of orchestration."""

    request_id: str
    model_used: str
    response: Any
    latency_ms: float
    cost: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class ModelRegistry:
    """
    Registry for managing specialized models.

    Maintains model specifications, capabilities, and availability status.
    """

    def __init__(self):
        self.models: Dict[str, ModelSpec] = {}
        self.capability_index: Dict[ModelCapability, Set[str]] = {}
        self.type_index: Dict[ModelType, Set[str]] = {}
        self._initialize_builtin_models()

    def _initialize_builtin_models(self):
        """Initialize built-in model specifications."""
        # Reasoning models
        self.register_model(
            ModelSpec(
                id="reasoning_gpt4",
                name="GPT-4 Reasoning",
                provider="openai",
                model_type=ModelType.REASONING,
                capabilities={
                    ModelCapability.REASONING,
                    ModelCapability.ANALYSIS,
                    ModelCapability.TEXT_GENERATION,
                },
                context_window=8192,
                max_tokens=4096,
                cost_per_token=0.03,
                avg_latency_ms=2000,
                accuracy_score=0.95,
            )
        )

        # Code models
        self.register_model(
            ModelSpec(
                id="code_gpt4",
                name="GPT-4 Code",
                provider="openai",
                model_type=ModelType.CODE,
                capabilities={
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.CODE_REVIEW,
                    ModelCapability.TEXT_GENERATION,
                },
                context_window=8192,
                max_tokens=4096,
                cost_per_token=0.03,
                avg_latency_ms=1500,
                accuracy_score=0.92,
            )
        )

        # Vision models
        self.register_model(
            ModelSpec(
                id="vision_gpt4v",
                name="GPT-4 Vision",
                provider="openai",
                model_type=ModelType.VISION,
                capabilities={
                    ModelCapability.IMAGE_ANALYSIS,
                    ModelCapability.TEXT_GENERATION,
                },
                context_window=8192,
                max_tokens=4096,
                cost_per_token=0.03,
                avg_latency_ms=3000,
                accuracy_score=0.88,
            )
        )

        # Mathematical models
        self.register_model(
            ModelSpec(
                id="math_specialist",
                name="Math Specialist",
                provider="anthropic",
                model_type=ModelType.MATHEMATICAL,
                capabilities={ModelCapability.MATHEMATICAL, ModelCapability.ANALYSIS},
                context_window=4096,
                max_tokens=2048,
                cost_per_token=0.04,
                avg_latency_ms=2500,
                accuracy_score=0.96,
            )
        )

        # General purpose fallback
        self.register_model(
            ModelSpec(
                id="general_gpt3",
                name="GPT-3.5 General",
                provider="openai",
                model_type=ModelType.GENERAL,
                capabilities={
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CONVERSATION,
                },
                context_window=4096,
                max_tokens=2048,
                cost_per_token=0.002,
                avg_latency_ms=800,
                accuracy_score=0.85,
            )
        )

    def register_model(self, model_spec: ModelSpec):
        """Register a new model."""
        self.models[model_spec.id] = model_spec

        # Update capability index
        for capability in model_spec.capabilities:
            if capability not in self.capability_index:
                self.capability_index[capability] = set()
            self.capability_index[capability].add(model_spec.id)

        # Update type index
        if model_spec.model_type not in self.type_index:
            self.type_index[model_spec.model_type] = set()
        self.type_index[model_spec.model_type].add(model_spec.id)

        logger.info(f"Registered model: {model_spec.name} ({model_spec.id})")

    def unregister_model(self, model_id: str):
        """Unregister a model."""
        if model_id in self.models:
            model = self.models[model_id]

            # Remove from capability index
            for capability in model.capabilities:
                if capability in self.capability_index:
                    self.capability_index[capability].discard(model_id)

            # Remove from type index
            if model.model_type in self.type_index:
                self.type_index[model.model_type].discard(model_id)

            del self.models[model_id]
            logger.info(f"Unregistered model: {model_id}")

    def get_model(self, model_id: str) -> Optional[ModelSpec]:
        """Get model specification by ID."""
        return self.models.get(model_id)

    def find_models_by_capability(self, capability: ModelCapability) -> List[ModelSpec]:
        """Find models that have a specific capability."""
        model_ids = self.capability_index.get(capability, set())
        return [self.models[mid] for mid in model_ids if mid in self.models]

    def find_models_by_type(self, model_type: ModelType) -> List[ModelSpec]:
        """Find models of a specific type."""
        model_ids = self.type_index.get(model_type, set())
        return [self.models[mid] for mid in model_ids if mid in self.models]

    def get_available_models(self) -> List[ModelSpec]:
        """Get all available models."""
        return [model for model in self.models.values() if model.is_available]

    def update_model_status(self, model_id: str, available: bool):
        """Update model availability status."""
        if model_id in self.models:
            self.models[model_id].is_available = available
            logger.info(f"Updated model {model_id} availability: {available}")


class AdaptiveRouter:
    """
    Intelligent router that selects the best model for each request.

    Uses multiple strategies and continuous learning to optimize routing decisions.
    """

    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.routing_history: List[RoutingDecision] = []
        self.performance_tracker = ModelPerformanceTracker()
        self.routing_weights = self._initialize_routing_weights()

    def _initialize_routing_weights(self) -> Dict[RoutingStrategy, float]:
        """Initialize routing strategy weights."""
        return {
            RoutingStrategy.PERFORMANCE: 0.3,
            RoutingStrategy.COST: 0.2,
            RoutingStrategy.LATENCY: 0.2,
            RoutingStrategy.ACCURACY: 0.2,
            RoutingStrategy.BALANCED: 0.1,
            RoutingStrategy.SPECIALIST: 0.0,  # Dynamic
            RoutingStrategy.FALLBACK: 0.0,  # Only when needed
        }

    def route_request(self, context: RequestContext) -> RoutingDecision:
        """
        Route a request to the best available model.

        Analyzes request context and selects optimal model using multiple strategies.
        """
        request_id = context.id

        # Analyze request requirements
        required_capabilities = self._analyze_capabilities(context)
        strategy = self._select_strategy(context, required_capabilities)

        # Find candidate models
        candidates = self._find_candidate_models(required_capabilities, context)

        if not candidates:
            # Fallback to general models
            candidates = self.model_registry.find_models_by_type(ModelType.GENERAL)
            strategy = RoutingStrategy.FALLBACK

        # Score and rank candidates
        scored_candidates = self._score_candidates(candidates, context, strategy)

        # Select best model
        if scored_candidates:
            selected_model, confidence, reasoning = scored_candidates[0]
            alternatives = [(mid, score) for mid, score, _ in scored_candidates[1:]]
        else:
            # Ultimate fallback
            general_models = self.model_registry.find_models_by_type(ModelType.GENERAL)
            if general_models:
                selected_model = general_models[0].id
                confidence = 0.1
                reasoning = (
                    "Fallback to general model - no suitable specialists available"
                )
            else:
                raise ValueError("No models available for routing")
            alternatives = []

        decision = RoutingDecision(
            request_id=request_id,
            selected_model=selected_model,
            strategy=strategy,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=alternatives,
        )

        self.routing_history.append(decision)
        self._update_routing_weights(decision)

        return decision

    def _analyze_capabilities(self, context: RequestContext) -> Set[ModelCapability]:
        """Analyze what capabilities are needed for the request."""
        capabilities = set()

        content_lower = context.content.lower()

        # Code-related capabilities
        if any(
            keyword in content_lower
            for keyword in ["code", "function", "class", "import", "def ", "print("]
        ):
            capabilities.add(ModelCapability.CODE_GENERATION)
            if "review" in content_lower or "bug" in content_lower:
                capabilities.add(ModelCapability.CODE_REVIEW)

        # Mathematical capabilities
        if any(
            keyword in content_lower
            for keyword in ["calculate", "solve", "equation", "math", "formula"]
        ):
            capabilities.add(ModelCapability.MATHEMATICAL)

        # Vision capabilities
        if (
            context.modality == "image"
            or "image" in content_lower
            or "picture" in content_lower
        ):
            capabilities.add(ModelCapability.IMAGE_ANALYSIS)

        # Scientific capabilities
        if any(
            keyword in content_lower
            for keyword in ["research", "scientific", "hypothesis", "experiment"]
        ):
            capabilities.add(ModelCapability.SCIENTIFIC)

        # Creative capabilities
        if any(
            keyword in content_lower
            for keyword in ["write", "story", "poem", "creative", "design"]
        ):
            capabilities.add(ModelCapability.CREATIVE_WRITING)

        # Reasoning capabilities (default for complex requests)
        if context.complexity > 0.7 or len(context.content) > 1000:
            capabilities.add(ModelCapability.REASONING)
            capabilities.add(ModelCapability.ANALYSIS)

        # Always include text generation as baseline
        capabilities.add(ModelCapability.TEXT_GENERATION)

        return capabilities

    def _select_strategy(
        self, context: RequestContext, capabilities: Set[ModelCapability]
    ) -> RoutingStrategy:
        """Select the best routing strategy for the request."""
        # Specialist strategy for specific capabilities
        specialist_caps = {
            ModelCapability.CODE_GENERATION,
            ModelCapability.MATHEMATICAL,
            ModelCapability.IMAGE_ANALYSIS,
            ModelCapability.SCIENTIFIC,
        }

        if capabilities.intersection(specialist_caps):
            return RoutingStrategy.SPECIALIST

        # Performance for urgent requests
        if context.urgency == "high":
            return RoutingStrategy.PERFORMANCE

        # Cost optimization for simple requests
        if context.complexity < 0.3:
            return RoutingStrategy.COST

        # Balanced for most cases
        return RoutingStrategy.BALANCED

    def _find_candidate_models(
        self, capabilities: Set[ModelCapability], context: RequestContext
    ) -> List[ModelSpec]:
        """Find models that match the required capabilities."""
        candidates = set()

        for capability in capabilities:
            models = self.model_registry.find_models_by_capability(capability)
            candidates.update(models)

        # Filter available models
        candidates = [m for m in candidates if m.is_available]

        # Prioritize based on context
        if context.domain:
            # Domain-specific prioritization could be implemented here
            pass

        return candidates

    def _score_candidates(
        self,
        candidates: List[ModelSpec],
        context: RequestContext,
        strategy: RoutingStrategy,
    ) -> List[Tuple[str, float, str]]:
        """Score and rank candidate models."""
        scored = []

        for model in candidates:
            score, reasoning = self._calculate_model_score(model, context, strategy)
            scored.append((model.id, score, reasoning))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _calculate_model_score(
        self, model: ModelSpec, context: RequestContext, strategy: RoutingStrategy
    ) -> Tuple[float, str]:
        """Calculate score for a model based on strategy."""
        base_score = 0.5
        reasoning_parts = []

        # Performance-based scoring
        perf = self.performance_tracker.get_performance(model.id)
        if perf:
            if strategy == RoutingStrategy.PERFORMANCE:
                # Weight by success rate and inverse latency
                success_weight = perf.successful_requests / max(perf.total_requests, 1)
                latency_score = 1.0 / (1.0 + perf.avg_latency_ms / 1000.0)  # Normalize
                base_score = (success_weight * 0.7) + (latency_score * 0.3)
                reasoning_parts.append(
                    f"Performance: {success_weight:.2f} success, {latency_score:.2f} latency"
                )

            elif strategy == RoutingStrategy.COST:
                cost_score = 1.0 / (1.0 + perf.avg_cost)  # Lower cost is better
                base_score = cost_score
                reasoning_parts.append(f"Cost: ${perf.avg_cost:.4f}/token")

            elif strategy == RoutingStrategy.LATENCY:
                latency_score = 1.0 / (1.0 + perf.avg_latency_ms / 1000.0)
                base_score = latency_score
                reasoning_parts.append(f"Latency: {perf.avg_latency_ms:.0f}ms")

            elif strategy == RoutingStrategy.ACCURACY:
                base_score = perf.avg_accuracy
                reasoning_parts.append(f"Accuracy: {perf.avg_accuracy:.2f}")

        # Specialist bonus
        if strategy == RoutingStrategy.SPECIALIST:
            if model.model_type != ModelType.GENERAL:
                base_score += 0.3
                reasoning_parts.append("Specialist model bonus")

        # Capability matching
        required_caps = self._analyze_capabilities(context)
        matched_caps = len(required_caps.intersection(model.capabilities))
        capability_score = matched_caps / len(required_caps) if required_caps else 0.5
        base_score = (base_score * 0.7) + (capability_score * 0.3)
        reasoning_parts.append(
            f"Capabilities: {matched_caps}/{len(required_caps)} matched"
        )

        # Availability bonus
        if model.is_available:
            base_score += 0.1
            reasoning_parts.append("Available")

        reasoning = "; ".join(reasoning_parts)
        return min(base_score, 1.0), reasoning

    def _update_routing_weights(self, decision: RoutingDecision):
        """Update routing weights based on decision outcomes."""
        # This would be updated based on actual performance feedback
        # For now, maintain static weights
        pass

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        if not self.routing_history:
            return {"total_decisions": 0}

        recent_decisions = self.routing_history[-100:]  # Last 100 decisions

        strategy_counts = {}
        for decision in recent_decisions:
            strategy_counts[decision.strategy.value] = (
                strategy_counts.get(decision.strategy.value, 0) + 1
            )

        avg_confidence = statistics.mean(d.confidence for d in recent_decisions)

        return {
            "total_decisions": len(self.routing_history),
            "recent_decisions": len(recent_decisions),
            "strategy_distribution": strategy_counts,
            "avg_confidence": avg_confidence,
        }


class ModelPerformanceTracker:
    """
    Tracks and analyzes model performance metrics.

    Maintains historical performance data and provides insights.
    """

    def __init__(self):
        self.performance: Dict[str, ModelPerformance] = {}
        self._lock = threading.Lock()

    def record_request(self, model_id: str, result: OrchestrationResult):
        """Record a request result."""
        with self._lock:
            if model_id not in self.performance:
                self.performance[model_id] = ModelPerformance(model_id=model_id)

            perf = self.performance[model_id]
            perf.total_requests += 1
            perf.last_used = result.timestamp

            if result.response is not None:  # Assuming success if response exists
                perf.successful_requests += 1
            else:
                perf.failed_requests += 1

            # Update rolling averages
            perf.avg_latency_ms = self._update_average(
                perf.avg_latency_ms, result.latency_ms, perf.total_requests
            )
            perf.avg_cost = self._update_average(
                perf.avg_cost, result.cost, perf.total_requests
            )

            # Estimate accuracy (would be provided by evaluation)
            perf.avg_accuracy = self._update_average(
                perf.avg_accuracy, result.confidence, perf.total_requests
            )

            perf.error_rate = perf.failed_requests / perf.total_requests

            # Keep metrics history
            perf.metrics_history.append(
                {
                    "timestamp": result.timestamp.isoformat(),
                    "latency_ms": result.latency_ms,
                    "cost": result.cost,
                    "confidence": result.confidence,
                    "success": result.response is not None,
                }
            )

            # Limit history size
            if len(perf.metrics_history) > 1000:
                perf.metrics_history = perf.metrics_history[-1000:]

    def get_performance(self, model_id: str) -> Optional[ModelPerformance]:
        """Get performance metrics for a model."""
        return self.performance.get(model_id)

    def get_top_models(
        self, metric: str = "success_rate", limit: int = 5
    ) -> List[Tuple[str, float]]:
        """Get top performing models by metric."""
        rankings = []

        for model_id, perf in self.performance.items():
            if perf.total_requests < 10:  # Minimum requests for ranking
                continue

            if metric == "success_rate":
                score = perf.successful_requests / perf.total_requests
            elif metric == "latency":
                score = 1.0 / (
                    1.0 + perf.avg_latency_ms / 1000.0
                )  # Lower latency is better
            elif metric == "cost":
                score = 1.0 / (1.0 + perf.avg_cost)  # Lower cost is better
            elif metric == "accuracy":
                score = perf.avg_accuracy
            else:
                continue

            rankings.append((model_id, score))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings[:limit]

    def _update_average(
        self, current_avg: float, new_value: float, count: int
    ) -> float:
        """Update rolling average."""
        return ((current_avg * (count - 1)) + new_value) / count

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        if not self.performance:
            return {"total_models": 0}

        total_requests = sum(p.total_requests for p in self.performance.values())
        total_successful = sum(p.successful_requests for p in self.performance.values())

        return {
            "total_models": len(self.performance),
            "total_requests": total_requests,
            "overall_success_rate": (
                total_successful / total_requests if total_requests > 0 else 0
            ),
            "avg_latency_ms": statistics.mean(
                p.avg_latency_ms
                for p in self.performance.values()
                if p.total_requests > 0
            ),
            "avg_cost": statistics.mean(
                p.avg_cost for p in self.performance.values() if p.total_requests > 0
            ),
        }


class SpecialistWrapper:
    """
    Wrapper for specialized model interactions.

    Provides unified interface for different model types and handles
    model-specific optimizations and post-processing.
    """

    def __init__(self, model_spec: ModelSpec, provider_client=None):
        self.model_spec = model_spec
        self.provider_client = provider_client or self._get_provider_client()
        self.call_count = 0
        self.total_latency = 0.0

    def _get_provider_client(self):
        """Get appropriate provider client."""
        # This would integrate with actual provider clients
        # For now, return a mock client
        return MockProviderClient(self.model_spec.provider)

    async def execute(self, request: RequestContext) -> OrchestrationResult:
        """Execute request using the specialized model."""
        start_time = time.time()

        try:
            # Pre-process request for this model type
            processed_request = self._preprocess_request(request)

            # Execute with provider
            raw_response = await self.provider_client.call_model(
                model_id=self.model_spec.id,
                request=processed_request,
                max_tokens=self.model_spec.max_tokens,
            )

            # Post-process response
            processed_response = self._postprocess_response(raw_response, request)

            latency_ms = (time.time() - start_time) * 1000

            # Estimate cost
            estimated_cost = self._estimate_cost(processed_request, processed_response)

            result = OrchestrationResult(
                request_id=request.id,
                model_used=self.model_spec.id,
                response=processed_response,
                latency_ms=latency_ms,
                cost=estimated_cost,
                confidence=self._estimate_confidence(processed_response),
                metadata={
                    "model_type": self.model_spec.model_type.value,
                    "capabilities_used": [
                        cap.value for cap in self.model_spec.capabilities
                    ],
                    "tokens_used": self._estimate_tokens(
                        processed_request, processed_response
                    ),
                },
            )

            # Update statistics
            self.call_count += 1
            self.total_latency += latency_ms

            return result

        except Exception as exc:
            latency_ms = (time.time() - start_time) * 1000
            return OrchestrationResult(
                request_id=request.id,
                model_used=self.model_spec.id,
                response=None,
                latency_ms=latency_ms,
                cost=0.0,
                confidence=0.0,
                metadata={"error": str(exc)},
            )

    def _preprocess_request(self, request: RequestContext) -> Dict[str, Any]:
        """Pre-process request for this model type."""
        processed = {"content": request.content, "context": request.metadata}

        # Model-specific preprocessing
        if self.model_spec.model_type == ModelType.CODE:
            processed["content"] = f"```code\n{request.content}\n```"
            processed["temperature"] = 0.1  # Lower temperature for code

        elif self.model_spec.model_type == ModelType.MATHEMATICAL:
            processed["content"] = (
                f"Please solve this mathematical problem step by step:\n{request.content}"
            )
            processed["temperature"] = 0.0  # Deterministic for math

        elif self.model_spec.model_type == ModelType.VISION:
            if request.modality == "image":
                processed["content"] = f"Analyze this image: {request.content}"
                processed["image_data"] = request.metadata.get("image_data")

        elif self.model_spec.model_type == ModelType.REASONING:
            processed["content"] = (
                f"Analyze and reason about this request step by step:\n{request.content}"
            )
            processed["temperature"] = 0.3

        return processed

    def _postprocess_response(self, raw_response: Any, request: RequestContext) -> Any:
        """Post-process response from the model."""
        if isinstance(raw_response, str):
            response = raw_response.strip()
        else:
            response = raw_response

        # Model-specific postprocessing
        if self.model_spec.model_type == ModelType.CODE:
            # Ensure code is properly formatted
            if not response.startswith("```"):
                response = f"```\n{response}\n```"

        elif self.model_spec.model_type == ModelType.MATHEMATICAL:
            # Add verification note
            response += "\n\n**Note:** Please verify calculations independently."

        elif self.model_spec.model_type == ModelType.VISION:
            # Add confidence indicators
            response += "\n\n*Analysis based on visual input processing.*"

        return response

    def _estimate_cost(self, request: Dict, response: Any) -> float:
        """Estimate cost of the request."""
        # Rough estimation based on tokens
        request_tokens = (
            len(str(request.get("content", ""))) / 4
        )  # Rough token estimation
        response_tokens = len(str(response)) / 4

        total_tokens = request_tokens + response_tokens
        return total_tokens * self.model_spec.cost_per_token

    def _estimate_confidence(self, response: Any) -> float:
        """Estimate confidence in the response."""
        # Simple heuristic based on response characteristics
        response_str = str(response).strip()

        if len(response_str) < 10:
            return 0.3  # Too short
        elif len(response_str) > 1000:
            return 0.9  # Comprehensive response
        else:
            return 0.7  # Standard response

    def _estimate_tokens(self, request: Dict, response: Any) -> int:
        """Estimate total tokens used."""
        request_tokens = len(str(request.get("content", ""))) / 4
        response_tokens = len(str(response)) / 4
        return int(request_tokens + response_tokens)

    def get_statistics(self) -> Dict[str, Any]:
        """Get wrapper statistics."""
        return {
            "model_id": self.model_spec.id,
            "call_count": self.call_count,
            "avg_latency_ms": self.total_latency / max(self.call_count, 1),
            "model_type": self.model_spec.model_type.value,
        }


class MockProviderClient:
    """Mock provider client for demonstration."""

    def __init__(self, provider: str):
        self.provider = provider

    async def call_model(
        self, model_id: str, request: Dict[str, Any], max_tokens: int
    ) -> str:
        """Mock model call."""
        await asyncio.sleep(0.1)  # Simulate network delay

        content = request.get("content", "")

        if "code" in model_id:
            return f"```python\n# Generated code for: {content[:50]}...\ndef solution():\n    return 'Hello World'\n```"
        elif "math" in model_id:
            return f"Mathematical solution for: {content[:50]}...\nStep 1: Analyze the problem\nStep 2: Apply formula\nResult: 42"
        elif "vision" in model_id:
            return f"Image analysis: {content[:50]}...\nDetected objects: person, computer, desk\nScene: office environment"
        elif "reasoning" in model_id:
            return f"Reasoning analysis: {content[:50]}...\nKey insights:\n1. Problem identification\n2. Solution approach\n3. Implementation steps"
        else:
            return f"General response to: {content[:50]}...\nThis is a helpful response from {model_id}"


class EvaluationLoop:
    """
    Continuous evaluation and improvement loop.

    Monitors performance, collects feedback, and refines routing decisions.
    """

    def __init__(
        self, router: AdaptiveRouter, performance_tracker: ModelPerformanceTracker
    ):
        self.router = router
        self.performance_tracker = performance_tracker
        self.evaluation_interval = 300  # 5 minutes
        self.feedback_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        self.evaluation_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the evaluation loop."""
        if self.is_running:
            return

        self.is_running = True
        self.evaluation_task = asyncio.create_task(self._evaluation_loop())
        logger.info("Started evaluation loop")

    async def stop(self):
        """Stop the evaluation loop."""
        self.is_running = False
        if self.evaluation_task:
            self.evaluation_task.cancel()
            try:
                await self.evaluation_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped evaluation loop")

    async def submit_feedback(self, request_id: str, rating: float, comments: str = ""):
        """Submit user feedback for evaluation."""
        await self.feedback_queue.put(
            {
                "request_id": request_id,
                "rating": rating,
                "comments": comments,
                "timestamp": datetime.now(),
            }
        )

    async def _evaluation_loop(self):
        """Main evaluation loop."""
        while self.is_running:
            try:
                # Wait for evaluation interval
                await asyncio.sleep(self.evaluation_interval)

                # Process accumulated feedback
                await self._process_feedback()

                # Analyze performance trends
                await self._analyze_performance()

                # Update routing strategies
                await self._update_routing_strategies()

                logger.debug("Completed evaluation cycle")

            except Exception as exc:
                logger.error(f"Evaluation loop error: {exc}")

    async def _process_feedback(self):
        """Process user feedback."""
        feedback_items = []

        # Collect all pending feedback
        while not self.feedback_queue.empty():
            try:
                feedback = self.feedback_queue.get_nowait()
                feedback_items.append(feedback)
            except asyncio.QueueEmpty:
                break

        if not feedback_items:
            return

        # Analyze feedback patterns
        avg_rating = statistics.mean(f["rating"] for f in feedback_items)

        # Update model preferences based on feedback
        # This is a simplified implementation
        logger.info(
            f"Processed {len(feedback_items)} feedback items, avg rating: {avg_rating:.2f}"
        )

    async def _analyze_performance(self):
        """Analyze performance trends."""
        summary = self.performance_tracker.get_performance_summary()

        # Identify underperforming models
        underperformers = []
        for model_id, perf in self.performance_tracker.performance.items():
            if perf.total_requests > 50:  # Minimum sample size
                if perf.error_rate > 0.1:  # High error rate
                    underperformers.append(model_id)

        if underperformers:
            logger.warning(f"Underperforming models detected: {underperformers}")

        # Log performance insights
        logger.info(f"Performance summary: {summary}")

    async def _update_routing_strategies(self):
        """Update routing strategies based on evaluation."""
        # Get top performing models
        top_latency = self.performance_tracker.get_top_models("latency", 3)
        top_accuracy = self.performance_tracker.get_top_models("accuracy", 3)

        # Adjust routing weights based on performance
        # This would implement more sophisticated learning

        logger.debug("Updated routing strategies based on evaluation")


class HybridLLMOrchestrator:
    """
    Hybrid LLM orchestrator.
    """

    def __init__(self, config_manager=None):
        self.config_manager = config_manager

        # Core components
        self.model_registry = ModelRegistry()
        self.router = AdaptiveRouter(self.model_registry)
        self.performance_tracker = self.router.performance_tracker
        self.evaluation_loop = EvaluationLoop(self.router, self.performance_tracker)

        # Model wrappers
        self.wrappers: Dict[str, SpecialistWrapper] = {}
        self._initialize_wrappers()

        # Request processing
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.processing_tasks: Set[asyncio.Task] = set()
        self.max_concurrent_requests = 10
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        # Telemetry
        self.telemetry_enabled = True
        self.request_history: List[OrchestrationResult] = []

        logger.info("Hybrid LLM Orchestrator initialized")

    def _initialize_wrappers(self):
        """Initialize model wrappers."""
        for model_spec in self.model_registry.models.values():
            self.wrappers[model_spec.id] = SpecialistWrapper(model_spec)

    async def start(self):
        """Start the orchestrator."""
        await self.evaluation_loop.start()
        logger.info("Hybrid LLM Orchestrator started")

    async def stop(self):
        """Stop the orchestrator."""
        await self.evaluation_loop.stop()

        # Cancel all processing tasks
        for task in self.processing_tasks:
            task.cancel()

        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        logger.info("Hybrid LLM Orchestrator stopped")

    async def process_request(self, content: str, **kwargs) -> OrchestrationResult:
        """
        Process a request through the hybrid orchestration system.

        Args:
            content: Request content
            **kwargs: Additional request parameters

        Returns:
            Orchestration result
        """
        async with self.semaphore:
            # Create request context
            request_id = str(uuid.uuid4())
            context = RequestContext(
                id=request_id,
                content=content,
                modality=kwargs.get("modality", "text"),
                domain=kwargs.get("domain"),
                complexity=kwargs.get("complexity", 0.5),
                urgency=kwargs.get("urgency", "normal"),
                user_id=kwargs.get("user_id"),
                session_id=kwargs.get("session_id"),
                metadata=kwargs.get("metadata", {}),
            )

            try:
                # Route request
                routing_decision = self.router.route_request(context)

                # Get model wrapper
                wrapper = self.wrappers.get(routing_decision.selected_model)
                if not wrapper:
                    raise ValueError(
                        f"No wrapper available for model {routing_decision.selected_model}"
                    )

                # Execute request
                result = await wrapper.execute(context)

                # Record performance
                self.performance_tracker.record_request(
                    routing_decision.selected_model, result
                )

                # Store result
                self.request_history.append(result)
                if len(self.request_history) > 10000:  # Limit history
                    self.request_history = self.request_history[-10000:]

                return result

            except Exception as exc:
                logger.error(f"Request processing failed: {exc}")
                return OrchestrationResult(
                    request_id=request_id,
                    model_used="error",
                    response=None,
                    latency_ms=0.0,
                    cost=0.0,
                    confidence=0.0,
                    metadata={"error": str(exc)},
                )

    async def submit_feedback(self, request_id: str, rating: float, comments: str = ""):
        """Submit feedback for a request."""
        await self.evaluation_loop.submit_feedback(request_id, rating, comments)

    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        routing_stats = self.router.get_routing_stats()
        performance_summary = self.performance_tracker.get_performance_summary()

        return {
            "routing": routing_stats,
            "performance": performance_summary,
            "models": {
                "total": len(self.model_registry.models),
                "available": len(self.model_registry.get_available_models()),
            },
            "requests": {
                "total_processed": len(self.request_history),
                "active": len(self.processing_tasks),
            },
            "evaluation": {"running": self.evaluation_loop.is_running},
        }

    def get_model_stats(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific model."""
        perf = self.performance_tracker.get_performance(model_id)
        wrapper = self.wrappers.get(model_id)

        if not perf and not wrapper:
            return None

        stats = {}

        if perf:
            stats.update(
                {
                    "performance": {
                        "total_requests": perf.total_requests,
                        "success_rate": perf.successful_requests
                        / max(perf.total_requests, 1),
                        "avg_latency_ms": perf.avg_latency_ms,
                        "avg_cost": perf.avg_cost,
                        "avg_accuracy": perf.avg_accuracy,
                        "error_rate": perf.error_rate,
                    }
                }
            )

        if wrapper:
            stats.update({"wrapper": wrapper.get_statistics()})

        return stats

    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models with their specs."""
        return [
            {
                "id": model.id,
                "name": model.name,
                "type": model.model_type.value,
                "capabilities": [cap.value for cap in model.capabilities],
                "context_window": model.context_window,
                "max_tokens": model.max_tokens,
                "cost_per_token": model.cost_per_token,
                "avg_latency_ms": model.avg_latency_ms,
                "accuracy_score": model.accuracy_score,
                "available": model.is_available,
            }
            for model in self.model_registry.models.values()
        ]


# Register with provider registry
def create_hybrid_llm_orchestrator(config_manager=None, **kwargs):
    """Factory function for HybridLLMOrchestrator."""
    return HybridLLMOrchestrator(config_manager=config_manager)


provider_registry.register_lazy(
    "adaptive_intelligence",
    "llm_orchestrator",
    "mia.adaptive_intelligence.hybrid_llm_orchestration",
    "create_hybrid_llm_orchestrator",
    default=True,
)
