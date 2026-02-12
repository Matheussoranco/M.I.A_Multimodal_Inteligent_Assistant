"""
Guardrails — Structured output validation & safety for M.I.A
=============================================================

Provides output validation, content safety checks, and structured
response formatting so the agent's answers are reliable and safe.

Key features:
- **Output schemas**: Define expected response structure (JSON schemas)
  and auto-retry if the LLM output doesn't match.
- **Content filters**: Block or flag harmful, off-topic, or hallucinated
  content before it reaches the user.
- **Tool-call validation**: Verify tool arguments match expected types
  and value ranges before execution.
- **Rate limiting**: Prevent runaway tool loops and token abuse.
- **PII redaction**: Detect and mask sensitive data in outputs.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ── Severity levels ─────────────────────────────────────────────────────────


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCK = "block"


@dataclass
class GuardrailViolation:
    """A detected guardrail violation."""

    rule: str
    message: str
    severity: Severity
    context: Optional[str] = None

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.rule}: {self.message}"


# ── Output Schema Validation ───────────────────────────────────────────────


@dataclass
class FieldSpec:
    """Specification for a single field in a structured output."""

    name: str
    field_type: str  # "str", "int", "float", "bool", "list", "dict"
    required: bool = True
    description: str = ""
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    allowed_values: Optional[List[Any]] = None


@dataclass
class OutputSchema:
    """A schema that the LLM's response must conform to."""

    name: str
    fields: List[FieldSpec] = field(default_factory=list)
    description: str = ""

    def to_prompt_instruction(self) -> str:
        """Convert schema to an LLM-readable instruction."""
        lines = [
            f"You MUST respond with a JSON object matching this schema:",
            f"Schema: {self.name}",
        ]
        if self.description:
            lines.append(f"Description: {self.description}")
        lines.append("Fields:")
        for f in self.fields:
            req = "required" if f.required else "optional"
            desc = f" — {f.description}" if f.description else ""
            constraint = ""
            if f.allowed_values:
                constraint = f" (one of: {f.allowed_values})"
            if f.min_length is not None:
                constraint += f" (min length: {f.min_length})"
            if f.max_length is not None:
                constraint += f" (max length: {f.max_length})"
            lines.append(
                f'  - "{f.name}" ({f.field_type}, {req}){desc}{constraint}'
            )
        lines.append("\nRespond with ONLY valid JSON. No markdown fences.")
        return "\n".join(lines)

    def validate(self, data: Any) -> List[GuardrailViolation]:
        """Validate parsed data against this schema."""
        violations: List[GuardrailViolation] = []

        if not isinstance(data, dict):
            violations.append(
                GuardrailViolation(
                    rule="schema_type",
                    message=f"Expected dict, got {type(data).__name__}",
                    severity=Severity.BLOCK,
                )
            )
            return violations

        for f_spec in self.fields:
            value = data.get(f_spec.name)

            if value is None and f_spec.required:
                violations.append(
                    GuardrailViolation(
                        rule="required_field",
                        message=f"Missing required field: {f_spec.name}",
                        severity=Severity.BLOCK,
                    )
                )
                continue

            if value is None:
                continue

            # Type check
            expected = {
                "str": str, "int": int, "float": (int, float),
                "bool": bool, "list": list, "dict": dict,
            }
            expected_type = expected.get(f_spec.field_type)
            if expected_type and not isinstance(value, expected_type):
                violations.append(
                    GuardrailViolation(
                        rule="field_type",
                        message=(
                            f"Field '{f_spec.name}': expected {f_spec.field_type}, "
                            f"got {type(value).__name__}"
                        ),
                        severity=Severity.BLOCK,
                    )
                )

            # Length checks
            if f_spec.min_length is not None and hasattr(value, "__len__"):
                if len(value) < f_spec.min_length:
                    violations.append(
                        GuardrailViolation(
                            rule="min_length",
                            message=(
                                f"Field '{f_spec.name}': length {len(value)} "
                                f"< minimum {f_spec.min_length}"
                            ),
                            severity=Severity.WARNING,
                        )
                    )

            if f_spec.max_length is not None and hasattr(value, "__len__"):
                if len(value) > f_spec.max_length:
                    violations.append(
                        GuardrailViolation(
                            rule="max_length",
                            message=(
                                f"Field '{f_spec.name}': length {len(value)} "
                                f"> maximum {f_spec.max_length}"
                            ),
                            severity=Severity.WARNING,
                        )
                    )

            # Enum checks
            if f_spec.allowed_values and value not in f_spec.allowed_values:
                violations.append(
                    GuardrailViolation(
                        rule="allowed_values",
                        message=(
                            f"Field '{f_spec.name}': value '{value}' "
                            f"not in {f_spec.allowed_values}"
                        ),
                        severity=Severity.BLOCK,
                    )
                )

        return violations


# ── Content Safety ──────────────────────────────────────────────────────────

# Patterns that indicate potentially harmful content
_UNSAFE_PATTERNS: List[Dict[str, Any]] = [
    {
        "name": "credential_leak",
        "pattern": r"(?:password|secret|api.?key|token)\s*[:=]\s*\S+",
        "severity": Severity.BLOCK,
        "message": "Potential credential exposure detected",
    },
    {
        "name": "path_traversal",
        "pattern": r"\.\.[/\\]",
        "severity": Severity.WARNING,
        "message": "Path traversal pattern detected in output",
    },
    {
        "name": "sql_injection",
        "pattern": r"(?:DROP|DELETE|UPDATE|INSERT)\s+(?:TABLE|FROM|INTO)",
        "severity": Severity.WARNING,
        "message": "SQL mutation pattern detected in output",
    },
    {
        "name": "shell_injection",
        "pattern": r"(?:rm\s+-rf\s+/|:;\s*\)|&&\s*rm\b|;\s*cat\s+/etc/)",
        "severity": Severity.BLOCK,
        "message": "Dangerous shell command pattern detected",
    },
]

# PII detection patterns
_PII_PATTERNS: Dict[str, str] = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}


class ContentFilter:
    """Checks output for safety violations and PII."""

    def __init__(
        self,
        enable_pii_redaction: bool = True,
        custom_blocked_words: Optional[Set[str]] = None,
        extra_patterns: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.enable_pii_redaction = enable_pii_redaction
        self.blocked_words = custom_blocked_words or set()
        self.patterns = list(_UNSAFE_PATTERNS)
        if extra_patterns:
            self.patterns.extend(extra_patterns)

    def check(self, text: str) -> List[GuardrailViolation]:
        """Run all content safety checks on the text."""
        violations: List[GuardrailViolation] = []

        # Pattern-based checks
        for pat_info in self.patterns:
            if re.search(pat_info["pattern"], text, re.IGNORECASE):
                violations.append(
                    GuardrailViolation(
                        rule=pat_info["name"],
                        message=pat_info["message"],
                        severity=pat_info["severity"],
                        context=text[:200],
                    )
                )

        # Blocked word check
        text_lower = text.lower()
        for word in self.blocked_words:
            if word.lower() in text_lower:
                violations.append(
                    GuardrailViolation(
                        rule="blocked_word",
                        message=f"Blocked word detected: '{word}'",
                        severity=Severity.BLOCK,
                    )
                )

        return violations

    def redact_pii(self, text: str) -> str:
        """Replace detected PII with placeholders."""
        if not self.enable_pii_redaction:
            return text
        result = text
        for pii_type, pattern in _PII_PATTERNS.items():
            result = re.sub(
                pattern,
                f"[REDACTED_{pii_type.upper()}]",
                result,
            )
        return result

    def is_safe(self, text: str) -> bool:
        """Quick check: does the text contain any BLOCK-level violations?"""
        violations = self.check(text)
        return not any(v.severity == Severity.BLOCK for v in violations)


# ── Tool Argument Guardrails ───────────────────────────────────────────────


# Dangerous tool+arg combinations that need extra scrutiny
_DANGEROUS_TOOL_PATTERNS: Dict[str, List[str]] = {
    "run_command": [
        r"rm\s+-rf",
        r"del\s+/[sfq]",
        r"format\b",
        r"mkfs\b",
        r"dd\s+if=",
        r">\s*/dev/",
        r"shutdown",
        r"reboot",
    ],
    "delete_file": [
        r"[/\\]\.\.",  # path traversal
        r"^[/\\]$",  # root directory
        r"system32",
        r"/etc/",
    ],
    "write_file": [
        r"\.ssh[/\\]",
        r"\.env$",
        r"id_rsa",
    ],
}


class ToolGuardrail:
    """Validates tool calls before execution for safety."""

    def __init__(
        self,
        blocked_tools: Optional[Set[str]] = None,
        require_confirmation: Optional[Set[str]] = None,
    ) -> None:
        self.blocked_tools = blocked_tools or set()
        self.require_confirmation = require_confirmation or {
            "run_command", "delete_file", "send_email",
        }
        self._call_counts: Dict[str, int] = {}
        self._call_window_start = time.time()

    def validate_tool_call(
        self, tool_name: str, args: Dict[str, Any]
    ) -> List[GuardrailViolation]:
        """Validate a tool call for safety before execution."""
        violations: List[GuardrailViolation] = []

        # Blocked tool check
        if tool_name in self.blocked_tools:
            violations.append(
                GuardrailViolation(
                    rule="blocked_tool",
                    message=f"Tool '{tool_name}' is blocked by policy",
                    severity=Severity.BLOCK,
                )
            )
            return violations

        # Dangerous argument patterns
        patterns = _DANGEROUS_TOOL_PATTERNS.get(tool_name, [])
        args_str = json.dumps(args)
        for pattern in patterns:
            if re.search(pattern, args_str, re.IGNORECASE):
                violations.append(
                    GuardrailViolation(
                        rule="dangerous_args",
                        message=(
                            f"Tool '{tool_name}' called with potentially "
                            f"dangerous arguments matching: {pattern}"
                        ),
                        severity=Severity.WARNING,
                        context=args_str[:200],
                    )
                )

        # Rate limiting: reset window every 60 seconds
        now = time.time()
        if now - self._call_window_start > 60:
            self._call_counts.clear()
            self._call_window_start = now

        self._call_counts[tool_name] = self._call_counts.get(tool_name, 0) + 1
        if self._call_counts[tool_name] > 20:
            violations.append(
                GuardrailViolation(
                    rule="rate_limit",
                    message=(
                        f"Tool '{tool_name}' called {self._call_counts[tool_name]} "
                        f"times in the current window — possible runaway loop"
                    ),
                    severity=Severity.WARNING,
                )
            )

        return violations

    def needs_confirmation(self, tool_name: str) -> bool:
        """Check if a tool call should prompt for user confirmation."""
        return tool_name in self.require_confirmation


# ── Structured Output Enforcer ──────────────────────────────────────────────


class StructuredOutputEnforcer:
    """Wraps LLM calls to enforce structured JSON output with retry."""

    def __init__(
        self,
        llm_query: Callable[[str], str],
        max_retries: int = 2,
    ) -> None:
        self.llm_query = llm_query
        self.max_retries = max_retries

    def query_with_schema(
        self,
        prompt: str,
        schema: OutputSchema,
    ) -> Dict[str, Any]:
        """Query the LLM and enforce that the response matches a schema.

        Retries with corrective feedback if the output doesn't validate.
        """
        full_prompt = f"{prompt}\n\n{schema.to_prompt_instruction()}"

        for attempt in range(1, self.max_retries + 2):
            raw = self.llm_query(full_prompt)
            parsed = self._parse_json(raw)

            if parsed is None:
                full_prompt = (
                    f"{prompt}\n\n{schema.to_prompt_instruction()}\n\n"
                    f"Your previous response was not valid JSON:\n{raw[:300]}\n"
                    f"Please respond with ONLY valid JSON."
                )
                continue

            violations = schema.validate(parsed)
            blocking = [v for v in violations if v.severity == Severity.BLOCK]

            if not blocking:
                return parsed

            # Build corrective feedback
            issues = "\n".join(f"  - {v.message}" for v in blocking)
            full_prompt = (
                f"{prompt}\n\n{schema.to_prompt_instruction()}\n\n"
                f"Your previous response had issues:\n{issues}\n"
                f"Please fix these and respond with valid JSON."
            )

        # Last resort: return whatever we got
        return parsed or {}

    @staticmethod
    def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM output with fallbacks."""
        # Strip markdown fences
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")

        try:
            result = json.loads(cleaned)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass
        return None


# ── Guardrails Manager ──────────────────────────────────────────────────────


class GuardrailsManager:
    """Unified interface for all guardrails in the agent pipeline."""

    def __init__(
        self,
        enable_content_filter: bool = True,
        enable_tool_guardrails: bool = True,
        enable_pii_redaction: bool = True,
        blocked_tools: Optional[Set[str]] = None,
    ) -> None:
        self.content_filter = (
            ContentFilter(enable_pii_redaction=enable_pii_redaction)
            if enable_content_filter
            else None
        )
        self.tool_guardrail = (
            ToolGuardrail(blocked_tools=blocked_tools)
            if enable_tool_guardrails
            else None
        )

    def check_output(self, text: str) -> tuple[str, List[GuardrailViolation]]:
        """Check agent output for safety and optionally redact PII.

        Returns the (possibly redacted) text and any violations found.
        """
        violations: List[GuardrailViolation] = []

        if self.content_filter:
            violations.extend(self.content_filter.check(text))
            text = self.content_filter.redact_pii(text)

        return text, violations

    def check_tool_call(
        self, tool_name: str, args: Dict[str, Any]
    ) -> List[GuardrailViolation]:
        """Check a tool call for safety before execution."""
        if self.tool_guardrail:
            return self.tool_guardrail.validate_tool_call(tool_name, args)
        return []

    def is_output_safe(self, text: str) -> bool:
        """Quick check if output has any BLOCK-level violations."""
        if self.content_filter:
            return self.content_filter.is_safe(text)
        return True
