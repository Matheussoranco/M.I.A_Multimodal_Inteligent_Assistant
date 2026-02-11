"""
Persistent Learning — Cross-session memory and skill acquisition for M.I.A
==========================================================================

Stores successful tool-call patterns as reusable "skills", persists
conversation summaries across sessions, and provides RAG-style retrieval
of relevant past interactions.

Key features:
- **Skill Library**: When a tool sequence solves a task, it's saved as a
  named skill that can be recalled and reused.
- **Session Summaries**: At session end, the conversation is summarised
  and stored so future sessions can reference past context.
- **Interaction Log**: Every user↔assistant exchange is logged with
  embeddings for semantic search.
- **RAG Retrieval**: Before the agent responds, relevant past interactions
  are retrieved and injected as context.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Data structures ─────────────────────────────────────────────────────────


@dataclass
class Skill:
    """A reusable tool-call pattern learned from a successful interaction."""

    name: str
    description: str
    tool_sequence: List[Dict[str, Any]]  # [{tool, args_template}, …]
    tags: List[str] = field(default_factory=list)
    success_count: int = 1
    last_used: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SessionSummary:
    """Summary of a past conversation session."""

    session_id: str
    summary: str
    topics: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    turn_count: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionSummary":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class InteractionRecord:
    """A single user↔assistant exchange for the interaction log."""

    query: str
    response: str
    tools_used: List[str] = field(default_factory=list)
    success: bool = True
    timestamp: float = field(default_factory=time.time)
    embedding_hash: str = ""  # lightweight text hash for similarity

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InteractionRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ── Persistent Memory ──────────────────────────────────────────────────────


class PersistentMemory:
    """Cross-session memory with skill learning and RAG retrieval."""

    def __init__(
        self,
        storage_dir: str = "memory/persistent",
        max_skills: int = 500,
        max_sessions: int = 200,
        max_interactions: int = 5000,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.max_skills = max_skills
        self.max_sessions = max_sessions
        self.max_interactions = max_interactions

        self.skills: List[Skill] = []
        self.sessions: List[SessionSummary] = []
        self.interactions: List[InteractionRecord] = []

        self._ensure_dirs()
        self._load_all()

    # ── Skill Library ───────────────────────────────────────────────

    def learn_skill(
        self,
        name: str,
        description: str,
        tool_sequence: List[Dict[str, Any]],
        tags: Optional[List[str]] = None,
    ) -> Skill:
        """Record a successful tool sequence as a reusable skill."""
        # Check for existing skill with same name
        existing = self.get_skill(name)
        if existing:
            existing.success_count += 1
            existing.last_used = time.time()
            self._save_skills()
            return existing

        skill = Skill(
            name=name,
            description=description,
            tool_sequence=tool_sequence,
            tags=tags or [],
        )
        self.skills.append(skill)

        # Trim oldest if over limit
        if len(self.skills) > self.max_skills:
            self.skills.sort(key=lambda s: s.last_used, reverse=True)
            self.skills = self.skills[: self.max_skills]

        self._save_skills()
        logger.info("Learned new skill: %s", name)
        return skill

    def get_skill(self, name: str) -> Optional[Skill]:
        """Look up a skill by exact name."""
        for s in self.skills:
            if s.name == name:
                return s
        return None

    def find_relevant_skills(
        self, query: str, top_k: int = 3
    ) -> List[Skill]:
        """Find skills relevant to a query via keyword matching."""
        query_words = set(query.lower().split())
        scored: List[tuple] = []
        for skill in self.skills:
            skill_words = set(skill.description.lower().split())
            skill_words.update(t.lower() for t in skill.tags)
            skill_words.update(skill.name.lower().split("_"))
            overlap = len(query_words & skill_words)
            if overlap > 0:
                score = overlap + (skill.success_count * 0.1)
                scored.append((score, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]

    # ── Session Summaries ───────────────────────────────────────────

    def save_session_summary(
        self,
        session_id: str,
        summary: str,
        topics: Optional[List[str]] = None,
        tools_used: Optional[List[str]] = None,
        turn_count: int = 0,
    ) -> SessionSummary:
        """Store a summary of the current session for future reference."""
        sess = SessionSummary(
            session_id=session_id,
            summary=summary,
            topics=topics or [],
            tools_used=tools_used or [],
            turn_count=turn_count,
        )
        self.sessions.append(sess)

        if len(self.sessions) > self.max_sessions:
            self.sessions = self.sessions[-self.max_sessions :]

        self._save_sessions()
        logger.info("Saved session summary: %s", session_id)
        return sess

    def get_recent_sessions(self, n: int = 5) -> List[SessionSummary]:
        """Retrieve the most recent session summaries."""
        return self.sessions[-n:]

    # ── Interaction Log ─────────────────────────────────────────────

    def log_interaction(
        self,
        query: str,
        response: str,
        tools_used: Optional[List[str]] = None,
        success: bool = True,
    ) -> InteractionRecord:
        """Log a single interaction for future RAG retrieval."""
        record = InteractionRecord(
            query=query,
            response=response,
            tools_used=tools_used or [],
            success=success,
            embedding_hash=self._text_hash(query),
        )
        self.interactions.append(record)

        if len(self.interactions) > self.max_interactions:
            self.interactions = self.interactions[-self.max_interactions :]

        # Save periodically (every 10 interactions) to avoid constant I/O
        if len(self.interactions) % 10 == 0:
            self._save_interactions()

        return record

    def retrieve_relevant_context(
        self, query: str, top_k: int = 3
    ) -> List[InteractionRecord]:
        """RAG-style retrieval: find past interactions relevant to the query.

        Uses lightweight keyword + hash similarity.  A real production system
        would use vector embeddings here, but this works well enough without
        requiring a running embedding model.
        """
        query_words = set(query.lower().split())
        query_hash = self._text_hash(query)

        scored: List[tuple] = []
        for rec in self.interactions:
            rec_words = set(rec.query.lower().split())
            word_overlap = len(query_words & rec_words)
            hash_sim = 1.0 if rec.embedding_hash == query_hash else 0.0
            success_bonus = 0.5 if rec.success else 0.0
            score = word_overlap + hash_sim + success_bonus
            if score > 0.5:
                scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

    def get_context_prompt(self, query: str) -> Optional[str]:
        """Build a context injection prompt from past interactions and skills.

        Returns None if there's nothing relevant to inject.
        """
        parts: List[str] = []

        # Relevant past interactions
        past = self.retrieve_relevant_context(query, top_k=2)
        if past:
            parts.append("Relevant past interactions:")
            for rec in past:
                parts.append(f"  Q: {rec.query[:150]}")
                parts.append(f"  A: {rec.response[:150]}")

        # Relevant skills
        skills = self.find_relevant_skills(query, top_k=2)
        if skills:
            parts.append("Previously learned skills you can reuse:")
            for skill in skills:
                tools_str = " → ".join(
                    step.get("tool", "?") for step in skill.tool_sequence
                )
                parts.append(f"  • {skill.name}: {skill.description} [{tools_str}]")

        # Recent session context
        recent = self.get_recent_sessions(2)
        if recent:
            parts.append("Recent session summaries:")
            for sess in recent:
                parts.append(f"  • {sess.summary[:150]}")

        return "\n".join(parts) if parts else None

    # ── Persistence ─────────────────────────────────────────────────

    def flush(self) -> None:
        """Force save all data to disk."""
        self._save_skills()
        self._save_sessions()
        self._save_interactions()

    def _ensure_dirs(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _load_all(self) -> None:
        self.skills = self._load_json_list(
            self.storage_dir / "skills.json", Skill.from_dict
        )
        self.sessions = self._load_json_list(
            self.storage_dir / "sessions.json", SessionSummary.from_dict
        )
        self.interactions = self._load_json_list(
            self.storage_dir / "interactions.json", InteractionRecord.from_dict
        )

    def _save_skills(self) -> None:
        self._save_json_list(
            self.storage_dir / "skills.json",
            [s.to_dict() for s in self.skills],
        )

    def _save_sessions(self) -> None:
        self._save_json_list(
            self.storage_dir / "sessions.json",
            [s.to_dict() for s in self.sessions],
        )

    def _save_interactions(self) -> None:
        self._save_json_list(
            self.storage_dir / "interactions.json",
            [r.to_dict() for r in self.interactions],
        )

    @staticmethod
    def _load_json_list(path: Path, from_dict: Any) -> list:
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [from_dict(item) for item in data if isinstance(item, dict)]
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)
            return []

    @staticmethod
    def _save_json_list(path: Path, data: List[Dict]) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Failed to save %s: %s", path, e)

    @staticmethod
    def _text_hash(text: str) -> str:
        """Lightweight text fingerprint for similarity deduplication."""
        normalized = " ".join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:12]
