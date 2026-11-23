"""Ephemeral in-memory session store (REQ-PRIV-01).

Guarantees zero persistent disk logging, ephemeral in-memory state only,
and explicit purge_session() memory wiping bound to pipeline teardown/disconnect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
import time
from typing import Any
import uuid

from .schemas import AFSADiagnosticOutput, SoftMemoryContext, SomaticTelemetryPayload


@dataclass
class Session:
    """Ephemeral in-memory conversational session container."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[AFSADiagnosticOutput] = field(default_factory=list)
    somatic_telemetry: SomaticTelemetryPayload | None = None
    soft_memory: SoftMemoryContext = field(default_factory=SoftMemoryContext)
    induction_phrase: str = "Hey love"
    created_at: float = field(default_factory=time.time)
    is_active: bool = True

    def purge(self) -> None:
        """Zero all in-memory state and invalidate session."""
        self.state.clear()
        self.diagnostics.clear()
        self.somatic_telemetry = None
        self.soft_memory = SoftMemoryContext()
        self.induction_phrase = ""
        self.is_active = False


class InMemorySessionStore:
    """Thread-safe ephemeral session store."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = RLock()

    def create(
        self,
        session_id: str | None = None,
        induction_phrase: str = "Hey love",
    ) -> Session:
        sid = session_id or str(uuid.uuid4())
        session = Session(session_id=sid, induction_phrase=induction_phrase)
        with self._lock:
            self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def purge(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is not None:
                session.purge()
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            for session in self._sessions.values():
                session.purge()
            self._sessions.clear()

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._sessions) == 0

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)


_sessions = InMemorySessionStore()


def get_default_session_store() -> InMemorySessionStore:
    """Return the global default session store."""
    return _sessions


def purge_session(session_id: str | None = None, store: InMemorySessionStore | None = None) -> bool:
    """Purge one session, or all sessions when no ID is supplied (zeroes in-memory state)."""
    target = store or _sessions
    if session_id is None:
        target.clear()
        return True
    return target.purge(session_id)


__all__ = [
    "InMemorySessionStore",
    "Session",
    "get_default_session_store",
    "purge_session",
]
