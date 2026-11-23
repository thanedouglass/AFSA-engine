"""Ephemeral in-memory session state."""

from dataclasses import dataclass, field
from threading import RLock
from typing import Any
import uuid


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: dict[str, Any] = field(default_factory=dict)


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = RLock()

    def create(self, session_id: str | None = None) -> Session:
        session = Session(session_id=session_id or str(uuid.uuid4()))
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def purge(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()


_sessions = InMemorySessionStore()


def purge_session(session_id: str | None = None, store: InMemorySessionStore | None = None) -> bool:
    """Purge one session, or all sessions when no ID is supplied."""
    target = store or _sessions
    if session_id is None:
        target.clear()
        return True
    return target.purge(session_id)
