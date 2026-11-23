"""Small, provider-independent payload schemas for the AFSA vertical slice."""

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping


def _bounded(value: float, low: float, high: float, name: str) -> float:
    value = float(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


@dataclass
class ErrorBounds:
    """Confidence and uncertainty bounds for a classification."""

    lower: float = 0.0
    upper: float = 1.0

    def __post_init__(self) -> None:
        self.lower = _bounded(self.lower, 0.0, 1.0, "lower")
        self.upper = _bounded(self.upper, 0.0, 1.0, "upper")
        if self.lower > self.upper:
            raise ValueError("lower confidence bound cannot exceed upper bound")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class CoreMotion:
    """Somatic/core-motion payload."""

    activation: float = 0.0
    tension: float = 0.0
    posture: str = "neutral"
    dialect: str = "neutral"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.activation = _bounded(self.activation, 0.0, 1.0, "activation")
        self.tension = _bounded(self.tension, 0.0, 1.0, "tension")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CAS:
    """Cognitive/affective/self-regulatory continuum payload."""

    score: float = 0.0
    label: str = "unknown"
    error_bounds: ErrorBounds = field(default_factory=ErrorBounds)
    evidence: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.score = _bounded(self.score, 0.0, 1.0, "score")

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "label": self.label,
            "error_bounds": self.error_bounds.to_dict(),
            "evidence": list(self.evidence),
        }


@dataclass
class CPD:
    """Communicative participation/dialogue continuum payload."""

    choice: float = 0.0
    label: str = "unknown"
    error_bounds: ErrorBounds = field(default_factory=ErrorBounds)
    evidence: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.choice = _bounded(self.choice, 0.0, 1.0, "choice")

    def to_dict(self) -> dict[str, Any]:
        return {
            "choice": self.choice,
            "label": self.label,
            "error_bounds": self.error_bounds.to_dict(),
            "evidence": list(self.evidence),
        }


@dataclass
class AICL:
    """Adaptive interaction/conversational-learning continuum payload."""

    score: float = 0.0
    label: str = "unknown"
    core_motion: CoreMotion = field(default_factory=CoreMotion)
    error_bounds: ErrorBounds = field(default_factory=ErrorBounds)
    evidence: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.score = _bounded(self.score, 0.0, 1.0, "score")

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "label": self.label,
            "core_motion": self.core_motion.to_dict(),
            "error_bounds": self.error_bounds.to_dict(),
            "evidence": list(self.evidence),
        }


Payload = CAS | CPD | AICL | CoreMotion | ErrorBounds


def payload_to_dict(payload: Payload) -> dict[str, Any]:
    if isinstance(payload, (CAS, CPD, AICL, CoreMotion, ErrorBounds)):
        return payload.to_dict()
    raise TypeError(f"Unsupported AFSA payload: {type(payload).__name__}")
