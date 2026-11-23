"""Bounded Pydantic payload schemas for the AFSA vertical slice.

Strictly targets the Ryan, Arredondo, & Sandoval (2005) continua:
- CAS: Cultural Adaptation Stress
- CPD: Code-Switching Prosodic Dexterity
- AICL: Acculturation-Induced Cognitive Load
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

MirrorPosture = Literal[
    "hold_space_minimal",
    "gentle_grounding",
    "dialectal_mirror",
    "other",
]


class ErrorBounds(BaseModel):
    """Confidence and uncertainty bounds for a classification (REQ-VAR-01)."""

    lower: float = Field(default=0.0, ge=0.0, le=1.0, description="Lower confidence bound")
    upper: float = Field(default=1.0, ge=0.0, le=1.0, description="Upper confidence bound")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence level")
    hardware_jitter: float = Field(default=0.0, ge=0.0, description="Sensor/codec hardware jitter margin")
    environmental_margin: float = Field(default=0.0, ge=0.0, description="Environmental acoustic noise margin")

    @model_validator(mode="after")
    def validate_bounds(self) -> ErrorBounds:
        if self.lower > self.upper:
            raise ValueError("lower confidence bound cannot exceed upper bound")
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class CoreMotion(BaseModel):
    """Somatic/core-motion kinematic payload."""

    pitch: float = 0.0
    roll: float = 0.0
    accel_variance: float = Field(default=0.0, ge=0.0)
    activation: float = Field(default=0.0, ge=0.0, le=1.0)
    tension: float = Field(default=0.0, ge=0.0, le=1.0)
    posture: str = "neutral"
    dialect: str = "neutral"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class SomaticTelemetryPayload(BaseModel):
    """AirPods CoreMotion IMU jaw-clench metrics, optional rPPG/EMG, and fatigue nuisance parameter."""

    pitch: float = Field(default=0.0, description="AirPods IMU pitch angle")
    roll: float = Field(default=0.0, description="AirPods IMU roll angle")
    accel_variance: float = Field(
        default=0.0, ge=0.0, description="Jaw-clench kinematic acceleration variance"
    )
    yaw: float = Field(default=0.0, description="AirPods IMU yaw angle")
    rppg_hrv: float | None = Field(
        default=None, ge=0.0, description="Optional client-side rPPG blood volume pulse / HRV proxy"
    )
    facial_emg: float | None = Field(
        default=None, ge=0.0, description="Optional facial masseter/corrugator micro-EMG proxy"
    )
    eta: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fatigue nuisance parameter (circadian/physical exhaustion to factor out)",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class SoftMemoryContext(BaseModel):
    """Lightweight 16-token m_k metadata protocol defaulting to a no-op pass-through (Phase 3 hook)."""

    m_k: list[float] = Field(
        default_factory=lambda: [0.0] * 16,
        description="16 soft memory tokens for speaker dialectal embedding airlock",
    )
    is_active: bool = Field(
        default=False, description="Flag indicating if soft memory token conditioning is active"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("m_k")
    @classmethod
    def validate_token_length(cls, v: list[float]) -> list[float]:
        if len(v) != 16:
            raise ValueError(f"m_k must contain exactly 16 tokens, got {len(v)}")
        return v

    def is_noop(self) -> bool:
        """Return True if this context acts as a no-op pass-through."""
        return not self.is_active or all(x == 0.0 for x in self.m_k)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class CAS(BaseModel):
    """Cultural Adaptation Stress (CAS) continuum payload (Ryan, Arredondo, & Sandoval, 2005)."""

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    label: str = "unknown"
    error_bounds: ErrorBounds = Field(default_factory=ErrorBounds)
    evidence: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class CPD(BaseModel):
    """Code-Switching Prosodic Dexterity (CPD) continuum payload (Ryan, Arredondo, & Sandoval, 2005)."""

    choice: float = Field(default=0.0, ge=0.0, le=1.0)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    label: str = "unknown"
    error_bounds: ErrorBounds = Field(default_factory=ErrorBounds)
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def sync_score_choice(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "choice" in data and "score" not in data:
                data["score"] = data["choice"]
            elif "score" in data and "choice" not in data:
                data["choice"] = data["score"]
        return data

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class AICL(BaseModel):
    """Acculturation-Induced Cognitive Load (AICL) continuum payload (Ryan, Arredondo, & Sandoval, 2005)."""

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    label: str = "unknown"
    core_motion: CoreMotion = Field(default_factory=CoreMotion)
    error_bounds: ErrorBounds = Field(default_factory=ErrorBounds)
    evidence: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class AFSADiagnosticOutput(BaseModel):
    """Complete typed AFSA diagnostic output vector (REQ-DEC-01, REQ-VAR-01).

    Encompasses:
    - CAS, CPD, AICL calibrated scores (0.0 to 1.0)
    - Noul: acute acculturation friction vs. circadian exhaustion gate probability
    - Choice: mirror posture route (hold_space_minimal, gentle_grounding, dialectal_mirror, other)
    - ErrorBounds: explicit confidence bounds and hardware/environmental margins
    """

    cas: float = Field(default=0.0, ge=0.0, le=1.0, description="Cultural Adaptation Stress score")
    cpd: float = Field(default=0.0, ge=0.0, le=1.0, description="Code-Switching Prosodic Dexterity score")
    aicl: float = Field(default=0.0, ge=0.0, le=1.0, description="Acculturation-Induced Cognitive Load score")
    noul: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Noul fatigue-vs-stress gate probability (acute friction vs circadian eta)",
    )
    choice: str = Field(
        default="hold_space_minimal",
        description="Mirror posture route (hold_space_minimal, gentle_grounding, dialectal_mirror, other)",
    )
    error_bounds: ErrorBounds = Field(
        default_factory=ErrorBounds, description="Explicit confidence and error bounds (REQ-VAR-01)"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall decision confidence")
    hardware_jitter: float = Field(default=0.0, ge=0.0, description="AirPods/codec hardware jitter margin")
    environmental_margin: float = Field(
        default=0.0, ge=0.0, description="Environmental acoustic noise margin"
    )
    somatic_telemetry: SomaticTelemetryPayload | None = Field(
        default=None, description="Client-side somatic telemetry snapshot"
    )
    soft_memory: SoftMemoryContext | None = Field(
        default=None, description="Soft memory context protocol snapshot"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def map_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "Noul" in data and "noul" not in data:
                val = data["Noul"]
                data["noul"] = val if isinstance(val, (int, float)) else 0.5
            if "Choice" in data and "choice" not in data:
                val = data["Choice"]
                data["choice"] = str(val) if isinstance(val, str) else "hold_space_minimal"
            if "Score" in data and "cas" not in data:
                val = data["Score"]
                if isinstance(val, (int, float)):
                    data["cas"] = float(val)
        return data

    @property
    def Noul(self) -> float:
        """Alias for Noul fatigue-vs-stress gate probability."""
        return self.noul

    @property
    def Choice(self) -> str:
        """Alias for mirror posture route Choice."""
        return self.choice

    @property
    def Score(self) -> float:
        """Alias for primary continuum Score (CAS)."""
        return self.cas

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump()
        d["Noul"] = self.noul
        d["Choice"] = self.choice
        d["Score"] = self.cas
        return d


Classification = AFSADiagnosticOutput

Payload = (
    CAS
    | CPD
    | AICL
    | CoreMotion
    | ErrorBounds
    | SomaticTelemetryPayload
    | SoftMemoryContext
    | AFSADiagnosticOutput
)


def payload_to_dict(payload: Payload) -> dict[str, Any]:
    if hasattr(payload, "to_dict"):
        return payload.to_dict()
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    raise TypeError(f"Unsupported AFSA payload: {type(payload).__name__}")
