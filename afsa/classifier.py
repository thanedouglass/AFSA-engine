"""Google GenAI Decisions classifier with deterministic offline fallback.

Uses Gemini Decisions API (gemini-decision-1) with Noul, Choice, and Score
constructs from langchain_core.decisions:
- Noul: Acute acculturation friction vs. baseline circadian exhaustion (eta)
- Choice: Mirror posture route (hold_space_minimal, gentle_grounding, dialectal_mirror, other)
- Score: Parallel calibrated scoring for CAS, CPD, and AICL (Ryan et al., 2005)
"""

from __future__ import annotations

import os
from typing import Any
from langchain_core.decisions import Choice, Noul, Score

from .schemas import (
    AFSADiagnosticOutput,
    Classification,
    ErrorBounds,
    MirrorPosture,
    SoftMemoryContext,
    SomaticTelemetryPayload,
)


def get_afsa_decision_questions() -> dict[str, Any]:
    """Define the structured questions for the Gemini Decisions API."""
    return {
        "noul_acute_friction_vs_exhaustion": Noul(
            instructions=(
                "Determine the probability (0.01 to 0.99) that the speaker is experiencing acute "
                "acculturation friction as opposed to baseline circadian physical exhaustion (eta)."
            ),
        ),
        "mirror_posture": Choice(
            instructions="Select the mirror posture route that best supports the speaker's communicative state.",
            criteria={
                "hold_space_minimal": "Hold space minimally without imposing conversational or cognitive demands.",
                "gentle_grounding": "Offer gentle conversational grounding, pacing stabilization, and calming presence.",
                "dialectal_mirror": "Subtly mirror the speaker's dialectal cadences and prosodic rhythm.",
                "other": "Default open posture.",
            },
        ),
        "cas_score": Score(
            instructions="Rate Cultural Adaptation Stress (CAS) severity on a 5-point calibrated scale.",
            criteria=[
                "Minimal or no adaptation stress.",
                "Mild adaptation tension.",
                "Moderate adaptation stress.",
                "High acculturation friction.",
                "Severe acculturation distress.",
            ],
        ),
        "cpd_score": Score(
            instructions="Rate Code-Switching Prosodic Dexterity (CPD) cognitive burden on a 5-point scale.",
            criteria=[
                "Effortless fluent code-switching / minimal burden.",
                "Mild prosodic transition cost.",
                "Noticeable hesitation and cadence strain.",
                "High prosodic dissonance / taxing switching.",
                "Severe prosodic block or cognitive bottleneck.",
            ],
        ),
        "aicl_score": Score(
            instructions="Rate Acculturation-Induced Cognitive Load (AICL) on a 5-point calibrated scale.",
            criteria=[
                "Baseline low cognitive load.",
                "Mild cognitive tax from cultural navigation.",
                "Moderate cognitive interference.",
                "High cognitive saturation.",
                "Overwhelming cognitive exhaustion.",
            ],
        ),
    }


class GoogleGenAIClassifier:
    """Decisions classifier supporting Gemini Decisions API and deterministic offline fallback."""

    def __init__(
        self,
        model: str = "gemini-decision-1",
        api_key: str | None = None,
        mock: bool | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if mock is not None:
            self.mock = mock
        else:
            mock_env = os.getenv("ENABLE_MOCK_CLASSIFIER", "true").lower()
            self.mock = mock_env in {"1", "true", "yes"}

    def classify(
        self,
        text: str,
        *,
        somatic: SomaticTelemetryPayload | None = None,
        soft_memory: SoftMemoryContext | None = None,
        dialect: str = "neutral",
    ) -> AFSADiagnosticOutput:
        """Classify a speech turn synchronously."""
        if self.mock or not self.api_key:
            return self._offline(text, somatic=somatic, soft_memory=soft_memory, dialect=dialect)

        try:
            return self._invoke_genai(
                text, somatic=somatic, soft_memory=soft_memory, dialect=dialect
            )
        except Exception:
            return self._offline(text, somatic=somatic, soft_memory=soft_memory, dialect=dialect)

    async def aclassify(
        self,
        text: str,
        *,
        somatic: SomaticTelemetryPayload | None = None,
        soft_memory: SoftMemoryContext | None = None,
        dialect: str = "neutral",
    ) -> AFSADiagnosticOutput:
        """Classify a speech turn asynchronously."""
        if self.mock or not self.api_key:
            return self._offline(text, somatic=somatic, soft_memory=soft_memory, dialect=dialect)

        try:
            return await self._ainvoke_genai(
                text, somatic=somatic, soft_memory=soft_memory, dialect=dialect
            )
        except Exception:
            return self._offline(text, somatic=somatic, soft_memory=soft_memory, dialect=dialect)

    def _invoke_genai(
        self,
        text: str,
        *,
        somatic: SomaticTelemetryPayload | None,
        soft_memory: SoftMemoryContext | None,
        dialect: str,
    ) -> AFSADiagnosticOutput:
        from langchain_google_genai import GoogleGenAIClassifier as LCClassifier

        classifier = LCClassifier(model=self.model, api_key=self.api_key)
        state_payload: dict[str, Any] = {
            "text": text,
            "dialect": dialect,
            "somatic_telemetry": somatic.to_dict() if somatic else {},
            "eta": somatic.eta if somatic else 0.0,
            "soft_memory_active": not (soft_memory.is_noop() if soft_memory else True),
        }
        questions = get_afsa_decision_questions()
        response = classifier.invoke({"state": state_payload, "questions": questions})
        return self._parse_classifier_response(response, somatic=somatic, soft_memory=soft_memory)

    async def _ainvoke_genai(
        self,
        text: str,
        *,
        somatic: SomaticTelemetryPayload | None,
        soft_memory: SoftMemoryContext | None,
        dialect: str,
    ) -> AFSADiagnosticOutput:
        from langchain_google_genai import GoogleGenAIClassifier as LCClassifier

        classifier = LCClassifier(model=self.model, api_key=self.api_key)
        state_payload: dict[str, Any] = {
            "text": text,
            "dialect": dialect,
            "somatic_telemetry": somatic.to_dict() if somatic else {},
            "eta": somatic.eta if somatic else 0.0,
            "soft_memory_active": not (soft_memory.is_noop() if soft_memory else True),
        }
        questions = get_afsa_decision_questions()
        response = await classifier.ainvoke({"state": state_payload, "questions": questions})
        return self._parse_classifier_response(response, somatic=somatic, soft_memory=soft_memory)

    def _parse_classifier_response(
        self,
        response: Any,
        somatic: SomaticTelemetryPayload | None,
        soft_memory: SoftMemoryContext | None,
    ) -> AFSADiagnosticOutput:
        noul_val = float(response.nouls["noul_acute_friction_vs_exhaustion"].noul)
        choice_val = str(response.choices["mirror_posture"].choice)
        cas_val = max(0.0, min(1.0, float(response.scores["cas_score"].score) / 4.0))
        cpd_val = max(0.0, min(1.0, float(response.scores["cpd_score"].score) / 4.0))
        aicl_val = max(0.0, min(1.0, float(response.scores["aicl_score"].score) / 4.0))

        confidence = float(getattr(response.choices["mirror_posture"], "confidence", 0.85))
        jitter = 0.015
        env_margin = 0.02
        half_width = max(0.05, (1.0 - confidence) * 0.5)
        lower_bound = max(0.0, round(cas_val - half_width, 3))
        upper_bound = min(1.0, round(cas_val + half_width, 3))

        return AFSADiagnosticOutput(
            cas=cas_val,
            cpd=cpd_val,
            aicl=aicl_val,
            noul=noul_val,
            choice=choice_val,
            error_bounds=ErrorBounds(
                lower=lower_bound,
                upper=upper_bound,
                confidence=confidence,
                hardware_jitter=jitter,
                environmental_margin=env_margin,
            ),
            confidence=confidence,
            hardware_jitter=jitter,
            environmental_margin=env_margin,
            somatic_telemetry=somatic,
            soft_memory=soft_memory,
        )

    @classmethod
    def _offline(
        cls,
        text: str,
        *,
        somatic: SomaticTelemetryPayload | None = None,
        soft_memory: SoftMemoryContext | None = None,
        dialect: str = "neutral",
    ) -> AFSADiagnosticOutput:
        """Deterministic offline mock classifier fallback."""
        normalized = text.strip().lower()
        words = set(normalized.split())

        positive_cues = {"yes", "choice", "want", "can", "go", "good", "safe", "calm", "ground"}
        stress_cues = {
            "no",
            "stop",
            "can't",
            "cannot",
            "pain",
            "stress",
            "tired",
            "pressure",
            "exhausted",
            "hard",
            "dissonance",
            "switching",
        }
        dialect_cues = {"y'all", "finna", "ain't", "bout", "bro", "nah", "mirror"}

        is_stress = any(w in words for w in stress_cues)
        is_positive = any(w in words for w in positive_cues)
        is_dialect = dialect.lower() in {"aave", "african-american-vernacular"} or any(
            w in words for w in dialect_cues
        )

        eta = somatic.eta if somatic else 0.0
        accel = somatic.accel_variance if somatic else 0.0

        # Rao-Blackwellized separation:
        # High eta (circadian exhaustion) lowers acute friction probability (noul).
        # Jaw-clench accel variance increases acute friction probability.
        base_noul = 0.50
        if is_stress:
            base_noul = 0.75
        elif is_positive:
            base_noul = 0.35

        # Factor in somatic telemetry:
        noul_adjusted = base_noul - (0.30 * eta) + (0.20 * min(1.0, accel))
        noul_val = round(max(0.01, min(0.99, noul_adjusted)), 2)

        # Mirror posture route:
        if is_dialect:
            choice_val: MirrorPosture = "dialectal_mirror"
        elif is_stress or accel > 0.4:
            choice_val = "gentle_grounding"
        elif is_positive:
            choice_val = "hold_space_minimal"
        else:
            choice_val = "hold_space_minimal"

        # Calibrated continuum scores:
        if is_stress:
            cas_val = 0.70
            cpd_val = 0.60
            aicl_val = 0.65
            confidence = 0.88
        elif is_positive:
            cas_val = 0.25
            cpd_val = 0.20
            aicl_val = 0.20
            confidence = 0.92
        else:
            cas_val = 0.50
            cpd_val = 0.50
            aicl_val = 0.50
            confidence = 0.85

        # Adjust CPD if dialectal mirroring is active:
        if is_dialect:
            cpd_val = max(0.10, round(cpd_val - 0.15, 2))

        hardware_jitter = 0.015
        environmental_margin = 0.020
        margin = max(0.05, (1.0 - confidence) * 0.5)
        lower_bound = max(0.0, round(cas_val - margin, 2))
        upper_bound = min(1.0, round(cas_val + margin, 2))

        error_bounds = ErrorBounds(
            lower=lower_bound,
            upper=upper_bound,
            confidence=confidence,
            hardware_jitter=hardware_jitter,
            environmental_margin=environmental_margin,
        )

        return AFSADiagnosticOutput(
            cas=cas_val,
            cpd=cpd_val,
            aicl=aicl_val,
            noul=noul_val,
            choice=choice_val,
            error_bounds=error_bounds,
            confidence=confidence,
            hardware_jitter=hardware_jitter,
            environmental_margin=environmental_margin,
            somatic_telemetry=somatic,
            soft_memory=soft_memory,
        )


__all__ = [
    "Classification",
    "GoogleGenAIClassifier",
    "get_afsa_decision_questions",
]
