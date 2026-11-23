"""Google GenAI classifier with a deterministic offline fallback."""

from dataclasses import dataclass
import os
from typing import Any, Mapping

from .schemas import AICL, CAS, CPD, ErrorBounds


@dataclass(frozen=True)
class Classification:
    """Three-way classification output described by the AFSA interface."""

    Noul: str = "unknown"
    Choice: float = 0.0
    Score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"Noul": self.Noul, "Choice": self.Choice, "Score": self.Score}


class GoogleGenAIClassifier:
    """Generate a Noul/Choice/Score classification."""

    def __init__(self, api_key: str | None = None, mock: bool | None = None) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.mock = (
            mock
            if mock is not None
            else os.getenv("ENABLE_MOCK_CLASSIFIER", "").lower() in {"1", "true", "yes"}
        )

    def classify(self, text: str, *, dialect: str = "neutral") -> Classification:
        if self.mock or not self.api_key:
            return self._offline(text, dialect=dialect)
        try:
            from google import genai

            client = genai.Client(api_key=self.api_key)
            prompt = (
                "Return only JSON with keys Noul, Choice, Score. "
                "Noul is a short label, Choice is 0..1, and Score is 0..1. "
                f"dialect={dialect}\ntext={text}"
            )
            response = client.models.generate_content(
                model=os.getenv("GOOGLE_GENAI_MODEL", "gemini-2.0-flash"),
                contents=prompt,
            )
            import json

            data = json.loads(str(response.text).strip())
            return Classification(
                Noul=str(data.get("Noul", "unknown")),
                Choice=self._bounded(data.get("Choice", 0.0)),
                Score=self._bounded(data.get("Score", 0.0)),
            )
        except Exception:
            return self._offline(text, dialect=dialect)

    async def aclassify(self, text: str, *, dialect: str = "neutral") -> Classification:
        return self.classify(text, dialect=dialect)

    @staticmethod
    def _bounded(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _offline(cls, text: str, *, dialect: str = "neutral") -> Classification:
        normalized = text.strip().lower()
        positive = ("yes", "choice", "want", "can", "go")
        negative = ("no", "stop", "can't", "cannot", "pain")
        if any(word in normalized.split() for word in positive):
            return Classification("noul-positive", 0.8, 0.7)
        if any(word in normalized.split() for word in negative):
            return Classification("noul-negative", 0.2, 0.7)
        return Classification("noul-neutral", 0.5, 0.5)
