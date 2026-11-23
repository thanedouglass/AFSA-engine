"""Pipecat worker and runner abstractions for the AFSA vertical slice."""

import os
from typing import Callable


DIALECT_VAD_MS = {
    "neutral": 800,
    "north-american": 750,
    "british": 850,
    "general": 800,
}


def dialect_vad_silence_ms(dialect: str = "neutral") -> int:
    """Return a dialect-adjusted, range-safe trailing-silence interval."""
    configured = os.getenv("VAD_SILENCE_MS")
    try:
        silence = int(configured) if configured is not None else DIALECT_VAD_MS.get(dialect.lower(), 800)
    except (TypeError, ValueError):
        silence = DIALECT_VAD_MS.get(dialect.lower(), 800)
    return max(500, min(2500, silence))


class PipelineWorker:
    """Minimal worker wrapper that remains usable without Pipecat installed."""

    def __init__(
        self,
        process: Callable[[bytes, str], dict] | None = None,
        dialect: str = "neutral",
        vad_silence_ms: int | None = None,
    ) -> None:
        self.process = process or (lambda audio, dialect: {"dialect": dialect})
        self.dialect = dialect
        configured = vad_silence_ms if vad_silence_ms is not None else dialect_vad_silence_ms(dialect)
        self.vad_silence_ms = max(500, min(2500, int(configured)))

    def run(self, audio: bytes) -> dict:
        return self.process(audio, self.dialect)

    async def arun(self, audio: bytes) -> dict:
        return self.run(audio)


class WorkerRunner:
    """Lifecycle runner for one or more AFSA pipeline workers."""

    def __init__(self, workers: PipelineWorker | list[PipelineWorker] | None = None) -> None:
        if workers is None:
            workers = []
        elif isinstance(workers, PipelineWorker):
            workers = [workers]
        self.workers = list(workers)
        self.running = False

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def run(self, audio: bytes) -> list[dict]:
        self.start()
        try:
            return [worker.run(audio) for worker in self.workers]
        finally:
            self.stop()

    async def arun(self, audio: bytes) -> list[dict]:
        self.start()
        try:
            return [await worker.arun(audio) for worker in self.workers]
        finally:
            self.stop()
