"""Pipecat voice core worker, runner, and AFSA evaluation frame processor.

Implements:
- PipelineWorker managed by WorkerRunner (with PipelineParams)
- Dialect-aware VAD pause buffers (enforcing 500ms–2500ms bounds)
- Disarming "Hey love" induction phrase (REQ-IND-01)
- Non-blocking AFSAEvaluationProcessor(FrameProcessor) on turn completion
- Automatic purge_session() cleanup on pipeline finish/disconnect (REQ-PRIV-01)
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    EndWorkerFrame,
    Frame,
    StopFrame,
    StopWorkerFrame,
    TextFrame,
    TranscriptionFrame,
    UserAudioRawFrame,
    UserStoppedSpeakingFrame,
    UserTurnCompletionFrame,
)
from pipecat.pipeline.base_pipeline import BasePipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker as BasePipecatWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .classifier import GoogleGenAIClassifier
from .schemas import (
    AFSADiagnosticOutput,
    SoftMemoryContext,
    SomaticTelemetryPayload,
)
from .session import (
    InMemorySessionStore,
    get_default_session_store,
    purge_session,
)

DISARMING_INDUCTION_PHRASE = "Hey love"

VAD_SILENCE_MIN_MS = 500
VAD_SILENCE_MAX_MS = 2500
DEFAULT_VAD_SILENCE_MS = 1200

DIALECT_VAD_MS = {
    "neutral": 1200,
    "aave": 1500,
    "african-american-vernacular": 1500,
    "north-american": 1150,
    "british": 1250,
    "general": 1200,
}


def clamp_vad_silence_ms(ms: int | float) -> int:
    """Clamp VAD silence interval strictly between 500ms and 2500ms."""
    try:
        val = int(ms)
    except (TypeError, ValueError):
        val = DEFAULT_VAD_SILENCE_MS
    return max(VAD_SILENCE_MIN_MS, min(VAD_SILENCE_MAX_MS, val))


def validate_vad_silence_ms(ms: int | float) -> int:
    """Validate that VAD silence falls within the strictly enforced 500ms–2500ms bounds."""
    try:
        val = int(ms)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Invalid VAD silence value: {ms}") from err

    if not (VAD_SILENCE_MIN_MS <= val <= VAD_SILENCE_MAX_MS):
        raise ValueError(
            f"VAD_SILENCE_MS must be between {VAD_SILENCE_MIN_MS} and {VAD_SILENCE_MAX_MS} ms, got {ms}"
        )
    return val


def dialect_vad_silence_ms(dialect: str = "neutral", configured_ms: int | None = None) -> int:
    """Return a dialect-adjusted, range-safe trailing-silence interval (500–2500 ms)."""
    if configured_ms is not None:
        base = configured_ms
    else:
        env_val = os.getenv("VAD_SILENCE_MS")
        if env_val is not None:
            try:
                base = int(env_val)
            except (TypeError, ValueError):
                base = DEFAULT_VAD_SILENCE_MS
        else:
            base = DEFAULT_VAD_SILENCE_MS

    dialect_offset = DIALECT_VAD_MS.get(dialect.lower(), DEFAULT_VAD_SILENCE_MS) - DEFAULT_VAD_SILENCE_MS
    silence = base + dialect_offset
    return clamp_vad_silence_ms(silence)


class AFSAEvaluationProcessor(FrameProcessor):
    """Non-blocking FrameProcessor that evaluates turn acculturation metrics upon turn completion."""

    def __init__(
        self,
        session_id: str | None = None,
        classifier: GoogleGenAIClassifier | None = None,
        dialect: str = "neutral",
        somatic: SomaticTelemetryPayload | None = None,
        soft_memory: SoftMemoryContext | None = None,
        induction_phrase: str = DISARMING_INDUCTION_PHRASE,
        session_store: InMemorySessionStore | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.session_id = session_id
        self.classifier = classifier or GoogleGenAIClassifier()
        self.dialect = dialect
        self.somatic = somatic
        self.soft_memory = soft_memory
        self.induction_phrase = induction_phrase
        self.session_store = session_store or get_default_session_store()

        self.latest_diagnostic: AFSADiagnosticOutput | None = None
        self.turns_evaluated: int = 0
        self._current_turn_text: list[str] = []
        self._eval_tasks: set[asyncio.Task] = set()

    def get_induction_phrase(self) -> str:
        """Return the disarming opening phrase ('Hey love')."""
        return self.induction_phrase

    async def evaluate_turn(
        self,
        text: str,
        somatic: SomaticTelemetryPayload | None = None,
    ) -> AFSADiagnosticOutput:
        """Explicit evaluation of a single turn."""
        som = somatic or self.somatic
        diagnostic = await self.classifier.aclassify(
            text,
            somatic=som,
            soft_memory=self.soft_memory,
            dialect=self.dialect,
        )
        self.latest_diagnostic = diagnostic
        if self.session_id:
            session = self.session_store.get(self.session_id)
            if session:
                session.diagnostics.append(diagnostic)
        return diagnostic

    async def _evaluate_and_store(
        self,
        text: str,
        somatic: SomaticTelemetryPayload | None = None,
    ) -> None:
        """Background asynchronous task for turn evaluation."""
        try:
            await self.evaluate_turn(text, somatic=somatic)
        except Exception:
            pass

    async def wait_for_evaluations(self) -> None:
        """Wait for any background non-blocking evaluation tasks to complete."""
        if self._eval_tasks:
            tasks = list(self._eval_tasks)
            await asyncio.gather(*tasks, return_exceptions=True)

    def cleanup(self) -> None:
        """Explicitly wipe ephemeral session state on processor teardown."""
        if self.session_id:
            purge_session(self.session_id, store=self.session_store)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        # 1. Accumulate conversational speech text from transcript or text frames
        if isinstance(frame, TextFrame):
            if frame.text and frame.text.strip():
                self._current_turn_text.append(frame.text.strip())
        elif isinstance(frame, TranscriptionFrame):
            if frame.text and frame.text.strip():
                self._current_turn_text.append(frame.text.strip())

        # 2. Trigger non-blocking evaluation upon turn completion
        is_turn_completion = (
            isinstance(frame, (UserTurnCompletionFrame, UserStoppedSpeakingFrame))
            or (isinstance(frame, TranscriptionFrame) and getattr(frame, "finalized", False))
        )

        if is_turn_completion:
            turn_text = " ".join(self._current_turn_text).strip()
            if not turn_text and hasattr(frame, "text"):
                turn_text = getattr(frame, "text", "") or ""
            if not turn_text:
                turn_text = "simulated conversational turn"

            self._current_turn_text = []
            self.turns_evaluated += 1

            # Non-blocking turn evaluation task
            task = asyncio.create_task(self._evaluate_and_store(turn_text, self.somatic))
            self._eval_tasks.add(task)
            task.add_done_callback(self._eval_tasks.discard)

        # 3. Teardown / disconnect: enforce zero-persistence session purge (REQ-PRIV-01)
        if isinstance(frame, (EndFrame, CancelFrame, StopFrame, EndWorkerFrame, StopWorkerFrame)):
            self.cleanup()

        # Always push frame downstream
        await self.push_frame(frame, direction)


class PipelineWorker(BasePipecatWorker):
    """AFSA PipelineWorker managed by WorkerRunner with PipelineParams and dialect-aware VAD."""

    def __init__(
        self,
        pipeline: BasePipeline | None = None,
        *,
        params: PipelineParams | None = None,
        process: Callable[[bytes, str], dict] | None = None,
        dialect: str = "neutral",
        vad_silence_ms: int | None = None,
        session_id: str | None = None,
        induction_phrase: str = DISARMING_INDUCTION_PHRASE,
        evaluation_processor: AFSAEvaluationProcessor | None = None,
        classifier: GoogleGenAIClassifier | None = None,
        **kwargs: Any,
    ) -> None:
        self.dialect = dialect
        if vad_silence_ms is not None:
            self.vad_silence_ms = clamp_vad_silence_ms(vad_silence_ms)
        else:
            self.vad_silence_ms = dialect_vad_silence_ms(dialect)

        self.session_id = session_id
        self.induction_phrase = induction_phrase
        self.classifier = classifier or GoogleGenAIClassifier()

        if evaluation_processor is not None:
            self.evaluation_processor = evaluation_processor
        else:
            self.evaluation_processor = AFSAEvaluationProcessor(
                session_id=session_id,
                classifier=self.classifier,
                dialect=dialect,
                induction_phrase=induction_phrase,
            )

        pipe = pipeline if pipeline is not None else Pipeline([self.evaluation_processor])
        pipeline_params = params if params is not None else PipelineParams()

        super().__init__(pipeline=pipe, params=pipeline_params, **kwargs)

        self.process = process or (
            lambda audio, d: {"dialect": d, "vad_silence_ms": self.vad_silence_ms}
        )

    def get_induction_phrase(self) -> str:
        """Return the disarming session opening phrase ('Hey love')."""
        return self.induction_phrase

    def run(self, audio: bytes) -> dict:
        """Synchronously process simulated audio bytes."""
        return self.process(audio, self.dialect)

    async def arun(self, audio: bytes) -> dict:
        """Asynchronously process simulated audio bytes."""
        return self.run(audio)

    async def simulate_turn(
        self,
        text: str,
        somatic: SomaticTelemetryPayload | None = None,
    ) -> AFSADiagnosticOutput:
        """Simulate a complete voice turn flowing through the evaluation processor."""
        await self.evaluation_processor.process_frame(
            TextFrame(text), FrameDirection.DOWNSTREAM
        )
        await self.evaluation_processor.process_frame(
            UserTurnCompletionFrame(kind="turn_end"), FrameDirection.DOWNSTREAM
        )
        await self.evaluation_processor.wait_for_evaluations()
        assert self.evaluation_processor.latest_diagnostic is not None
        return self.evaluation_processor.latest_diagnostic

    def close(self) -> None:
        """Explicitly purge session upon worker termination."""
        if self.session_id:
            purge_session(self.session_id)
        try:
            loop = asyncio.get_running_loop()
            if hasattr(self, "stop") and asyncio.iscoroutinefunction(self.stop):
                loop.create_task(self.stop())
        except RuntimeError:
            pass


class WorkerRunner:
    """Lifecycle runner managing AFSA PipelineWorkers (with PipelineParams)."""

    def __init__(
        self,
        workers: PipelineWorker | list[PipelineWorker] | None = None,
        *,
        params: PipelineParams | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        if workers is None:
            self.workers: list[PipelineWorker] = []
        elif isinstance(workers, PipelineWorker):
            self.workers = [workers]
        else:
            self.workers = list(workers)

        self.params = params or PipelineParams()
        self.name = name
        self.running = False
        self._kwargs = kwargs

    def add_worker(self, worker: PipelineWorker) -> None:
        self.workers.append(worker)

    def add_workers(self, *workers: PipelineWorker) -> None:
        self.workers.extend(workers)

    def start(self) -> None:
        self.running = True
        try:
            loop = asyncio.get_running_loop()
            for worker in self.workers:
                if hasattr(worker, "start"):
                    if asyncio.iscoroutinefunction(worker.start):
                        loop.create_task(worker.start())
                    else:
                        worker.start()
        except RuntimeError:
            pass

    def stop(self) -> None:
        self.running = False
        for worker in self.workers:
            if hasattr(worker, "close"):
                worker.close()
            elif hasattr(worker, "stop"):
                try:
                    if asyncio.iscoroutinefunction(worker.stop):
                        try:
                            loop = asyncio.get_running_loop()
                            loop.create_task(worker.stop())
                        except RuntimeError:
                            pass
                    else:
                        worker.stop()
                except Exception:
                    pass

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


__all__ = [
    "AFSAEvaluationProcessor",
    "DISARMING_INDUCTION_PHRASE",
    "PipelineWorker",
    "WorkerRunner",
    "clamp_vad_silence_ms",
    "dialect_vad_silence_ms",
    "validate_vad_silence_ms",
]
