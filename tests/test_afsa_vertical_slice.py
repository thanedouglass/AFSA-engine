"""Vertical slice test suite for the AFSA engine.

Proves:
(a) A simulated voice turn flows through AFSAEvaluationProcessor
(b) Structured Noul/Choice/Score + error bounds validate against AFSADiagnosticOutput
(c) VAD_SILENCE_MS bounds (500–2500) are enforced
(d) purge_session() zeroes all in-memory session state
"""

import pytest
from pipecat.frames.frames import EndFrame, TextFrame, UserTurnCompletionFrame
from pipecat.processors.frame_processor import FrameDirection

from afsa.classifier import Classification, GoogleGenAIClassifier, get_afsa_decision_questions
from afsa.pipeline import (
    AFSAEvaluationProcessor,
    DISARMING_INDUCTION_PHRASE,
    PipelineWorker,
    WorkerRunner,
    clamp_vad_silence_ms,
    dialect_vad_silence_ms,
    validate_vad_silence_ms,
)
from afsa.schemas import (
    AICL,
    CAS,
    CPD,
    AFSADiagnosticOutput,
    CoreMotion,
    ErrorBounds,
    SoftMemoryContext,
    SomaticTelemetryPayload,
    payload_to_dict,
)
from afsa.session import InMemorySessionStore, purge_session


# ---------------------------------------------------------------------------
# Requirement (b): Structured Noul/Choice/Score + bounds validate
# ---------------------------------------------------------------------------
def test_payloads_serialize_and_bounds_validate():
    """Verify individual schema models, serialization, and range assertions."""
    bounds = ErrorBounds(lower=0.1, upper=0.9, confidence=0.88, hardware_jitter=0.01)
    assert bounds.to_dict()["upper"] == 0.9
    assert bounds.to_dict()["confidence"] == 0.88

    cas = CAS(score=0.6, error_bounds=bounds)
    assert cas.to_dict()["score"] == 0.6
    assert payload_to_dict(cas)["score"] == 0.6

    cpd = CPD(choice=0.7)
    assert cpd.to_dict()["choice"] == 0.7
    assert cpd.score == 0.7

    cm = CoreMotion(pitch=1.2, roll=0.5, accel_variance=0.3, activation=0.4)
    aicl = AICL(score=0.5, core_motion=cm)
    assert aicl.to_dict()["core_motion"]["activation"] == 0.4

    # Out-of-bounds checks
    with pytest.raises(Exception):
        CAS(score=1.1)

    with pytest.raises(ValueError):
        ErrorBounds(lower=0.8, upper=0.2)


def test_somatic_and_soft_memory_schemas():
    """Verify SomaticTelemetryPayload and SoftMemoryContext schemas."""
    somatic = SomaticTelemetryPayload(
        pitch=12.4,
        roll=-3.1,
        accel_variance=0.55,
        yaw=0.8,
        rppg_hrv=68.5,
        facial_emg=0.22,
        eta=0.15,
    )
    assert somatic.pitch == 12.4
    assert somatic.accel_variance == 0.55
    assert somatic.eta == 0.15
    assert somatic.to_dict()["eta"] == 0.15

    # Eta out of bounds (0..1)
    with pytest.raises(Exception):
        SomaticTelemetryPayload(eta=1.5)

    # SoftMemoryContext no-op default (16 tokens)
    soft_mem = SoftMemoryContext()
    assert len(soft_mem.m_k) == 16
    assert soft_mem.is_noop() is True

    # Invalid token count
    with pytest.raises(ValueError):
        SoftMemoryContext(m_k=[0.1] * 8)


def test_structured_noul_choice_score_validates_diagnostic_output():
    """Verify (b): structured Noul/Choice/Score + error bounds validate against AFSADiagnosticOutput."""
    bounds = ErrorBounds(lower=0.55, upper=0.85, confidence=0.90, hardware_jitter=0.015)
    diag = AFSADiagnosticOutput(
        cas=0.70,
        cpd=0.60,
        aicl=0.65,
        noul=0.82,
        choice="gentle_grounding",
        error_bounds=bounds,
        confidence=0.90,
    )

    # Validate continuum scores (Ryan et al., 2005)
    assert 0.0 <= diag.cas <= 1.0
    assert 0.0 <= diag.cpd <= 1.0
    assert 0.0 <= diag.aicl <= 1.0

    # Validate Noul acute-friction gate probability (0.01..0.99)
    assert 0.0 <= diag.Noul <= 1.0
    assert diag.Noul == 0.82

    # Validate Choice mirror posture route
    assert diag.Choice in {"hold_space_minimal", "gentle_grounding", "dialectal_mirror", "other"}
    assert diag.Choice == "gentle_grounding"

    # Validate Score alias and dictionary representation
    assert diag.Score == 0.70
    as_dict = diag.to_dict()
    assert as_dict["Noul"] == 0.82
    assert as_dict["Choice"] == "gentle_grounding"
    assert as_dict["Score"] == 0.70
    assert as_dict["error_bounds"]["lower"] == 0.55
    assert as_dict["error_bounds"]["upper"] == 0.85

    # Out-of-range validations
    with pytest.raises(Exception):
        AFSADiagnosticOutput(cas=1.5)
    with pytest.raises(Exception):
        AFSADiagnosticOutput(noul=-0.1)


def test_offline_classifier_is_deterministic(monkeypatch):
    """Verify deterministic offline classification and Decisions API question structure."""
    monkeypatch.setenv("ENABLE_MOCK_CLASSIFIER", "true")

    questions = get_afsa_decision_questions()
    assert "noul_acute_friction_vs_exhaustion" in questions
    assert "mirror_posture" in questions
    assert "cas_score" in questions
    assert "cpd_score" in questions
    assert "aicl_score" in questions

    classifier = GoogleGenAIClassifier()
    somatic = SomaticTelemetryPayload(accel_variance=0.65, eta=0.1)

    # Stress turn triggering gentle_grounding
    res_stress = classifier.classify("I feel intense stress and code switching pressure", somatic=somatic)
    assert isinstance(res_stress, Classification)
    assert isinstance(res_stress, AFSADiagnosticOutput)
    assert res_stress.Choice == "gentle_grounding"
    assert res_stress.Noul > 0.5
    assert res_stress.cas == 0.70
    assert res_stress.error_bounds.lower <= res_stress.error_bounds.upper

    # Dialect turn triggering dialectal_mirror
    res_dialect = classifier.classify("y'all finna see what happens", dialect="aave")
    assert res_dialect.Choice == "dialectal_mirror"

    # Positive / calm turn triggering hold_space_minimal
    res_calm = classifier.classify("I want to choose yes and go ahead")
    assert res_calm.Choice == "hold_space_minimal"
    assert res_calm.Score == 0.25
    assert res_calm.to_dict()["Choice"] == "hold_space_minimal"


# ---------------------------------------------------------------------------
# Requirement (c): VAD_SILENCE_MS bounds (500–2500) are enforced
# ---------------------------------------------------------------------------
def test_vad_silence_bounds_enforced(monkeypatch):
    """Verify (c): VAD_SILENCE_MS bounds (500–2500) are strictly enforced."""
    # Underflow clamped to 500
    monkeypatch.setenv("VAD_SILENCE_MS", "300")
    assert dialect_vad_silence_ms("neutral") == 500
    assert clamp_vad_silence_ms(300) == 500

    # Overflow clamped to 2500
    monkeypatch.setenv("VAD_SILENCE_MS", "3200")
    assert dialect_vad_silence_ms("neutral") == 2500
    assert clamp_vad_silence_ms(3200) == 2500

    # In-range default
    monkeypatch.setenv("VAD_SILENCE_MS", "1200")
    assert dialect_vad_silence_ms("neutral") == 1200

    # Dialect adjustments stay clamped within 500–2500
    assert dialect_vad_silence_ms("aave") == 1500
    assert dialect_vad_silence_ms("british") == 1250

    # Strict validator raises ValueError on out-of-bound inputs
    with pytest.raises(ValueError):
        validate_vad_silence_ms(499)
    with pytest.raises(ValueError):
        validate_vad_silence_ms(2501)
    assert validate_vad_silence_ms(1200) == 1200

    # PipelineWorker clamps configured VAD values to bounds
    worker_low = PipelineWorker(vad_silence_ms=200)
    assert worker_low.vad_silence_ms == 500

    worker_high = PipelineWorker(vad_silence_ms=3500)
    assert worker_high.vad_silence_ms == 2500

    # WorkerRunner test execution
    monkeypatch.setenv("VAD_SILENCE_MS", "800")
    worker = PipelineWorker(dialect="british")
    assert worker.vad_silence_ms == 850
    runner = WorkerRunner(worker)
    assert runner.run(b"audio") == [{"dialect": "british", "vad_silence_ms": 850}]


# ---------------------------------------------------------------------------
# Requirement (a): Simulated voice turn flows through AFSAEvaluationProcessor
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_simulated_voice_turn_flows_through_evaluation_processor(monkeypatch):
    """Verify (a): simulated voice turn flows through AFSAEvaluationProcessor."""
    monkeypatch.setenv("ENABLE_MOCK_CLASSIFIER", "true")

    store = InMemorySessionStore()
    session = store.create("session-turn-flow-test")

    processor = AFSAEvaluationProcessor(
        session_id=session.session_id,
        dialect="aave",
        induction_phrase=DISARMING_INDUCTION_PHRASE,
        session_store=store,
    )

    # Verify disarming session opening ("Hey love")
    assert processor.get_induction_phrase() == "Hey love"

    # Simulate speech frames arriving from VAD / STT
    turn_speech = "Navigating corporate spaces with this code switch is really tiring today"
    await processor.process_frame(TextFrame(turn_speech), FrameDirection.DOWNSTREAM)

    # Simulate turn completion signal (UserTurnCompletionFrame)
    await processor.process_frame(
        UserTurnCompletionFrame(kind="turn_end"), FrameDirection.DOWNSTREAM
    )

    # Wait for non-blocking evaluation task
    await processor.wait_for_evaluations()

    # Verify diagnostic output produced
    assert processor.turns_evaluated == 1
    diag = processor.latest_diagnostic
    assert diag is not None
    assert isinstance(diag, AFSADiagnosticOutput)
    assert diag.cas > 0.0
    assert diag.Choice in {"hold_space_minimal", "gentle_grounding", "dialectal_mirror", "other"}
    assert len(session.diagnostics) == 1
    assert session.diagnostics[0] is diag


@pytest.mark.asyncio
async def test_pipeline_worker_simulate_turn_and_induction(monkeypatch):
    """Verify PipelineWorker simulate_turn helper and disarming induction."""
    monkeypatch.setenv("ENABLE_MOCK_CLASSIFIER", "true")

    worker = PipelineWorker(dialect="neutral")
    assert worker.get_induction_phrase() == "Hey love"

    diag = await worker.simulate_turn("I am doing okay and feel safe")
    assert isinstance(diag, AFSADiagnosticOutput)
    assert diag.Choice == "hold_space_minimal"
    assert diag.cas == 0.25


# ---------------------------------------------------------------------------
# Requirement (d): purge_session() zeroes all in-memory session state
# ---------------------------------------------------------------------------
def test_session_is_ephemeral_and_purgeable():
    """Verify (d): purge_session() zeroes all in-memory session state."""
    store = InMemorySessionStore()
    session = store.create("test-session")
    session.state["secret"] = "temporary_biometric_data"
    session.diagnostics.append(
        AFSADiagnosticOutput(cas=0.5, cpd=0.5, aicl=0.5, noul=0.5, choice="hold_space_minimal")
    )
    session.somatic_telemetry = SomaticTelemetryPayload(pitch=1.0, accel_variance=0.2)

    assert store.get("test-session") is session
    assert len(session.state) == 1
    assert len(session.diagnostics) == 1
    assert session.is_active is True

    # Purge specific session
    assert purge_session("test-session", store) is True
    assert store.get("test-session") is None

    # Prove in-memory state is zeroed out
    assert len(session.state) == 0
    assert len(session.diagnostics) == 0
    assert session.somatic_telemetry is None
    assert session.is_active is False
    assert session.induction_phrase == ""


@pytest.mark.asyncio
async def test_purge_session_bound_to_pipeline_teardown():
    """Verify that pipeline finish/disconnect (EndFrame or worker.close) triggers purge_session()."""
    store = InMemorySessionStore()
    session = store.create("teardown-bound-session")
    session.state["turn_cache"] = "audio_temp"

    processor = AFSAEvaluationProcessor(
        session_id=session.session_id,
        session_store=store,
    )

    # Verify session active before end frame
    assert store.get("teardown-bound-session") is not None

    # Process EndFrame (pipeline finish/disconnect)
    await processor.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)

    # Session must be purged and zeroed out
    assert store.get("teardown-bound-session") is None
    assert len(session.state) == 0
    assert session.is_active is False
