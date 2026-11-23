from afsa.classifier import Classification, GoogleGenAIClassifier
from afsa.pipeline import PipelineWorker, WorkerRunner, dialect_vad_silence_ms
from afsa.schemas import AICL, CAS, CPD, CoreMotion, ErrorBounds
from afsa.session import InMemorySessionStore, purge_session


def test_payloads_serialize_and_bounds_validate():
    bounds = ErrorBounds(lower=0.1, upper=0.9)
    assert bounds.to_dict()["upper"] == 0.9
    assert CAS(score=0.6, error_bounds=bounds).to_dict()["score"] == 0.6
    assert CPD(choice=0.7).to_dict()["choice"] == 0.7
    assert AICL(core_motion=CoreMotion(activation=0.4)).to_dict()["core_motion"]["activation"] == 0.4

    try:
        CAS(score=1.1)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range score must fail")


def test_offline_classifier_is_deterministic(monkeypatch):
    monkeypatch.setenv("ENABLE_MOCK_CLASSIFIER", "true")
    classifier = GoogleGenAIClassifier()
    result = classifier.classify("I want to choose yes")
    assert isinstance(result, Classification)
    assert result.to_dict() == {"Noul": "noul-positive", "Choice": 0.8, "Score": 0.7}


def test_worker_runner_and_vad_bounds(monkeypatch):
    monkeypatch.setenv("VAD_SILENCE_MS", "300")
    assert dialect_vad_silence_ms("neutral") == 500
    monkeypatch.setenv("VAD_SILENCE_MS", "800")
    worker = PipelineWorker(dialect="british")
    assert worker.vad_silence_ms == 850
    runner = WorkerRunner(worker)
    assert runner.run(b"audio") == [{"dialect": "british"}]


def test_session_is_ephemeral_and_purgeable():
    store = InMemorySessionStore()
    session = store.create("test-session")
    session.state["secret"] = "temporary"
    assert store.get("test-session") is session
    assert purge_session("test-session", store) is True
    assert store.get("test-session") is None
