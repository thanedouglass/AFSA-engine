# AFSA Engine (Acculturation Fluidity & Stress Assessment)

`afsa-engine` is a privacy-preserving, real-time conversational assessment runtime vertical slice. It evaluates human acculturation dynamics without relying on homogenized baseline psychometrics or deficit models.

The engine strictly targets the **Ryan, Arredondo, & Sandoval (2005)** continua (`REQ-ACC-01`):

- **CAS (Cultural Adaptation Stress):** Evaluates acute adaptation tension, cultural dissonance, and systemic navigating strain.
- **CPD (Code-Switching Prosodic Dexterity):** Measures the cognitive and prosodic cost of navigating between non-standard (e.g., AAVE) and dominant cultural dialects.
- **AICL (Acculturation-Induced Cognitive Load):** Quantifies processing saturation, hesitation latency, and communicative strain induced by real-time cultural translation.

## Headless Mirror Stance (`REQ-IND-01`)

AFSA-Engine operates as a minimalist, ambient acoustic and somatic mirror:
- **Zero Avatars & Zero Transcript UI:** Complete elimination of anthropomorphic visual avatars and real-time transcripts to prevent self-consciousness and observer-effect masking.
- **Disarming Induction Opening:** Sessions initiate with a minimal, disarming vocal opening (`"Hey love"`) rather than corporate or clinical scripts.
- **Dialect-Aware Adaptive VAD (`REQ-VCE-02`):** Trailing silence thresholds are dynamically calibrated within **500ms – 2500ms** to respect natural dialectal pauses without premature turn cuts.

## Privacy & Neuro-Sovereignty (`REQ-PRIV-01`)

- **Ephemeral In-Memory State:** All session telemetry, kinematic states, and diagnostic outputs exist solely in volatile RAM.
- **Zero Persistent External Logging:** Raw audio, video frames, transcripts, and PII are never persisted to disk or external stores by default.
- **Explicit `purge_session()` Teardown:** Bound to pipeline disconnect and termination (`EndFrame`, `CancelFrame`), immediately wiping all in-memory buffers and zeroing session state.

## Architecture & Core Modules

- `afsa/schemas.py`: Typed Pydantic models for `SomaticTelemetryPayload` (AirPods CoreMotion IMU jaw-clench kinematics, optional rPPG/EMG, and circadian exhaustion nuisance parameter $\eta$), `SoftMemoryContext` (16-token $m_k$ metadata protocol; Phase 3 hook), and `AFSADiagnosticOutput` (bounded `CAS`, `CPD`, `AICL` scores, `Noul` fatigue-vs-stress gate probability, `Choice` mirror posture route, and explicit confidence/error bounds `REQ-VAR-01`).
- `afsa/classifier.py`: Google Gemini Decisions API integration (`GoogleGenAIClassifier(model="gemini-decision-1")`) utilizing `Noul`, `Choice`, and `Score` from `langchain_core.decisions`, with a deterministic offline mock fallback when `GOOGLE_API_KEY` is unset or `ENABLE_MOCK_CLASSIFIER=true`.
- `afsa/pipeline.py`: Real-time voice core featuring `PipelineWorker` managed by `WorkerRunner` (`PipelineParams`), configurable dialect-aware VAD (`500ms–2500ms`), disarming `"Hey love"` induction, and non-blocking `AFSAEvaluationProcessor(FrameProcessor)` on turn completion.
- `afsa/session.py`: Thread-safe `InMemorySessionStore` with explicit zeroing via `purge_session()`.

## Configuration

Copy `.env.example` to `.env`:

```bash
# Optional Google GenAI credentials (for live gemini-decision-1)
GOOGLE_API_KEY=

# Voice activity detection trailing silence (valid range: 500-2500 ms)
VAD_SILENCE_MS=1200

# Use deterministic local/offline classification instead of live Google GenAI
ENABLE_MOCK_CLASSIFIER=true
```

## Running Tests

Run the full vertical slice test suite using `pytest`:

```bash
uv run pytest -v
```
