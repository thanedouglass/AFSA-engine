# Implementation Plan: Acculturation Assessment Agent (AFSA-Engine)

> **Boundary Note:** This plan governs the `afsa-engine` Python codebase only. Do **not** fork Pipecat—install `pipecat-ai` as a standard dependency in `pyproject.toml`. The 4-agent Gemini operations workflow (Coordinator, Literature Synthesizer, Telemetry Triager, AfriCHI Architect) lives in `docs/OPS_WORKFLOW.md` and is **not** implemented in Python.

## Phase 1: Repo Foundation & Voice + Decisions Vertical Slice
1. **Repository & Dependency Scaffold:** Finalize `pyproject.toml` (`pipecat-ai`, `langchain`, `langchain-google-genai>=4.5.0`, `google-genai>=3.2.0`, `pydantic`, `pytest`), `.env.example`, and `docs/` (`PRD.md`, `RTM.csv`, `PLAN.md`).
2. **Scope & Privacy Documentation:** Document Ryan, Arredondo, & Sandoval (2005) continua (`CAS`, `CPD`, `AICL`), headless mirror stance (`"Hey love"`), and zero-persistence privacy rules in `README.md`.
3. **Bounded Schemas (`afsa/schemas.py`):** Define typed Pydantic models for `CAS`, `CPD`, `AICL`, somatic/CoreMotion IMU payloads (`eta` hook), `SoftMemoryContext` (`m_k` no-op protocol), and explicit variance/error bounds.
4. **Decisions Classifier (`afsa/classifier.py`):** Implement `GoogleGenAIClassifier(model="gemini-decision-1")` (`Noul`, `Choice`, `Score`) with a deterministic offline mock fallback when `GOOGLE_API_KEY` is unset or `ENABLE_MOCK_CLASSIFIER=true`.
5. **Dialect-Aware VAD & Worker Lifecycle (`afsa/pipeline.py`):** Wire Pipecat `PipelineWorker`, `WorkerRunner`, and `PipelineParams` with configurable VAD silence buffers (`500ms–2500ms`), disarming opening (`"Hey love"`), and a non-blocking `AFSAEvaluationProcessor(FrameProcessor)`.
6. **Ephemeral Sessions & Purge (`afsa/session.py`):** Implement in-memory session state with `purge_session()` bound to pipeline teardown/disconnect and zero external persistence.
7. **Vertical-Slice Validation (`tests/test_afsa_vertical_slice.py`):** Verify a simulated voice turn end-to-end through `AFSAEvaluationProcessor`, structured `Noul`/`Choice`/`Score` outputs + error bounds, and confirmed state wipe via `purge_session()`.

## Phase 2: Somatic Conditioning & CoreMotion / WASM Telemetry
- Build local AirPods CoreMotion (`CMHeadphoneMotionManager`) IMU jaw-clench bridge (`127.0.0.1` socket/WebSocket) alongside optional client-side WASM rPPG/HRV and facial EMG proxies.
- Implement the Rao-Blackwellized conditioning layer ($\tilde{\theta} = E[\hat{\theta} \mid T]$) to separate baseline circadian fatigue ($\eta$) from acute acculturation stress signals.

## Phase 3: Memorilla Soft-Token Airlock & AAE Prosody Grounding
- Connect `snap-stanford/memorilla` (`MemoryModule`) and `my_data/data/aae_prosody/` corpora (Project Elevate Black Voices) into the `SoftMemoryContext` slot ($m_k$) once Phase 1–2 turn-taking and scoring are stable.

## Phase 4: Interactive Test Harness & AfriCHI 2027 Submission Package (Due Dec 14, 2026 AoE)
- Run repeatable interactive sessions, hand sanitized JSON summaries to the Gemini Ops Loop (`afsa-telemetry-triager` -> Google Sheets), and harden demo edge cases for the in-person University of Lagos workshop.
- Finalize the AfriCHI 2027 workshop submission (title, abstract, subject areas, positionality statement, ≤250-word participation call, and ≤2,500-word proposal PDF covering all 10 required workshop elements).
