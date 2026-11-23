# Product Requirements Document (PRD): AFSA-Agent

## Executive Summary
The Acculturation Fluidity & Stress Assessment Agent (AFSA-Agent) is a real-time, voice-first diagnostic engine designed to evaluate human acculturation dynamics—specifically measuring acculturation stress, dynamic code-switching efficiency, and cultural adaptation friction without relying on homogenized baseline psychometrics. By integrating client-side somatic telemetry, a soft-latent memory airlock, and the Google Gemini Decisions API (`GoogleGenAIClassifier`), AFSA-Agent measures the physical and cognitive cost of navigating dominant cultural spaces with typed, probabilistic rigor while preserving neuro-sovereignty.

## Core Product Objectives
- **Domain Isolation:** Exclusively target acculturation measurement (Ryan, Arredondo, & Sandoval, 2005 continua: CAS, CPD, AICL), completely bypassing legacy IQ, youth-safety, and standard psychometric validation traps.
- **Ambient Hardware Mirror / Minimalist Interface:** Function as a headless acoustic and somatic mirror designed for ambient hardware form factors. The sole visual output is an optional, non-distracting 2D responsive colored gradient cosine wave (zero avatars, zero text fluff, zero UI clutter).
- **Encoder Bias Bypassing:** Utilize soft latent memory tokens injected directly into the LLM embedding space to bypass base encoder tokenization penalties on non-standard phonology (e.g., unaccommodated African American Vernacular English).
- **Somatic Signal Decomposition:** Differentiate between biological circadian fatigue and acute acculturation stress by pairing acoustic analysis with client-side kinematic (AirPods CoreMotion) and vision (rPPG/EMG/oculomotor) telemetry.
- **Typed Probabilistic Classification:** Leverage `GoogleGenAIClassifier` (`gemini-decision-1`) to output zero-output-token discrete probabilities, categorical choices, and scored severity levels (`Noul`, `Choice`, `Score`) rather than opaque generative text, with a deterministic offline fallback for local testing.
- **Non-Extractive Telemetry:** Enforce ephemeral, in-memory local session-state management with explicit `purge_session()` teardown and zero persistent external logging.

## User Interface & Experience
- **Ambient Voice-Live Ingestion:** Real-time bidirectional WebRTC / WebSocket audio streaming.
- **Minimalist Visual Feedback:** Optional responsive 2D canvas gradient cosine wave modulated by real-time audio amplitude and frequency spectrum.
- **Zero-Facade Disarming Induction:** The agent initiates interaction without clinical facades or corporate scripts, using a minimal, disarming prompt baseline (`"Hey love"`) to minimize observer-effect masking.
- **No Transcripts / No Avatars:** Complete elimination of visual avatars, text transcripts, and clinical scoring interfaces during evaluation to prevent self-consciousness and anthropomorphic priming.

## Key Functional Requirements
1. **Dynamic VAD & Turn-Taking:** Adaptive Voice Activity Detection (VAD) featuring configurable, dialect-aware silence thresholds (`500ms` – `2500ms`) to accommodate natural turn-taking pauses without premature interruption.
2. **Latent Memory Airlock (Memorilla Layer):** Lightweight `SoftMemoryContext` interface (no-op in Phase 1) scaling to `snap-stanford/memorilla` `MemoryModule` ingestion of community-grounded speech datasets (e.g., Project Elevate Black Voices) to distill speaker dialectal embeddings into 16 soft memory tokens ($m_k$).
3. **Multimodal Somatic Noise Conditioning:** Client-side integration of AirPods CoreMotion (`CMHeadphoneMotionManager`) IMU jaw-clench kinematics (supporting camera-off headless operation), paired with optional WASM/OpenCV facial masseter/corrugator EMG proxies, melanin-inclusive rPPG blood volume pulse tracking, and oculomotor metrics.
4. **Rao-Blackwellized Noise Filtering Engine:** Mathematical conditioning of raw acoustic/prosodic metrics against somatic sufficient statistics ($T$), factoring out baseline circadian exhaustion (nuisance parameter $\eta$) to eliminate false-positive cognitive or cultural deficit flags.
5. **Typed Decision & Metric Classification (`GoogleGenAIClassifier`):** Parallel evaluation of Cultural Adaptation Stress (CAS), Code-Switching Prosodic Dexterity (CPD), and Acculturation-Induced Cognitive Load (AICL) via LangChain's `GoogleGenAIClassifier` (`Noul`, `Choice`, `Score`) with a deterministic offline mock fallback when no API key is configured.
6. **Variance & Error Margin Engine:** Explicit calculation and output of confidence bounds, hardware jitter, and codec/environmental error margins alongside every diagnostic vector.
7. **Client-Side Privacy Airlock:** Local execution of biometric/kinematic telemetry, ephemeral in-memory session storage, explicit `purge_session()` memory wipe on disconnect, and zero raw audio/video/PII persistence.
