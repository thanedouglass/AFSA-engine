# AFSA Engine

AFSA is a privacy-preserving, real-time conversational assessment scaffold.  It
uses the Ryan et al. (2005) continua to organize language output along three
axes:

- **CAS** — cognitive/affective/self-regulatory state;
- **CPD** — communicative participation and dialogue development; and
- **AICL** — adaptive interaction and conversational learning.

The project is designed as a vertical slice rather than a clinical or diagnostic
instrument.  It does not infer medical diagnoses and must not be used as a
stand-alone basis for treatment decisions.

## Privacy

Sessions are ephemeral and held in memory only.  `purge_session()` removes a
session and all of its payloads.  Audio and transcript payloads are not written
to disk by the engine, and deployments should disable provider-side retention
for keys used with the optional Google GenAI classifier.

## Configuration

Copy `.env.example` to `.env` and provide provider credentials as needed.
Set `VAD_SILENCE_MS` to a value between 500 and 2500 milliseconds.
Set `ENABLE_MOCK_CLASSIFIER=true` for deterministic offline operation.