# AFSA Engine Product Requirements Document

AFSA Engine provides a vertical slice for privacy-preserving conversational
assessment.  It accepts text or audio worker input, emits CAS, CPD, AICL, and
somatic payload data, and keeps session state only in process memory.

The implementation is non-diagnostic and follows the Ryan et al. (2005)
continuum framing.  Deployment teams are responsible for consent, retention,
and use policies.
