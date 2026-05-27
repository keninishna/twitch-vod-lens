# SpeakerID / Persistent Streamer Intelligence Plan — Index

> **For Hermes:** This is the context-light hub. Load only the phase doc you are actively executing, plus this index. Avoid loading the old full plan into context unless auditing cross-phase consistency.

**Goal:** Add speaker attribution and persistent streamer intelligence to VOD Lens so future VOD analyses can reuse learned voice thumbprints, streamer personality/context, recurring chatters, community in-jokes, and prior evidence.

**Core policy:** Speaker attribution is prompt context for Qwen inference and title/report accuracy. Do **not** add deterministic Stage-2 speaker-specific penalties or hard gates unless Ken explicitly asks for that later.

---

## Phase Dependency Graph

```text
Phase 00 — Overview / contracts
   │
   v
Phase 01 — Foundations / diarization / transcript alignment
   │
   v
Phase 02 — Voiceprints / speaker recognition / name inference / artifact
   │
   v
Phase 03 — Phase4 + clip context + Qwen speaker-framing + outputs
   │
   ├──────────────┐
   v              v
Phase 04 — Persistent streamer intelligence
   │
   v
Phase 05 — Docs / WSL validation / rollout
```

---

## Document Map

| Phase | File | Load when... |
|---|---|---|
| 00 | `docs/plans/speakerid/00-overview-and-contracts.md` | You need architecture, research summary, data contracts, and dynamic name-inference rules. |
| 01 | `docs/plans/speakerid/01-foundations-diarization-alignment.md` | You are adding dependencies, Pydantic models, audio extraction, pyannote diarization, or transcript alignment. |
| 02 | `docs/plans/speakerid/02-voiceprints-recognition-artifact.md` | You are implementing voice enrollment, ECAPA recognition, text-based speaker name inference, or `speaker_attribution_<VOD_ID>.json`. |
| 03 | `docs/plans/speakerid/03-pipeline-context-qwen-framing.md` | You are integrating speaker attribution into phase4 prep, clip context, Qwen prompts, or final selected clip outputs. |
| 04 | `docs/plans/speakerid/04-persistent-streamer-intelligence.md` | You are building `data/streamer_intelligence/<streamer_id>/`, profile context rendering, or profile update proposals. |
| 05 | `docs/plans/speakerid/05-docs-validation-rollout.md` | You are writing runbooks, WSL validation scripts, streamer-id guardrails, preprocessing hardening, Bee/extraction reliability fixes, or rollout pitfalls. |

---

## Execution Guidance

1. Start with Phase 00 once to understand contracts.
2. Execute phases in order unless doing docs-only work.
3. Within each phase, execute one task at a time and commit after each task.
4. Use focused tests listed in the active phase.
5. Do not load every phase into context for normal implementation.
6. After any phase doc changes, update this index if file names, dependencies, or scope changed.

---

## Quick Start for an AI Agent

```text
Load:
1. docs/plans/speakerid.md
2. One phase file from docs/plans/speakerid/
3. docs/CLIP-INTELLIGENCE-PIPELINE.md only if you need current pipeline contract context

Then implement only the active phase's next unchecked task.
```

---

## Current Implementation Status (May 27, 2026 — updated)

### Completed

- **Phase 01 (Tasks 1–5): complete**
  - Added optional SpeakerID dependency manifest + gitignore rules.
  - Added Pydantic contracts in `src/preprocessing/types.py`:
    - `SpeakerTurn`, `SpeakerRecognitionResult`, `SpeakerNameCandidate`, `SpeakerClusterSummary`, `ClipSpeakerStats`, `SpeakerAttributionResult`.
  - Added audio extraction utilities: `src/preprocessing/audio_segments.py`.
  - Added pyannote diarization backend wrapper: `src/preprocessing/speaker_diarization.py`.
  - Added transcript alignment utilities: `src/preprocessing/speaker_alignment.py`.

- **Phase 02 (Tasks 6–9): complete**
  - Voice profile enrollment implemented:
    - `src/preprocessing/speaker_profiles.py`
    - `src/preprocessing/speaker_enroll.py`
  - Speaker recognition against profiles implemented:
    - `src/preprocessing/speaker_recognition.py`
  - Text-based name inference implemented:
    - `src/preprocessing/speaker_name_inference.py`
  - Artifact orchestrator + CLI implemented:
    - `src/preprocessing/speaker_attribution.py`
    - `src/preprocessing/run_speaker_attribution.py`
    - `tests/test_speaker_attribution.py`

- **Phase 03 (Tasks 10–13): complete**
  - Integrated optional speaker attribution into phase4 prep and validation:
    - `src/preprocessing/prepare_phase4.py`
    - `src/preprocessing/validate_phase4_inputs.py`
    - flags: `--enable-speaker-id`, `--speaker-profiles-dir`, `--require-speaker-id`
  - Extended clip context with speaker stats and prompt warning rendering:
    - `src/synthesis/clip_context.py`
    - `src/synthesis/schemas/clip_intelligence_stages.py`
  - Added Stage 1/Stage 3 speaker-framing inference prompt contract:
    - `src/synthesis/qwen_clip_analyzer_progressive.py`
  - Preserved per-clip speaker attribution in final selected output:
    - `src/synthesis/title_dedup.py`
    - `src/synthesis/schemas/clip_intelligence_stages.py`

### In Progress / Pending

- **Phase 04: complete (Tasks 14–19)**
  - **Task 14 complete:** persistent intelligence data contracts added (`src/intelligence/types.py`, `tests/test_streamer_intelligence_types.py`).
  - **Task 15 complete:** streamer profile store APIs added (`src/intelligence/streamer_store.py`, `tests/test_streamer_store.py`).
  - **Task 16 complete:** compact profile-context renderer + Stage-1 prompt injection with CLI enable wiring (`src/intelligence/profile_context.py`, prompt wiring in `qwen_clip_analyzer_progressive.py`, `tests/test_profile_context.py`, prompt-template updates).
  - **Task 17 complete:** proposal helpers + end-to-end proposal emission/auto-merge orchestration (`src/intelligence/profile_update.py`, `tests/test_profile_update.py`, synthesis runtime wiring).
  - **Task 18 complete:** persistent voice-profile reuse wired into recognition flow (`load_profiles_from_paths`, `load_persistent_voice_profiles`, `generate_speaker_attribution(..., profiles=...)`, CLI hook in `run_speaker_attribution.py`) with focused tests.
  - **Task 19 complete:** full CLI/pipeline flag wiring for persistent-intelligence modes across `prepare_phase4.py`, `qwen_clip_analyzer_progressive.py`, and `validate_phase4_inputs.py`.

- **Phase 05: in progress (Tasks 20–28)**
  - **Task 22 complete:** streamer-id metadata resolution + override mismatch guardrail is implemented (`resolve_streamer_id_context`) and wired through prep/validation/synthesis outputs.
  - **Task 23 complete:** WSL artifact-first validation harness is implemented (`scripts/validate_persistent_intelligence_wsl.sh`).
  - Remaining priority items: **Task 20**, **Task 25**, **Task 26**, **Task 27**, **Task 28**.
  - **Task 24** is largely complete in code via typed `fuse(...)` + retained legacy `fuse_signals(...)`; keep regression checks and cleanup follow-through as needed.

### Verification snapshot

Focused SpeakerID + Phase-03 integration tests passing:

```bash
pytest -q \
  tests/test_phase4_validation.py \
  tests/test_clip_context.py \
  tests/test_prompt_templates.py \
  tests/test_stage_schemas.py \
  tests/test_title_dedup.py \
  tests/test_stage1_discovery.py
```

Result: **27 passed**.

Phase-04 incremental checks (Task 17 helper slice + Task 18 voice-profile reuse wiring):

```bash
pytest -q \
  tests/test_profile_update.py \
  tests/test_speaker_profiles.py \
  tests/test_streamer_store.py \
  tests/test_speaker_attribution.py
```

Result: **21 passed**.

SpeakerID foundation suite (previous checkpoint):

```bash
pytest -q \
  tests/test_speaker_attribution.py \
  tests/test_speaker_attribution_types.py \
  tests/test_audio_segments.py \
  tests/test_speaker_diarization.py \
  tests/test_speaker_alignment.py \
  tests/test_speaker_profiles.py \
  tests/test_speaker_recognition.py \
  tests/test_speaker_name_inference.py
```

Result: **29 passed, 1 skipped**.

---

## Next Implementation Plan for AI Agent (May 27, 2026)

> **For Hermes / future agent:** Load this hub plus `docs/plans/speakerid/05-docs-validation-rollout.md`. Execute one task at a time, write/adjust tests first, and commit after each green task. Do not add deterministic Stage-2 speaker-specific penalties or hard gates; attribution remains prompt context and reporting evidence.

### Execution order

1. **Task 20 — Runbook documentation artifacts**
   - Add/update operator docs and create the missing references:
     - `docs/references/speaker-attribution.md`
     - `docs/references/persistent-streamer-intelligence.md`
   - Keep policy explicit: speaker attribution is prompt/report context, not deterministic Stage-2 speaker gating.

2. **Task 24 — Modern preprocessing contract cleanup follow-through**
   - Keep `fuse(...)` as canonical typed path and `fuse_signals(...)` as legacy compatibility.
   - Add/keep focused regression checks so phase4 prep does not depend on fallback behavior.

3. **Task 25 — Phase4 manifest quality hardening**
   - Upgrade `prepare_phase4.py` manifest generation from baseline deterministic windows to YOLO-aware candidate scoring/ranking.
   - Keep output schema compatible with `clip_manifest.json` consumers.

4. **Task 26 — Reproducible preprocessing/runtime environment contract**
   - Document and pin which steps run in Docker vs local Python.
   - Add missing requirements/notes so future WSL validation does not depend on ad-hoc installs.

5. **Task 27 — Bee startup/health reliability**
   - Add a safe managed-start/preflight path for Bee on port `8082`, or a clear opt-in `--start-bee`/env command, so analysis does not silently start before the vision backend is ready.

6. **Task 28 — Raw VOD path canonicalization for extraction/upload**
   - Make extraction resolve the raw MP4 from phase4 metadata/standard paths instead of fragile `raw/<VOD_ID>.mp4` assumptions.
   - Add dry-run tests for missing/raw path resolution.

### Required verification before declaring complete

Run the focused local suites relevant to touched files, then perform one WSL validation pass:

```bash
pytest -q \
  tests/test_streamer_store.py \
  tests/test_phase4_validation.py \
  tests/test_profile_context.py \
  tests/test_profile_update.py \
  tests/test_clip_context.py \
  tests/test_stage_schemas.py
```

For WSL validation, prove at minimum:
- `qwen_vision_progressive.json` contains resolved streamer metadata and, when enabled, a `profile_update` block.
- `profile_update_proposal_<VOD_ID>.json` is written under the metadata-derived streamer profile unless an explicit override was used.
- If override conflicts with metadata, logs/output contain a visible warning and record both IDs.
- Final selected clips still preserve `speaker_attribution` audit payloads.
- No profile observation is promoted without confidence and evidence refs.

---

## Current Split Status

The original monolithic `speakerid.md` plan has been split into phase documents to reduce context load. This hub is the canonical entry point.
