# SpeakerID Phase 04 — Persistent Streamer Intelligence

> **For Hermes:** Load only this phase doc plus `INDEX.md` unless you need cross-phase details. Use `subagent-driven-development` to execute one task at a time.

**Goal:** Add reusable per-streamer intelligence profiles: voice profile refs, personality/lore, common chatters, guests, inside jokes, clip-quality lessons, compact prompt context, and post-VOD update proposals.

**Depends on:** Phase 02 voice profiles and Phase 03 prompt integration points.

**Tasks covered:** 14–19.

## Current status (May 24, 2026)

- ✅ **Task 14 complete:** persistent intelligence type contracts in `src/intelligence/types.py`.
- ✅ **Task 15 complete:** profile/observation storage APIs in `src/intelligence/streamer_store.py`.
- ✅ **Task 16 complete:** compact profile context renderer + Stage-1 prompt injection, with CLI enable flag wiring in synthesis.
- ✅ **Task 17 complete:** helper-level proposal logic plus end-to-end pipeline orchestration (proposal file emission and auto-merge wiring).
- ✅ **Task 18 complete:** persistent voice-profile reuse wired into speaker recognition flow (path-based profile loading + attribution passthrough + CLI hook) with focused tests.
- ✅ **Task 19 complete:** persistent-intelligence CLI/pipeline mode wiring across prep/synthesis/validation entry points.

---

### Task 14: Add persistent streamer intelligence models

**Objective:** Define strict profile/observation contracts before building storage or prompt injection.

**Files:**
- Modify: `src/preprocessing/types.py` or create `src/intelligence/types.py`
- Test: `tests/test_streamer_intelligence_types.py`

**Models to add:**
- `StreamerProfile`
- `VoiceProfileRef`
- `StreamerObservation`
- `PersonalityTrait`
- `CommunityChatterSummary`
- `InsideJoke`
- `ContentPattern`
- `ProfileUpdateProposal`

**Validation rules:**
- Every durable claim must include `confidence`, `evidence_refs`, and `updated_at` or `created_at`.
- Observations must include `vod_id`, `timestamp_start`, `timestamp_end`, `type`, `claim`, `evidence`, and `source`.
- Reject confidence outside `[0, 1]` and invalid timestamp ranges.

### Task 15: Implement streamer intelligence storage

**Objective:** Load/save per-streamer profiles and append immutable observation records.

**Files:**
- Create: `src/intelligence/streamer_store.py`
- Test: `tests/test_streamer_store.py`

**Public API:**
```python
def resolve_streamer_id(vod_meta: dict, override: str | None = None) -> str: ...
def load_streamer_profile(streamer_id: str, root: Path) -> StreamerProfile: ...
def save_streamer_profile(profile: StreamerProfile, root: Path) -> Path: ...
def append_observations(streamer_id: str, observations: list[StreamerObservation], root: Path) -> Path: ...
def load_recent_observations(streamer_id: str, root: Path, limit: int = 200) -> list[StreamerObservation]: ...
```

**Implementation notes:**
- Use atomic writes for `profile.json` (`tmp` file then rename).
- Append observations as JSONL.
- Add a simple file lock or lockfile to prevent concurrent profile writes.
- Create missing profiles with empty sections and `profile_version=1`.

### Task 16: Render compact streamer profile context for prompts

**Objective:** Give Stage 1/3 useful historical context without flooding Qwen or overriding current evidence.

**Files:**
- Create: `src/intelligence/profile_context.py`
- Modify: `src/synthesis/qwen_clip_analyzer_progressive.py`
- Test: `tests/test_profile_context.py`

**Function:**
```python
def render_streamer_profile_context(profile: StreamerProfile, max_chars: int = 2000) -> str: ...
```

**Rules:**
- Include only facts with `confidence >= 0.65` and non-empty evidence refs.
- Prioritize voice profile refs, known guests, high-confidence inside jokes, high-value/low-value clip lessons, and common chatters seen recently.
- Always label the block as `evidence-backed, advisory`.
- Add a conflict rule: current VOD evidence beats profile context.

### Task 17: Generate profile-update proposals after each VOD

**Objective:** Convert run output into candidate persistent observations with evidence.

**Files:**
- ✅ Created: `src/intelligence/profile_update.py`
- ✅ Added: `tests/test_profile_update.py`
- ✅ Integrated runtime wiring: `src/synthesis/qwen_clip_analyzer_progressive.py` (proposal emission + auto-merge orchestration)

**Inputs:**
- `qwen_vision_progressive.json`
- `speaker_attribution_<VOD_ID>.json` when present
- `fusion_result_<VOD_ID>.json` / chat summary
- existing `StreamerProfile`

**Outputs:**
- ✅ Candidate observations can now be generated deterministically from final clip payloads.
- ✅ `profile_update_proposal_<VOD_ID>.json` is written by synthesis runtime when profile updates are enabled.
- ✅ `auto` mode merges accepted observations and persists updates to `profile.json` + `observations.jsonl`.

**Proposal schema:**
```json
{
  "vod_id": "2776101332",
  "streamer_id": "skitch",
  "candidate_observations": [
    {
      "type": "inside_joke",
      "claim": "The abrasive donation alert is a recurring community bit.",
      "confidence": 0.86,
      "evidence": ["..."],
      "source_clip_ids": ["120-240"],
      "promote_to_profile": true
    }
  ]
}
```

**Merge policy:**
- Auto-accept observations with confidence `>=0.80`, at least two evidence lines, and no conflict.
- Queue `0.60-0.80` observations in proposal file but do not promote unless repeated in later VODs or manually approved.
- Never persist sensitive personal details unless explicitly approved.
- De-duplicate by normalized claim text + type + nearby evidence.

**Current implementation note:**
- `src/intelligence/profile_update.py` now covers deterministic helper logic (claim normalization, sensitivity checks, dedupe, partitioning, proposal assembly) and auto-merge application helpers.
- `src/synthesis/qwen_clip_analyzer_progressive.py` now orchestrates proposal write-out and optional auto-merge persistence (`append_observations` + `save_streamer_profile`) when enabled.

### Task 18: Use persistent voice profiles in speaker recognition

**Objective:** Reuse streamer voice thumbprints across VODs without manually passing profile paths every time.

**Files:**
- ✅ Modified: `src/preprocessing/speaker_profiles.py` (`load_profiles_from_paths`)
- ✅ Modified: `src/preprocessing/run_speaker_attribution.py` (CLI hook + runtime profile merge)
- ✅ Modified: `src/intelligence/streamer_store.py` (`load_persistent_voice_profiles`)
- ✅ Modified: `src/preprocessing/speaker_attribution.py` (explicit `profiles` passthrough)
- ✅ Tests: `tests/test_speaker_profiles.py`, `tests/test_streamer_store.py`, `tests/test_speaker_attribution.py`

**Behavior:**
- Resolve `streamer_id` first.
- Load `profile.voice_profiles[*].path` and pass those profiles into speaker recognition.
- When manual enrollment creates a new profile, optionally attach it to `StreamerProfile.voice_profiles` with evidence refs.
- If no voice profile exists, run diarization/name inference but mark streamer recognition unavailable.

### Task 19: Add persistent-intelligence CLI flags and pipeline integration

**Objective:** Wire persistent profile load/update into normal phase4/synthesis runs while keeping it optional.

**Implementation status:** ✅ Complete (CLI flags + runtime wiring in prep/synthesis/validation entry points).

**Files:**
- Modify: `src/preprocessing/prepare_phase4.py`
- Modify: `src/synthesis/qwen_clip_analyzer_progressive.py`
- Modify: `src/preprocessing/validate_phase4_inputs.py`
- Test: `tests/test_phase4_validation.py`, `tests/test_prompt_templates.py`

**Flags:**
```bash
--streamer-id <id>
--profile-root data/streamer_intelligence
--enable-persistent-intelligence
--update-streamer-profile
--profile-update-mode propose|auto|off
```

**Integration points:**
1. During phase4 prep, resolve streamer ID and validate/create profile directory.
2. During speaker attribution, load persistent voice profile refs.
3. Before Stage 1, render compact `STREAMER PROFILE CONTEXT` into analysis prompts.
4. After final output, generate `profile_update_proposal_<VOD_ID>.json`.
5. If mode is `auto`, merge accepted high-confidence observations.
