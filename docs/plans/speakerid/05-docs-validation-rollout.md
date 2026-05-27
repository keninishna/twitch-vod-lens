# SpeakerID Phase 05 — Documentation, WSL Validation, Rollout, and Pitfalls

> **For Hermes:** Load only this phase doc plus `INDEX.md` unless you need cross-phase details. Use `subagent-driven-development` to execute one task at a time.

**Goal:** Document operator workflows, validate on WSL2, run focused tests, and track rollout pitfalls/future enhancements.

**Depends on:** Phases 01–04.

**Tasks covered:** 20–28 plus test matrix, operational pitfalls, rollout hardening, and next implementation slice.

---

### Task 20: Add runbook documentation

**Objective:** Document setup, enrollment, run commands, and failure modes.

**Files:**
- Modify: `docs/CLIP-INTELLIGENCE-PIPELINE.md`
- Create: `docs/references/speaker-attribution.md`
- Create: `docs/references/persistent-streamer-intelligence.md`

**Docs must include:**
- HF token setup (`HF_TOKEN` / `HUGGINGFACE_TOKEN`).
- How to enroll streamer voice.
- How to run speaker attribution for a VOD.
- How persistent streamer intelligence is loaded before analysis and updated after analysis.
- How Qwen speaker-framing inference works without deterministic speaker penalties/gates.
- How dynamic name inference works and its limitations.
- Privacy note: voice profiles are biometric-ish artifacts; profile facts must be evidence-backed and sensitive personal details should not be persisted without approval.

### Task 21: Add WSL2 validation workflow

**Objective:** Validate on the real environment, not only unit tests.

**Files:**
- Create: `scripts/validate_speakerid_wsl.sh` or document commands in `docs/references/speaker-attribution.md`

**Commands:**
```bash
cd ~/twitch-vod-analyzer
export HF_TOKEN=<token>
PYTHONPATH=. python3 -m src.preprocessing.run_speaker_attribution \
  --vod-id <KNOWN_VOD_ID> \
  --vod-media vods/phase4_<KNOWN_VOD_ID>/raw/<KNOWN_VOD_ID>.mp4 \
  --transcript vods/phase4_<KNOWN_VOD_ID>/transcript.json \
  --chat vods/phase4_<KNOWN_VOD_ID>/chat.json \
  --profiles-dir data/streamer_intelligence/<STREAMER_ID>/voice_profiles \
  --output vods/phase4_<KNOWN_VOD_ID>/speaker_attribution_<KNOWN_VOD_ID>.json

PYTHONPATH=. python3 src/synthesis/qwen_clip_analyzer_progressive.py \
  --vod-id <KNOWN_VOD_ID> --skip-audio \
  --streamer-id <STREAMER_ID> \
  --profile-root data/streamer_intelligence \
  --enable-persistent-intelligence \
  --profile-update-mode propose
```

**Acceptance criteria:**
- `speaker_attribution_<VOD_ID>.json` exists and validates.
- At least one speaker cluster is recognized as streamer when a profile exists.
- Clip contexts include speaker stats.
- Qwen output contains `speaker_framing_assessment` / attribution-risk fields where expected.
- Final selected clips include `speaker_attribution` blocks.
- `data/streamer_intelligence/<STREAMER_ID>/profile_update_proposal_<VOD_ID>.json` exists in propose mode.
- No profile fact is promoted without evidence refs.


---

### Task 22: Add streamer-ID metadata resolution and override guardrail

**Objective:** Prevent persistent profile pollution by resolving `streamer_id` from VOD metadata by default and treating `--streamer-id` as an explicit override with mismatch warnings.

**Why this is first:** Real-run evidence showed VOD `2776101332` belonged to `LostGirls27`, but the run used manual `--streamer-id asyajade`. That can write profile proposals/observations into the wrong streamer directory.

**Files:**
- Modify: `src/intelligence/streamer_store.py`
- Modify: `src/preprocessing/prepare_phase4.py`
- Modify: `src/preprocessing/validate_phase4_inputs.py`
- Modify: `src/synthesis/qwen_clip_analyzer_progressive.py`
- Test: `tests/test_streamer_store.py`
- Test: `tests/test_phase4_validation.py`
- Test: `tests/test_profile_update.py` if proposal/output metadata shape changes

**Implementation steps:**
1. Extend streamer ID resolution helpers in `src/intelligence/streamer_store.py`:
   - Keep `_sanitize_streamer_id(...)` as the canonical sanitizer.
   - Add or refactor into a helper that returns metadata-derived ID separately from override-derived ID, for example:
     ```python
     @dataclass(frozen=True)
     class ResolvedStreamerId:
         streamer_id: str
         source: Literal["metadata", "override", "unknown"]
         metadata_streamer_id: str | None = None
         override_streamer_id: str | None = None
         mismatch_warning: str | None = None
     ```
   - Metadata keys should include, in priority order where available: `streamer_id`, `streamer`, `channel_login`, `channel`, `channel_name`, `user_login`, `uploader_id`, `uploader`, `owner`.
   - Default behavior with no override: use metadata-derived ID, else `unknown_streamer`.
   - Behavior with override: use sanitized override, but if metadata-derived ID exists and differs, populate `mismatch_warning`.
2. Wire the resolver into phase4 prep:
   - Read `vod_meta` from fusion/preprocessing outputs before profile path creation.
   - Record resolved streamer metadata in any phase4 validation/prep summary output/log.
   - If mismatch occurs, print a clear warning containing both IDs and the source of the override.
3. Wire the resolver into synthesis:
   - Load `vod_meta` from `fusion_result_<VOD_ID>.json` before persistent intelligence profile load/update.
   - Use metadata-derived ID when `--streamer-id` is omitted.
   - If `--streamer-id` conflicts with metadata, keep the explicit override but write a warning into `qwen_vision_progressive.json` under a stable metadata block, for example:
     ```json
     {
       "streamer_identity": {
         "streamer_id": "asyajade",
         "source": "override",
         "metadata_streamer_id": "lostgirls27",
         "override_streamer_id": "asyajade",
         "mismatch_warning": "--streamer-id override differs from VOD metadata"
       }
     }
     ```
4. Ensure profile update output follows the resolved identity:
   - `profile_update_proposal_<VOD_ID>.json` should include the resolved `streamer_id`, source, and mismatch warning if any.
   - Auto-merge should write under `data/streamer_intelligence/<resolved_streamer_id>/` only.

**Tests to add/update:**
- `resolve_streamer_id` returns metadata ID when no override is supplied.
- Override uses override ID but records mismatch when metadata differs.
- Override matching metadata does not warn.
- Missing metadata + no override returns `unknown_streamer`.
- Phase4 validation can assert expected streamer ID when persistent intelligence is enabled.

**Verification commands:**
```bash
pytest -q tests/test_streamer_store.py tests/test_phase4_validation.py tests/test_profile_update.py
```

**Acceptance criteria:**
- A LostGirls27 VOD with no override writes/loads profile data under `data/streamer_intelligence/lostgirls27/`.
- A LostGirls27 VOD with `--streamer-id asyajade` logs and persists a mismatch warning.
- No persistent profile proposal/auto-merge happens without a visible resolved streamer identity in run output.

---

### Task 23: Add WSL persistent-intelligence validation harness

**Objective:** Provide a repeatable real-environment validation path that checks artifacts, not just process logs.

**Files:**
- Create: `scripts/validate_persistent_intelligence_wsl.sh` or extend `scripts/validate_speakerid_wsl.sh`
- Modify: `docs/references/persistent-streamer-intelligence.md` if created by Task 20
- Modify: `docs/CLIP-INTELLIGENCE-PIPELINE.md` only if operator commands change

**Implementation steps:**
1. Script should accept: `VOD_ID`, optional `STREAMER_ID`, `PROFILE_ROOT`, `MODE=propose|auto`, and `SKIP_AUDIO=1`.
2. Run phase4 validation first:
   ```bash
   PYTHONPATH=. python3 src/preprocessing/validate_phase4_inputs.py \
     --vod-id "$VOD_ID" \
     --enable-persistent-intelligence \
     --profile-root "${PROFILE_ROOT:-data/streamer_intelligence}"
   ```
3. Run synthesis with persistent intelligence enabled. If `STREAMER_ID` is unset, omit `--streamer-id` to validate metadata-derived default behavior.
4. After run, inspect artifacts with a small Python check:
   - `vods/phase4_<VOD_ID>/qwen_vision_progressive.json` exists.
   - output contains `streamer_identity` and, when update enabled, `profile_update`.
   - proposal file exists in propose/auto mode.
   - accepted observations in auto mode appear in both `observations.jsonl` and `profile.json` only when `accepted > 0`.

**Verification commands:**
```bash
bash scripts/validate_persistent_intelligence_wsl.sh <KNOWN_VOD_ID>
```

**Acceptance criteria:**
- Script exits non-zero when an expected artifact is missing or stale.
- Script prints resolved streamer ID/source and proposal/auto-merge counts.
- Script distinguishes `accepted=0` as expected policy behavior when candidates are below threshold.

---

### Task 24: Clean up modern preprocessing contract drift

**Objective:** Make the modern preprocessing path run without falling back because of internal API mismatch.

**Files:**
- Modify: `src/preprocessing/pipeline.py`
- Modify: `src/preprocessing/fusion.py`
- Modify: `src/preprocessing/__main__.py`
- Modify: `src/preprocessing/prepare_phase4.py` if it special-cases fallback behavior
- Test: add/update `tests/test_preprocessing_pipeline.py` or nearest existing preprocessing tests

**Known issue:** `pipeline.py` expects `fuse(...)` while `fusion.py` exports `fuse_signals(...)`. `prepare_phase4.py` currently tolerates fallback behavior, but the normal path should be contract-clean.

**Implementation steps:**
1. Decide the canonical function name (`fuse_signals(...)` preferred if already exported and tested).
2. Update `pipeline.py` imports/calls to use the canonical API.
3. Keep a small compatibility shim only if needed:
   ```python
   def fuse(*args, **kwargs):
       return fuse_signals(*args, **kwargs)
   ```
   Add a comment that it is legacy compatibility, not the primary API.
4. Ensure script entrypoints avoid stdlib `types` shadowing by preserving repo-root bootstrap behavior.
5. Add a unit/smoke test that imports and runs the modern pipeline with tiny fake transcript/scene/chat inputs without requiring Docker.

**Verification commands:**
```bash
pytest -q tests/test_preprocessing_pipeline.py tests/test_phase4_validation.py
PYTHONPATH=. python3 -m src.preprocessing --help
PYTHONPATH=. python3 src/preprocessing/prepare_phase4.py --help
```

**Acceptance criteria:**
- Modern preprocessing imports do not raise `ImportError`/`AttributeError` for fusion.
- Fallback path is not used for the normal in-repo smoke test.
- Existing phase4 validation tests still pass.

---

### Task 25: Harden phase4 manifest generation with YOLO-aware ranking

**Objective:** Improve `clip_manifest.json` candidate quality while preserving the current schema expected by synthesis.

**Files:**
- Modify or create: `src/preprocessing/clip_manifest.py`
- Modify: `src/preprocessing/prepare_phase4.py`
- Test: create `tests/test_clip_manifest_generation.py`
- Optional docs: `docs/CLIP-INTELLIGENCE-PIPELINE.md`

**Implementation steps:**
1. Extract manifest generation into a testable pure function if it is currently embedded in `prepare_phase4.py`.
2. Inputs should include fusion timeline/transcript/chat, optional `yolo_detections.json`, frame timestamps, and VOD duration.
3. Keep 120s candidate windows unless Ken changes the candidate policy.
4. Add deterministic scoring features:
   - speech density / has speech,
   - chat intensity / spikes,
   - scene/activity changes,
   - YOLO object presence weighted for content usefulness (person/face/devices/food/screens, etc.),
   - penalties for long silence / empty windows.
5. Emit required schema fields for every clip:
   - `start`, `end`, `title`, `score`, `objects_detected`, `summary`, `has_speech`, `chat_intensity`, `label`.
6. Add debug metadata only under non-breaking optional keys (for example `score_breakdown`).

**Tests to add:**
- YOLO-positive window ranks above otherwise-equal no-object window.
- No-transcript/no-chat windows are either omitted or score low.
- Output validates against expected `clip_manifest.json` contract.

**Verification commands:**
```bash
pytest -q tests/test_clip_manifest_generation.py tests/test_phase4_validation.py
```

**Acceptance criteria:**
- Existing synthesis still loads the manifest without schema changes.
- Manifest generation remains deterministic and auditable.
- `validate_phase4_inputs.py` can report manifest count and basic score range.

---

### Task 26: Make preprocessing/runtime environment reproducible

**Objective:** Remove ad-hoc WSL dependency assumptions by documenting/pinning which dependencies are local vs Dockerized.

**Files:**
- Modify/create: `requirements-preprocessing.txt`
- Modify/create: `requirements-speakerid.txt` if speaker-specific deps need updates
- Modify: `docs/CLIP-INTELLIGENCE-PIPELINE.md`
- Modify: `docs/references/speaker-attribution.md` / `persistent-streamer-intelligence.md` if created

**Implementation steps:**
1. Split dependency docs by runtime:
   - local Python lightweight utilities,
   - Docker WhisperX image,
   - Docker YOLO/worker image,
   - vLLM audio image,
   - optional SpeakerID deps (`pyannote.audio`, `speechbrain`, etc.).
2. Add commands to verify WSL readiness:
   ```bash
   docker image ls | grep -E 'whisperx|vod-lens-worker|vllm'
   ffmpeg -version
   yt-dlp --version
   python3 --version
   ```
3. Add failure messages/runbook notes for missing Docker images vs missing Python packages.

**Acceptance criteria:**
- A new agent can tell whether a failure is missing Docker image, missing HF token, missing local package, or Bee not running.
- Docs do not imply the empty `vod-lens-venv` has GPU preprocessing deps installed.

---

### Task 27: Add Bee startup and health reliability path

**Objective:** Prevent analysis from silently racing the Bee vision backend.

**Files:**
- Modify: `src/synthesis/qwen_clip_analyzer_progressive.py`
- Optional create: `src/synthesis/bee_server.py`
- Test: `tests/test_bee_server.py` or `tests/test_prompt_templates.py` only if no better location
- Docs: `docs/CLIP-INTELLIGENCE-PIPELINE.md`

**Implementation steps:**
1. Preserve existing `wait_for_bee_api(timeout=300, interval=5)` behavior.
2. Add opt-in managed start, not surprising default process spawning. Example CLI/env design:
   - `--start-bee` to enable startup if health check fails.
   - `--bee-start-command <shell command>` or `BEE_START_COMMAND` to define exact launch command.
   - `--bee-url` to avoid hardcoded `100.97.240.34:8082` where possible.
3. If `--start-bee` is not set and health check fails, fail with actionable command text instead of continuing into empty model responses.
4. Tests should mock subprocess/requests; do not launch Bee in unit tests.

**Acceptance criteria:**
- Pipeline never proceeds to Stage 1 after Bee health failure.
- Managed start is explicit and logs the command source without printing secrets.
- Existing WSL manual Bee workflow still works.

---

### Task 28: Canonicalize raw VOD path resolution for extraction/upload

**Objective:** Make clip extraction find the raw MP4 reliably from phase4 data instead of fragile working-directory assumptions.

**Files:**
- Modify: `src/synthesis/extract_and_upload_clips.py`
- Modify: `src/preprocessing/prepare_phase4.py` if it should write a canonical raw path into phase4 metadata
- Test: create/update `tests/test_extract_and_upload_clips.py`
- Docs: `docs/CLIP-INTELLIGENCE-PIPELINE.md`

**Implementation steps:**
1. Add a resolver function, for example:
   ```python
   def resolve_raw_vod_path(vod_id: str, phase4_dir: Path, explicit_path: Path | None = None) -> Path:
       ...
   ```
2. Resolution order:
   - explicit `--vod` path, if supplied and exists,
   - `phase4_<VOD_ID>/raw/<VOD_ID>.mp4`,
   - raw path recorded in phase4/fusion metadata, if present,
   - legacy repo-level `raw/<VOD_ID>.mp4`, with warning.
3. In `--dry-run`, print the resolved path and whether it exists.
4. Fail with clear remediation commands if no path exists.

**Verification commands:**
```bash
pytest -q tests/test_extract_and_upload_clips.py
python src/synthesis/extract_and_upload_clips.py \
  --json vods/phase4_<VOD_ID>/qwen_vision_progressive.json \
  --vod-id <VOD_ID> \
  --dry-run
```

**Acceptance criteria:**
- New VODs prepared under `vods/phase4_<VOD_ID>/raw/` do not require manual repo-root `raw/` copies.
- Dry-run reports the exact raw MP4 path that would be used.
- Missing raw MP4 errors are actionable.


---

## Test Matrix

Run fast tests after each task:

```bash
pytest -q \
  tests/test_speaker_attribution_types.py \
  tests/test_audio_segments.py \
  tests/test_speaker_alignment.py \
  tests/test_speaker_profiles.py \
  tests/test_speaker_recognition.py \
  tests/test_speaker_name_inference.py \
  tests/test_speaker_attribution.py \
  tests/test_streamer_intelligence_types.py \
  tests/test_streamer_store.py \
  tests/test_profile_context.py \
  tests/test_profile_update.py \
  tests/test_clip_context.py \
  tests/test_scoring.py \
  tests/test_stage_schemas.py
```

Run optional integration only on WSL2 with deps/token:

```bash
RUN_SPEAKERID_INTEGRATION=1 HF_TOKEN=<token> pytest -q tests/test_speaker_diarization.py -m integration
```

---

## Operational Pitfalls

1. **HF gated model access:** pyannote community-1 requires accepting model terms and using a token. Error messages must say this explicitly.
2. **Dependency isolation:** Do not make ordinary repo imports require pyannote/SpeechBrain. Lazy-import inside speaker modules.
3. **GPU contention:** Diarization/embedding may compete with Bee/Qwen/Omni. Run speaker attribution during preprocessing, before Bee-heavy synthesis, or provide CPU fallback.
4. **Overlapping speech:** Use pyannote exclusive diarization for transcript assignment; keep raw overlapping turns for diagnostics.
5. **Chat greeting false positives:** Twitch streamers greet chat users constantly. Only assign names to voiced speakers when a diarized voice response follows.
6. **Voiceprint privacy:** Treat enrolled embeddings as biometric-like data; do not commit them by default.
7. **Threshold calibration:** Default cosine thresholds are starting points; validate on Ken’s real VODs and tune with false-positive/false-negative examples.
8. **Streamer may not be dominant:** Guest-heavy clips can still be good, but title/report must attribute correctly. Use Qwen prompt inference/reframing for streamer-reaction mismatches; do not add deterministic speaker-specific penalties/gates.
9. **Persistent profile drift:** Do not let one bad LLM summary poison future runs. Persistent intelligence must be evidence-backed, confidence-scored, and conflict-aware.
10. **Streamer-ID override drift:** Manual `--streamer-id` values can pollute the wrong profile when they disagree with VOD metadata. Default to metadata-derived IDs, and make overrides explicit/audited.
11. **Privacy and chat persistence:** Common chatter tracking is useful for attribution, but avoid storing sensitive personal claims or raw chat dumps in the profile. Prefer aggregate counters, roles, and evidence refs.
12. **Prompt contamination:** Inject only compact high-confidence profile context. Full profile dumps will bias Qwen and waste context.

---

## Future Enhancements

1. Add NeMo Sortformer backend behind `--diarization-backend sortformer` for benchmark comparison.
2. Add auto-enrollment suggestions: pick high-confidence solo streamer segments from VOD intro, ask operator to approve before saving profile.
3. Add per-streamer profile calibration report with similarity histograms.
4. Add face/person visual cross-check later if streams include facecam and guest windows.
5. Add UI/report section: “Speaker attribution confidence and evidence.”
6. Add profile review UI where Ken can approve/reject proposed persistent observations before promotion.
7. Add profile decay / staleness handling so old chatters or retired bits do not dominate current VOD analysis.

---

## Recommended Next Implementation Slice

For the next coding pass, implement in this order:

1. **Task 22 first** — streamer-id metadata resolution and override guardrail. This directly prevents wrong-profile writes.
2. **Task 23 second** — WSL persistent-intelligence validation harness, so later changes can be verified on real artifacts.
3. **Task 24 third** — modern preprocessing contract cleanup, to reduce fallback-dependent behavior before manifest hardening.
4. Then choose between:
   - **Task 25** if candidate quality is the priority, or
   - **Tasks 26–28** if operator/run reliability is the priority.

Do not start with speculative speaker-scoring gates. The approved policy remains: speaker attribution is prompt context and audit metadata; no deterministic Stage-2 speaker-specific penalties/hard gates unless Ken explicitly changes the requirement.
