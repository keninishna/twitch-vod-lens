# VOD Lens — Clip Intelligence Pipeline (Agent Context Brief)

> **Status:** Active development  
> **Last Updated:** May 27, 2026  
> **Repo:** https://github.com/keninishna/twitch-vod-lens

This is the compressed, AI-agent-facing pipeline contract. Keep this file short and operational.
For deep implementation details, load:
- `docs/plans/speakerid.md` (+ one phase file)
- skill reference: `clip-intelligence-pipeline`

---

## 1) Purpose

Given a Twitch VOD, produce:
1. ranked clip candidates,
2. deterministic score/gate decisions,
3. final clip titles + intelligence rationale,
4. optional speaker attribution + persistent streamer profile updates,
5. extraction/upload-ready outputs.

---

## 2) Canonical Flow

```text
0    Preprocessing (download, transcript, scenes, chat, YOLO, manifest, frames)
0.5  Optional speaker attribution + persistent profile context load
1    Stage 1 discovery (LLM): clip analysis + failure modes + draft title context
1.5  Deterministic stitching
1.5b Audio normalization to structured flags
2    Deterministic scoring + penalties + hard gate (>= 3)
3    Final verification + title finalization + dedup + intelligence report
Post RMS fallback only for unresolved full 120s windows, then mandatory rescoring
Post+ Optional profile update proposal/auto-merge (mode-gated)
```

---

## 3) Stage Contracts (Non-Negotiable)

1. **Stage 1 is discovery-only** (no final posting decisions).
2. **Stage 2 is deterministic authority** for score, penalties, and gate.
3. **Stage 3 is finalization** for titles, dedup, and final report payloads.
4. **Any trim mutation (RMS/manual) requires rescoring + re-gating**.
5. **Speaker attribution remains prompt context and reporting evidence** (no Stage-2 speaker-specific hard gates unless explicitly requested).

---

## 4) Inputs Contract (phase4 directory)

Expected under `vods/phase4_<VOD_ID>/`:
- `fusion_result_<VOD_ID>.json`
- `clip_manifest.json`
- `frames/`
- `raw/<VOD_ID>.mp4`
- optional `speaker_attribution_<VOD_ID>.json`

Do **not** reuse/symlink phase4 data across different VOD IDs.

---

## 5) Deterministic Policies

### 5.1 Dead Air
- Precomputed in Python from transcript timings.
- Policy:
  - longest gap `>20s` -> penalty `-3`, score cap `<=6`
  - total silence `>30%` -> score cap `<=5`
- Trims that include substantial dead-air must be narrowed or rejected.

### 5.2 Duration
- Retention-first shortest valid trim.
- Penalties:
  - `<=60s`: `0`
  - `61–75s`: `-1`
  - `76–90s`: `-2`
  - `>90s`: `-3`

### 5.3 Clip Criticism Penalty
- Stage 1 provides `failure_modes[].suggested_penalty`.
- Stage 2 sums penalties with hard cap `-5.0`.

### 5.4 Hard Gate
- Stage-2 pass threshold: **`final_score >= 3`**.

### 5.5 RMS Fallback
- Run only when selected trim is still full candidate span **and** candidate width is exactly `120s`.
- After RMS boundary changes: recompute duration/dead-air/final score/eligibility.

---

## 6) Speaker Attribution + Persistent Intelligence Status

### Implemented
- **Phases 01–03 complete:** diarization/alignment, voiceprints/recognition/name inference, phase4 + prompt/output integration.
- **Phase 04 complete (Tasks 14–19):**
  - persistent intelligence contracts (`src/intelligence/types.py`)
  - profile store (`src/intelligence/streamer_store.py`)
  - profile context rendering (`src/intelligence/profile_context.py`)
  - profile update proposal + auto-merge wiring (`src/intelligence/profile_update.py`)
  - voice-profile reuse integration
  - CLI mode wiring across prep/validation/synthesis

### Phase 05 status (Tasks 20–28)

**Completed (code-verified):**
1. **Task 22:** Streamer-ID metadata resolution + override mismatch guardrail is implemented (`resolve_streamer_id_context` wired across prep/validation/synthesis + output `streamer_identity` metadata).
2. **Task 23:** WSL artifact-first persistent-intelligence validation harness is implemented (`scripts/validate_persistent_intelligence_wsl.sh`).
3. **Task 24 (largely complete):** modern preprocessing `fuse`/`fuse_signals` contract drift is addressed via typed `fuse(...)` path in `pipeline.py` with legacy `fuse_signals(...)` retained in `fusion.py`.
4. **Task 27:** Bee managed startup/reliability path is implemented (`--bee-url`, `--start-bee`, `--bee-start-command` / `BEE_START_COMMAND`) with strict fail-fast Bee preflight before Stage 1.

**Still open / in progress:**
5. **Task 20:** runbook docs split remains incomplete (`docs/references/speaker-attribution.md` and `docs/references/persistent-streamer-intelligence.md` are not present).
6. **Task 25:** YOLO-aware phase4 manifest quality hardening not yet implemented (manifest still baseline deterministic windows + speech/chat heuristics).
7. **Task 26:** preprocessing/runtime environment contract is now documented in this brief; rollout should still validate real WSL image/tag naming in active environments.
8. **Task 28:** raw VOD extraction path resolution is partially improved but not yet fully canonicalized against phase4 metadata contract.

---

## 7) Canonical Source Files

### Preprocessing
- `src/preprocessing/prepare_phase4.py`
- `src/preprocessing/validate_phase4_inputs.py`
- `src/preprocessing/pipeline.py`
- `src/preprocessing/fusion.py`

Speaker attribution:
- `src/preprocessing/speaker_diarization.py`
- `src/preprocessing/speaker_profiles.py`
- `src/preprocessing/speaker_recognition.py`
- `src/preprocessing/speaker_name_inference.py`
- `src/preprocessing/speaker_attribution.py`
- `src/preprocessing/run_speaker_attribution.py`

Persistent intelligence:
- `src/intelligence/types.py`
- `src/intelligence/streamer_store.py`
- `src/intelligence/profile_context.py`
- `src/intelligence/profile_update.py`

### Synthesis
- `src/synthesis/qwen_clip_analyzer_progressive.py`
- `src/synthesis/stage1_discovery.py`
- `src/synthesis/stitching.py`
- `src/synthesis/audio_normalization.py`
- `src/synthesis/scoring.py`
- `src/synthesis/title_dedup.py`
- `src/synthesis/clip_context.py`
- `src/synthesis/schemas/clip_intelligence_stages.py`
- `src/synthesis/extract_and_upload_clips.py`

---

## 8) Minimal Runbook

### 8.1 Prepare + Validate
```bash
cd ~/twitch-vod-analyzer
PYTHONPATH=. python3 src/preprocessing/prepare_phase4.py \
  --url https://www.twitch.tv/videos/<VOD_ID> \
  --vod-id <VOD_ID>

PYTHONPATH=. python3 src/preprocessing/validate_phase4_inputs.py --vod-id <VOD_ID>
```

### 8.2 Analyze
```bash
PYTHONPATH=. python3 src/synthesis/qwen_clip_analyzer_progressive.py --vod-id <VOD_ID>
```

Optional modes:
```bash
# vision-only
PYTHONPATH=. python3 src/synthesis/qwen_clip_analyzer_progressive.py --vod-id <VOD_ID> --skip-audio

# persistent intelligence enabled
PYTHONPATH=. python3 src/synthesis/qwen_clip_analyzer_progressive.py \
  --vod-id <VOD_ID> \
  --enable-persistent-intelligence \
  --update-streamer-profile \
  --profile-update-mode propose|auto|off \
  --streamer-id <STREAMER_ID> \
  --profile-root data/streamer_intelligence

# managed Bee startup (optional)
PYTHONPATH=. python3 src/synthesis/qwen_clip_analyzer_progressive.py \
  --vod-id <VOD_ID> \
  --bee-url <BEE_URL> \
  --start-bee \
  --bee-start-command "<launch command>"
```

### 8.3 Extract + Upload
```bash
python src/synthesis/extract_and_upload_clips.py \
  --json qwen_vision_progressive.json \
  --vod raw/<VOD_ID>.mp4 \
  --min-score 7 \
  --output-dir ./clips
```

---

## 9) Output Contract

Primary output:
- `vods/phase4_<VOD_ID>/qwen_vision_progressive.json`

Must include:
- `stage1_5_stitched`
- `stage2_scored`
- `stage3_final_selected`
- `final_ranking.final_selected_clips` (canonical final set)

Optional artifacts:
- `speaker_attribution_<VOD_ID>.json` (flag-gated)
- `profile_update_proposal_<VOD_ID>.json` (mode-gated)

Per final clip expected fields:
- `score`/`final_score`
- `suggested_trim_start` / `suggested_trim_end`
- `clip_point`
- `platform_scores`
- `platform_recommendations`
- `intelligence_report`
- optional `speaker_attribution` payload

---

## 10) Preprocessing / Runtime Environment Contract

### 10.1 Local Python venv
Use `requirements-preprocessing.txt` for lightweight local orchestration and validation only:
- phase4 prep/validation CLIs,
- manifest/frame/context utilities,
- JSON/data-model validation,
- non-GPU helper scripts.

Do **not** assume a fresh `vod-lens-venv` contains GPU-heavy preprocessing dependencies.

### 10.2 Dockerized / external runtimes
These are not guaranteed by the lightweight local venv:
- WhisperX transcription image / runtime,
- YOLO / `vod-lens-worker` image for object/frame processing,
- vLLM audio image for Omni/audio analysis,
- Bee/Qwen vision backend service on the configured API URL.

### 10.3 Optional SpeakerID runtime
Speaker attribution uses `requirements-speakerid.txt` and is optional/gated:
- `pyannote.audio`, `speechbrain`, `torch`/`torchaudio`, `soundfile`, `librosa`,
- `HF_TOKEN` or `HUGGINGFACE_TOKEN`,
- accepted HuggingFace terms for gated pyannote models.

Keep this stack separate from lightweight preprocessing because it is GPU-sensitive and model-access dependent.

### 10.4 WSL readiness checks
```bash
docker image ls | grep -E 'whisperx|vod-lens-worker|vllm'
ffmpeg -version
yt-dlp --version
python3 --version
```

### 10.5 Failure diagnosis
- Missing Docker image: preprocessing step fails before producing expected phase4 artifacts; pull/build the relevant WhisperX, worker/YOLO, or vLLM image.
- Missing local Python package: phase4 prep/validation import fails; install `requirements-preprocessing.txt` in the local venv.
- Missing HF token or gated-model access: SpeakerID diarization fails; export `HF_TOKEN`/`HUGGINGFACE_TOKEN` and accept pyannote model terms.
- Bee not running/unhealthy: synthesis should fail preflight or produce model connection errors; start/check Bee separately (Task 27 owns managed Bee startup hardening).

---

## 11) Current Open Risks

1. Task-20 doc artifacts are still missing (speaker attribution + persistent-intelligence reference docs).
2. `clip_manifest.json` generation is functional but not yet YOLO-aware ranking parity.
3. Bee managed startup is implemented, but launch-command quality and environment-specific startup correctness remain operator-dependent.
4. Raw VOD path resolution for extraction is improved but not yet fully canonicalized to phase4 metadata.
5. Runtime/dependency contract is now documented, but real WSL image/tag naming should still be validated during rollout.

---

## 12) Related Docs

- `docs/plans/speakerid.md` (hub)
- `docs/plans/speakerid/05-docs-validation-rollout.md` (current next-phase tasks)
- `~/.hermes/skills/mlops/clip-intelligence-pipeline/SKILL.md`
