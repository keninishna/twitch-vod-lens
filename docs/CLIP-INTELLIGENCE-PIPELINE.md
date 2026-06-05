# VOD Lens — Clip Intelligence Pipeline (Agent Context Brief)

> **Status:** Active development / Phase 1 validated  
> **Last Updated:** June 05, 2026  
> **Repo:** https://github.com/keninishna/twitch-vod-lens

This is the compressed, AI-agent-facing pipeline contract. Keep this file short and operational.
For deep implementation details, load:
- `docs/plans/fastpass.md` (flag-gated Gemma fast-pass: Gemma enrichment via `llama.cpp`, then targeted Qwen vision)
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
      Fast-pass variant (flag-gated):
        1a  Gemma 4 12B local multimodal enrichment over transcript windows
        1b  Qwen text-only triage over enriched chunks (no images)
        1c  Deterministic diverse shortlist selection for Qwen vision
        1d  Targeted Qwen vision criticism + full intelligence analysis on shortlist
1.5  Deterministic stitching
1.5b Audio normalization to structured flags
2    Deterministic scoring + penalties + hard gate (>= 3)
3    Final verification + title finalization + dedup + intelligence report
Post RMS fallback only for unresolved full 120s windows, then mandatory rescoring
Post+ Optional profile update proposal/auto-merge (mode-gated)
```

**Fast-pass status (June 05):** Validated end-to-end on real VOD `2788478641` (197 clips, ~6.5 hr VOD). Gemma enrichment processed 262 windows, ~48 min sequential. Text triage, vision shortlist, and Stage 1–3 produced a final ranking with 10 selected clips + 14 rejected. Concurrency (`-np 3` + `GEMMA_CONCURRENT_WORKERS=3`) added to reduce Gemma enrichment wall time ~3×.

**Runtime stack:**
- **Gemma 4 12B** → upstream `llama.cpp` / `build_compat` server on port **8084** (`-np 3`)
- **Qwen 3.6 27B DFlash** → BeeLlama v0.3.1 prebuilt CUDA 13.1 on port **8082** (GPU mmproj, `--reasoning on`, `--chat-template-kwargs '{"preserve_thinking":true}'`)
- Both run on WSL2 / RTX 5090, fit in ~32 GiB VRAM

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

### Phase 05 (Tasks 20–28) — Complete
All eight tasks verified on real WSL artifacts:
1. **Task 20:** Runbook documentation (`docs/references/speaker-attribution.md`, `docs/references/persistent-streamer-intelligence.md`).
2. **Task 22:** Streamer-ID metadata resolution + override mismatch guardrail.
3. **Task 23:** WSL artifact-first validation harness (`scripts/validate_persistent_intelligence_wsl.sh`).
4. **Task 24:** Modern preprocessing `fuse`/`fuse_signals` contract drift addressed (typed `fuse(...)` path, legacy `fuse_signals(...)` retained).
5. **Task 25:** YOLO-aware phase4 manifest hardening (`src/preprocessing/clip_manifest.py`, `tests/test_clip_manifest_generation.py`).
6. **Task 26:** Preprocessing/runtime environment contract documented with split requirements files.
7. **Task 27:** Bee managed startup/reliability path (`--bee-url`, `--start-bee`, `--bee-start-command` / `BEE_START_COMMAND`).
8. **Task 28:** Extraction raw-VOD resolver (`--vod` override -> phase4 raw -> fusion metadata -> legacy) with `--dry-run` reporting.

**No remaining open items in Phase 05.**

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
- `src/synthesis/gemma_enrichment.py` (Gemma multimodal enrichment, concurrent workers)
- `src/synthesis/fastpass_triage.py`
- `src/synthesis/bee_server.py`
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

### 8.2 Analyze — Full synthesis (vision-only, no audio)
```bash
# Default: Bee on http://localhost:8082, Gemma on http://localhost:8084
PYTHONPATH=. python3 src/synthesis/qwen_clip_analyzer_progressive.py \
  --vod-id <VOD_ID> \
  --skip-audio \
  --batch-size 1
```

### 8.3 Analyze — Fast-pass mode (Gemma + targeted Qwen vision)
```bash
PYTHONPATH=. python3 src/synthesis/qwen_clip_analyzer_progressive.py \
  --vod-id <VOD_ID> \
  --fast-pass \
  --fast-pass-mode gemma-enriched \
  --skip-audio \
  --batch-size 1
```

### 8.4 Fast-pass with persistent intelligence
```bash
PYTHONPATH=. python3 src/synthesis/qwen_clip_analyzer_progressive.py \
  --vod-id <VOD_ID> \
  --fast-pass \
  --fast-pass-mode gemma-enriched \
  --skip-audio \
  --batch-size 1 \
  --enable-persistent-intelligence \
  --update-streamer-profile \
  --profile-update-mode propose \
  --streamer-id <STREAMER_ID>
```

### 8.5 Extract + Upload
```bash
python src/synthesis/extract_and_upload_clips.py \
  --json vods/phase4_<VOD_ID>/qwen_vision_progressive.json \
  --vod-id <VOD_ID> \
  --dry-run
```

### 8.6 Gemma concurrent workers override
```bash
# Default is 3; reduce to 1 for sequential, increase for more parallelism
GEMMA_CONCURRENT_WORKERS=1 PYTHONPATH=. python3 src/synthesis/qwen_clip_analyzer_progressive.py ...
```

### 8.7 Starting the backends (launch before pipeline runs)

**Gemma server (ln -np 3 for concurrent window processing):**
```bash
/home/john/llama.cpp/build_compat/bin/llama-server \
  -m /home/john/models/gemma-4-12b/gemma-4-12B-it-Q4_K_M.gguf \
  --mmproj /home/john/models/gemma-4-12b/mmproj-gemma-4-12B-it-Q8_0.gguf \
  --host 0.0.0.0 --port 8084 \
  -ngl all -np 3 --kv-unified -c 32768 \
  -b 2048 -ub 512 -fa on --jinja \
  --temp 0.2 --top-p 0.95 --top-k 64 --repeat-penalty 1.0
```

**Bee/Qwen server (prebuilt CUDA 13.1, GPU mmproj, DFlash):**
```bash
# Launch script at: ~/twitch-vod-analyzer/scripts/start_bee_prebuilt_wsl.sh
bash ~/twitch-vod-analyzer/scripts/start_bee_prebuilt_wsl.sh

# Equivalent manual launch:
export LD_LIBRARY_PATH="/home/john/.local/lib/python3.12/site-packages/nvidia/cu13/lib:/home/john/beellama-prebuilt/v0.3.1:${LD_LIBRARY_PATH:-}"
nohup /home/john/beellama-prebuilt/v0.3.1/llama-server \
  -m /home/john/models/bee-qwen36-27b/Qwen3.6-27B-Q5_K_S.gguf \
  --mmproj /home/john/models/bee-qwen36-27b/mmproj-BF16.gguf \
  --spec-draft-model /home/john/models/bee-qwen36-27b/dflash-draft-3.6-q4_k_m.gguf \
  --spec-type dflash --spec-dflash-cross-ctx 1024 \
  --host 0.0.0.0 --port 8082 -np 1 --kv-unified \
  -ngl all --spec-draft-ngl all \
  -b 2048 -ub 512 --ctx-size 102400 \
  --cache-type-k q5_0 --cache-type-v q4_1 \
  --flash-attn on --jinja --no-mmap --mlock --no-host \
  --reasoning on --chat-template-kwargs '{"preserve_thinking":true}' \
  --temp 0.6 --top-k 20 --top-p 1.0 --min-p 0.0 \
  > /home/john/bee_prebuilt_v031.log 2>&1 &
```

Optional modes:
```bash
# vision-only (no audio phase)
PYTHONPATH=. python3 src/synthesis/qwen_clip_analyzer_progressive.py --vod-id <VOD_ID> --skip-audio

# Gemma fast-pass dry-run (Gemma + text triage only, no vision)
PYTHONPATH=. python3 src/synthesis/qwen_clip_analyzer_progressive.py \
  --vod-id <VOD_ID> \
  --fast-pass \
  --fast-pass-mode gemma-enriched \
  --fast-pass-dry-run

# Managed Bee startup (optional — use when Bee is not already running)
PYTHONPATH=. python3 src/synthesis/qwen_clip_analyzer_progressive.py \
  --vod-id <VOD_ID> \
  --bee-url http://localhost:8082 \
  --start-bee \
  --bee-start-command "bash /home/john/twitch-vod-analyzer/scripts/start_bee_prebuilt_wsl.sh"
```

### 8.8 Comparison harness (fast-pass vs baseline)
```bash
PYTHONPATH=. python3 scripts/compare_fastpass_recall.py \
  --baseline vods/phase4_<VOD_ID>/baseline_full_vision.json \
  --fastpass vods/phase4_<VOD_ID>/qwen_vision_progressive.json \
  --gemma vods/phase4_<VOD_ID>/gemma_multimodal_annotations.json \
  --triage vods/phase4_<VOD_ID>/text_triage_candidates.json \
  --shortlist vods/phase4_<VOD_ID>/vision_shortlist.json
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
- `gemma_multimodal_annotations.json` (fast-pass evidence artifact)
- `text_triage_candidates.json` (fast-pass text triage)
- `vision_shortlist.json` (fast-pass shortlist)

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
- Bee/Qwen vision backend service on port 8082 (prebuilt CUDA 13.1 binary, not Docker).
- Gemma 4 12B enrichment service on port 8084 (upstream llama.cpp build).

### 10.3 Optional SpeakerID runtime
Speaker attribution uses `requirements-speakerid.txt` and is optional/gated:
- `pyannote.audio`, `speechbrain`, `torch`/`torchaudio`, `soundfile`, `librosa`,
- `HF_TOKEN` or `HUGGINGFACE_TOKEN`,
- accepted HuggingFace terms for gated pyannote models.

Keep this stack separate from lightweight preprocessing because it is GPU-sensitive and model-access dependent.

### 10.4 WSL readiness checks
```bash
# Backend servers
curl -s http://localhost:8082/v1/models      # Bee/Qwen
curl -s http://localhost:8084/v1/models      # Gemma

# Tools
ffmpeg -version
yt-dlp --version
python3 --version

# GPU
/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

### 10.5 WSL validation workflow
1. Run readiness checks and confirm both backend servers are healthy.
2. Prepare and validate a single phase4 VOD.
3. Run the Gemma smoke test (`scripts/smoke_test_gemma12b_llamacpp.py`).
4. Run a full fast-pass synthesis pass.
5. Inspect `qwen_vision_progressive.json` for `final_ranking.final_selected_clips` and `rejected_clips`.

### 10.6 Failure diagnosis
- Missing Docker image: preprocessing step fails before producing expected phase4 artifacts; pull/build the relevant image.
- Missing local Python package: phase4 prep/validation import fails; install `requirements-preprocessing.txt`.
- Missing HF token or gated-model access: SpeakerID diarization fails; export `HF_TOKEN`/`HUGGINGFACE_TOKEN`.
- Bee not running/unhealthy: synthesis preflight fails; check Bee server logs at `/home/john/bee_prebuilt_v031.log`.
- Gemma not running/unhealthy: fast-pass Gemma enrichment errors; check Gemma server logs.
- GPU memory pressure: check `nvidia-smi`. Both Bee (~24 GiB) + Gemma (~7 GiB) fit on RTX 5090 (~32 GiB).

---

## 11) Current Open Risks

1. **Gemma enrichment parse rate:** Real VOD run showed 39/262 (15%) successful Gemma annotations; 223/262 returned "empty response". The enrichment still drives text triage via chunk coverage regardless of parse success, but parse rate needs improvement for future quality gains.

2. **Qwen thinking/output contract (vLLM only):** vLLM Qwen3.6 NVFP4 requires `enable_thinking=False` + `response_format=json_object` to produce usable output. BeeLlama handles thinking mode correctly. Only relevant if the backend switches back to vLLM.

3. **Gemma concurrent workers:** Default is `GEMMA_CONCURRENT_WORKERS=3` with `-np 3` on Gemma. Tuning this higher may help further if ffmpeg audio extraction becomes the bottleneck.

4. **Audio phase (Omni 7B) not validated on this RTX 5090 stack:** Currently gated behind `--skip-audio`. The Omni container and audio pipeline are carryover from a 3090/vLLM era and may need revalidation on the current setup.

---

## 12) Related Docs

- `docs/plans/speakerid.md` (hub)
- `docs/plans/speakerid/05-docs-validation-rollout.md`
- `docs/plans/fastpass.md` (Gemma fast-pass; Qwen+Bee backend guidance)
- `scripts/start_bee_prebuilt_wsl.sh` (Bee launch helper for current working config)
- `~/.hermes/skills/mlops/clip-intelligence-pipeline/SKILL.md`
