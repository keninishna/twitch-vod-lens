# VOD Lens — Clip Intelligence Pipeline (Agent Context Brief)

> **Status:** Active development / Phase 1 validated  
> **Last Updated:** June 11, 2026  
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
Post++ Optional artifact cleanup (intermediate or post-extraction)
```

**Fast-pass status (June 11):** Validated on VOD `2792837521` (129 clips, ~4.2 hr VOD). Gemma enrichment processed **172 windows, 162 successful in 21.6 min** at ~148 tok/s generation with 32K context. No MTP draft — the upstream `llama.cpp` build lacks the `gemma4-assistant` architecture from the `llama.cpp-mtp` fork (deleted during cleanup). Raw text observations + deterministic Python parser (`parse_gemma_raw_output`) eliminates all parse failures. Concurrency: single-worker (GEMMA_CONCURRENT_WORKERS=1) due to `-np auto = 4`, but serializes one window at a time to avoid context collisions.

**Runtime stack (sequential loading):**
- **Gemma 4 12B** → upstream `llama.cpp` build on port **8084** using **Unsloth QAT GGUF** (`gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` + `mmproj-F16.gguf`). Context **32768** (required to fit images + audio + text in KV cache). No MTP draft — upstream lacks `gemma4-assistant` architecture. New build at `/home/john/llama.cpp/build/bin/llama-server`; fallback `build_compat` at `/home/john/llama.cpp/build_compat/bin/llama-server`.
- **Qwen 3.6 27B DFlash** → BeeLlama v0.3.1 prebuilt CUDA 13.1 on port **8082** (GPU mmproj, `--reasoning on`, `--ctx-size 102400`).
- Both backends **never run simultaneously**. Pipeline loads them sequentially:
  1. Fast-pass path: Gemma loaded first → enrichment → `shutdown_gemma()` → Bee loaded → Qwen vision
  2. Standard path: only Bee loaded (Gemma not needed)
- After pipeline completes: auto-`shutdown_bee()` when `--start-bee` was used. Output verification via `_verify_pipeline_output()` checks for valid JSON and >=1 clip in `final_selected_clips` (exits code 3 on failure).
- Managed by `ensure_gemma_api_ready()` / `shutdown_gemma()` in `gemma_enrichment.py` and `ensure_bee_api_ready()` / `shutdown_bee()` in `bee_server.py`.
- Both require `LD_LIBRARY_PATH` to include `/home/john/.local/lib/python3.12/site-packages/nvidia/cu13/lib` for CUDA backend discovery. The `start_bee_prebuilt_wsl.sh` script handles this; `ensure_gemma_api_ready()` passes it via explicit `env` param.

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
- Twitch clips have a **hard max of 60 seconds** via API. Clips >60s cannot be created programmatically on Twitch.
- Penalties (for pipeline's internal score only; clips >60s require manual trim before Twitch upload):
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
- `src/synthesis/gemma_enrichment.py` (Gemma multimodal enrichment, raw text → deterministic parser via `parse_gemma_raw_output()`, concurrent workers)
- `src/synthesis/fastpass_triage.py`
- `src/synthesis/bee_server.py` (Bee lifecycle: `ensure_bee_api_ready`, `shutdown_bee`)
- `src/synthesis/schemas/clip_intelligence_stages.py`
- `src/synthesis/extract_and_upload_clips.py`

### Artifact cleanup
- `src/artifacts/__init__.py`
- `src/artifacts/cleanup.py`
- `scripts/cleanup_phase4_artifacts.py`

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
# Note: GEMMA_CONCURRENT_WORKERS only affects the Python-side ThreadPoolExecutor
# for parallel window requests. The Gemma server's -np must be >= your concurrent workers count.
# When using --model-draft (MTP), -np must be 1, so set GEMMA_CONCURRENT_WORKERS=1 as well.
GEMMA_CONCURRENT_WORKERS=1 PYTHONPATH=. python3 src/synthesis/qwen_clip_analyzer_progressive.py ...
```

### 8.7 Starting the backends (sequential — never both at once)

Backends are loaded **sequentially** to avoid VRAM contention. The pipeline manages this automatically — you only need to ensure the environment has the CUDA library path:

```bash
export LD_LIBRARY_PATH="/home/john/.local/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
```

**Gemma server (fast-pass enrichment only, no MTP draft, 32K context):**
```bash
# The pipeline auto-starts Gemma via ensure_gemma_api_ready() with --fast-pass.
# Manual launch for testing:
tmux new-session -d -s gemma "cd /home/john && \
  LD_LIBRARY_PATH=/home/john/.local/lib/python3.12/site-packages/nvidia/cu13/lib \
  /home/john/llama.cpp/build/bin/llama-server \
  --host 0.0.0.0 --port 8084 \
  --model /home/john/models/gemma-4-12b/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
  --mmproj /home/john/models/gemma-4-12b/mmproj-F16.gguf \
  -c 32768 -ngl 999 --no-mmap --flash-attn on --reasoning on --no-host \
  > /home/john/gemma_mtp_server.log 2>&1"

# Or let the pipeline auto-start via ensure_gemma_api_ready() with --fast-pass.
# ensure_gemma_api_ready() passes LD_LIBRARY_PATH via explicit env param
# and redirects output to /home/john/gemma_mtp_server.log.
```

**Bee/Qwen server (all paths):**
```bash
# Launch script at: ~/twitch-vod-analyzer/scripts/start_bee_prebuilt_wsl.sh
bash ~/twitch-vod-analyzer/scripts/start_bee_prebuilt_wsl.sh
# This sets LD_LIBRARY_PATH internally and writes logs to /home/john/bee_prebuilt_v031.log
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

# Managed Bee startup (probes for 10s before starting — was 300s)
PYTHONPATH=. python3 src/synthesis/qwen_clip_analyzer_progressive.py \\
  --vod-id <VOD_ID> \\
  --bee-url http://localhost:8082 \\
  --start-bee \\
  --bee-start-command "bash /home/john/twitch-vod-analyzer/scripts/start_bee_prebuilt_wsl.sh"

# After pipeline completes: backends auto-shutdown if started via --start-bee.
# Output is verified — exits with code 3 if qwen_vision_progressive.json is
# missing, corrupt, or contains zero final_selected_clips.
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

### 8.9 Cleanup phase4 artifacts after extraction/upload

After clips are extracted/uploaded and ``share_links.json`` is saved, remove large intermediate artifacts:

```bash
python src/synthesis/extract_and_upload_clips.py \
  --json vods/phase4_<VOD_ID>/qwen_vision_progressive.json \
  --vod-id <VOD_ID> \
  --output-dir clips/<VOD_ID> \
  --cleanup-artifacts \
  --cleanup-write-report
```

Manual standalone dry run:

```bash
PYTHONPATH=. python3 scripts/cleanup_phase4_artifacts.py \
  --vod-id <VOD_ID> \
  --phase4-dir vods/phase4_<VOD_ID> \
  --mode post-extraction \
  --dry-run
```

Three modes:

- **intermediate** — keeps raw VOD, frames, fusion, manifest. Removes per-step temp artifacts (audio batch I/O, fast-pass debug JSONs). Safe mid-pipeline.
- **post-extraction** (default) — removes raw VOD, frames, and all intermediates. Requires ``qwen_vision_progressive.json`` to exist.
- **aggressive** — additionally removes fusion, manifest, transcript, scenes, chat, YOLO, speaker attribution. Only when you're done with the VOD.

Flags: ``--keep-raw``, ``--keep-frames``, ``--write-report``.

### Output contract after cleanup

After cleanup in ``post-extraction`` or ``aggressive`` mode, only these remain in the ``phase4_<VOD_ID>`` directory:

- ``qwen_vision_progressive.json``
- ``profile_update_proposal_<VOD_ID>.json`` (if profile proposal mode was enabled)
- ``cleanup_report_<VOD_ID>.json`` (if report writing was enabled)
- ``data/streamer_intelligence/`` (persistent profiles, always protected)
- Extracted clips in the output directory (not under ``vods/``)

### 8.10 Create clips from intelligence output (via Twitch API)

⚠️ **Not currently part of the pipeline.** The Twitch ``POST /helix/videos/clips`` endpoint is unreliable — ~30% of VOD positions (early offsets under ~10 min plus random dead zones throughout) return ``202`` but the clip never materializes. This is a known Twitch-side issue documented in their forums since 2019.

If you do need to create a Twitch clip from a pipeline-suggested offset, these notes may help:

1. **Editor role** on the broadcaster's channel. The broadcaster adds you via Dashboard → Community → Roles Manager → Add Role → Editor.
2. **OAuth token with ``editor:manage:clips`` scope** (plus ``clips:edit``) — not available on TwitchTokenGenerator. Obtain via a custom dev app:
   - Register at https://dev.twitch.tv/console/apps (OAuth Redirect URL: ``http://localhost``)
   - Authorize at:
     ```
     https://id.twitch.tv/oauth2/authorize?client_id=YOUR_CLIENT_ID
       &redirect_uri=http://localhost&response_type=token
       &scope=editor%3Amanage%3Aclips+clips%3Aedit
     ```
3. **Add 30s to the intended clip start time** when passing ``vod_offset``. WhisperX transcript timestamps drift from VOD player position (~0.5% cumulative), so the pipeline's suggested offset lands early. Adding a flat +30s compensates for most of this drift across the VOD.

   ```bash
   curl -X POST "https://api.twitch.tv/helix/videos/clips
     ?broadcaster_id=BROADCASTER_ID
     &editor_id=YOUR_USER_ID
     &vod_id=VOD_ID
     &vod_offset=$((TRANSCRIPT_OFFSET + 30))
     &duration=DURATION
     &title=TITLE" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Client-ID: YOUR_CLIENT_ID"
   ```

4. **Dead-zone workaround:** If a clip returns 202 but never materializes, retry at an offset +-5s away. Some offsets are permanently dead — skip those clips.

**Credentials persisted at:** ``/home/hermeswebui/.hermes/twitch_credentials.json``

**GQL fallback** (creates 30s clips, no editor scope needed):
```python
import requests
r = requests.post("https://gql.twitch.tv/gql",
    headers={"Client-ID": "kimne78kx3ncx6brgo4mv6wki5h1ko",
             "Authorization": f"OAuth {browser_auth_token}",
             "Content-Type": "text/plain;charset=UTF-8"},
    json={"query": f'mutation {{ createClip(input: {{ broadcasterID: "{id}", videoID: "{vod}", offsetSeconds: {offset} }}) {{ clip {{ id }} }} }}'})
```
Then set title via ``publishClip`` with ``segments: []``.

**Limits:** 60s max duration. Clips are bound to the broadcaster's Twitch channel — they don't leave the platform.

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
- **`suggested_trim_start_hms`** / **`suggested_trim_end_hms`** — HH:MM:SS format
- **`start_hms`** / **`end_hms`** — HH:MM:SS format
- **`vod_url`** — direct link to VOD at clip window start (e.g. `https://www.twitch.tv/videos/<VOD_ID>?t=1h44m31s`)
- `clip_point`
- `platform_scores`
- `platform_recommendations`
- `intelligence_report`
- optional `speaker_attribution` payload

**Pipeline verification:** After the final save, `_verify_pipeline_output()` checks that the output file exists, is valid JSON, and has ≥1 clip in `final_selected_clips`. Exits with code 3 on failure.

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
- Gemma 4 12B enrichment service on port 8084 (upstream llama.cpp build using Unsloth QAT GGUF; typically kept alive in `tmux` session `gemma`).

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
1. Run readiness checks and confirm tools/tools are available.
2. Prepare and validate a single phase4 VOD.
3. Run the Gemma smoke test (`scripts/smoke_test_gemma12b_llamacpp.py`) — this verifies Gemma can load on GPU.
4. Run a full fast-pass synthesis pass (sequential loading: Gemma → enrichment → shutdown → Bee → Qwen vision).
5. Inspect `qwen_vision_progressive.json` for `final_ranking.final_selected_clips` and `rejected_clips`.

### 10.6 Failure diagnosis
- Missing Docker image: preprocessing step fails before producing expected phase4 artifacts; pull/build the relevant image.
- Missing local Python package: phase4 prep/validation import fails; install `requirements-preprocessing.txt`.
- Missing HF token or gated-model access: SpeakerID diarization fails; export `HF_TOKEN`/`HUGGINGFACE_TOKEN`.
- Bee not running/unhealthy: synthesis preflight fails; check Bee server logs at `/home/john/bee_prebuilt_v031.log`.
- Gemma not running/unhealthy: fast-pass Gemma enrichment errors; check Gemma server logs (`/home/john/gemma_mtp_server.log`) and verify `tmux ls | grep gemma`.
- GPU memory pressure: check `nvidia-smi`. Backends load sequentially — only one is in VRAM at a time. Gemma (~11 GiB) loads first for enrichment, then is killed before Bee (~20 GiB) starts. Never both simultaneously. If Bee fails to start after Gemma, check that `shutdown_gemma()` freed VRAM and the CUDA library path (`LD_LIBRARY_PATH`) includes `/home/john/.local/lib/python3.12/site-packages/nvidia/cu13/lib`.
- **Gemma enrichment returns 500 / context exceeded errors:** Gemma server was started with insufficient `-c` context size. Enrichment requests include images + audio + text which can exceed 4096 tokens. Ensure `ensure_gemma_api_ready()` uses `-c 32768`.
- **Pipeline exits with code 3:** `_verify_pipeline_output()` failed. The output file is missing, corrupt JSON, or `final_selected_clips` is empty. Check the pipeline log for specific error messages.

---

## 11) Notes

- **Twitch clip API is unreliable.** `POST /helix/videos/clips` has ~30% failure rate on VOD clipping (early offsets under ~10min and random dead zones throughout). Known Twitch-side issue since 2019. Not suitable for pipeline automation. Clip creation is manual/one-off. See Section 8.10 for the +30s workaround if needed.
- **MTP draft model unavailable.** The `gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf` (architecture `gemma4-assistant`) only works with the `llama.cpp-mtp` fork. The fork was deleted during cleanup. Upstream `llama.cpp` does not support this architecture. Gemma runs without draft at ~148 tok/s generation.

---

## 12) Related Docs

- `docs/plans/speakerid.md` (hub)
- `docs/plans/speakerid/05-docs-validation-rollout.md`
- `docs/plans/fastpass.md` (Gemma fast-pass; Qwen+Bee backend guidance)
- `scripts/start_bee_prebuilt_wsl.sh` (Bee launch helper for current working config)
- `~/.hermes/skills/mlops/clip-intelligence-pipeline/SKILL.md`
- [llama.cpp PR #23398](https://github.com/ggml-org/llama.cpp/pull/23398) — Gemma 4 MTP (merged June 07, 2026)
