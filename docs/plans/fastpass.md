# Gemma-Enriched Fast-Pass Clip Intelligence Implementation Plan

> **For Hermes:** Use `subagent-driven-development` skill to implement this plan task-by-task. Load this document plus `docs/CLIP-INTELLIGENCE-PIPELINE.md`; do not load every historical plan unless a task explicitly asks for it.

**Goal:** Validate the latest upstream `llama.cpp` + Gemma 4 12B Unified as a local audio+image enrichment pass, then use its annotations to reduce Stage 1 vision cost while improving candidate recall and clip narrative quality.

**Architecture:** Fast-pass Stage 1 becomes a flag-gated, three-pass discovery pipeline: (1) Gemma 4 12B Unified on the latest upstream local `llama.cpp` enriches transcript windows with factual audio/visual annotations, (2) Qwen 3.6 text-only reads transcript/chat/speaker/profile/Gemma annotations to select high-recall candidate arcs, and (3) Qwen vision performs final criticism + full intelligence analysis only on a deterministic shortlist. Stage 2 remains the deterministic authority for score, penalties, hard gate `final_score >= 3`, and rescoring after any trim mutation.

**Tech Stack:** Python 3, existing phase4 artifacts, latest upstream local `llama.cpp` OpenAI-compatible server for Gemma enrichment, `ggml-org/gemma-4-12B-it-GGUF`, existing Bee/Qwen OpenAI-compatible API for Stage-1 vision/final criticism, `src/synthesis/fastpass_triage.py`, `qwen_clip_analyzer_progressive.py`, `stage1_discovery.py`, `stitching.py`, `scoring.py`, `title_dedup.py`, pytest.

**Progress:** Tasks 1–2 from the earlier fast-pass plan are complete and should **not** be undone. They are implemented in `src/synthesis/fastpass_triage.py` with coverage in `tests/test_fastpass_triage.py`. New tasks below continue at Task 3 and adapt the plan from text-only first pass to Gemma-enriched first pass.

> **Runtime split:** Gemma fast-pass runs on the latest upstream `llama.cpp` baseline (`build_compat`). Bee/BeeLLaMA v0.3.1 prebuilt CUDA 13.1 is the Qwen-side backend on port 8082. Both run on the same RTX 5090.

---

## Non-Negotiable Policy

1. **Gemma enrichment is factual evidence, not final judgment.** It labels what is heard/seen; it does not decide final clips.
2. **Fast pass is high-recall triage, not final judgment.** It decides where expensive Qwen vision budget goes; it does not decide final clips.
3. **Stage 1 remains discovery-only.** No final posting decisions, no final platform recommendations, no Stage-2 authority inside prompts.
4. **Stage 2 remains deterministic authority.** Keep deterministic scoring/penalties/hard gate exactly where they are.
5. **Preserve visual-only and audio-only rescue lanes.** Do not shortlist solely by Qwen text rank; include Gemma audio/visual flags, chat spikes, YOLO/scene novelty, and sentinel coverage.
6. **Every artifact must be auditable.** If a clip is missed, operators must be able to see whether Gemma enrichment failed, Qwen triage failed, shortlist selection dropped it, Qwen vision mis-scored it, Stage 2 rejected it, or Stage 3 dedup/finalization removed it.
7. **Default rollout must be safe.** Implement behind CLI flags first; existing full-vision behavior remains available as baseline until real WSL validation proves recall and quality.
8. **Gemma 4 MTP support is now available.** PR #23398 ([Gemma 4 MTP](https://github.com/ggml-org/llama.cpp/pull/23398)) was merged into upstream `llama.cpp` master on June 07, 2026. Multimodal MTP (images + draft model) works correctly. Use `--spec-type draft-mtp --spec-draft-n-max 4 --reasoning on` with the Janvitos MTP draft model. Note that MTP requires `-np 1` (single slot), so `GEMMA_CONCURRENT_WORKERS` should be set to 1 as well. For maximum parallel window throughput, use non-MTP mode with `-np 3 + GEMMA_CONCURRENT_WORKERS=3`.

---

## Target Pipeline Flow

```text
0    Preprocessing (download, transcript, scenes, chat, YOLO, manifest, frames)
0.5  Optional speaker attribution + persistent profile context load
1a   Gemma 4 12B Unified local multimodal enrichment over transcript windows
     - input: transcript chunk + selected frames + <=30s audio clip
     - output: factual audio/visual/speaker/emotion annotations
1b   Qwen 3.6 text-only candidate triage over enriched transcript/chunks
     - no images
     - selects high-recall candidate arcs
1c   Deterministic diverse shortlist selection for Qwen vision budget
1d   Targeted Qwen vision criticism + full intelligence analysis on shortlist
1.5  Deterministic stitching
1.5b Audio normalization / optional existing Omni audio on top candidates if still enabled
2    Deterministic scoring + penalties + hard gate (>= 3)
3    Final verification + title finalization + dedup + intelligence report
Post RMS fallback only for unresolved full 120s windows, then mandatory rescoring
Post+ Optional profile update proposal/auto-merge (mode-gated)
```

If the external pipeline labels remain flat, represent `1a/1b/1c/1d` as internal Stage 1 discovery modes, not as new user-facing pipeline stages.

---

## Completed Tasks 1–2 Compatibility Decision

### Keep Task 1

Task 1 created pure selection helpers:

- `compute_vision_budget(...)`
- `normalize_triage_candidate(...)`
- `select_vision_shortlist(...)`

These remain useful. Do **not** undo them. Extend the candidate contract with Gemma annotation fields, but preserve backward-compatible behavior for tests and existing callers.

### Keep Task 2

Task 2 created transcript/chat chunking helpers:

- `build_triage_chunks(...)`
- `summarize_chunk_signals(...)`

These remain useful. Do **not** undo them. Extend chunks to optionally reference audio/frame inputs and Gemma annotations, but keep existing chunk tests passing.

### Required Compatibility Rule

All completed Task 1–2 tests must continue to pass unchanged unless a test is explicitly extended to cover new optional fields.

```bash
PYTHONPATH=. pytest tests/test_fastpass_triage.py -v
```

---

## New / Revised Artifacts

Expected under `vods/phase4_<VOD_ID>/` when Gemma fast-pass is enabled:

- `llama_gemma12b_smoke_tests.json`
  - backend capability report: server build, model ID, text/image/audio/audio+image results, timings, parse success
- `gemma_multimodal_annotations.json`
  - factual enrichment windows produced by Gemma 4 12B Unified
  - includes audio events, visual events, speaker/emotion nuance, confidence, request metadata, and failure state per window
- `text_triage_candidates.json`
  - keep this artifact name for compatibility
  - now means Qwen text-only candidate triage using transcript/chat/signals plus Gemma annotations
- `vision_shortlist.json`
  - deterministic shortlist selected for Qwen vision review
  - includes `selection_reason`, Gemma evidence references, and source scores
- `qwen_vision_progressive.json`
  - existing primary output path remains canonical
  - should include fast-pass/Gemma metadata summarizing settings, counters, runtime, and backend health

Recommended metadata block inside final output:

```json
{
  "fast_pass": {
    "enabled": true,
    "mode": "gemma_enriched",
    "gemma_backend": {
      "provider": "llama_cpp",
      "base_url": "http://localhost:8084/v1",
      "model_repo": "ggml-org/gemma-4-12B-it-GGUF",
      "quant": "Q4_K_M",
      "speculative_mode": "none|ngram-mod",
      "smoke_test_passed": true
    },
    "gemma_annotation_windows": 42,
    "gemma_failed_windows": 0,
    "qwen_text_calls": 24,
    "qwen_vision_calls": 40,
    "qwen_images_sent": 120,
    "total_manifest_candidates": 200,
    "text_triage_candidates": 60,
    "vision_shortlist_candidates": 40,
    "selection_reasons": {
      "qwen_text_top_rank": 24,
      "gemma_audio_alert_or_laughter": 5,
      "gemma_visual_reaction": 4,
      "chat_spike": 3,
      "yolo_visual_novelty": 2,
      "sentinel_coverage": 2
    },
    "wall_clock_seconds": 2700,
    "artifact_paths": {
      "gemma_annotations": "vods/phase4_<VOD_ID>/gemma_multimodal_annotations.json",
      "text_triage": "vods/phase4_<VOD_ID>/text_triage_candidates.json",
      "vision_shortlist": "vods/phase4_<VOD_ID>/vision_shortlist.json"
    }
  }
}
```

---

## Suggested Defaults

Initial defaults for WSL validation:

```text
`--fast-pass`                         validated on real VOD 2788478641
`--fast-pass-mode`                    gemma-enriched|text-only
--gemma-url                         http://localhost:8084/v1
--gemma-model                       gemma-4-12B-it
--gemma-window-seconds              30
--gemma-window-stride-seconds       30
--gemma-max-windows                 0       # 0 = all selected windows
--gemma-frames-per-window           2
--gemma-audio-format                wav
--gemma-audio-max-seconds           30
--gemma-response-timeout-seconds    180
--gemma-concurrent-workers          3       # ThreadPoolExecutor + Gemma -np N
--fast-pass-chunk-seconds           600
--fast-pass-overlap-seconds         60
--fast-pass-max-triage-candidates   60
--fast-pass-vision-ratio            0.20
--fast-pass-min-vision-candidates   25
--fast-pass-max-vision-candidates   50
--fast-pass-vision-frames           3
--fast-pass-sentinel-ratio          0.05
--fast-pass-dry-run                 false
--gemma-smoke-test-only             false
```

Recommended latest-upstream `llama.cpp` baseline command for WSL/RTX 5090 validation (do NOT use `--chat-template-kwargs '{"enable_thinking":false}'` — Gemma 4 needs thinking on for raw text observation output):

```bash
cd ~/llama.cpp
./build/bin/llama-server \
  -hf ggml-org/gemma-4-12B-it-GGUF:Q4_K_M \
  --host 0.0.0.0 \
  --port 8084 \
  -ngl all \
  -np 3 \
  --kv-unified \
  -c 32768 \
  -b 2048 \
  -ub 512 \
  -fa on \
  --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --temp 0.2 \
  --top-p 0.95 \
  --top-k 64 \
  --repeat-penalty 1.0
```

Recommended `ngram-mod` variant to benchmark after baseline works:

```bash
--spec-type ngram-mod \
--spec-ngram-mod-n-match 16 \
--spec-ngram-mod-n-min 24 \
--spec-ngram-mod-n-max 48
```

---

## Gemma Annotation Contract

A Gemma annotation window must be factual, timestamped, and safe to treat as fallible evidence.

```json
{
  "window_id": "gemma_0001230_0001260",
  "start": 1230.0,
  "end": 1260.0,
  "source_refs": {
    "transcript_segment_ids": [],
    "chat_message_ids": [],
    "frame_paths": [],
    "audio_path": ""
  },
  "audio_events": [
    {
      "timestamp": 1238.2,
      "type": "streamer_speech|non_streamer_speech|donation_alert|tts_alert|game_audio|music|laugh|silence|unknown",
      "speaker_guess": "streamer|chat_tts|game_character|unknown",
      "confidence": 0.0,
      "evidence": "brief factual description"
    }
  ],
  "visual_events": [
    {
      "timestamp": 1241.0,
      "type": "streamer_visible|face_visible|laughing|surprised|focused|gameplay_event|scene_change|visual_payoff|unknown",
      "confidence": 0.0,
      "evidence": "brief factual description"
    }
  ],
  "speaker_nuance": {
    "primary_speaker": "streamer|non_streamer|mixed|unknown",
    "streamer_led_likelihood": 0.0,
    "non_streamer_voice_present": false,
    "non_streamer_voice_type": "tts_alert|game_character|other|unknown"
  },
  "emotion_nuance": {
    "streamer_affect": "amused|surprised|confused|flat|performative|focused|unknown",
    "organic_reaction_likelihood": 0.0,
    "transactional_alert_likelihood": 0.0,
    "evidence": "brief factual description"
  },
  "clip_relevance_notes": [
    "short factual notes, no final clip judgment"
  ],
  "risk_flags": [
    "possible_alert_reaction",
    "game_audio_dominant",
    "visual_context_required",
    "speaker_uncertain"
  ],
  "parse_ok": true,
  "error": null
}
```

Mapping rules:

- Gemma `confidence` is evidence confidence, not Stage-2 score.
- `transactional_alert_likelihood` may inform Qwen criticism and Stage-1 failure modes, not a hard gate by itself.
- `streamer_led_likelihood` is prompt/report context, not deterministic Stage-2 speaker gating.
- If Gemma fails for a window, record the failure and continue with transcript/chat-only fallback for that window.

---

## Qwen Text Triage Candidate Contract

Keep the artifact name `text_triage_candidates.json`, but the prompt now receives enriched evidence. A candidate must still be convertible into existing Stage 1 discovery payloads.

```json
{
  "candidate_id": "triage_1234",
  "start": 1234,
  "end": 1294,
  "suggested_trim_start": 1241,
  "suggested_trim_end": 1272,
  "narrative_type": "storytelling|chat_banter|transactional_reaction|organic_reaction|gameplay_event|ambient|other",
  "trigger": "What starts the moment",
  "payoff": "What resolves or lands",
  "evidence_lines": [
    "transcript 1241s: ...",
    "chat 1243s user: ...",
    "gemma audio 1244s: possible TTS alert, confidence 0.83",
    "gemma visual 1248s: streamer laughs, confidence 0.78"
  ],
  "gemma_annotation_refs": ["gemma_0001230_0001260"],
  "risk_flags": [
    "requires_visual_context",
    "possibly_transactional",
    "weak_payoff",
    "speaker_attribution_uncertain"
  ],
  "triage_score": 1.0,
  "triage_confidence": 0.0,
  "vision_need": "none|verify_expression|verify_scene|verify_audio_visual_alignment|critical",
  "selection_reasons": []
}
```

Important mapping rules:

- `triage_confidence` maps to discovery `confidence`, not final score.
- `triage_score` is only for deterministic shortlist ranking.
- `suggested_trim_*` may be used as a hint but must still pass deterministic Stage 2 validation after Qwen vision review.
- `risk_flags` should become evidence for `failure_modes` or prompt context, not direct Stage-2 hard gates unless explicitly implemented later.

---

## Task Checklist

### Task 1: Add Fast-Pass Types and Pure Selection Helpers

**Objective:** Define pure, testable data contracts for text triage candidates and deterministic vision-shortlist selection.

**Status:** Complete — implemented in `src/synthesis/fastpass_triage.py` and verified in `tests/test_fastpass_triage.py`.

**Do not undo:** Keep existing functions and tests passing. Later tasks may extend candidates with optional Gemma fields.

**Verification:**

```bash
PYTHONPATH=. pytest tests/test_fastpass_triage.py -v
```

---

### Task 2: Build Text/Signal Chunking Utilities

**Objective:** Convert full-VOD transcript/chat/speaker context into long overlapping chunks suitable for cheap text-only Qwen calls.

**Status:** Complete — implemented in `src/synthesis/fastpass_triage.py` and verified in `tests/test_fastpass_triage.py`.

**Do not undo:** Keep existing chunk helpers and tests passing. Later tasks may extend chunks with optional frame/audio/Gemma references.

**Verification:**

```bash
PYTHONPATH=. pytest tests/test_fastpass_triage.py -v
```

---

### Task 3: Add `llama.cpp` Gemma 12B Smoke-Test Script

**Objective:** Prove the local backend can process text, image, audio, and combined audio+image+text before touching pipeline logic.

**Files:**
- Create: `scripts/smoke_test_gemma12b_llamacpp.py`
- Modify: `docs/plans/fastpass.md` only if results reveal changed assumptions

**Implementation notes:**

Create a script that accepts:

```text
--base-url http://localhost:8084/v1
--model gemma-4-12B-it
--image PATH
--audio PATH
--output PATH
--timeout 180
```

The script should run four OpenAI-compatible chat tests:

1. text-only JSON
2. image + text JSON using `image_url`
3. audio + text JSON using `input_audio` base64 `wav` or `mp3`
4. audio + image + transcript JSON

Use `response_format: {"type": "json_object"}` when supported. If unsupported, fall back to prompt-only JSON and record that in the output.

**Output:**

```text
vods/phase4_<VOD_ID>/llama_gemma12b_smoke_tests.json
```

or a manually provided `--output` path for non-VOD tests.

**Tests:**

- Add a pure unit test for payload construction if there is an existing script-test pattern.
- Otherwise keep this script manually verified and record smoke-test output as artifact.

**Verification:**

```bash
python3 scripts/smoke_test_gemma12b_llamacpp.py \
  --base-url http://localhost:8084/v1 \
  --model gemma-4-12B-it \
  --image /path/to/test_frame.jpg \
  --audio /path/to/test_30s.wav \
  --output /tmp/llama_gemma12b_smoke_tests.json
```

Expected: all four tests return parseable JSON. If any modality fails, do not proceed to Task 8 pipeline integration; keep the fallback text-only fast-pass path.

---

### Task 4: Add Media Window Builder for Gemma Enrichment

**Objective:** Build deterministic per-window transcript/frame/audio inputs from phase4 artifacts.

**Files:**
- Modify: `src/synthesis/fastpass_triage.py`
- Modify: `tests/test_fastpass_triage.py`

**Implementation notes:**

Add pure helpers such as:

```python
def build_gemma_annotation_windows(
    triage_chunks: list[dict],
    manifest_clips: list[dict],
    *,
    window_seconds: int = 30,
    stride_seconds: int = 30,
    max_windows: int = 0,
) -> list[dict]: ...

def select_gemma_frames_for_window(
    window: dict,
    frames_dir: str,
    frames_per_window: int = 2,
) -> list[str]: ...

def build_gemma_audio_extract_command(
    vod_mp4: str,
    window: dict,
    output_wav: str,
) -> list[str]: ...
```

Window selection rules:

- Start with transcript/chat chunks from Task 2.
- Prefer manifest-backed candidate windows when manifest exists.
- Include chat-spike windows even if transcript is sparse.
- Include sentinel/even-coverage windows for silent visual/audio moments.
- Cap audio window length at `<=30s` for Gemma 4 12B audio input.

**Tests:**

- windows cover manifest starts deterministically
- max window cap is deterministic
- selected frame times are start/mid or nearest available frames
- audio extraction command uses `ffmpeg` and writes `16kHz mono wav`
- no window exceeds configured `gemma_audio_max_seconds`

**Verification:**

```bash
PYTHONPATH=. pytest tests/test_fastpass_triage.py -v
```

---

### Task 5: Add Gemma Annotation Schema Normalizers

**Objective:** Normalize Gemma output into a stable artifact even when the model returns missing or malformed fields.

**Files:**
- Modify: `src/synthesis/fastpass_triage.py`
- Modify: `tests/test_fastpass_triage.py`

**Implementation notes:**

Add helpers:

```python
def normalize_gemma_annotation(raw: dict, window: dict) -> dict: ...

def merge_gemma_annotations_into_chunk(chunk: dict, annotations: list[dict]) -> dict: ...

def summarize_gemma_signals_for_triage(annotations: list[dict]) -> dict: ...
```

Normalization must:

- clamp timestamps to the window
- clamp confidence scores to `[0.0, 1.0]`
- dedupe risk flags
- preserve raw parse errors without throwing
- keep factual evidence lines concise

**Tests:**

- malformed Gemma JSON becomes `parse_ok=false` annotation
- confidence/timestamps are clamped
- alert/TTS/game-audio flags survive normalization
- streamer-led and transactional likelihood are preserved as evidence fields
- existing Task 1–2 tests still pass unchanged

**Verification:**

```bash
PYTHONPATH=. pytest tests/test_fastpass_triage.py -v
```

---

### Task 6: Add Gemma Client and Prompt Builder

**Objective:** Add a reusable local `llama.cpp` Gemma call path that sends transcript, frames, and audio and returns normalized annotations.

**Files:**
- Create or modify: `src/synthesis/gemma_enrichment.py`
- Modify: `tests/test_fastpass_triage.py` or create `tests/test_gemma_enrichment.py`

**Implementation notes:**

Suggested functions:

```python
def build_gemma_enrichment_prompt(window: dict) -> str: ...

def build_gemma_chat_payload(
    *,
    model: str,
    prompt: str,
    image_paths: list[str],
    audio_path: str | None,
    max_tokens: int = 1200,
) -> dict: ...

def call_gemma_llamacpp(
    *,
    base_url: str,
    payload: dict,
    timeout: int,
) -> dict: ...
```

Prompt requirements:

- Ask for factual timestamped annotations only.
- Prohibit final clip scoring and final platform recommendations.
- Ask specifically about:
  - donation/TTS/audio alert vs streamer speech
  - game audio vs human reaction
  - streamer-led vs non-streamer-led moment
  - laughter/surprise/confusion/focused affect
  - visual evidence of reaction or payoff
- Require JSON only.

OpenAI-compatible payload rules:

- images use `{"type":"image_url", "image_url":{"url":"data:image/jpeg;base64,..."}}`
- audio uses `{"type":"input_audio", "input_audio":{"data":"<base64>", "format":"wav"}}`
- do not use `audio_url`; llama-server does not implement it
- **do NOT use `response_format: {"type":"json_object"}`** — Gemma 4 + llama.ccp guided JSON grammar produces empty responses with multimodal input. Instead, ask for raw natural-language observations with labeled sections and parse deterministically.

**Tests:**

- payload builder includes `input_audio` for wav
- payload builder includes `image_url` for frames
- no local file paths leak into remote JSON except debug metadata
- prompt contains “factual annotations only” and “no final clip decisions”
- mocked HTTP response is parsed and normalized

**Verification:**

```bash
PYTHONPATH=. pytest tests/test_gemma_enrichment.py tests/test_fastpass_triage.py -v
```

---

### Task 7: Add Gemma Enrichment Artifact Writer

**Objective:** Run Gemma annotation over selected windows and write `gemma_multimodal_annotations.json`.

**Files:**
- Modify: `src/synthesis/gemma_enrichment.py`
- Modify: `src/synthesis/qwen_clip_analyzer_progressive.py`
- Add tests if an integration-test pattern exists

**Implementation notes:**

Add orchestration function:

```python
def run_gemma_enrichment(
    *,
    base_url: str,
    model: str,
    phase4_dir: str,
    fusion: dict,
    manifest: dict,
    frames_dir: str,
    raw_vod_path: str,
    window_seconds: int,
    stride_seconds: int,
    frames_per_window: int,
    max_windows: int,
    timeout: int,
    concurrent_workers: int = 1,
) -> dict: ...
```

Behavior:

1. build windows
2. extract temporary wav files with ffmpeg
3. call Gemma for each window (concurrently if concurrent_workers > 1)
4. normalize each result
5. write artifact
6. continue on individual window failures
7. include timing/call counters

Artifact shape:

```json
{
  "vod_id": "<VOD_ID>",
  "model": "gemma-4-12B-it",
  "backend": "llama_cpp",
  "created_at": "ISO-8601",
  "windows": [],
  "stats": {
    "total_windows": 0,
    "successful_windows": 0,
    "failed_windows": 0,
    "wall_clock_seconds": 0.0
  },
  "errors": []
}
```

**Verification:**

```bash
PYTHONPATH=. python3 src/synthesis/qwen_clip_analyzer_progressive.py \
  --vod-id <KNOWN_PHASE4_VOD_ID> \
  --fast-pass \
  --fast-pass-mode gemma-enriched \
  --gemma-url http://localhost:8084/v1 \
  --gemma-max-windows 3 \
  --fast-pass-dry-run
```

Expected: `gemma_multimodal_annotations.json` exists, contains 3 windows, and has parseable normalized annotations.

---

### Task 8: Revise Qwen Text Triage to Consume Gemma Annotations

**Objective:** Replace the old text-only first pass with Qwen text-only candidate selection over enriched transcript/chat/Gemma evidence.

**Files:**
- Modify: `src/synthesis/fastpass_triage.py`
- Modify: `src/synthesis/qwen_clip_analyzer_progressive.py`
- Modify: `tests/test_fastpass_triage.py`

**Implementation notes:**

This supersedes the old Task 3 from the previous plan. Do **not** implement a separate old-style text-only triage as the only path. Instead:

- keep `--fast-pass-mode text-only` as fallback if Gemma smoke tests fail
- make `--fast-pass-mode gemma-enriched` use `gemma_multimodal_annotations.json`
- keep artifact name `text_triage_candidates.json` for compatibility

Suggested function:

```python
def run_enriched_text_triage(
    *,
    qwen_call,
    model: str,
    manifest: dict,
    fusion: dict,
    triage_chunks: list[dict],
    gemma_annotations: dict | None,
    speaker_attribution: dict | None,
    streamer_profile_context: str,
    max_triage_candidates: int,
) -> dict: ...
```

Prompt must include:

- transcript excerpt with timestamps
- chat excerpt with timestamps/usernames when available
- speaker attribution context when available
- streamer profile context when enabled
- dead-air summary
- chat density/spike summary
- Gemma audio events and visual events
- Gemma streamer-led/transactional likelihood as evidence
- explicit instruction to preserve borderline setup/payoff arcs

Prompt must prohibit:

- final posting decisions
- final platform recommendations
- claiming visual details beyond Gemma evidence
- over-filtering transactional reactions that include inside-joke explanation arcs

**Tests:**

- mocked Qwen call receives no `image_url`
- prompt includes Gemma alert/game-audio/laughter evidence when provided
- prompt falls back cleanly when Gemma annotations are missing
- malformed Qwen result records an error and continues
- `max_triage_candidates` caps output deterministically after normalization/ranking

**Verification:**

```bash
PYTHONPATH=. pytest tests/test_fastpass_triage.py -v
```

---

### Task 9: Emit Deterministic Vision Shortlist Artifact with Gemma Rescue Lanes

**Objective:** Use enriched Qwen triage plus Gemma-specific rescue lanes to produce `vision_shortlist.json`.

**Files:**
- Modify: `src/synthesis/fastpass_triage.py`
- Modify: `src/synthesis/qwen_clip_analyzer_progressive.py`
- Modify: `tests/test_fastpass_triage.py`

**Implementation notes:**

Extend selection lane handling to support:

```text
qwen_text_top_rank
gemma_audio_alert_or_laughter
gemma_game_audio_or_non_streamer_voice
gemma_visual_reaction
gemma_visual_payoff
chat_spike
yolo_visual_novelty
sentinel_coverage
```

Shortlist entries should include:

```json
{
  "start": 1234,
  "end": 1294,
  "suggested_trim_start": 1241,
  "suggested_trim_end": 1272,
  "triage_score": 8.1,
  "triage_confidence": 0.84,
  "vision_need": "verify_audio_visual_alignment",
  "selection_reason": "gemma_audio_alert_or_laughter",
  "source_candidate_id": "triage_1234",
  "gemma_annotation_refs": ["gemma_0001230_0001260"],
  "evidence_lines": []
}
```

If an enriched triage candidate does not align exactly with a manifest candidate, map it to the nearest overlapping manifest clip or construct a bounded synthetic candidate only if downstream code can safely handle it. Prefer manifest-backed candidates for initial rollout.

**Tests:**

- shortlist writes stable JSON
- no duplicate starts
- every shortlist item maps to a manifest candidate for initial rollout
- Gemma audio/visual rescue candidates can enter shortlist even if Qwen text rank is lower
- selected candidates preserve Gemma evidence refs for Qwen vision prompt context

**Verification:**

```bash
PYTHONPATH=. pytest tests/test_fastpass_triage.py -v
```

---

### Task 10: Add Fast-Pass CLI Flags and Dry-Run Mode

**Objective:** Make Gemma fast-pass configurable and validate artifacts without running expensive Qwen vision.

**Files:**
- Modify: `src/synthesis/qwen_clip_analyzer_progressive.py`
- Modify: parser tests if present, otherwise add focused tests

**CLI flags:**

```text
--fast-pass
--fast-pass-mode gemma-enriched|text-only
--fast-pass-dry-run
--gemma-smoke-test-only
--gemma-url http://localhost:8084/v1
--gemma-model gemma-4-12B-it
--gemma-window-seconds 30
--gemma-window-stride-seconds 30
--gemma-max-windows 0
--gemma-frames-per-window 2
--gemma-audio-max-seconds 30
--gemma-response-timeout-seconds 180
--fast-pass-chunk-seconds 600
--fast-pass-overlap-seconds 60
--fast-pass-max-triage-candidates 60
--fast-pass-vision-ratio 0.20
--fast-pass-min-vision-candidates 25
--fast-pass-max-vision-candidates 50
--fast-pass-vision-frames 3
--fast-pass-sentinel-ratio 0.05
```

Dry-run behavior:

1. optionally run Gemma smoke test if `--gemma-smoke-test-only` or first Gemma run on a VOD
2. run Gemma enrichment unless existing artifact is explicitly reused
3. write `gemma_multimodal_annotations.json`
4. run enriched Qwen text triage
5. write `text_triage_candidates.json`
6. write `vision_shortlist.json`
7. print summary table
8. exit before Qwen vision calls

Summary table should include:

- Gemma backend health
- Gemma windows / failures
- manifest candidates
- triage chunks
- triage candidates
- vision shortlist candidates
- selection reason counts
- estimated image count compared with baseline

**Verification:**

```bash
PYTHONPATH=. python3 src/synthesis/qwen_clip_analyzer_progressive.py \
  --vod-id <KNOWN_PHASE4_VOD_ID> \
  --fast-pass \
  --fast-pass-mode gemma-enriched \
  --fast-pass-dry-run \
  --gemma-url http://localhost:8084/v1
```

Expected: Gemma/text/shortlist artifacts are written and no image payloads are sent to Qwen vision.

---

### Task 11: Route Qwen Vision Review Through the Gemma-Enriched Shortlist

**Objective:** When `--fast-pass` is enabled, run existing Stage 1 Qwen vision analysis only on `vision_shortlist` instead of every manifest clip.

**Files:**
- Modify: `src/synthesis/qwen_clip_analyzer_progressive.py`
- Modify: `src/synthesis/stage1_discovery.py` only if prompt context is centralized there
- Modify tests or create focused integration tests

**Implementation notes:**

When fast-pass is enabled:

1. ensure Gemma enrichment has run or been loaded
2. ensure enriched text triage artifact exists
3. ensure deterministic shortlist exists
4. set Stage 1 Qwen vision loop source to shortlist-mapped clips
5. include enriched evidence in each Qwen vision prompt

Prompt context additions for shortlisted candidates:

- Qwen triage trigger/payoff
- Qwen triage evidence lines
- Gemma audio events and visual events
- Gemma risk flags
- instruction: verify/correct Gemma evidence, do not blindly trust it
- instruction: produce criticism/failure modes/full intelligence analysis as current Stage 1 expects

**Tests:**

- with fast-pass disabled, all manifest clips are sent to Stage 1 loop
- with fast-pass enabled, only shortlisted clips are sent
- each fast-pass vision prompt includes Gemma/Qwen triage evidence/risk context
- no image payloads are sent during Gemma/Qwen text triage; image payloads are sent during targeted Qwen vision review

**Verification:**

```bash
PYTHONPATH=. pytest tests/test_fastpass_triage.py tests/test_stage1_discovery.py -v
```

---

### Task 12: Reduce Targeted Qwen Vision Frame Count Safely

**Objective:** Use fewer initial frames for targeted Qwen vision review while retaining extra-frame escalation for uncertain candidates.

**Files:**
- Modify: `src/synthesis/qwen_clip_analyzer_progressive.py`
- Add or modify tests as appropriate

**Implementation notes:**

When fast-pass is enabled, sample frames around likely trigger/midpoint/payoff:

1. `suggested_trim_start` or candidate start
2. midpoint of suggested trim/candidate
3. `suggested_trim_end` or candidate end

Keep existing full-baseline frame count behavior when `--fast-pass` is not set.

Escalation rule for uncertain candidates:

- If Qwen vision returns `vision_need=critical`, `failure_modes` mention missing context, or confidence is below threshold, allow extra frames up to the old full-frame count.
- Record escalations in fast-pass metadata.

**Tests:**

- fast-pass frame sampler returns 3 frames when available
- missing exact frame files fall back to nearest existing sampled frame
- uncertain candidates can escalate to more frames
- full baseline mode still uses current configured frame count

**Verification:**

```bash
PYTHONPATH=. pytest tests/test_fastpass_triage.py -v
```

---

### Task 13: Preserve Stage 2/Stage 3 Contracts

**Objective:** Ensure Gemma fast-pass does not bypass deterministic scoring, hard gates, stitching, title dedup, audio normalization, or RMS rescoring policy.

**Files:**
- Modify: `src/synthesis/qwen_clip_analyzer_progressive.py`
- Modify: `tests/test_scoring.py` only if a real contract gap appears
- Modify: `tests/test_title_dedup.py` only if finalization contract changes

**Checklist:**

- [ ] `stitch_discoveries(...)` still receives discovery payloads for every Qwen-vision-reviewed candidate.
- [ ] `normalize_clip_analysis(...)` still runs after audio injection.
- [ ] hard gate remains `final_score >= 3`.
- [ ] `finalize_stage3_candidates(...)` remains final deterministic title/dedup payload builder.
- [ ] RMS fallback, if applied, still triggers rescoring/regating.
- [ ] speaker attribution remains prompt/report context, not a deterministic hard gate.
- [ ] Gemma annotations remain evidence, not final scoring authority.

**Verification:**

```bash
PYTHONPATH=. pytest tests/test_stage1_discovery.py tests/test_stitching.py tests/test_scoring.py tests/test_title_dedup.py -v
```

---

### Task 14: Add Fast-Pass Metadata to Final Output

**Objective:** Record runtime/cost counters and artifact paths in `qwen_vision_progressive.json`.

**Files:**
- Modify: `src/synthesis/qwen_clip_analyzer_progressive.py`
- Add or modify tests if final-output tests exist

**Metadata must include:**

- fast-pass enabled/disabled
- fast-pass mode
- Gemma backend settings excluding secrets
- Gemma smoke-test pass/fail
- Gemma window/call counts
- Gemma failure count
- Qwen text call count
- Qwen vision call count
- images sent
- escalated frame count
- artifact paths
- selection reason counts
- wall-clock seconds

**Verification:**

Run a dry run and inspect metadata:

```bash
python3 - << 'PY'
import json
p='vods/phase4_<VOD_ID>/qwen_vision_progressive.json'
data=json.load(open(p))
print(json.dumps(data.get('fast_pass'), indent=2)[:4000])
PY
```

---

### Task 15: Add WSL Baseline Comparison Harness

**Objective:** Compare Gemma fast-pass output against full-vision baseline on real WSL artifacts before making fast-pass default.

**Files:**
- Create: `scripts/compare_fastpass_recall.py`
- Optionally create: `scripts/validate_gemma_fastpass_wsl.sh`

**Comparison inputs:**

- full baseline `qwen_vision_progressive.json`
- Gemma `gemma_multimodal_annotations.json`
- fast-pass `text_triage_candidates.json`
- fast-pass `vision_shortlist.json`
- fast-pass `qwen_vision_progressive.json`

**Metrics:**

- baseline final selected clips
- percentage of baseline final clips present in Gemma windows
- percentage of baseline final clips present in Qwen text triage candidates
- percentage of baseline final clips present in fast-pass vision shortlist
- percentage of baseline final clips present in fast-pass final selected clips
- runtime delta
- Gemma/Qwen call delta
- image count delta
- missed baseline clips with nearest Gemma/triage/shortlist explanations

**Acceptance threshold for initial rollout:**

```text
Gemma annotation coverage against baseline final clips: >= 95%
Fast-pass vision shortlist recall against baseline final clips: >= 90%
Fast-pass final selected recall against baseline final clips: inspect manually; no automatic threshold until quality reviewed
Runtime reduction on 200-candidate VOD: target >= 50%, ideally >= 75%
No obvious quality regression in manually inspected top clips
```

**Verification:**

```bash
PYTHONPATH=. python3 scripts/compare_fastpass_recall.py \
  --baseline vods/phase4_<VOD_ID>/baseline/qwen_vision_progressive.json \
  --fastpass vods/phase4_<VOD_ID>/qwen_vision_progressive.json \
  --gemma vods/phase4_<VOD_ID>/gemma_multimodal_annotations.json \
  --triage vods/phase4_<VOD_ID>/text_triage_candidates.json \
  --shortlist vods/phase4_<VOD_ID>/vision_shortlist.json
```

---

### Task 16: Update Pipeline Docs and Operator Runbook

**Objective:** Document Gemma fast-pass mode, artifacts, validation expectations, backend setup, and known risks.

**Files:**
- Modify: `docs/CLIP-INTELLIGENCE-PIPELINE.md`
- Modify: `docs/plans/fastpass.md` if implementation details change
- Optionally create/modify: `docs/references/gemma-fastpass.md` if operator docs grow too large
- Patch skill after implementation if behavior becomes durable: `clip-intelligence-pipeline`

**Required doc updates:**

- canonical flow mentions Gemma-enriched fast-pass as planned/flag-gated until implemented
- output contract mentions `gemma_multimodal_annotations.json`, `text_triage_candidates.json`, and `vision_shortlist.json`
- runbook includes llama.cpp server launch command
- runbook includes Gemma smoke-test command
- runbook includes fast-pass dry-run and full-run commands
- open risks mention Gemma audio experimental quality, unsupported 12B assistant/MTP in upstream llama.cpp, and WSL recall validation

**Verification:**

```bash
git diff -- docs/CLIP-INTELLIGENCE-PIPELINE.md docs/plans/fastpass.md
```

---

## Validation Strategy

### Unit Tests

Run after each implementation task:

```bash
PYTHONPATH=. pytest tests/test_fastpass_triage.py -v
```

Run broader synthesis contract tests before handoff:

```bash
PYTHONPATH=. pytest \
  tests/test_fastpass_triage.py \
  tests/test_stage1_discovery.py \
  tests/test_stitching.py \
  tests/test_scoring.py \
  tests/test_title_dedup.py \
  tests/test_clip_context.py \
  -v
```

### Backend Smoke Validation

Before any full VOD integration, verify local Gemma:

1. Start llama.cpp with `ggml-org/gemma-4-12B-it-GGUF:Q4_K_M`.
2. Run `scripts/smoke_test_gemma12b_llamacpp.py`.
3. Confirm text/image/audio/audio+image all return parseable JSON.
4. Repeat with `ngram-mod` enabled and compare latency/output quality.
5. Do not proceed to full integration if audio or image fails.

### WSL Artifact Validation

Use a known VOD with many candidates, ideally one with a prior full baseline run.

1. Run baseline full mode or locate existing baseline output.
2. Run Gemma smoke test.
3. Run Gemma fast-pass dry run with `--gemma-max-windows` small.
4. Inspect `gemma_multimodal_annotations.json` manually.
5. Run full Gemma fast-pass dry run.
6. Inspect `text_triage_candidates.json` and `vision_shortlist.json` manually.
7. Run fast-pass full mode.
8. Run comparison harness.
9. Manually inspect missed baseline clips, visual-only candidates, and alert/TTS/game-audio cases.

### Success Criteria

Initial implementation is acceptable when:

- existing default full-vision mode still works
- Tasks 1–2 tests continue passing
- Gemma smoke test passes text/image/audio/audio+image on WSL
- `--fast-pass-dry-run --fast-pass-mode gemma-enriched` writes all three artifacts
- dry-run sends no image payloads to Qwen vision
- `--fast-pass` runs Qwen vision only on the shortlist
- Qwen vision prompts include Gemma and Qwen triage evidence
- final output includes fast-pass metadata
- deterministic Stage 2/3 contracts are unchanged
- comparison harness reports Gemma coverage and shortlist recall against baseline final clips
- real WSL run shows substantial runtime reduction without obvious clip-quality regression

---

## Known Risks and Mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| Gemma audio path is experimental in llama.cpp | Audio alert/game-audio labels may be wrong | Treat as evidence only; Qwen vision critiques; compare with existing audio phase when needed |
| Gemma 12B MTP assistant unsupported in upstream llama.cpp | Speculative decoding with `google/gemma-4-12B-it-assistant` may not load | Use baseline or `ngram-mod`; add MTP only after compatible assistant GGUF/runtime is proven |
| Gemma misses visual-only clips | Silent reactions/glitches may lack transcript/audio cues | Include visual sentinel, YOLO novelty, chat spike, and even-coverage lanes |
| Gemma over-labels donation/TTS reactions | Could cause Qwen triage to reject inside-joke clips | Prompt Qwen to distinguish generic alert reaction from explanation/setup/payoff narrative |
| Qwen text triage over-selects transactional reactions | Donation/sub reactions can look exciting in transcript/chat | Preserve narrative-first prompt rules and Stage 2 criticism penalties |
| Shortlist loses baseline winners | Fast mode could be faster but worse | Compare against full baseline; target >=90% shortlist recall before defaulting |
| Prompt starts finalizing too early | Stage 1 must remain discovery-only | Prompts output evidence/candidate arcs, not final posting decisions |
| Artifact drift | Future agents may not know why a candidate was selected | Store selection reason, evidence lines, risk flags, Gemma refs, source IDs |
| Runtime reduction is less than expected | Gemma pass might add too much cost | Benchmark baseline vs ngram; cap windows; cache artifacts; record counters |
| Fast-pass breaks audio phase assumptions | Existing audio phase selects top clips from Stage 1 analyses | Ensure fast-pass analyses expose compatible `clip_worthiness` and fields |

---

## Future-Proofing / Explicitly Skip for First Rollout

Do **not** implement these in the first slice unless explicitly requested:

- replacing Qwen final vision with Gemma final judgment
- using `google/gemma-4-12B-it-assistant` in llama.cpp until runtime support is proven
- making fast-pass default before WSL recall comparison
- removing the full-vision baseline path
- training a learned ranker
- embedding/vector search over transcript/chat
- deterministic speaker hard gates
- rewriting the whole analyzer into a new pipeline framework
- optimizing with Bee/fork-specific Gemma MTP before upstream baseline works

---

## Recommended First Implementation Slice

The initial slice should validate Gemma before changing candidate routing:

1. **Task 3:** Add and run Gemma smoke-test script.
2. **Task 4:** Add deterministic Gemma window/media builder.
3. **Task 5:** Add Gemma annotation normalizers.
4. **Task 6:** Add Gemma client and prompt builder.
5. **Task 7:** Write `gemma_multimodal_annotations.json` for a small `--gemma-max-windows 3` dry run.

Do **not** route Qwen vision through the shortlist until Gemma artifacts look sane on at least one real WSL phase4 VOD.
