# VOD Lens — Clip Intelligence Pipeline (Compressed)

> **Status:** Active Development
> **Last Updated:** May 17, 2026 (Switched from vLLM to BeeLlama.cpp for inference)
> **Repo:** https://github.com/keninishna/twitch-vod-lens

## Purpose

This document is a compact operator reference for the current clip-intelligence pipeline:
- discover clip-worthy moments,
- apply deterministic scoring/gates,
- generate final titles/intelligence reports,
- extract/upload/share selected clips.

---

## Pipeline (Current Contract)

```text
Preprocessing (transcript + scenes + YOLO + chat)
  -> Stage 1: Discovery (LLM, no final platform decisions; emits draft title fields AND failure mode IDs for carryover)
  -> Stage 1.5: Deterministic cross-window stitching
  -> Stage 1.5b: Audio normalization to structured flags
  -> Stage 2: Deterministic scoring + penalties (duration, dead-air, clip criticism) + hard gate
  -> Stage 3: Final verification + title generation + dedup + intelligence report
  -> Post: RMS trim fallback (only unresolved full 120s windows) + mandatory rescoring
  -> Clip extraction + Nextcloud upload + public share links
```

### Current Workstream (May 18, 2026 — Stage 2 retuned)

1. **Inference backend switched to BeeLlama.cpp** (replaces vLLM)
   - Precision setup: Qwen3.6-27B-Q5_K_S target + DFlash Q4_K_M draft
   - Port changed from `8000` → **`8082`** (already updated in pipeline code)
   - Vision tower in VRAM via `--mmproj mmproj-BF16.gguf` (no `--no-mmproj-offload`)
   - DFlash speculative decoding active (~145 tok/s, ~40% draft acceptance)
   - Thinking mode enabled: `reasoning_content` and `content` fields split
   - Context size: **200K tokens** via TurboQuant (`turbo4` K, `turbo3_tcq` V)
   - Bee auto-restart integrated into audio phase (replaces docker stop/run cycle)
   - **Max tokens bumped to 16384** for all API calls (was 4096/8192) — gives Qwen room to think before outputting JSON

2. **Stage 1 title quality pass (active)**
   - Added a compact research brief to Stage 1 prompt (YouTube/NNg/curiosity-gap guidance).
   - Stage 1 now emits direct draft title fields:
     - `clip_point`
     - `title_why`
   - Removed provisional-named fields/fallback layering for this path.

3. **Endpoints (current)**
   - Bee inference API: `100.97.240.34:8082`
   - Model: `Qwen3.6-27B-Q5_K_S.gguf` (not HuggingFace ID)

### Stage Invariants (Non-Negotiable)

1. **Stage 1 is discovery-only**
   - No final platform posting decisions
   - Draft `clip_point` / `title_why` allowed for carryover context
2. **Stage 2 is deterministic enforcement** — aggregates all scoring and penalties:
   - Duration/dead-air penalties
   - **Clip criticism penalty** — failure modes classified in Stage 1 are deducted from final_score (capped at -5.0)
   - Hard gate: `score >= 3` to proceed
3. **Stage 3 is finalization**
   - Final titles, dedup, intelligence report
4. **Any RMS trim mutation invalidates prior score**
   - Must rescore + re-gate before final output

---

## Key Quality Controls

### 1) Dead Air Enforcement

- Dead air gaps are computed in Python from transcript timing (not left to model inference).
- Injected warning format: `⚠️ DEAD AIR DETECTED: ...`
- Policy:
  - single gap `>10s` -> **-5 penalty**, cap score at **<=5**
  - total silence `>30%` -> score **<=5**
  - trims crossing dead-air regions are invalid and must be narrowed or dropped

### 2) Chat Attribution Enforcement

- Chat/transcript matching injects `⚠️ CHAT-READ FLAGS`.
- If streamer reads a viewer message aloud, title/summary must attribute story to chatter:
  - ✅ "reads a chat message about ..."
  - ❌ "streamer reveals she ..."

### 3) Title Quality + Dedup

- Final title generation only in Stage 3.
- Title style guide uses 5 patterns (reaction, "the moment", question bait, etc.).
- Dedup is 3-layered:
  - `clip_id` anchors
  - `title_given` carryover context
  - explicit "no duplicate concept/title" rule + deterministic dedup pass

#### 3.1) Stage 3 Title Contract (Phase-1 Evidence -> Click Hook)

Title generation must be evidence-driven, not metadata-driven:

1. **Source of truth for title content**
   - `trigger`, `payoff`, `narrative_arc`, and evidence lines discovered in Stage 1/1.5.
   - Do **not** build titles from boilerplate wrappers like `"chat message from ..."`.

2. **Output split (important)**
   - `clip_point` = click-worthy hook (curiosity + specificity).
   - `intelligence_report.*` = dry factual reasoning is allowed and preferred.

3. **Chat-read title rule**
   - Keep attribution (story belongs to chatter), but avoid dry phrasing.
   - Preferred: hooky attribution forms (e.g., `"What happens when chat drops a message about ...?"`).
   - Avoid: `"Streamer reads a chat message about ..."` unless rewritten to be hooky.

4. **Deterministic post-check**
   - If model emits dry/recursive titles (e.g., `"...about chat message from ..."`), Stage 3 sanitizer rewrites topic wording before final output.

5. **Prompt requirement**
   - FINAL/PROVISIONAL prompts must explicitly tell Qwen to derive title angle from trigger+payoff evidence and score dry titles low.

### 4) Duration/Trim Policy

- Retention-first: shortest trim that preserves setup + payoff + standalone clarity.
- Penalty bands:
  - `<=60s`: 0
  - `61-75s`: -1
  - `76-90s`: -2
  - `>90s`: -3
- Prompt guidance is advisory; deterministic scorer is authoritative.

### 5) RMS Fallback Policy

RMS fallback runs **only** when all are true:
- selected trim still equals full candidate span,
- candidate width is exactly `120s`.

If RMS changes boundaries, pipeline must recompute:
- duration,
- dead-air penalties,
- final score,
- eligibility.

### Clip Criticism Penalty (Runs Inside Stage 2)

A deterministic sub-step that subtracts score based on **failure modes classified by Qwen in Stage 1**. Qwen outputs a `suggested_penalty` per failure; Stage 2 sums them with a -5.0 cap.

| Failure Category | Examples | Suggested Penalty Range |
|-----------------|----------|-----------------------|
| **A: Structural** | No hook, dead air front, front-loaded, no climax | -2.0 to -5.0 |
| **B: Context** | Context required, inside joke, chat-dependent | -1.5 to -5.0 |
| **C: Pacing/Length** | Too long, pacing slow, wrong length | -1.0 to -3.0 |
| **D: Transactional** | Transactional reaction, energy without content | -1.0 to -4.0 |
| **E: Technical** | Audio issues, no caption compat, wrong ratio | -0.5 to -3.0 |

**Total cap: -5.0.** No dedup rules in Python — Qwen judges overlap holistically. If penalty drops score below 3, clip is rejected with a `criticism_penalty` reason code.

**Reference:** `docs/references/clip-failure-classification-guide.md` for full taxonomy, prompt injection patterns, and implementation spec.

---

## Implementation Map (Source of Truth)

- Main pipeline: `src/synthesis/qwen_clip_analyzer_progressive.py`
- Stage schemas/contracts: `src/synthesis/schemas/clip_intelligence_stages.py`
- Stage 1 discovery helpers: `src/synthesis/stage1_discovery.py`
- Stage 1.5 stitching: `src/synthesis/stitching.py`
- Stage 1.5b audio normalization: `src/synthesis/audio_normalization.py`
- Shared context builder (dead-air + chat-read flags): `src/synthesis/clip_context.py`
- Deterministic scoring/gates: `src/synthesis/scoring.py`
- Title dedup/finalization: `src/synthesis/title_dedup.py`
- Extraction/upload/share automation: `src/synthesis/extract_and_upload_clips.py`

---

## Outputs

Primary run output:
- `phase4_<VOD_ID>/qwen_vision_progressive.json`

Expected major sections:
- `stage1_5_stitched`
- `stage2_scored`
- `stage3_final_selected`
- `final_ranking.final_selected_clips` (canonical post-gating set)

Per selected clip (final):
- `score` / `final_score`
- `suggested_trim_start`, `suggested_trim_end`
- `clip_point` (title)
- `platform_scores`
- `platform_recommendations`
- `intelligence_report`:
  - `why_selected`
  - `narrative_arc`
  - `evidence`
  - `trim_rationale`
  - `duration_fit`
  - `platform_fit`
  - `risks`
  - `streamer_feedback`

---

## Minimal Runbook

### Run analysis pipeline (WSL2)

```bash
cd ~/twitch-vod-analyzer
python3 src/synthesis/qwen_clip_analyzer_progressive.py --vod-id <VOD_ID>
```

Preflight check (Bee API readiness on port 8082):

```bash
curl -sS --max-time 3 http://100.97.240.34:8082/v1/models
```

If this fails, wait/retry before running analysis (cold start often takes ~1-2 minutes even when container status is `Up`).

Vision-only mode:

```bash
python3 src/synthesis/qwen_clip_analyzer_progressive.py --vod-id <VOD_ID> --skip-audio
```

Optional: limit audio review scope:

```bash
python3 src/synthesis/qwen_clip_analyzer_progressive.py --vod-id <VOD_ID> --top-clips 5
```

### Extract + upload + generate links

```bash
python src/synthesis/extract_and_upload_clips.py \
  --json qwen_vision_progressive.json \
  --vod raw/<VOD_ID>.mp4 \
  --min-score 7 \
  --output-dir ./clips
```

Dry-run:

```bash
python src/synthesis/extract_and_upload_clips.py \
  --json qwen_vision_progressive.json \
  --vod raw/<VOD_ID>.mp4 \
  --min-score 7 \
  --dry-run
```

Browser-compatible extraction settings (when done manually):
- H.264 Main profile, AAC audio
- `scale=854:480`
- `-movflags +faststart`

---

## Known Limitations

1. LLM can still drift on attribution/title nuance in edge cases.
2. Prompt constraints alone are not sufficient; deterministic post-filters are required.
3. Full end-to-end one-command orchestration (VOD ID -> share links) is still being hardened.
4. Integration tests may depend on optional runtime packages/environment not present in every container.
5. **Bee service must be started before pipeline run** — no auto-start yet (manual via `~/beellama.cpp/build/bin/llama-server …`). Cold start ~60s.
6. `response_format: json_object` not supported by llama.cpp — rely on prompt enforcement + `safe_json_parse` fallback. Verified working: clean JSON output when system prompt says "output ONLY valid JSON".

---

## Dev Workflow

### Pre-run: Ensure Bee server is running on WSL

```bash
# Check if running
curl -sS --max-time 3 http://100.97.240.34:8082/v1/models

# If not, start it (from WSL)
~/beellama.cpp/build/bin/llama-server \
  -m ~/models/bee-qwen36-27b/Qwen3.6-27B-Q5_K_S.gguf \
  --mmproj ~/models/bee-qwen36-27b/mmproj-BF16.gguf \
  --spec-draft-model ~/models/bee-qwen36-27b/dflash-draft-3.6-q4_k_m.gguf \
  --spec-type dflash --spec-dflash-cross-ctx 1024 \
  --port 8082 -np 1 --kv-unified -ngl all --spec-draft-ngl all \
  -b 2048 -ub 256 --ctx-size 200000 \
  --cache-type-k turbo4 --cache-type-v turbo3_tcq \
  --flash-attn on --cache-ram 0 --jinja --no-mmap --mlock \
  --reasoning on \
  --chat-template-kwargs '{"preserve_thinking":true}' \
  --temp 0.6 --top-k 20 --min-p 0.0
```

### Git workflow

- Hermes repo is source-of-truth for edits.
- Push to GitHub from Hermes.
- Pull on WSL2 before runs.

```bash
# Hermes side
cd /workspace/twitch-vod-lens
git add -A && git commit -m "..." && git push

# WSL2 side
cd ~/twitch-vod-analyzer && git pull
```

---

## Related Internal References

- `docs/CLIP-INTELLIGENCE-PIPELINE.md` (this compressed doc)
- `~/.hermes/skills/mlops/clip-intelligence-pipeline/SKILL.md`
- Skill references:
  - `references/stage-contracts-and-context-refactor.md`
  - `references/rms-rescore-and-audio-ordering.md`
  - `references/trim-gating-and-length-policy.md`
  - `references/nextcloud-clip-upload.md`
  - `references/viral-short-form-framework-research.md`
  - `references/clip-failure-classification-guide.md`
