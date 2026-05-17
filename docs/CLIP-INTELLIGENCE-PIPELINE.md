# VOD Lens — Clip Intelligence Pipeline (Compressed)

> **Status:** Active Development
> **Last Updated:** May 17, 2026 (Stage 1 direct title contract + WSL2/Qwen readiness troubleshooting)
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
  -> Stage 1: Discovery (LLM, no final platform decisions; emits draft title fields for carryover)
  -> Stage 1.5: Deterministic cross-window stitching
  -> Stage 1.5b: Audio normalization to structured flags
  -> Stage 2: Deterministic scoring + penalties + hard gate
  -> Stage 3: Final verification + title generation + dedup + intelligence report
  -> Post: RMS trim fallback (only unresolved full 120s windows) + mandatory rescoring
  -> Clip extraction + Nextcloud upload + public share links
```

### Current Workstream (May 17, 2026)

1. **Stage 1 title quality pass (active)**
   - Added a compact research brief to Stage 1 prompt (YouTube/NNg/curiosity-gap guidance).
   - Stage 1 now emits direct draft title fields:
     - `clip_point`
     - `title_why`
   - Removed provisional-named fields/fallback layering for this path.

2. **Qwen/vLLM reliability hardening (active)**
   - Verified current WSL2 endpoint path is `100.97.240.34:8000`.
   - Observed real failure mode: WSL2/Tailscale offline + vLLM cold start window.
   - During cold start, container can be `Up` while `/v1/models` still returns connection refused for ~1-2 minutes.
   - Next stabilization step: add deterministic preflight wait-for-readiness gate before batch analysis.

### Stage Invariants (Non-Negotiable)

1. **Stage 1 is discovery-only**
   - No final platform posting decisions
   - Draft `clip_point` / `title_why` allowed for carryover context
2. **Stage 2 is deterministic enforcement**
   - Hard gate: `score >= 8` to proceed
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

WSL2/Qwen readiness preflight (recommended):

```bash
curl -sS --max-time 3 http://100.97.240.34:8000/v1/models
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
5. Qwen service readiness lag: after WSL/container restart, `/v1/models` may be unavailable briefly while vLLM loads.

---

## Dev Workflow

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
