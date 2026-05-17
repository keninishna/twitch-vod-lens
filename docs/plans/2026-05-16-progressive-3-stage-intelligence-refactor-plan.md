# VOD Lens Clip Intelligence Refactor — Research-Driven Implementation Plan (Merged)

> **For Hermes:** Execute task-by-task with verification gates. Keep this as the source-of-truth implementation plan.

**Mission Statement:**
VOD Lens is a **research-driven intelligence pipeline that extracts multimodal intelligence from Twitch VODs to select the best clips**. The system should fuse transcript timing, chat behavior, frame/scene evidence, audio signals, and deterministic policy enforcement to consistently select clips with clear narrative payoff, clean trims, and strong platform fit.

**Primary integration target:** `src/synthesis/qwen_clip_analyzer_progressive.py`

---

## 1) Architecture and Stage Boundaries

### Stage 1 — Discovery (model-assisted, no final authority)
- Input: candidate windows from preprocessing/manifest
- Output: discovery evidence only (`trigger`, `payoff`, narrative type, confidence)
- **No title generation**
- **No final platform recommendations**

### Stage 1.5 — Cross-window stitching (deterministic Python)
- Merge adjacent discoveries that represent one story arc.
- Preserve provenance: `source_candidate_ids`, `source_windows`, `merge_reasons`.

### Stage 2 — Deterministic scoring + policy enforcement (Python)
- Compute `final_score` in code, not prompt text.
- Apply hard caps and rejection gates.
- Enforce hard gate: **only `final_score >= 8` moves forward**.

### Stage 3 — Final verification + title generation + intelligence report
- Generate titles only after Stage 2 gate.
- Suppress duplicate/near-duplicate concepts.
- Emit `intelligence_report` for each final clip.

### Stage 3.5 — Audio intelligence enrichment
- Keep model swap flow, but normalize outputs into structured fields.
- Audio augments scoring context; audio does not directly decide clip selection.

### Post-process — RMS trim fallback safety net
- Run only on unresolved full-window returns per existing gating policy.
- If fallback changes trim, recompute duration metadata and penalties.

---

## 2) Deterministic Scoring and Gating Contract

Implement in `src/synthesis/scoring.py`:

```python
def normalize_clip_analysis(candidate, analysis, context, audio=None) -> dict:
    ...
```

### Required normalized fields
- `raw_score`
- `normalized_score`
- `penalties[]`
- `hard_gates[]`
- `rejection_reasons[]`
- `eligible_for_final`
- `trim_source` (`qwen` | `rms_fallback` | `python_corrected`)

### Mandatory deterministic rules
- Single dead-air gap >10s: apply `-5`, cap score at 5.
- Total dead-air >30%: cap score at 5.
- Dead-air inside suggested trim: reject or force <=3 unless trim excludes gap.
- Transactional reaction without narrative arc: cap at 4.
- No narrative payoff: cap at 5.
- Context-required/not standalone: cap at 5.
- Invalid trim (`end <= start`) or impossible bounds: reject.
- Full unresolved 120s window without justification: cap at 5 and likely reject.
- Platform recommendations must be subset of platforms with score >=6.

### Duration penalty policy (retain)
- 25–60s: 0
- 20–24s or 61–75s: -1
- 15–19s or 76–90s: -2
- <15s or >90s: -3

### Hard shortlist gate
- `eligible_for_final == True` only if all are true:
  - `normalized_score >= 8`
  - clear trigger/payoff evidence
  - trim valid
  - no hard reject

---

## 3) Stage JSON Schemas (must be explicit)

Create `src/synthesis/schemas/clip_intelligence_stages.py` with validators for:

1. `DiscoveryCandidate`
2. `StitchedCandidate`
3. `ScoredCandidate`
4. `FinalSelectedClip`
5. `ClipContext`

Add `tests/test_stage_schemas.py` with strict field/typing validation.

---

## 4) Files and Module Split

### New modules
- `src/synthesis/clip_context.py`
  - Build structured context from transcript/chat/scenes/objects.
  - Emit `dead_air_gaps`, `dead_air_ratio`, `chat_read_flags`.

- `src/synthesis/stitching.py`
  - Deterministic cross-window merge logic + provenance.

- `src/synthesis/scoring.py`
  - Penalties, hard caps, normalization, eligibility gate.

- `src/synthesis/title_dedup.py`
  - Exact + normalized + near-duplicate suppression.

- `src/synthesis/schemas/clip_intelligence_stages.py`
  - Stage output contracts + validation helpers.

### Existing file modifications
- `src/synthesis/qwen_clip_analyzer_progressive.py`
  - Convert orchestration to explicit stage transitions.
  - Call scoring normalization after model outputs.
  - Re-run normalization after RMS fallback if trims changed.

---

## 5) Execution Plan (Bite-Sized Tasks)

## Phase A — Contracts first
1. Add stage schema module and tests.
2. Add typed context contract.
3. Add schema validation hooks at stage boundaries.

## Phase B — Discovery pipeline split
4. Extract `context_for_time()` into `clip_context.py`.
5. Ensure Stage 1 emits discovery-only payload (no titles/platform recs).
6. Add tests for dead-air detection and chat-read attribution flags.

## Phase C — Deterministic merge + scoring
7. Implement `stitching.py` with merge rules and provenance.
8. Implement `scoring.py` normalization + penalties + hard caps.
9. Apply hard `>=8` gate in code.
10. Capture `rejected_clips` with reason codes.

## Phase D — Finalization quality
11. Implement Stage 3 title generation post-gate only.
12. Implement title dedup (exact + near-duplicate).
13. Emit `intelligence_report` with required keys:
    - `why_selected`, `narrative_arc`, `evidence`, `trim_rationale`,
      `duration_fit`, `platform_fit`, `risks`, `streamer_feedback`.

## Phase E — RMS + audio normalization
14. Keep RMS fallback gated; if trim changes, recompute duration penalty and normalized score.
15. Normalize audio phase into structured fields (`speech_clarity`, `laughter_present`, etc.).

## Phase F — Regression and runbook
16. Add fixture-based regression set for VOD `2770929139`.
17. Add evaluation script (`scripts/regression_compare.py` or `src/synthesis/evaluate_clip_run.py`).
18. Compare before/after metrics: selected count, score distribution, duplicate rate, trim widths, RMS usage.
19. Update `docs/CLIP-INTELLIGENCE-PIPELINE.md` with actual implementation status.

---

## 6) Acceptance Criteria (Definition of Done)

- Stage 1 is discovery-only (no title/platform-finalization leakage).
- Stage 1.5 stitching works with provenance.
- Stage 2 deterministic scoring is authoritative.
- Only `final_score >= 8` reaches final outputs.
- Stage 3 emits deduped titles + full intelligence report.
- RMS fallback never overrides non-full-window trims incorrectly.
- JSON outputs are schema-valid.
- Regression shows improved consistency (fewer duplicates, fewer bad trims, better gate compliance).

---

## 7) Risks and Mitigations

- **Model non-compliance with prompt rules** → enforce all policy in Python.
- **JSON drift across prompt edits** → schema validation and defaults/repair path.
- **Chat attribution false positives/negatives** → keep deterministic matching + add fuzzy normalization tests.
- **Over-penalizing due to transcript sparsity** → combine transcript gaps with audio-confirmed silence when available.
- **RMS latching onto loud alerts** → keep RMS as fallback only; track `trim_source` explicitly.
- **Monolith maintenance risk** → split into `context`, `stitching`, `scoring`, `schemas`, `title_dedup`, orchestration.

---

## 8) Immediate Next Work Item

**Start here:** Build `src/synthesis/schemas/clip_intelligence_stages.py` and `tests/test_stage_schemas.py`, then wire validation into the current pipeline path before behavior changes.