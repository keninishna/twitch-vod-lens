## 1) Mission Statement

VOD Lens should be implemented as a **research-driven intelligence pipeline that extracts multimodal intelligence from Twitch VODs to select the best clips**: it fuses transcript timing, chat behavior, scene/frame evidence, object detections, audio facts, and deterministic scoring rules into a staged decision system that finds self-contained, high-retention Twitch moments with clear setup, payoff, platform fit, and extractable trim boundaries.

## 2) Target Architecture And Stage Boundaries

### Stage 1: Preprocessing Intelligence

Inputs: Twitch VOD, chat, audio/video stream.  
Outputs: `FusionResult`, `clip_manifest.json`.

Responsibilities:

- Download VOD and chat.
- Generate transcript with word/segment timestamps.
- Detect scenes and YOLO objects.
- Produce candidate windows.
- Preserve enough timing fidelity for later hard gates.

Boundary rule: this stage proposes candidate windows; it should not decide final clip quality.

### Stage 2: Candidate Context Builder

Primary code target: `context_for_time()` in `src/synthesis/qwen_clip_analyzer_progressive.py`.

Responsibilities:

- Build transcript excerpt with explicit timestamps.
- Attach nearby chat messages.
- Detect dead air gaps.
- Detect streamer-reading-chat attribution.
- Include candidate metadata: scene, objects, chat intensity, transcript density.

Implementation direction:

- Move `context_for_time()` out of `run()` into a testable helper module, e.g. `src/synthesis/clip_context.py`.
- Return structured data, not just prompt strings:
  - `transcript_lines`
  - `dead_air_gaps`
  - `total_dead_air_seconds`
  - `dead_air_ratio`
  - `chat_read_flags`
  - `chat_messages`
  - `objects_detected`

### Stage 3: Per-Clip Multimodal Analysis

Primary code target: `ANALYSIS_PROMPT`, `qwen_call()`, `sample_clip_frames()`.

Responsibilities:

- Analyze 6 sampled frames plus structured context.
- Generate narrative classification, title, trim suggestion, platform scores.
- Return raw model judgment only.

Boundary rule: Qwen proposes; Python validates and corrects.

### Stage 3.5: Audio Intelligence

Current design is correct: run top N candidates through Qwen2.5-Omni or equivalent audio model.

Responsibilities:

- Detect laughter, excitement, silence, alert sounds, music, speech clarity.
- Add audio facts to synthesis context.
- Do not let audio model directly select clips.

Implementation direction:

- Normalize audio results into structured fields instead of free text:
  - `speech_clarity`
  - `laughter_present`
  - `alert_sound_present`
  - `music_only`
  - `dead_air_confirmed`
  - `audio_energy_summary`
  - `audio_clip_value_delta`

### Stage 4: Deterministic Score Normalization

This should become an explicit Python stage between model output and synthesis.

Responsibilities:

- Apply hard caps.
- Apply penalties.
- Fix trim duration metadata.
- Enforce platform recommendation consistency.
- Add machine-readable rejection reasons.

Suggested module: `src/synthesis/scoring.py`.

### Stage 5: Provisional Synthesis

Responsibilities:

- Rank normalized per-clip analyses.
- Request extra frames only for uncertainty.
- No final authority until deterministic post-checks run again.

### Stage 6: Frame Review

Responsibilities:

- Resolve visual uncertainty for at most 3 clips per round.
- Merge revised evidence into the clip analysis.

Implementation note: cap `MAX_FRAME_REQUEST_ROUNDS` much lower than 20, likely `2`, because the prompt itself says up to 3 uncertain clips and runaway review adds little value.

### Stage 7: Final Synthesis And Post-Processing

Responsibilities:

- Produce final selected clips.
- Re-run deterministic gates on final output.
- Apply RMS fallback only for unresolved full-window trims.
- Save final contract.

Important change: after RMS changes `suggested_trim_start/end`, recompute:

- `trim_duration_seconds`
- `duration_penalty_applied`
- final score if duration penalty changes
- `trim_source = "qwen" | "rms_fallback" | "python_corrected"`

## 3) Deterministic Scoring/Gating Design

Implement a Python function:

```python
def normalize_clip_analysis(candidate, analysis, context, audio=None) -> dict:
    ...
```

It should return:

```json
{
  "raw_score": 8,
  "normalized_score": 5,
  "hard_gates": [],
  "penalties": [],
  "rejection_reasons": [],
  "eligible_for_final": false
}
```

Concrete gates:

| Rule | Action |
|---|---|
| Single dead air gap > 10s inside candidate | `-5`, max score `5` |
| Total dead air > 30% of window | max score `5` |
| Dead air inside suggested trim | reject or force score `<=3` unless trim excludes it |
| `transactional_reaction` without explanation arc | max score `4` |
| `has_narrative_payoff == false` | max score `5` |
| `requires_context == true` and not standalone | max score `5` |
| Generic title | max score `5` |
| Full 120s trim without strong justification | max score `5`; likely exclude |
| Invalid trim or end <= start | reject |
| Trim outside candidate bounds | clamp and add penalty |
| Duplicate `clip_point` | require rewrite or demote duplicate |
| Platform recommendation score < 6 | remove that platform |

Duration penalty should stay as currently encoded:

```python
25-60s   -> 0
20-24s   -> -1
61-75s   -> -1
15-19s   -> -2
76-90s   -> -2
<15/>90s -> -3
```

Final selection rule:

```python
eligible_for_final = (
    normalized_score >= 7
    and not hard_rejected
    and has_clear_trigger
    and has_clear_payoff
    and trim_valid
)
```

Borderline exception:

```python
score in [5, 6] may pass only if:
- one platform score >= 8
- narrative payoff is true
- no dead-air or trim gate fired
- final output marks it as platform-specific
```

## 4) JSON Schema Contracts By Stage

### Per-Clip Analysis Contract

```json
{
  "clip_start": 0,
  "clip_end": 120,
  "person_visible": true,
  "face_visible": true,
  "primary_expression": "laughing",
  "visible_objects": [],
  "scene_description": "string",
  "streamer_activity": "string",
  "emotional_energy": 1,
  "visual_interest": 1,
  "clip_worthiness": 1,
  "clip_point": "string",
  "narrative_type": "storytelling",
  "has_narrative_payoff": true,
  "requires_context": false,
  "suggested_trim_start": 12,
  "suggested_trim_end": 52,
  "trim_duration_seconds": 40,
  "duration_penalty_applied": 0,
  "trim_start_reason": "string",
  "trim_end_reason": "string",
  "narrative_arc": "string",
  "comparative_note": "string",
  "platform_scores": {
    "tiktok": 7,
    "shorts": 7,
    "twitter": 8,
    "twitch": 9,
    "reels": 7
  },
  "platform_reasoning": {
    "tiktok": "string",
    "shorts": "string",
    "twitter": "string",
    "twitch": "string",
    "reels": "string"
  },
  "platform_recommendations": ["tiktok"],
  "reason": "string"
}
```

### Context Contract

```json
{
  "clip_start": 0,
  "clip_end": 120,
  "transcript_lines": [
    {"start": 10.2, "end": 14.8, "text": "string"}
  ],
  "chat_messages": [
    {"timestamp": 12.0, "user": "name", "message": "string"}
  ],
  "chat_read_flags": [
    {
      "timestamp": 12.0,
      "user": "name",
      "message": "string",
      "matched_transcript": "string"
    }
  ],
  "dead_air_gaps": [
    {"start": 30.0, "end": 43.0, "duration": 13.0}
  ],
  "total_dead_air_seconds": 13.0,
  "dead_air_ratio": 0.108,
  "objects_detected": ["person"]
}
```

### Normalized Scoring Contract

```json
{
  "clip_start": 0,
  "raw_score": 8,
  "normalized_score": 5,
  "duration_seconds": 40,
  "duration_penalty": 0,
  "penalties": [
    {"code": "single_dead_air_gt_10s", "points": -5}
  ],
  "hard_gates": [
    {"code": "dead_air_inside_trim", "action": "reject"}
  ],
  "rejection_reasons": [],
  "eligible_for_final": false,
  "trim_source": "qwen"
}
```

### Final Ranking Contract

```json
{
  "vod_id": "2770929139",
  "final_selected_clips": [
    {
      "rank": 1,
      "start": 100,
      "end": 220,
      "score": 8,
      "raw_score": 9,
      "normalized_score": 8,
      "why": "string",
      "clip_point": "string",
      "narrative_type": "chat_banter",
      "suggested_trim_start": 118,
      "suggested_trim_end": 158,
      "trim_duration_seconds": 40,
      "duration_penalty_applied": 0,
      "trim_source": "qwen",
      "platform_scores": {},
      "platform_recommendations": ["tiktok", "twitch"],
      "penalties": [],
      "hard_gates": []
    }
  ],
  "rejected_clips": [
    {
      "start": 652,
      "score": 4,
      "rejection_reasons": ["single_dead_air_gt_10s"]
    }
  ],
  "overall_vod_assessment": "string",
  "total_clips_evaluated": 0
}
```

## 5) Evaluation And Regression Methodology

Create a small golden dataset under something like:

```text
tests/fixtures/vods/2770929139/
```

Include:

- `fusion_result_sample.json`
- `clip_manifest_sample.json`
- representative `qwen_raw_outputs.json`
- expected `normalized_outputs.json`

Regression tests should cover:

1. Dead air gap >10s demotes score to max 5.
2. Dead air inside trim rejects or forces score <=3.
3. Transactional reaction without explanation caps at 4.
4. No narrative payoff caps at 5.
5. Invalid trim is rejected.
6. Duration penalties are recomputed correctly.
7. Duplicate titles are detected.
8. Platform recommendations are removed when score <6.
9. Chat-read flags prevent wrong attribution.
10. RMS fallback updates trim duration and trim source.

Add an offline evaluation command:

```bash
python -m src.synthesis.evaluate_clip_run \
  --input vods/phase4_2770929139/qwen_vision_progressive.json \
  --fusion vods/phase4_2770929139/fusion_result_2770929139.json
```

It should report:

- selected count
- rejected count
- average score
- score changes from penalties
- clips with full-window trims
- duplicate titles
- clips with dead air in final trim
- platform recommendation inconsistencies

## 6) Execution Roadmap In Bite-Sized Tasks

1. Extract `context_for_time()` into `src/synthesis/clip_context.py`.
2. Add `ClipContext` dataclass or plain typed dict.
3. Add `src/synthesis/scoring.py` with `duration_penalty_seconds()`, hard gates, and score normalization.
4. Call normalization immediately after each Qwen per-clip response.
5. Store both `raw_analysis` and `normalized_analysis`.
6. Update `build_analysis_log_entry()` to include normalized score, penalties, and gates.
7. Normalize provisional synthesis outputs before frame requests.
8. Normalize final synthesis outputs before saving.
9. After RMS fallback, recompute trim duration, duration penalty, and score.
10. Add `rejected_clips` to output JSON.
11. Add schema validation with `jsonschema` or Pydantic.
12. Add fixture-based unit tests for scoring and context extraction.
13. Add an evaluation CLI for regression summaries.
14. Reduce frame request rounds from `20` to `2`.
15. Replace free-text audio analysis with structured audio fields.

## 7) Risks And Mitigations

| Risk | Mitigation |
|---|---|
| Qwen ignores scoring rules | Treat Qwen output as raw evidence; enforce all caps in Python |
| Prompt JSON drifts over time | Add schema validation and repair/default paths |
| Good clips are over-penalized by transcript gaps | Distinguish transcript silence from audio-confirmed silence when audio data exists |
| Chat attribution remains brittle | Replace substring-only matching with normalized fuzzy matching |
| RMS fallback picks loud alert instead of good moment | Only use RMS for full-window trims and label `trim_source = "rms_fallback"` |
| Duplicate titles pass synthesis | Track normalized title strings and reject/rewrite duplicates |
| Platform scores become decorative | Enforce `platform_recommendations ⊆ scores >= 6` |
| Model swap is operationally fragile | Isolate audio phase behind a resumable artifact: `audio_batch_input.json` → `audio_batch_output.json` |
| Regression quality is subjective | Maintain golden clips with expected gates, scores, and reject reasons |
| Long script becomes hard to maintain | Split into `context`, `scoring`, `schemas`, `audio`, `prompts`, and `runner` modules |