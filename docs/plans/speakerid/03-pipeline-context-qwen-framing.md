# SpeakerID Phase 03 — Phase4 Integration, Clip Context, Qwen Speaker-Framing, and Final Outputs

> **For Hermes:** Load only this phase doc plus `INDEX.md` unless you need cross-phase details. Use `subagent-driven-development` to execute one task at a time.

**Goal:** Wire speaker attribution into phase4 validation, clip context rendering, Qwen prompts, and final selected clip outputs.

**Important policy:** Do **not** add deterministic Stage-2 speaker-specific penalties or hard gates. Qwen should infer and reframe streamer-reaction mismatches from prompt context.

**Depends on:** Phase 02 artifact.

**Tasks covered:** 10–13.

---

### Task 10: Integrate optional speaker attribution into phase4 prep

**Objective:** Let phase4 prep produce speaker attribution when requested.

**Files:**
- Modify: `src/preprocessing/prepare_phase4.py`
- Modify: `src/preprocessing/validate_phase4_inputs.py`
- Test: `tests/test_phase4_validation.py`

**Flags:**
```bash
--enable-speaker-id
--speaker-profiles-dir data/speaker_profiles
--require-speaker-id
```

**Validation rules:**
- Existing phase4 validation should pass without speaker attribution.
- If `--require-speaker-id`, require `speaker_attribution_<VOD_ID>.json` and validate schema.
- If artifact exists, include it in validation summary.

### Task 11: Extend clip context with speaker stats

**Objective:** Make speaker attribution available to Stage 1/2/3 without prompt hacks.

**Files:**
- Modify: `src/synthesis/schemas/clip_intelligence_stages.py`
- Modify: `src/synthesis/clip_context.py`
- Test: `tests/test_clip_context.py`

**Add fields to `ClipContext`:**
- `speaker_turns: list[SpeakerTurnLite]`
- `primary_speaker_label: str | None`
- `primary_speaker_identity: str | None`
- `primary_speaker_name: str | None`
- `streamer_speaking_seconds: float`
- `streamer_speaking_ratio: float`
- `streamer_speaking_confidence: float`
- `off_streamer_voice_detected: bool`
- `speaker_name_evidence: list[str]`

**Implementation notes:**
- Add optional `speaker_attribution` arg to `build_clip_context(...)`.
- Compute stats by overlap with clip window.
- Render prompt warning in `render_prompt_context(...)` when non-streamer is dominant:
  ```text
  ⚠️ SPEAKER ATTRIBUTION: Primary voice is SPEAKER_01 / guest / Skitch, not streamer. Do not title as streamer reaction unless streamer speech is the payoff.
  ```

### Task 12: Add Qwen speaker-framing inference prompts

**Objective:** Prompt Qwen to infer when a clip is titled/framed as a streamer reaction but the streamer is not actually speaking, without adding deterministic speaker penalties/gates.

**Files:**
- Modify: `src/synthesis/qwen_clip_analyzer_progressive.py`
- Modify: `src/synthesis/stage1_discovery.py` if Stage 1 prompt text is split there
- Test: `tests/test_prompt_templates.py`, `tests/test_stage1_discovery.py`

**Prompt behavior:**

Add speaker-attribution instructions to Stage 1 / Stage 3 prompts:

1. Use `speaker_attribution` context to infer whether the active speaker is the streamer, a guest, chat, or unknown.
2. If a draft/final title frames the clip as a streamer reaction but the streamer is not actually speaking, Qwen should reframe the title/report to the actual speaker or the situation.
3. If the clip's value depends on the streamer reaction but speaker evidence shows the streamer is absent, Qwen should explain the attribution risk in analysis fields and lower its own `clip_worthiness` only if that weakens the clip's standalone narrative value.
4. If the clip is guest-led but still has a clear story/payoff, Qwen may keep it as a valid clip with accurate attribution.
5. Do **not** add Python hard gates or deterministic speaker-specific score penalties for this behavior.

**Suggested output fields:**
```json
"speaker_framing_assessment": {
  "primary_speaker_identity": "streamer|guest|chat|unknown|mixed",
  "is_framed_as_streamer_reaction": false,
  "streamer_actually_speaking": true,
  "attribution_risk": "none|low|medium|high",
  "recommended_title_framing": "Attribute this to the guest, not the streamer.",
  "evidence": ["SPEAKER_01 dominates the trim; streamer voice not detected"]
}
```

**Verification:** prompt-template tests assert the instructions say to infer/reframe speaker attribution and explicitly do not add deterministic speaker penalties/gates.

### Task 13: Preserve speaker attribution in final outputs

**Objective:** Make reports and selected clips auditable.

**Files:**
- Modify: `src/synthesis/schemas/clip_intelligence_stages.py`
- Modify: `src/synthesis/title_dedup.py`
- Modify: `src/synthesis/qwen_clip_analyzer_progressive.py`
- Test: `tests/test_title_dedup.py`, `tests/test_stage_schemas.py`

**Add to final selected clips:**
```json
"speaker_attribution": {
  "primary_speaker_identity": "streamer",
  "primary_speaker_name": "Skitch",
  "streamer_speaking_ratio": 0.31,
  "streamer_speaking_confidence": 0.89,
  "off_streamer_voice_detected": true,
  "evidence": ["SPEAKER_00 matched streamer_skitch profile sim=0.84"]
}
```

**Stage 3 title rule:** if primary speaker is not streamer, title must not imply streamer is the active speaker unless streamer’s reaction is clearly present in the trim.
