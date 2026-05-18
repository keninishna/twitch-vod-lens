# Clip Failure Classification Guide (for Qwen Injection)

> **Purpose:** A structured taxonomy of *why* a short-form clip fails as standalone content. Injected into Stage 1/3 prompts so Qwen can classify what a clip is MISSING and suggest a penalty. Stage 2 applies the suggestion (sanity-capped).
>
> **Research Sources:** Clypse.ai (Twitch clip virality), Livecounts.io (Shorts retention), GetKoro.app (algorithm diagnostics), Shortimize (2026 retention benchmarks), Slate (clip economy/context collapse), LinkedIn/Instagram creator case studies, Logan E. Smith HPC framework.
>
> **Last Updated:** May 18, 2026

---

## Principles (Injected Context)

A standalone short-form clip from a stream fails for one or more of these reasons:

1. **The viewer does not understand what they are watching within 1-3 seconds.**
2. **The viewer has no reason to keep watching after the first 3 seconds.**
3. **The payoff does not justify the time spent watching.**
4. **The clip requires outside knowledge (stream lore, chat context, backstory, inside jokes).**
5. **The clip lacks a fundamental structural element (hook, progression, or climax).**
6. **The clip has correctable technical failures (length, pacing, audio, captions).**

---

## Failure Mode Categories

Each failure is classified by **Type** (what is wrong), **Severity** (how damaging), **Detection Signal** (what to look for), and **suggested penalty range** (reference for the model — the model outputs its own judgment).

### Category A: Structural Failures (Fatal without re-trim)

These are the most common killers. A clip with any of these is unlikely to retain viewers even with good content.

| ID | Failure Mode | Definition | Severity | Detection Signal | Suggested Penalty |
|----|-------------|-----------|----------|-----------------|-------------------|
| A1 | **No Hook** | First 3 seconds lack pattern interrupt, curiosity, or topic signal. Viewer can't tell what the clip is about. | **Critical** | Starts with silence, slow setup, greeting ("hey guys"), establishing shot, or unrelated preamble. First frame is visually ambiguous. | -3.0 to -5.0 |
| A2 | **Dead Air Front** | Extended silence or dead time in the opening 10 seconds. No speech, no action, no audio event. | **Critical** | Transcript shows `[silence]` or gaps >2s in first 10s. | -3.0 to -5.0 |
| A3 | **Front-Loaded (Best Moment First)** | The most interesting or climactic moment occurs in the first 10 seconds. Nothing that follows surpasses it. | **High** | Trigger/payoff score in first segment exceeds all later segments. Trigger and payoff are the same timestamp. | -2.5 to -4.0 |
| A4 | **No Progression** | Clip has flat energy throughout. No build, no tension curve, no narrative arc. It's a single emotional level start to finish. | **High** | Narrative_type is "reaction" without setup. Trigger and payoff are identical concepts. Progression score <3. | -2.0 to -3.5 |
| A5 | **No Climax / Flat Payoff** | Clip builds toward nothing. The ending is no more abrupt than satisfying. Payoff doesn't match setup. | **High** | Last 5 seconds contain no emotional peak, laugh, or resolution. Ending is mid-sentence or mid-action with no punchline. | -2.0 to -3.5 |
| A6 | **Broken HPC** | Two of the three HPC elements (Hook, Progression, Climax) are missing or weak. | **Critical** | HPC Score < 4. Multiple structural elements are absent. | -3.5 to -5.0 |

### Category B: Context Failures (Standalone Viability)

These determine whether a clip works outside the stream community.

| ID | Failure Mode | Definition | Severity | Detection Signal | Suggested Penalty |
|----|-------------|-----------|----------|-----------------|-------------------|
| B1 | **Context Required** | Clip only makes sense with knowledge of stream lore, ongoing inside jokes, or earlier conversation not in the clip. | **Critical** | References to past stream events, named entities without introduction, reactions to unseen earlier segment. | -3.0 to -5.0 |
| B2 | **Inside Joke Dependence** | The humor/impact relies on knowing a specific recurring bit, community meme, or streamer personality trait. | **High** | Chat reactions reference past events. Streamer refers to "like last time" or "as always." Clip needs a title to explain the joke. | -2.0 to -3.5 |
| B3 | **Chat-Dependent Context** | The moment only makes sense if you see chat reactions (not included in clip). Streamer reacting to something chat said that isn't in the transcript. | **High** | Streamer responds to chat but the chat message being responded to isn't visible in frames. Read chat message with no setup. | -2.0 to -3.5 |
| B4 | **Topic Too Narrow** | The clip topic is interesting only to an extremely niche audience (e.g., a specific game mechanic only 0.1% of viewers understand). | **Medium** | Topic requires domain expertise (esports meta, obscure game tech, industry jargon). Platform_fitness low except Twitch. | -1.5 to -2.5 |

### Category C: Pacing & Length Failures

Even good content fails if it takes too long to deliver.

| ID | Failure Mode | Definition | Severity | Detection Signal | Suggested Penalty |
|----|-------------|-----------|----------|-----------------|-------------------|
| C1 | **Too Long for Content** | Clip duration exceeds the amount of interesting content. Viewer hits an empty tail. | **Medium** | After payoff, there are >3 seconds of dead time, outro, or trailing reaction. | -1.5 to -3.0 |
| C2 | **Pacing Too Slow** | Gaps between interesting moments are too long. Viewer gets bored between beats. | **Medium** | Transcript has multiple gaps >3s between narrative-relevant dialogue. | -1.5 to -3.0 |
| C3 | **Pacing Too Fast / Jarring** | Cuts are too rapid. Viewer can't follow the narrative. | **Low** | Frame-to-frame changes are 100% unrelated. No continuity between segments. | -1.0 to -2.0 |
| C4 | **Wrong Length for Platform** | Too long for TikTok/Shorts retention curves, too short for a story-driven clip to land. | **Medium** | 15-30s optimal for single-moment. >30s needs story justification. <10s rarely works. | -1.0 to -2.0 |

### Category D: Transactional / Low-Value Content

These clips score well on "something happened" but fail retention because they're generic.

| ID | Failure Mode | Definition | Severity | Detection Signal | Suggested Penalty |
|----|-------------|-----------|----------|-----------------|-------------------|
| D1 | **Transactional Reaction** | Generic reaction to a donation/sub/alert with no narrative framing. Laughing at a donation on screen is not a story. | **High** | Streamer reacts to on-screen text donation. No framing, no explanation, no follow-up. | -2.0 to -4.0 |
| D2 | **Generic Interaction** | Chat interaction that is standard/expected. No surprise, no humor, no unexpected outcome. | **Medium** | Streamer reads chat message, responds as expected. Nothing surprising happens. | -1.5 to -2.5 |
| D3 | **Uninteresting Gameplay** | Clip is just gameplay achievements without emotional moment. No narrative beyond "good play happened." | **Medium** | Focus is on screen action rather than personality/interaction. Trigger is gameplay event with no human reaction. | -1.0 to -2.0 |
| D4 | **Energy Without Content** | High emotional energy (screaming, laughing, overreacting) but the underlying reason is trivial or not communicated. | **High** | Strong laughter/screaming but no clear trigger in the clip that explains why. Viewer feels left out. | -2.0 to -3.5 |

### Category E: Technical/Format Failures

These are platform-specific issues that prevent distribution.

| ID | Failure Mode | Definition | Severity | Detection Signal | Suggested Penalty |
|----|-------------|-----------|----------|-----------------|-------------------|
| E1 | **No Caption Compatibility** | Dialogue-heavy clip that requires captions to follow (85% watch muted) but pipeline can't add captions. | **Medium** | Dense dialogue clip. Platform score for TikTok/Shorts high but no text-to-speech/caption pipeline. | -1.0 to -2.0 |
| E2 | **Audio Quality Issue** | Echo, background noise, music drowning voice, mic clipping. Audio makes clip unpleasant. | **Medium** | Transcript has garbled sections. Music detected at >30% of RMS peaks. Multiple "inaudible" markers. | -1.5 to -3.0 |
| E3 | **Wrong Aspect Ratio** | Content that cannot be cropped to 9:16 without losing key visual information. | **Low** | Frame analysis shows key action at extreme horizontal edges. Horizontal-native layout. | -0.5 to -1.5 |

---

## Penalty Calculation (Stage 2 Sanity Cap)

**No lookup table or dedup logic in Python.** Qwen owns the penalty by outputting a `suggested_penalty` per failure mode. Stage 2 just sums them with a safety cap.

```python
def apply_criticism_penalty(clip: dict) -> dict:
    """Apply clip criticism penalties from Stage 1 failure analysis.

    Qwen outputs a `suggested_penalty` per failure mode (e.g. -3.0).
    Stage 2 sums them, caps at -5.0 as a safety limit, and
    subtracts from final_score.

    Args:
        clip: Clip dict with clip['failure_modes'] array

    Returns:
        Updated clip dict with criticism_penalty and updated final_score
    """
    failures = clip.get('failure_modes', [])
    if not failures:
        clip['criticism_penalty'] = 0.0
        return clip

    # Sum suggested penalties from model output
    total = sum(
        min(f.get('suggested_penalty', 0.0), 0.0)  # ensure negative
        for f in failures
    )

    # Safety cap so one overzealous run can't zero everything
    total = max(total, -5.0)

    clip['criticism_penalty'] = total
    clip['criticism_raw_failures'] = [f['failure_id'] for f in failures]
    clip['final_score'] = clip.get('final_score', 10.0) + total  # total is negative

    # Re-check eligibility
    if 'eligible_for_final' in clip:
        clip['eligible_for_final'] = clip['final_score'] >= 3

    return clip
```

**Why this approach:**
- Qwen sees the full clip context (frames + transcript + chat) and can judge severity holistically
- Two "No Progression" clips can have different penalties depending on how badly the pacing fails
- No brittle lookup table or dedup rules to maintain
- The -5.0 sanity cap is the only safety net, preventing an over-critical run from destroying all scores

---

## Prompt Injection Pattern

Add the following block to Stage 1 (ANALYSIS_PROMPT) and Stage 3 (FINAL_SYNTHESIS_PROMPT):

```
### CLIP CRITICISM CLASSIFICATION

Analyze this clip for FAILURE MODES that would cause it to underperform as standalone
short-form content. Identify ALL applicable failure modes from the taxonomy below.

For each identified failure, output:
- failure_id: The ID code (e.g. "A1")
- failure_name: Short name
- severity: Critical / High / Medium / Low
- suggested_penalty: Float from -1.0 to -5.0 indicating how much this
  failure should reduce the clip's score. Be aggressive — a clip with
  critical failures should not pass the gate. Harsher is safer here.
- evidence: Specific evidence from transcript, frames, or clip metadata
- fix_suggestion: What would fix it (re-trim, add context, discard, etc.)

Output format (JSON array):
[
  {
    "failure_id": "A1",
    "failure_name": "No Hook",
    "severity": "Critical",
    "suggested_penalty": -4.0,
    "evidence": "Clip opens with 4s of silence before streamer speaks",
    "fix_suggestion": "Trim start to 4s to begin at first speech, or discard if no hook exists"
  }
]

Failure mode taxonomy:
(Inject the full Category A-E tables here)

IMPORTANT — Penalty guidance:
- The suggested_penalty values in the taxonomy are RANGES, not fixed amounts.
- Use your judgment: a clip with mild "No Hook" (misses by 1 second) = -3.0.
  A clip with severe "No Hook" (completely opaque opening) = -5.0.
- Overlap is fine — if multiple failures apply, each gets its own penalty.
  Stage 2 will sum them with a safety cap at -5.0 total.
- Be harsher rather than gentler. Clips that barely pass become
  forgettable content that wastes extraction effort.
```

---

## Stage 2 Integration

Runs as a sub-step inside **Stage 2**, after duration/dead-air penalties but before the hard gate.

```python
# In Stage 2 scoring pipeline:

def stage_2_scoring_pipeline(clip: dict) -> dict:
    """Full Stage 2 pipeline for one clip."""

    # 2a: Base penalties (duration, dead air)
    clip = apply_duration_penalty(clip)
    clip = apply_dead_air_penalty(clip)

    # 2b: Clip criticism (model-suggested)
    clip = apply_criticism_penalty(clip)

    # 2c: Hard gate
    clip['passed_gate'] = clip.get('final_score', 0) >= 3
    if not clip['passed_gate']:
        clip['gate_reason'] = (
            f"criticism_penalty={clip['criticism_penalty']}"
            if clip.get('criticism_penalty', 0) < 0
            else "below threshold"
        )

    return clip
```

---

## Diagnostic Logging

Each run emits a debug entry for operator review:

```json
{
  "stage": "2_criticism",
  "clip_id": 998,
  "failures": [
    {"failure_id": "A1", "suggested_penalty": -4.0},
    {"failure_id": "B1", "suggested_penalty": -2.0}
  ],
  "raw_sum": -6.0,
  "safety_cap_applied": true,
  "criticism_penalty_applied": -5.0,
  "score_before_criticism": 8.5,
  "score_after_criticism": 3.5,
  "passed_gate": false,
  "gate_reason": "criticism_penalty=-5.0"
}
```

---

## Flow Summary

```
Stage 1 (Discovery)
  ├── HPC evaluation (hook, progression, climax scores)
  └── Failure mode classification with suggested_penalty per failure

Stage 1.5 (Stitching)
Stage 1.5b (Audio Normalization)

Stage 2 (Scoring + Penalties + Hard Gate)
  ├── Duration penalty
  ├── Dead air penalty
  ├── Clip criticism penalty
  │     ├── Collect suggested_penalty from each failure mode
  │     ├── Sum with -5.0 safety cap (no dedup — Qwen handles overlap)
  │     ├── Subtract from final_score
  │     └── Re-check eligibility >= 3
  └── Hard gate (score >= 3 → proceed, < 3 → log and reject)

Stage 3 (Finalization)
```