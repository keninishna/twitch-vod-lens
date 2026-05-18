"""Stage 2 deterministic scoring and policy enforcement.

This module converts model analysis into deterministic scored candidates:
- penalties
- hard caps
- rejection reasons
- eligible_for_final gate
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.synthesis.schemas import validate_stage_payload


def _as_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value, default=0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _duration_penalty(trim_start: int, trim_end: int) -> Tuple[int, int]:
    """Return (penalty_points, duration_seconds)."""
    if trim_end <= trim_start:
        return 3, 0

    dur = trim_end - trim_start
    if 25 <= dur <= 60:
        return 0, dur
    if 20 <= dur <= 24 or 61 <= dur <= 75:
        return 1, dur
    if 15 <= dur <= 19 or 76 <= dur <= 90:
        return 2, dur
    return 3, dur


def _dead_air_inside_trim(dead_air_gaps: List[Dict], trim_start: int, trim_end: int) -> bool:
    for gap in dead_air_gaps:
        gs = _as_float(gap.get("start"), -1)
        ge = _as_float(gap.get("end"), -1)
        if ge <= gs:
            continue
        # overlap condition
        if gs < trim_end and ge > trim_start:
            return True
    return False


def normalize_clip_analysis(
    candidate: Dict,
    analysis: Dict,
    context: Dict,
    audio: Optional[Dict] = None,
) -> Dict:
    """Apply deterministic Stage 2 scoring and return ScoredCandidate payload."""

    audio = audio or {}

    start = _as_int(candidate.get("start"), 0)
    end = _as_int(candidate.get("end"), start)
    candidate_id = str(candidate.get("stitched_id") or candidate.get("candidate_id") or f"cand_{start}")

    raw_score = _as_float(analysis.get("clip_worthiness"), 0.0)
    raw_score = max(0.0, min(10.0, raw_score))

    trim_start = _as_int(analysis.get("suggested_trim_start"), start)
    trim_end = _as_int(analysis.get("suggested_trim_end"), end)

    penalties: List[Dict] = []
    hard_gates: List[Dict] = []
    rejection_reasons: List[str] = []
    hard_reject = False
    score_cap = 10.0

    def add_penalty(code: str, points: float):
        penalties.append({"code": code, "points": float(points)})

    def add_gate(code: str, action: str):
        hard_gates.append({"code": code, "action": action})

    def reject(code: str, action: str = "reject"):
        nonlocal hard_reject
        hard_reject = True
        if code not in rejection_reasons:
            rejection_reasons.append(code)
        add_gate(code, action)

    def cap(max_score: float, code: str):
        nonlocal score_cap
        score_cap = min(score_cap, float(max_score))
        add_gate(code, f"cap<={int(max_score) if float(max_score).is_integer() else max_score}")

    # Trim validation gates
    if end <= start:
        reject("invalid_candidate_window")

    if trim_end <= trim_start:
        reject("invalid_trim_range")

    if trim_start < start or trim_end > end:
        reject("trim_out_of_bounds")

    # Dead-air policies
    dead_air_gaps = context.get("dead_air_gaps") or []
    dead_air_ratio = _as_float(context.get("dead_air_ratio"), 0.0)

    longest_gap = 0.0
    for gap in dead_air_gaps:
        longest_gap = max(longest_gap, _as_float(gap.get("duration"), 0.0))

    if longest_gap > 10:
        add_penalty("dead_air_single_gap_gt_10", 5)
        cap(5, "dead_air_single_gap_gt_10")

    if dead_air_ratio > 0.30:
        cap(5, "dead_air_ratio_gt_30pct")

    if _dead_air_inside_trim(dead_air_gaps, trim_start, trim_end):
        cap(3, "dead_air_inside_trim")
        reject("dead_air_inside_trim")

    # Narrative/context gates
    has_payoff = bool(analysis.get("has_narrative_payoff", True))
    requires_context = bool(analysis.get("requires_context", False))
    transactional = bool(analysis.get("transactional_reaction", False))
    narrative_arc = str(analysis.get("narrative_arc") or "")

    if not has_payoff:
        cap(5, "no_narrative_payoff")
        reject("no_narrative_payoff")

    if requires_context:
        cap(5, "context_required")
        reject("context_required")

    if transactional and len(narrative_arc.strip()) < 20:
        cap(4, "transactional_without_narrative_arc")
        reject("transactional_without_narrative_arc")

    # Full unresolved 120s window policy
    candidate_width = end - start
    is_full_window = trim_start == start and trim_end == end
    trim_justification = str(analysis.get("trim_justification") or analysis.get("trim_start_reason") or "").strip()
    if candidate_width == 120 and is_full_window and len(trim_justification) < 20:
        cap(5, "full_window_unresolved")
        reject("full_window_unresolved")

    # Platform recommendation consistency
    platform_scores = analysis.get("platform_scores") or {}
    platform_recommendations = analysis.get("platform_recommendations") or []
    invalid_platform_recs = []
    for p in platform_recommendations:
        ps = _as_float(platform_scores.get(p), -1)
        if ps < 6:
            invalid_platform_recs.append(p)
    if invalid_platform_recs:
        reject("platform_recommendation_invalid", action="reject_bad_platform_recs")

    # Duration penalty
    duration_penalty, _dur = _duration_penalty(trim_start, trim_end)
    if duration_penalty > 0:
        add_penalty("duration_policy_penalty", duration_penalty)

    # Audio context integration (context signal, not direct selector).
    if bool(audio.get("dead_air_detected")):
        add_penalty("audio_dead_air_signal", 1)
    if bool(audio.get("music_only")) and not has_payoff:
        add_penalty("audio_music_only_without_payoff", 1)

    # Compute normalized/final score
    total_penalty = sum(_as_float(p.get("points"), 0.0) for p in penalties)
    final_score = raw_score - total_penalty
    final_score = min(final_score, score_cap)
    final_score = max(0.0, min(10.0, final_score))
    final_score = round(final_score, 4)

    trim_source = str(analysis.get("trim_source") or "qwen")
    if trim_source not in {"qwen", "rms_fallback", "python_corrected"}:
        trim_source = "python_corrected"

    eligible_for_final = (
        (not hard_reject)
        and final_score >= 3.0
        and trim_end > trim_start
        and start <= trim_start < trim_end <= end
    )

    scored = {
        "candidate_id": candidate_id,
        "start": start,
        "end": end,
        "final_score": final_score,
        "raw_score": raw_score,
        "eligible_for_final": eligible_for_final,
        "penalty_trace": penalties,
        "hard_gates": hard_gates,
        "rejection_reasons": rejection_reasons,
        "trim_source": trim_source,
    }

    validated = validate_stage_payload("scored", scored)
    return validated.model_dump()
