"""Stage 3 title generation + duplicate suppression helpers."""

from __future__ import annotations

import re
from typing import Dict, List

from src.synthesis.schemas import validate_stage_payload

_WORD_RE = re.compile(r"[a-z0-9']+")


def normalize_title(title: str) -> str:
    words = _WORD_RE.findall((title or "").lower())
    return " ".join(words)


def _normalize_token(token: str) -> str:
    t = token.lower()
    for suffix in ("ing", "ed", "es", "s"):
        if len(t) > 4 and t.endswith(suffix):
            return t[: -len(suffix)]
    return t


def _token_set(title: str):
    return {_normalize_token(w) for w in _WORD_RE.findall((title or "").lower()) if len(w) >= 3}


def is_near_duplicate_title(a: str, b: str, threshold: float = 0.75) -> bool:
    na = normalize_title(a)
    nb = normalize_title(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if na in nb or nb in na:
        return True

    sa = _token_set(a)
    sb = _token_set(b)
    if not sa or not sb:
        return False

    overlap = len(sa.intersection(sb))
    union = len(sa.union(sb))
    jaccard = overlap / union if union else 0.0
    return jaccard >= threshold


def _generate_title(stitched: Dict, analysis: Dict) -> str:
    clip_point = str(analysis.get("clip_point") or "").strip()
    if clip_point:
        return clip_point

    narrative_type = str(stitched.get("narrative_type") or "moment")
    trigger = str(stitched.get("trigger") or "a key moment")
    payoff = str(stitched.get("payoff") or "a strong payoff")

    if "chat" in narrative_type:
        return f"Streamer reads a chat message about {trigger[:60].lower()}"

    return f"The moment {trigger[:45].lower()} led to {payoff[:45].lower()}"


def finalize_stage3_candidates(
    scored_candidates: List[Dict],
    stitched_candidates: List[Dict],
    analysis_by_candidate: Dict[str, Dict],
    min_score: float = 8.0,
    max_clips: int = 20,
    fallback_top_n_when_empty: int = 0,
) -> List[Dict]:
    """Return ranked final clip payloads with duplicate suppression.

    Normal path: only eligible clips at/above min_score.
    Optional fallback: when normal path is empty, take top-N by final_score.
    """

    stitched_by_id = {s.get("stitched_id"): s for s in stitched_candidates}

    prelim = [
        s for s in scored_candidates
        if s.get("eligible_for_final") and float(s.get("final_score", 0)) >= float(min_score)
    ]

    if not prelim and fallback_top_n_when_empty > 0:
        scored_with_stitched = [
            s for s in scored_candidates if stitched_by_id.get(s.get("candidate_id")) is not None
        ]
        scored_with_stitched.sort(key=lambda x: float(x.get("final_score", 0)), reverse=True)
        prelim = scored_with_stitched[: int(fallback_top_n_when_empty)]

    prelim.sort(key=lambda x: float(x.get("final_score", 0)), reverse=True)

    selected: List[Dict] = []
    seen_titles: List[str] = []

    for scored in prelim:
        cid = scored.get("candidate_id")
        stitched = stitched_by_id.get(cid)
        if not stitched:
            continue

        analysis = analysis_by_candidate.get(cid, {})

        start = int(stitched.get("start", 0))
        end = int(stitched.get("end", start))

        trim_start = int(float(analysis.get("suggested_trim_start", start)))
        trim_end = int(float(analysis.get("suggested_trim_end", end)))

        # Final verification: trim must be valid and within candidate bounds.
        if trim_end <= trim_start:
            continue
        if trim_start < start or trim_end > end:
            continue

        title = _generate_title(stitched, analysis)

        if any(is_near_duplicate_title(title, prev) for prev in seen_titles):
            continue

        seen_titles.append(title)

        platform_scores = analysis.get("platform_scores") or {}
        platform_recommendations = analysis.get("platform_recommendations") or []

        dur = trim_end - trim_start
        if 25 <= dur <= 60:
            duration_fit = f"{dur}s optimal (25-60s band)"
        elif 20 <= dur <= 24 or 61 <= dur <= 75:
            duration_fit = f"{dur}s acceptable with minor duration penalty"
        elif 15 <= dur <= 19 or 76 <= dur <= 90:
            duration_fit = f"{dur}s risky duration band"
        else:
            duration_fit = f"{dur}s poor duration fit"

        if platform_recommendations:
            platform_fit = "Recommended for " + ", ".join(platform_recommendations)
        else:
            platform_fit = "No strong platform recommendation"

        risks: List[str] = []
        if bool(analysis.get("requires_context", False)):
            risks.append("context_required")
        if float(stitched.get("confidence", 0.0)) < 0.7:
            risks.append("low_discovery_confidence")
        if not platform_recommendations:
            risks.append("no_platform_recommendation")

        trigger = str(stitched.get("trigger") or "")
        payoff = str(stitched.get("payoff") or "")
        trim_start_reason = str(analysis.get("trim_start_reason") or "").strip()
        trim_end_reason = str(analysis.get("trim_end_reason") or "").strip()
        trim_rationale = (
            f"Start: {trim_start_reason or 'model boundary'}; "
            f"End: {trim_end_reason or 'model boundary'}"
        )

        payload = {
            "rank": len(selected) + 1,
            "clip_id": str(cid),
            "start": start,
            "end": end,
            "suggested_trim_start": trim_start,
            "suggested_trim_end": trim_end,
            "trim_source": scored.get("trim_source") or "qwen",
            "score": float(scored.get("final_score", 0.0)),
            "raw_score": float(scored.get("raw_score", 0.0)),
            "normalized_score": float(scored.get("final_score", 0.0)),
            "clip_point": title,
            "narrative_type": str(stitched.get("narrative_type") or "unknown"),
            "platform_scores": platform_scores,
            "platform_recommendations": platform_recommendations,
            "intelligence_report": {
                "why_selected": (
                    f"Deterministic score {float(scored.get('final_score', 0.0)):.1f}/10 "
                    f"with clear trigger/payoff evidence and non-duplicate title concept."
                ),
                "narrative_arc": f"trigger: {trigger} -> payoff: {payoff}",
                "evidence": (stitched.get("evidence_lines") or ["No evidence provided"])[:5],
                "trim_rationale": trim_rationale,
                "duration_fit": duration_fit,
                "platform_fit": platform_fit,
                "risks": risks,
                "streamer_feedback": (
                    "Keep setup concise, hit payoff quickly, and preserve the exact chat-to-reaction beat in the posted cut."
                ),
            },
        }

        validated = validate_stage_payload("final_selected", payload)
        selected.append(validated.model_dump())

        if len(selected) >= max_clips:
            break

    return selected
