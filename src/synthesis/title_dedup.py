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


def _extract_chat_topic(*texts: str) -> str:
    """Extract a concise chat-story topic from noisy trigger/title text.

    Handles patterns like:
    - "chat message from 'user' about a streak"
    - "chat message at 118s: '...mclaren owner...'"
    """

    for raw in texts:
        if not raw:
            continue

        # Split on stitch joins so we can pick the cleanest fragment.
        fragments = [frag.strip() for frag in str(raw).split("|") if frag.strip()]
        for frag in fragments:
            text = frag.strip(" .")
            if not text:
                continue

            # Drop repeated chat-message boilerplate + user/time metadata.
            text = re.sub(
                r"^chat message(?:\s+from\s+['\"`]?[^'\":|]+['\"`]?)?(?:\s+at\s+\d+s)?\s*[:,-]?\s*",
                "",
                text,
                flags=re.IGNORECASE,
            )

            # Prefer the semantic payload after "about ..." when available.
            about_match = re.search(r"\babout\b\s+(.+)", text, flags=re.IGNORECASE)
            candidate = about_match.group(1) if about_match else text

            # If the candidate still starts with chat metadata, strip it again.
            candidate = re.sub(
                r"^chat message(?:\s+from\s+['\"`]?[^'\":|]+['\"`]?)?(?:\s+at\s+\d+s)?\s*[:,-]?\s*",
                "",
                candidate,
                flags=re.IGNORECASE,
            )

            # Handle nested forms like "chat message from X about Y".
            nested_about = re.search(r"\babout\b\s+(.+)", candidate, flags=re.IGNORECASE)
            if nested_about and "chat message" in candidate.lower():
                candidate = nested_about.group(1)

            candidate = candidate.strip(" ' \".,;:-")
            if candidate.lower().startswith("about "):
                candidate = candidate[6:].strip(" ' \".,;:-")
            if not candidate:
                continue

            low = candidate.lower()
            if low in {"chat message", "message", "the message"}:
                continue

            # Collapse extra whitespace.
            candidate = re.sub(r"\s+", " ", candidate)
            if candidate:
                return candidate

    return "a viewer story"


def _sanitize_chat_title(title: str, trigger: str, payoff: str) -> str:
    low = (title or "").lower()

    # Repair only clearly broken/recursive forms like
    # "...about chat message from X about Y".
    if "reads a chat message about chat message" in low or "about chat message from" in low:
        topic = _extract_chat_topic(title, trigger, payoff)
        return f"What happens when chat drops a message about {topic[:62]}?"

    return title


def _generate_title(stitched: Dict, analysis: Dict) -> str:
    narrative_type = str(stitched.get("narrative_type") or "moment")
    trigger = str(stitched.get("trigger") or "a key moment")
    payoff = str(stitched.get("payoff") or "a strong payoff")

    clip_point = str(analysis.get("clip_point") or "").strip()
    if clip_point:
        # Keep Stage-1 title intent by default; only repair clearly broken chat recursion.
        if "chat" in narrative_type.lower():
            return _sanitize_chat_title(clip_point, trigger, payoff)
        return clip_point

    if "chat" in narrative_type.lower():
        topic = _extract_chat_topic(trigger, payoff)
        return f"What happens when chat drops a message about {topic[:62]}?"

    return f"The moment {trigger[:45].lower()} led to {payoff[:45].lower()}"


def _apply_speaker_title_guard(title: str, stitched: Dict, analysis: Dict) -> str:
    speaker = analysis.get("speaker_attribution") if isinstance(analysis, dict) else None
    if not isinstance(speaker, dict):
        return title

    primary_identity = str(speaker.get("primary_speaker_identity") or "unknown").lower()
    if primary_identity == "streamer":
        return title

    low = (title or "").lower()
    if not low:
        return title

    if "streamer" in low or "she " in low or "her " in low:
        trigger = str(stitched.get("trigger") or "a key moment")
        payoff = str(stitched.get("payoff") or "a payoff")
        return f"What happens when {trigger[:52].lower()} leads to {payoff[:42].lower()}?"

    return title


def finalize_stage3_candidates(
    scored_candidates: List[Dict],
    stitched_candidates: List[Dict],
    analysis_by_candidate: Dict[str, Dict],
    min_score: float = 3.0,
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

        # Prefer clamped trim values from Stage 2 scoring over raw Qwen analysis
        trim_start = scored.get("clamped_trim_start")
        trim_end = scored.get("clamped_trim_end")
        if trim_start is None or trim_end is None:
            trim_start = int(float(analysis.get("suggested_trim_start", start)))
            trim_end = int(float(analysis.get("suggested_trim_end", end)))

        # Final verification: trim must be valid and within candidate bounds.
        if trim_end <= trim_start:
            continue
        if trim_start < start or trim_end > end:
            continue

        title = _generate_title(stitched, analysis)
        title = _apply_speaker_title_guard(title, stitched, analysis)

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
            "title": title,
            "clip_point": title,
            "narrative_type": str(stitched.get("narrative_type") or "unknown"),
            "platform_scores": platform_scores,
            "platform_recommendations": platform_recommendations,
            "speaker_attribution": {
                "primary_speaker_identity": str(((analysis.get("speaker_attribution") or {}).get("primary_speaker_identity") or "unknown")),
                "primary_speaker_name": (analysis.get("speaker_attribution") or {}).get("primary_speaker_name"),
                "streamer_speaking_ratio": float(((analysis.get("speaker_attribution") or {}).get("streamer_speaking_ratio") or 0.0)),
                "streamer_speaking_confidence": float(((analysis.get("speaker_attribution") or {}).get("streamer_speaking_confidence") or 0.0)),
                "off_streamer_voice_detected": bool(((analysis.get("speaker_attribution") or {}).get("off_streamer_voice_detected") or False)),
                "evidence": list(((analysis.get("speaker_attribution") or {}).get("evidence") or []))[:5],
            },
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
