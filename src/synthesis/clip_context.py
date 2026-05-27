"""Clip context builder for synthesis stages.

Extracts transcript/chat context around a timestamp and computes deterministic
signals used by prompt construction and scoring:
- transcript lines with timestamps
- dead-air gap list + ratio
- chat messages
- chat-read attribution flags
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.synthesis.schemas import validate_stage_payload


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_text(value) -> str:
    if value is None:
        return ""
    return str(value)


def _tokenize_for_match(text: str) -> List[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [tok for tok in cleaned.split() if len(tok) >= 3]


def _looks_like_read_aloud(message: str, transcript_text_lower: str) -> bool:
    # Fast-path: existing exact-ish prefix check.
    prefix = message.lower()[:40]
    if prefix and prefix in transcript_text_lower:
        return True

    # Fuzzy fallback: token overlap for paraphrased read-aloud lines.
    msg_tokens = set(_tokenize_for_match(message))
    trn_tokens = set(_tokenize_for_match(transcript_text_lower))
    if not msg_tokens or not trn_tokens:
        return False

    overlap = msg_tokens & trn_tokens
    overlap_ratio = len(overlap) / max(1, len(msg_tokens))
    return len(overlap) >= 4 and overlap_ratio >= 0.35


def _overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _extract_speaker_context(
    lo: float,
    hi: float,
    speaker_attribution: Dict[str, Any] | None,
) -> Dict[str, Any]:
    if not isinstance(speaker_attribution, dict):
        return {
            "speaker_turns": [],
            "primary_speaker_label": None,
            "primary_speaker_identity": None,
            "primary_speaker_name": None,
            "streamer_speaking_seconds": 0.0,
            "streamer_speaking_ratio": 0.0,
            "streamer_speaking_confidence": 0.0,
            "off_streamer_voice_detected": False,
            "speaker_name_evidence": [],
        }

    segments = speaker_attribution.get("segments") or []
    if not isinstance(segments, list):
        segments = []

    speaker_turns: List[Dict[str, Any]] = []
    totals_by_label: Dict[str, float] = {}
    max_conf_by_label: Dict[str, float] = {}
    identity_by_label: Dict[str, str] = {}
    name_by_label: Dict[str, str | None] = {}

    streamer_speaking_seconds = 0.0

    for seg in segments:
        if not isinstance(seg, dict):
            continue

        ss = _safe_float(seg.get("start"), -1)
        se = _safe_float(seg.get("end"), -1)
        if se <= ss:
            continue

        overlap = _overlap_seconds(ss, se, lo, hi)
        if overlap <= 0:
            continue

        label = _normalize_text(seg.get("speaker_label", "UNKNOWN")) or "UNKNOWN"
        rec = seg.get("recognition") if isinstance(seg.get("recognition"), dict) else {}
        identity = _normalize_text(rec.get("identity", "unknown")).lower() or "unknown"
        if identity not in {"streamer", "guest", "unknown", "chatter", "mixed"}:
            identity = "unknown"
        conf = _safe_float(rec.get("confidence"), 0.0)
        inferred_name = seg.get("inferred_name")
        inferred_name = _normalize_text(inferred_name) if inferred_name is not None else None

        speaker_turns.append(
            {
                "start": max(lo, ss),
                "end": min(hi, se),
                "speaker_label": label,
                "identity": identity,
                "inferred_name": inferred_name,
                "confidence": max(0.0, min(1.0, conf)),
            }
        )

        totals_by_label[label] = totals_by_label.get(label, 0.0) + overlap
        max_conf_by_label[label] = max(max_conf_by_label.get(label, 0.0), conf)
        identity_by_label[label] = identity
        name_by_label[label] = inferred_name

        if identity == "streamer":
            streamer_speaking_seconds += overlap

    clip_duration = max(1.0, hi - lo)
    streamer_ratio = max(0.0, min(1.0, streamer_speaking_seconds / clip_duration))

    primary_speaker_label = None
    primary_speaker_identity = None
    primary_speaker_name = None
    streamer_confidence = 0.0

    if totals_by_label:
        primary_speaker_label = max(totals_by_label, key=lambda k: totals_by_label[k])
        primary_speaker_identity = identity_by_label.get(primary_speaker_label, "unknown")
        primary_speaker_name = name_by_label.get(primary_speaker_label)

    for label, ident in identity_by_label.items():
        if ident == "streamer":
            streamer_confidence = max(streamer_confidence, max_conf_by_label.get(label, 0.0))

    off_streamer = any(
        ident in {"guest", "chatter", "mixed"}
        for ident in identity_by_label.values()
    )

    speaker_name_evidence: List[str] = []
    clusters = speaker_attribution.get("speaker_clusters")
    if isinstance(clusters, dict):
        for label, summary in clusters.items():
            if not isinstance(summary, dict):
                continue
            if label != primary_speaker_label:
                continue
            candidates = summary.get("candidate_names") or []
            if isinstance(candidates, list):
                for c in candidates[:3]:
                    if not isinstance(c, dict):
                        continue
                    nm = _normalize_text(c.get("name", ""))
                    ev = c.get("evidence") or []
                    if nm:
                        if isinstance(ev, list) and ev:
                            speaker_name_evidence.append(f"{nm}: {str(ev[0])}")
                        else:
                            speaker_name_evidence.append(f"{nm}: inferred name candidate")

    return {
        "speaker_turns": speaker_turns,
        "primary_speaker_label": primary_speaker_label,
        "primary_speaker_identity": primary_speaker_identity,
        "primary_speaker_name": primary_speaker_name,
        "streamer_speaking_seconds": round(streamer_speaking_seconds, 3),
        "streamer_speaking_ratio": round(streamer_ratio, 6),
        "streamer_speaking_confidence": round(max(0.0, min(1.0, streamer_confidence)), 6),
        "off_streamer_voice_detected": bool(off_streamer),
        "speaker_name_evidence": speaker_name_evidence,
    }


def build_clip_context(
    seconds: float,
    transcript_segments: List[Dict],
    chat_messages: List[Dict],
    window: float = 120,
    objects_detected: List[str] | None = None,
    speaker_attribution: Dict[str, Any] | None = None,
) -> Dict:
    """Build structured context around a timestamp.

    Returns schema-aligned fields for `ClipContext` plus raw window bounds.
    """

    lo = max(0.0, _safe_float(seconds) - _safe_float(window))
    hi = max(lo + 1.0, _safe_float(seconds) + _safe_float(window))

    segs = [
        s
        for s in transcript_segments
        if lo <= _safe_float(s.get("start", 0.0)) <= hi
    ]

    transcript_lines = [
        {
            "start": _safe_float(s.get("start", 0.0)),
            "end": _safe_float(s.get("end", 0.0)),
            "text": _normalize_text(s.get("text", "")),
        }
        for s in segs
    ]

    sorted_segs = sorted(transcript_lines, key=lambda s: s["start"])
    dead_air_gaps = []
    for idx in range(1, len(sorted_segs)):
        prev_seg = sorted_segs[idx - 1]
        curr_seg = sorted_segs[idx]
        gap = curr_seg["start"] - prev_seg["end"]
        if gap > 5:
            dead_air_gaps.append(
                {
                    "start": prev_seg["end"],
                    "end": curr_seg["start"],
                    "duration": float(round(gap, 3)),
                }
            )

    total_dead_air_seconds = float(round(sum(g["duration"] for g in dead_air_gaps), 3))
    window_duration = max(hi - lo, 1.0)
    dead_air_ratio = min(1.0, max(0.0, total_dead_air_seconds / window_duration))

    chats = [
        m
        for m in chat_messages
        if lo <= _safe_float(m.get("timestamp", 0.0)) <= hi
    ]

    normalized_chats = [
        {
            "timestamp": _safe_float(m.get("timestamp", 0.0)),
            "user": _normalize_text(m.get("user", "?")),
            "message": _normalize_text(m.get("message", "")),
        }
        for m in chats
    ]

    full_transcript_text = " ".join(line["text"] for line in transcript_lines).lower()
    chat_read_flags = []
    for m in normalized_chats:
        msg = m["message"]
        if not msg:
            continue
        if len(msg) <= 20:
            continue
        if _looks_like_read_aloud(msg, full_transcript_text):
            chat_read_flags.append(
                {
                    "timestamp": m["timestamp"],
                    "user": m["user"],
                    "message": msg,
                    "matched_transcript": msg[:40].lower(),
                }
            )

    speaker_context = _extract_speaker_context(lo, hi, speaker_attribution)

    context_payload = {
        "clip_start": float(round(lo, 3)),
        "clip_end": float(round(hi, 3)),
        "transcript_lines": transcript_lines,
        "chat_messages": normalized_chats,
        "chat_read_flags": chat_read_flags,
        "dead_air_gaps": dead_air_gaps,
        "total_dead_air_seconds": total_dead_air_seconds,
        "dead_air_ratio": float(round(dead_air_ratio, 6)),
        "objects_detected": objects_detected or [],
        "speaker_turns": speaker_context["speaker_turns"],
        "primary_speaker_label": speaker_context["primary_speaker_label"],
        "primary_speaker_identity": speaker_context["primary_speaker_identity"],
        "primary_speaker_name": speaker_context["primary_speaker_name"],
        "streamer_speaking_seconds": speaker_context["streamer_speaking_seconds"],
        "streamer_speaking_ratio": speaker_context["streamer_speaking_ratio"],
        "streamer_speaking_confidence": speaker_context["streamer_speaking_confidence"],
        "off_streamer_voice_detected": speaker_context["off_streamer_voice_detected"],
        "speaker_name_evidence": speaker_context["speaker_name_evidence"],
    }

    validated = validate_stage_payload("context", context_payload)
    return validated.model_dump()


def render_prompt_context(context: Dict, transcript_char_limit: int = 2000) -> Tuple[str, str]:
    """Render transcript/chat strings for legacy prompt templates.

    Keeps current pipeline behavior while context production is now centralized.
    """

    transcript_lines = context.get("transcript_lines", [])
    chat_messages = context.get("chat_messages", [])
    dead_air_gaps = context.get("dead_air_gaps", [])
    chat_read_flags = context.get("chat_read_flags", [])
    primary_speaker_identity = context.get("primary_speaker_identity")
    primary_speaker_label = context.get("primary_speaker_label")
    primary_speaker_name = context.get("primary_speaker_name")
    streamer_ratio = float(context.get("streamer_speaking_ratio") or 0.0)

    txt_lines = [
        f"[{line.get('start', 0):.0f}s-{line.get('end', 0):.0f}s] {line.get('text', '')}"
        for line in transcript_lines
    ]
    transcript_text = "\n".join(txt_lines)

    if dead_air_gaps:
        total_dead = context.get("total_dead_air_seconds", 0.0)
        pct = context.get("dead_air_ratio", 0.0) * 100
        details = "; ".join(
            f"{gap.get('duration', 0):.0f}s gap at {gap.get('start', 0):.0f}s-{gap.get('end', 0):.0f}s"
            for gap in dead_air_gaps
        )
        window_duration = context.get("clip_end", 0) - context.get("clip_start", 0)
        transcript_text += (
            f"\n\n⚠️ DEAD AIR DETECTED: {total_dead:.0f}s silence "
            f"({pct:.0f}% of {window_duration:.0f}s window). Gaps: {details}"
        )

    if chat_read_flags:
        transcript_text += "\n\n⚠️ CHAT-READ FLAGS (streamer reading chat aloud — do NOT attribute to streamer):"
        for item in chat_read_flags:
            transcript_text += (
                f"\n  @{item.get('user','?')} at {item.get('timestamp', 0):.0f}s: "
                f"'{item.get('message','')[:80]}...' — streamer reads this aloud."
            )
        transcript_text += (
            "\nREMINDER: When a chat message appears in the transcript, it is often the "
            "CHATTER's story and the streamer may be reading it aloud. Keep attribution accurate "
            "and avoid framing it as the streamer's personal claim unless evidence supports that."
        )

    if primary_speaker_identity and primary_speaker_identity != "streamer":
        display_name = primary_speaker_name or primary_speaker_label or "unknown speaker"
        transcript_text += (
            "\n\n⚠️ SPEAKER ATTRIBUTION: "
            f"Primary voice is {display_name} ({primary_speaker_identity}), not streamer. "
            "Do not title this as a streamer reaction unless streamer speech is the payoff."
        )
    elif primary_speaker_identity == "streamer" and streamer_ratio < 0.15:
        transcript_text += (
            "\n\n⚠️ SPEAKER ATTRIBUTION: streamer identity detected but speaking ratio is low "
            f"({streamer_ratio:.2f}). Verify title framing before calling this a streamer-led moment."
        )

    transcript_text = transcript_text[:transcript_char_limit]

    chat_lines = [
        f"[{msg.get('timestamp', 0):.0f}s] @{msg.get('user', '?')}: {msg.get('message', '')}"
        for msg in chat_messages
    ]
    chat_text = "\n".join(chat_lines)
    return transcript_text, chat_text
