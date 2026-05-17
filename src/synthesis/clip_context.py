"""Clip context builder for synthesis stages.

Extracts transcript/chat context around a timestamp and computes deterministic
signals used by prompt construction and scoring:
- transcript lines with timestamps
- dead-air gap list + ratio
- chat messages
- chat-read attribution flags
"""

from __future__ import annotations

from typing import Dict, List, Tuple

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


def build_clip_context(
    seconds: float,
    transcript_segments: List[Dict],
    chat_messages: List[Dict],
    window: float = 120,
    objects_detected: List[str] | None = None,
) -> Dict:
    """Build structured context around a timestamp.

    Returns schema-aligned fields for `ClipContext` plus raw window bounds.
    """

    lo = _safe_float(seconds) - _safe_float(window)
    hi = _safe_float(seconds) + _safe_float(window)

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
            "\nIMPORTANT: When a chat message appears in the transcript, it is the "
            "CHATTER's story, not the streamer's. Titles MUST say 'reads a chat message about...'."
        )

    transcript_text = transcript_text[:transcript_char_limit]

    chat_lines = [
        f"[{msg.get('timestamp', 0):.0f}s] @{msg.get('user', '?')}: {msg.get('message', '')}"
        for msg in chat_messages
    ]
    chat_text = "\n".join(chat_lines)
    return transcript_text, chat_text
