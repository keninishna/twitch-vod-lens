"""Normalize free-form audio phase output into structured fields."""

from __future__ import annotations

import re
from typing import Dict, List


_FLOAT_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _extract_first_float(text: str):
    m = _FLOAT_RE.search(text or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def normalize_audio_result(audio_result: Dict) -> Dict:
    """Convert raw audio batch result into deterministic structured fields."""

    analysis = str(audio_result.get("analysis") or "")
    low = analysis.lower()

    lines = [ln.strip("- •\t ") for ln in analysis.splitlines() if ln.strip()]
    key_events: List[str] = [ln for ln in lines[:5]]

    dead_air_detected = any(k in low for k in ["dead air", "long silence", "extended silence", "silence gap"])
    laughter_detected = any(k in low for k in ["laugh", "laughter", "giggle", "cackle"])
    raised_voice_detected = any(k in low for k in ["shout", "yell", "raised voice", "screams"])
    music_only = ("music" in low and "speech" in low and ("no speech" in low or "mostly music" in low))

    confidence = audio_result.get("confidence")
    if confidence is None:
        confidence = _extract_first_float(low.split("confidence")[-1]) if "confidence" in low else None
    if confidence is None:
        confidence = 0.5

    confidence = max(0.0, min(1.0, float(confidence)))

    return {
        "summary": analysis[:1000],
        "key_events": key_events,
        "dead_air_detected": bool(dead_air_detected),
        "laughter_detected": bool(laughter_detected),
        "raised_voice_detected": bool(raised_voice_detected),
        "music_only": bool(music_only),
        "confidence": confidence,
        "extraction_time_seconds": audio_result.get("extraction_time_seconds"),
        "inference_time_seconds": audio_result.get("inference_time_seconds"),
    }
