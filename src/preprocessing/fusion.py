"""Signal fusion engines.

This module provides two fusion paths:

1) `fuse_signals(...)` (legacy): file-path based JSON fusion used by preprocess.py.
2) `fuse(...)` (typed): model-based fusion used by src/preprocessing/pipeline.py.
"""

from __future__ import annotations

import json
from typing import Iterable

from src.preprocessing.types import (
    ChatAnalysis,
    FusionResult,
    FusionTimeline,
    SceneClip,
    TranscriptResult,
    VodMeta,
)


def _chat_intensity_at(chat: ChatAnalysis, ts: float) -> float:
    """Compute a simple deterministic chat intensity at timestamp ts.

    Intensity is normalized message count per activity window and clamped to [0, 10].
    """
    for window in chat.activity:
        if window.window_start <= ts < window.window_end:
            width = max(1.0, window.window_end - window.window_start)
            return min(10.0, window.message_count / width)
    return 0.0


def _top_emotes_at(chat: ChatAnalysis, ts: float, window_seconds: float = 15.0) -> list[str]:
    """Collect up to 5 most frequent emotes around a timestamp."""
    start = ts - window_seconds
    end = ts + window_seconds

    counts: dict[str, int] = {}
    for msg in chat.messages:
        if start <= msg.timestamp <= end and msg.emotes:
            for em in msg.emotes:
                counts[em] = counts.get(em, 0) + 1

    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [em for em, _ in ranked[:5]]


def _scene_change_for_timestamp(scenes: Iterable[SceneClip], ts: float, tolerance: float = 2.0) -> tuple[bool, int | None]:
    """Return whether ts is near a scene boundary and that scene index."""
    for scene in scenes:
        # boundary-like points: near scene start
        if abs(ts - scene.start) <= tolerance:
            return True, scene.index
    return False, None


def fuse(
    *,
    vod_meta: VodMeta,
    transcript: TranscriptResult,
    scenes: list[SceneClip],
    chat: ChatAnalysis,
) -> FusionResult:
    """Typed fusion path used by pipeline.py.

    Produces a deterministic timeline keyed by transcript segment starts.
    """
    timeline: list[FusionTimeline] = []

    for seg in transcript.segments:
        scene_change, scene_index = _scene_change_for_timestamp(scenes, seg.start)
        timeline.append(
            FusionTimeline(
                timestamp=seg.start,
                transcript=seg,
                scene_change=scene_change,
                scene_index=scene_index,
                chat_intensity=_chat_intensity_at(chat, seg.start),
                top_emotes=_top_emotes_at(chat, seg.start),
            )
        )

    return FusionResult(
        vod_meta=vod_meta,
        transcript=transcript,
        scenes=scenes,
        chat=chat,
        timeline=timeline,
        processing_time_seconds=0.0,
    )


def fuse_signals(transcript_path: str, scenes_path: str,
                 chat_path: str, output_path: str) -> int:
    """Fuse all signals into scored moments (legacy file-path API).

    Args:
        transcript_path: Path to transcript.json.
        scenes_path: Path to scenes.json.
        chat_path: Path to chat_analysis.json.
        output_path: Path to write moments.json.

    Returns:
        int: Number of moments found.
    """
    with open(transcript_path) as f:
        transcript = json.load(f)
    with open(scenes_path) as f:
        scenes = json.load(f)
    with open(chat_path) as f:
        chat = json.load(f)

    moments = []

    # Chat spikes are strong signals
    for spike in chat.get("spikes", []):
        moments.append({
            "start": spike["start"],
            "end": spike["end"],
            "score": min(100, spike.get("intensity", 0) * 20),
            "signals": ["chat_spike"],
            "message_count": spike.get("message_count", 0),
        })

    # Scene changes that coincide with voice excitement
    for seg in transcript:
        text = seg.get("text", "")
        excitement = 0

        # ALL CAPS = excitement
        if text.isupper() and len(text) > 5:
            excitement += 20

        # Repeated punctuation = excitement
        if any(c * 3 in text.lower() for c in "!?."):
            excitement += 15

        # Short utterances = reactions (e.g. "LETS GO", "NO WAY")
        if len(text.split()) < 5:
            excitement += 10

        if excitement > 25:
            moments.append({
                "start": seg["start"],
                "end": seg["end"],
                "score": min(100, 50 + excitement),
                "signals": ["voice_excitement"],
                "text": text,
            })

    # Merge overlapping moments (within 5 seconds)
    moments.sort(key=lambda m: m["start"])
    merged = []
    for m in moments:
        if merged and m["start"] < merged[-1]["end"] + 5:
            merged[-1]["end"] = max(merged[-1]["end"], m["end"])
            merged[-1]["score"] = max(merged[-1]["score"], m["score"])
            # Merge signal lists
            merged[-1]["signals"] = list(set(merged[-1]["signals"] + m["signals"]))
        else:
            merged.append(m)

    # Sort by score descending
    merged.sort(key=lambda m: m["score"], reverse=True)

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    return len(merged)
