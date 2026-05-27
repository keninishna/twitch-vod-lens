"""Utilities to align diarization turns with transcript segments."""

from __future__ import annotations

from typing import Iterable

from src.preprocessing.types import SpeakerTurn


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Return overlap duration in seconds between two half-open intervals."""

    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers_to_transcript(
    transcript_segments: Iterable[dict],
    speaker_turns: list[SpeakerTurn],
) -> list[dict]:
    """Attach speaker labels to transcript segments based on largest overlap."""

    ordered_turns = sorted(speaker_turns, key=lambda t: (t.start, t.end, t.speaker_label))
    labeled_segments: list[dict] = []

    for segment in transcript_segments:
        seg = dict(segment)
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        seg_duration = max(0.0, seg_end - seg_start)

        best_turn: SpeakerTurn | None = None
        best_overlap = 0.0

        for turn in ordered_turns:
            ov = overlap_seconds(seg_start, seg_end, turn.start, turn.end)
            if ov > best_overlap:
                best_overlap = ov
                best_turn = turn

        overlap_ratio = best_overlap / seg_duration if seg_duration > 0 else 0.0
        seg["speaker_label"] = (
            best_turn.speaker_label if best_turn is not None and overlap_ratio >= 0.30 else "UNKNOWN"
        )
        labeled_segments.append(seg)

    return labeled_segments
