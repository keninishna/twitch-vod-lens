from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_signal(value: Any, full_scale: float) -> float:
    """Normalize arbitrary signal values to the 0..1 range deterministically."""
    if value is None:
        return 0.0

    if isinstance(value, bool):
        return 1.0 if value else 0.0

    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric <= 1.0:
            return _clamp(numeric, 0.0, 1.0)
        return _clamp(numeric / max(full_scale, 1.0), 0.0, 1.0)

    if isinstance(value, str):
        return 1.0 if value.strip() else 0.0

    if isinstance(value, Mapping):
        return _clamp(len(value) / max(full_scale, 1.0), 0.0, 1.0)

    if isinstance(value, Sequence):
        return _clamp(len(value) / max(full_scale, 1.0), 0.0, 1.0)

    return 0.0


def _unique_sorted_strings(values: Any) -> list[str]:
    if values is None:
        return []

    names: set[str] = set()

    if isinstance(values, str):
        cleaned = values.strip().lower()
        if cleaned:
            names.add(cleaned)
    elif isinstance(values, Mapping):
        for key, val in values.items():
            label = None
            if isinstance(val, Mapping):
                label = (
                    val.get("label")
                    or val.get("name")
                    or val.get("class")
                    or val.get("class_name")
                )
            if label is None and isinstance(key, str):
                label = key
            if label:
                names.add(str(label).strip().lower())
    elif isinstance(values, Sequence):
        for item in values:
            if isinstance(item, str):
                cleaned = item.strip().lower()
                if cleaned:
                    names.add(cleaned)
            elif isinstance(item, Mapping):
                label = (
                    item.get("label")
                    or item.get("name")
                    or item.get("class")
                    or item.get("class_name")
                )
                if label:
                    names.add(str(label).strip().lower())

    return sorted(names)


def _parse_frame_key_to_second(frame_key: str, default: int = 0) -> int:
    digits = re.findall(r"\d+", frame_key)
    if not digits:
        return default
    # Prefer the last numeric group for keys like frame_000123.jpg.
    return int(digits[-1])


def _parse_yolo_record(record: Any, fallback_index: int) -> tuple[int, list[str]]:
    if isinstance(record, Mapping):
        frame_idx = int(
            _to_float(
                record.get("timestamp_sec")
                or record.get("frame")
                or record.get("frame_idx")
                or record.get("frame_index")
                or record.get("second")
                or record.get("timestamp")
                or fallback_index,
                fallback_index,
            )
        )
        objects = _unique_sorted_strings(
            record.get("objects")
            or record.get("labels")
            or record.get("classes")
            or record.get("detections")
        )
        return frame_idx, objects

    if isinstance(record, Sequence) and not isinstance(record, str):
        return fallback_index, _unique_sorted_strings(record)

    return fallback_index, _unique_sorted_strings(record)


def load_yolo_detections(yolo_path: str | Path | None) -> dict[int, list[str]]:
    """Load YOLO detections into {timestamp_second: [sorted unique object labels]}.

    Supports JSON files containing either:
    - yolo_detect.py style: {"results": {"frame_*.jpg": {"timestamp_sec": N, "detections": [...]}}}
    - dict with a "frames" key
    - dict mapping frame indexes/timestamps to labels
    - list of frame records
    - JSONL (one JSON object/array per line)
    """
    if yolo_path is None:
        return {}

    path = Path(yolo_path)
    if not path.exists() or not path.is_file():
        return {}

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}

    merged: dict[int, set[str]] = {}

    def add_record(frame_idx: int, objects: list[str]) -> None:
        if frame_idx not in merged:
            merged[frame_idx] = set()
        merged[frame_idx].update(objects)

    def consume(data: Any) -> None:
        if isinstance(data, Mapping):
            if "results" in data:
                consume(data["results"])
                return
            if "frames" in data:
                consume(data["frames"])
                return

            for key, value in data.items():
                if key in {"results", "frames"}:
                    continue

                try:
                    fallback = int(float(key))
                except (TypeError, ValueError):
                    fallback = _parse_frame_key_to_second(str(key), default=0)

                if isinstance(value, Mapping) or (
                    isinstance(value, Sequence) and not isinstance(value, str)
                ):
                    frame_idx, objects = _parse_yolo_record(value, fallback_index=fallback)
                    add_record(frame_idx, objects)
                else:
                    add_record(fallback, _unique_sorted_strings(value))
            return

        if isinstance(data, Sequence) and not isinstance(data, str):
            for idx, record in enumerate(data):
                frame_idx, objects = _parse_yolo_record(record, fallback_index=idx)
                add_record(frame_idx, objects)
            return

        frame_idx, objects = _parse_yolo_record(data, fallback_index=0)
        add_record(frame_idx, objects)

    parsed = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        pass

    if parsed is not None:
        consume(parsed)
    else:
        # JSONL fallback
        for idx, line in enumerate(raw.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            frame_idx, objects = _parse_yolo_record(record, fallback_index=idx)
            add_record(frame_idx, objects)

    return {frame_idx: sorted(objects) for frame_idx, objects in sorted(merged.items())}


def _extract_windows(fusion: Any) -> tuple[list[Mapping[str, Any]], float]:
    duration_seconds = 0.0

    if isinstance(fusion, Mapping):
        windows = (
            fusion.get("windows")
            or fusion.get("segments")
            or fusion.get("clips")
            or []
        )
        duration_seconds = _to_float(
            fusion.get("duration_seconds")
            or (fusion.get("vod_meta", {}) or {}).get("duration_seconds"),
            0.0,
        )
    elif isinstance(fusion, Sequence) and not isinstance(fusion, str):
        windows = fusion
    else:
        windows = []

    normalized_windows: list[Mapping[str, Any]] = [
        w for w in windows if isinstance(w, Mapping)
    ]

    if not duration_seconds:
        for idx, window in enumerate(normalized_windows):
            start = _to_float(
                window.get("start")
                or window.get("start_seconds")
                or window.get("t0"),
                idx,
            )
            end = _to_float(
                window.get("end") or window.get("end_seconds") or window.get("t1"),
                start,
            )
            duration_seconds = max(duration_seconds, start, end)

    return normalized_windows, duration_seconds


def _extract_transcript_segments(fusion: Any) -> list[tuple[float, float]]:
    if not isinstance(fusion, Mapping):
        return []

    transcript = fusion.get("transcript", {})
    if isinstance(transcript, Mapping):
        segments = transcript.get("segments", [])
    elif isinstance(transcript, Sequence) and not isinstance(transcript, str):
        segments = transcript
    else:
        segments = []

    normalized: list[tuple[float, float]] = []
    for seg in segments:
        if not isinstance(seg, Mapping):
            continue

        text = seg.get("text")
        if text is not None and not str(text).strip():
            # Respect real transcript contracts where blank text implies no speech.
            continue

        start = _to_float(seg.get("start") or seg.get("start_seconds") or seg.get("t0"), 0.0)
        end = _to_float(seg.get("end") or seg.get("end_seconds") or seg.get("t1"), start)
        if end > start:
            normalized.append((start, end))

    normalized.sort(key=lambda pair: (pair[0], pair[1]))
    return normalized


def _extract_timeline_points(fusion: Any) -> list[tuple[float, float]]:
    if not isinstance(fusion, Mapping):
        return []

    timeline = fusion.get("timeline", [])
    if not isinstance(timeline, Sequence) or isinstance(timeline, str):
        return []

    points: list[tuple[float, float]] = []
    for row in timeline:
        if not isinstance(row, Mapping):
            continue
        ts = _to_float(row.get("timestamp") or row.get("time") or row.get("ts"), -1.0)
        if ts < 0:
            continue
        intensity = _to_float(row.get("chat_intensity"), 0.0)
        points.append((ts, _clamp(intensity, 0.0, 1.0)))

    points.sort(key=lambda pair: pair[0])
    return points


def _compute_duration_seconds(
    fusion: Any,
    windows: Sequence[Mapping[str, Any]],
    transcript_segments: Sequence[tuple[float, float]],
    timeline_points: Sequence[tuple[float, float]],
    duration_hint: float,
) -> float:
    duration_seconds = max(0.0, duration_hint)

    if isinstance(fusion, Mapping):
        vod_meta = fusion.get("vod_meta", {})
        if isinstance(vod_meta, Mapping):
            duration_seconds = max(duration_seconds, _to_float(vod_meta.get("duration_seconds"), 0.0))
        transcript = fusion.get("transcript", {})
        if isinstance(transcript, Mapping):
            duration_seconds = max(duration_seconds, _to_float(transcript.get("duration_seconds"), 0.0))

    for window in windows:
        start = _to_float(window.get("start") or window.get("start_seconds") or window.get("t0"), 0.0)
        end = _to_float(window.get("end") or window.get("end_seconds") or window.get("t1"), start)
        duration_seconds = max(duration_seconds, start, end)

    for _, end in transcript_segments:
        duration_seconds = max(duration_seconds, end)

    for ts, _ in timeline_points:
        duration_seconds = max(duration_seconds, ts)

    return duration_seconds


def _build_sliding_windows(
    duration_seconds: float,
    window_seconds: int,
    step_seconds: int,
) -> list[dict[str, float]]:
    duration_seconds = max(0.0, float(duration_seconds))
    if duration_seconds <= 0:
        return []

    window_len = max(1.0, float(max(1, int(window_seconds))))
    step_len = max(1.0, float(max(1, int(step_seconds))))

    windows: list[dict[str, float]] = []
    start = 0.0
    while start < duration_seconds:
        end = min(start + window_len, duration_seconds)
        windows.append({"start": start, "end": end})
        if end >= duration_seconds:
            break
        start += step_len

    return windows


def _speech_ratio_for_window(
    start: float,
    end: float,
    transcript_segments: Sequence[tuple[float, float]],
) -> float:
    window_duration = max(0.001, end - start)
    if not transcript_segments:
        return 0.0

    overlap = 0.0
    for seg_start, seg_end in transcript_segments:
        if seg_end <= start or seg_start >= end:
            continue
        overlap += max(0.0, min(end, seg_end) - max(start, seg_start))

    return _clamp(overlap / window_duration, 0.0, 1.0)


def _chat_intensity_for_window(
    start: float,
    end: float,
    timeline_points: Sequence[tuple[float, float]],
) -> float:
    if not timeline_points:
        return 0.0

    values = [value for ts, value in timeline_points if start <= ts <= end]
    if not values:
        return 0.0

    return _clamp(sum(values) / len(values), 0.0, 1.0)


def _objects_for_window(
    window: Mapping[str, Any],
    yolo_frames: Mapping[int, Sequence[str]] | None,
    start: float,
    end: float,
) -> list[str]:
    local_objects = _unique_sorted_strings(
        window.get("objects") or window.get("objects_detected") or window.get("detections")
    )

    if not yolo_frames:
        return local_objects

    yolo_set = set(local_objects)
    start_i, end_i = int(start), int(end)
    for second in range(start_i, max(start_i + 1, end_i + 1)):
        labels = yolo_frames.get(second, ())
        yolo_set.update(_unique_sorted_strings(labels))

    return sorted(yolo_set)


_USEFUL_OBJECT_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("face", 1.0),
    ("person", 0.8),
    ("monitor", 0.9),
    ("screen", 0.9),
    ("tv", 0.8),
    ("laptop", 0.8),
    ("phone", 0.8),
    ("cell phone", 0.8),
    ("keyboard", 0.7),
    ("mouse", 0.6),
    ("controller", 0.8),
    ("remote", 0.5),
    ("bottle", 0.5),
    ("cup", 0.5),
    ("drink", 0.6),
    ("food", 0.6),
    ("pizza", 0.6),
    ("sandwich", 0.6),
    ("cake", 0.5),
    ("bowl", 0.4),
)


def _useful_object_bonus(objects: Sequence[str]) -> tuple[float, bool]:
    total_weight = 0.0
    has_useful = False
    for obj in objects:
        label = str(obj).strip().lower()
        if not label:
            continue
        matched_weight = 0.0
        for token, weight in _USEFUL_OBJECT_WEIGHTS:
            if token in label:
                matched_weight = max(matched_weight, weight)
        if matched_weight > 0:
            has_useful = True
            total_weight += matched_weight

    return _clamp(total_weight, 0.0, 2.0), has_useful


def _label_for_score(score: float, has_speech: bool, chat_intensity: float, objects: list[str]) -> str:
    if score >= 8.0:
        return "hype"
    if score >= 5.0:
        return "highlight"
    if not has_speech and chat_intensity < 0.15 and not objects:
        return "dead"
    return "context"


def generate_clip_manifest(
    fusion: Any,
    *,
    vod_id: str,
    vod_title: str,
    streamer: str,
    window_seconds: int,
    step_seconds: int,
    yolo_frames: Mapping[int, Sequence[str]] | None = None,
) -> dict[str, Any]:
    windows, duration_hint = _extract_windows(fusion)
    transcript_segments = _extract_transcript_segments(fusion)
    timeline_points = _extract_timeline_points(fusion)
    duration_seconds = _compute_duration_seconds(
        fusion,
        windows,
        transcript_segments,
        timeline_points,
        duration_hint,
    )

    if not windows:
        windows = _build_sliding_windows(duration_seconds, window_seconds, step_seconds)

    clips: list[dict[str, Any]] = []

    for idx, window in enumerate(windows):
        start = _to_float(
            window.get("start") or window.get("start_seconds") or window.get("t0"),
            idx * step_seconds,
        )
        end = _to_float(
            window.get("end") or window.get("end_seconds") or window.get("t1"),
            start + window_seconds,
        )

        if end < start:
            start, end = end, start

        window_duration = max(0.001, end - start)

        if transcript_segments:
            speech_norm = _speech_ratio_for_window(start, end, transcript_segments)
        else:
            speech_source = (
                window.get("speech")
                or window.get("speech_score")
                or window.get("speech_ratio")
                or window.get("speech_activity")
                or window.get("transcript")
            )
            speech_norm = _normalize_signal(speech_source, full_scale=max(1.0, window_duration))

        speech_bonus = _clamp(speech_norm * 3.0, 0.0, 3.0)

        if timeline_points:
            chat_norm = _chat_intensity_for_window(start, end, timeline_points)
        else:
            chat_source = (
                window.get("chat")
                or window.get("chat_activity")
                or window.get("chat_messages")
                or window.get("message_count")
                or window.get("chat_intensity")
            )
            chat_norm = _normalize_signal(chat_source, full_scale=120.0)

        chat_bonus = _clamp(chat_norm * 3.0, 0.0, 3.0)

        objects = _objects_for_window(window, yolo_frames, start, end)
        yolo_bonus, useful_objects_present = _useful_object_bonus(objects)

        speech_present = speech_norm > 0.0
        chat_present = chat_norm > 0.0

        dead_window = (not speech_present) and (not chat_present) and (not useful_objects_present)
        empty_penalty = 2.0 if dead_window else 0.0

        base_score = 1.0
        raw_score = base_score + speech_bonus + chat_bonus + yolo_bonus - empty_penalty
        score = round(_clamp(raw_score, 0.0, 10.0), 2)

        has_speech = speech_present
        chat_intensity = round(chat_norm, 3)
        label = _label_for_score(score, has_speech, chat_norm, objects)

        title = window.get("title")
        if not isinstance(title, str) or not title.strip():
            title = f"{label.title()} clip {idx + 1}"

        summary = window.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            if label == "dead":
                summary = "Low activity window with minimal speech, chat, or detectable objects."
            else:
                summary = (
                    f"Speech {speech_norm:.2f}, chat {chat_norm:.2f}, "
                    f"objects {len(objects)} contribute to a {label} moment."
                )

        clips.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "title": title,
                "score": score,
                "objects_detected": objects,
                "summary": summary,
                "has_speech": has_speech,
                "chat_intensity": chat_intensity,
                "label": label,
                "score_breakdown": {
                    "base": base_score,
                    "speech_bonus": round(speech_bonus, 3),
                    "chat_bonus": round(chat_bonus, 3),
                    "yolo_bonus": round(yolo_bonus, 3),
                    "empty_penalty": round(empty_penalty, 3),
                    "raw_score": round(raw_score, 3),
                    "final_score": score,
                },
            }
        )

    clips.sort(key=lambda clip: (clip["start"], clip["end"]))

    if not duration_seconds and clips:
        duration_seconds = max(clip["end"] for clip in clips)

    return {
        "vod_id": vod_id,
        "vod_title": vod_title,
        "streamer": streamer,
        "duration_seconds": round(duration_seconds, 3),
        "clips": clips,
        "total_clips": len(clips),
    }
