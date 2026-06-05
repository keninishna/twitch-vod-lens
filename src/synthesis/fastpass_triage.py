"""Fast-pass Stage 1 triage helpers.

These helpers stay discovery-only and purely deterministic:
- compute a vision budget
- normalize raw triage candidates into safe dicts
- select a deterministic vision shortlist with rescue lanes

The module intentionally avoids any model calls or pipeline wiring.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


_ALLOWED_VISION_NEEDS = {"none", "verify_expression", "verify_scene", "critical"}
_YOLO_TAGS = {"yolo_visual_novelty", "yolo", "visual_novelty", "verify_scene"}
_SENTINEL_TAGS = {"sentinel_coverage", "sentinel", "coverage"}
_TEXT_TOP_TAG = "text_top_rank"
_GEMMA_AUDIO_SIGNAL_TYPES = {"donation_alert", "tts_alert", "laugh", "game_audio", "music", "non_streamer_speech"}
_GEMMA_SENTINEL_TYPES = {"visual_payoff", "scene_change", "streamer_visible", "face_visible", "gameplay_event"}
_GEMMA_SHORTLIST_LANES = {
    "gemma_audio_alert_or_laughter": {
        "gemma_audio_alert_or_laughter",
        "audio_alert",
        "audio_alert_or_laughter",
        "alert_or_laughter",
        "donation_alert",
        "tts_alert",
        "laugh",
        "laughter",
    },
    "gemma_game_audio_or_non_streamer_voice": {
        "gemma_game_audio_or_non_streamer_voice",
        "game_audio",
        "non_streamer_voice",
        "non_streamer_speech",
        "game_audio_or_non_streamer_voice",
    },
    "gemma_visual_reaction": {
        "gemma_visual_reaction",
        "visual_reaction",
        "reaction",
        "streamer_reaction",
        "face_visible",
        "streamer_visible",
        "laughing",
    },
    "gemma_visual_payoff": {
        "gemma_visual_payoff",
        "visual_payoff",
        "payoff",
        "scene_change",
        "gameplay_event",
    },
}


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


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _clean_text(value: Any, default: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or default


def _clean_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []

    cleaned: List[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return _dedupe_preserve_order(cleaned)


def _normalize_vision_need(value: Any) -> str:
    need = _clean_text(value, "none").lower()
    return need if need in _ALLOWED_VISION_NEEDS else "none"


def _score_key(candidate: Dict[str, Any]) -> Tuple[float, float, int, str]:
    return (
        -_as_float(candidate.get("triage_score"), 0.0),
        -_as_float(candidate.get("triage_confidence"), 0.0),
        _as_int(candidate.get("start"), 0),
        _clean_text(candidate.get("candidate_id"), ""),
    )


def _normalize_window(start: int, end: int, fallback_start: int, fallback_end: int) -> Tuple[int, int]:
    if fallback_end <= fallback_start:
        fallback_end = fallback_start + 1

    if end <= start:
        start = fallback_start
        end = fallback_end

    if end <= start:
        end = start + 1

    return start, end


def _clamp_within_window(value: Any, start: int, end: int, default: int) -> int:
    if value is None:
        return default
    coerced = _as_int(value, default)
    if coerced < start:
        return start
    if coerced > end:
        return end
    return coerced


def _derive_lane_reason(candidate: Dict[str, Any], lane: str) -> str | None:
    reasons = {
        str(reason).strip().lower()
        for reason in (candidate.get("selection_reasons") or [])
        if str(reason).strip()
    }
    if lane == "chat_spike":
        if "chat_spike" in reasons:
            return "chat_spike"
        if reasons & {"chat", "chat_banter", "conversation"}:
            return sorted(reasons & {"chat", "chat_banter", "conversation"})[0]
        return None
    if lane == "audio_signal":
        if "audio_signal" in reasons:
            return "audio_signal"
        if reasons & {"audio", "sound", "loud_reaction"}:
            return sorted(reasons & {"audio", "sound", "loud_reaction"})[0]
        return None
    if lane == "yolo_visual_novelty":
        if "yolo_visual_novelty" in reasons:
            return "yolo_visual_novelty"
        if reasons & _YOLO_TAGS:
            return sorted(reasons & _YOLO_TAGS)[0]
        return None
    if lane == "sentinel_coverage":
        if "sentinel_coverage" in reasons:
            return "sentinel_coverage"
        if reasons & _SENTINEL_TAGS:
            return sorted(reasons & _SENTINEL_TAGS)[0]
        return None
    if lane in _GEMMA_SHORTLIST_LANES:
        if lane in reasons:
            return lane
        if reasons & _GEMMA_SHORTLIST_LANES[lane]:
            return sorted(reasons & _GEMMA_SHORTLIST_LANES[lane])[0]
        return None
    return None


def _normalize_manifest_clip(raw: Dict[str, Any], index: int) -> Dict[str, Any]:
    start = _as_int(raw.get("start"), 0)
    end = _as_int(raw.get("end"), start + 1)
    if end <= start:
        end = start + 1
    clip_id = _clean_text(raw.get("clip_id") or raw.get("candidate_id"), f"manifest_{index}")
    return {
        "clip_id": clip_id,
        "start": start,
        "end": end,
        "raw": raw,
    }


def _best_manifest_match(candidate: Dict[str, Any], manifest_clips: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not manifest_clips:
        return None

    cand_start = _as_int(candidate.get("start"), 0)
    cand_end = _as_int(candidate.get("end"), cand_start + 1)
    if cand_end <= cand_start:
        cand_end = cand_start + 1

    best_overlap = -1
    best_distance = 10**12
    best_clip: Dict[str, Any] | None = None

    for clip in manifest_clips:
        clip_start = clip["start"]
        clip_end = clip["end"]
        overlap = max(0, min(cand_end, clip_end) - max(cand_start, clip_start))
        distance = abs(cand_start - clip_start)
        if overlap > best_overlap or (overlap == best_overlap and distance < best_distance):
            best_overlap = overlap
            best_distance = distance
            best_clip = clip

    return best_clip


def compute_vision_budget(total_candidates: int, ratio: float, min_candidates: int, max_candidates: int) -> int:
    total = max(0, _as_int(total_candidates, 0))
    if total == 0:
        return 0

    ratio = max(0.0, _as_float(ratio, 0.0))
    min_candidates = max(0, _as_int(min_candidates, 0))
    max_candidates = max(min_candidates, _as_int(max_candidates, min_candidates))

    budget = math.ceil(total * ratio)
    budget = max(min_candidates, budget)
    budget = min(max_candidates, budget)
    budget = min(total, budget)
    return max(0, budget)


def normalize_triage_candidate(raw: dict, fallback_start: int, fallback_end: int) -> dict:
    raw = raw if isinstance(raw, dict) else {}

    fallback_start = _as_int(fallback_start, 0)
    fallback_end = _as_int(fallback_end, fallback_start + 1)
    if fallback_end <= fallback_start:
        fallback_end = fallback_start + 1

    start = _as_int(raw.get("start"), fallback_start)
    end = _as_int(raw.get("end"), fallback_end)
    start, end = _normalize_window(start, end, fallback_start, fallback_end)

    suggested_trim_start = _clamp_within_window(raw.get("suggested_trim_start"), start, end, start)
    suggested_trim_end = _clamp_within_window(raw.get("suggested_trim_end"), start, end, end)
    if suggested_trim_end <= suggested_trim_start:
        suggested_trim_start, suggested_trim_end = start, end

    candidate_id = _clean_text(raw.get("candidate_id"), f"triage_{start}_{end}")

    evidence_lines = _clean_list(raw.get("evidence_lines"))
    if not evidence_lines:
        evidence_lines = [f"[{start}s] triage candidate from {start}s to {end}s"]

    risk_flags = _clean_list(raw.get("risk_flags"))
    selection_reasons = _clean_list(raw.get("selection_reasons"))

    triage_score = _as_float(raw.get("triage_score"), 0.0)
    triage_score = max(0.0, min(10.0, triage_score))

    triage_confidence = _as_float(raw.get("triage_confidence"), 0.0)
    triage_confidence = max(0.0, min(1.0, triage_confidence))

    normalized = {
        "candidate_id": candidate_id,
        "start": start,
        "end": end,
        "suggested_trim_start": suggested_trim_start,
        "suggested_trim_end": suggested_trim_end,
        "narrative_type": _clean_text(raw.get("narrative_type"), "other") or "other",
        "trigger": _clean_text(raw.get("trigger"), "What starts the moment"),
        "payoff": _clean_text(raw.get("payoff"), "What resolves or lands"),
        "evidence_lines": evidence_lines,
        "risk_flags": risk_flags,
        "triage_score": round(triage_score, 4),
        "triage_confidence": round(triage_confidence, 4),
        "vision_need": _normalize_vision_need(raw.get("vision_need")),
        "selection_reasons": selection_reasons,
    }

    gemma_annotation_refs = _clean_list(raw.get("gemma_annotation_refs"))
    if gemma_annotation_refs:
        normalized["gemma_annotation_refs"] = gemma_annotation_refs

    return normalized


def _build_selected_item(
    candidate: Dict[str, Any],
    selection_reason: str,
    *,
    manifest_clip: Dict[str, Any] | None = None,
    source_candidate_id: str | None = None,
    synthetic: bool = False,
) -> Dict[str, Any]:
    base_start = _as_int(candidate.get("start"), 0)
    base_end = _as_int(candidate.get("end"), base_start + 1)
    if base_end <= base_start:
        base_end = base_start + 1

    if manifest_clip is not None:
        start = manifest_clip["start"]
        end = manifest_clip["end"]
    else:
        start = base_start
        end = base_end

    suggested_trim_start = _clamp_within_window(candidate.get("suggested_trim_start"), start, end, start)
    suggested_trim_end = _clamp_within_window(candidate.get("suggested_trim_end"), start, end, end)
    if suggested_trim_end <= suggested_trim_start:
        suggested_trim_start, suggested_trim_end = start, end

    evidence_lines = list(candidate.get("evidence_lines") or [])
    if synthetic and not evidence_lines:
        clip_id = manifest_clip["clip_id"] if manifest_clip is not None else source_candidate_id or f"sentinel_{start}"
        evidence_lines = [f"[{start}s] sentinel coverage from manifest clip {clip_id}"]

    selected_selection_reasons = _dedupe_preserve_order(
        list(candidate.get("selection_reasons") or []) + [selection_reason]
    )

    item = {
        "candidate_id": candidate.get("candidate_id"),
        "source_candidate_id": source_candidate_id or candidate.get("candidate_id"),
        "start": start,
        "end": end,
        "suggested_trim_start": suggested_trim_start,
        "suggested_trim_end": suggested_trim_end,
        "triage_score": _as_float(candidate.get("triage_score"), 0.0),
        "triage_confidence": _as_float(candidate.get("triage_confidence"), 0.0),
        "vision_need": candidate.get("vision_need") or "none",
        "selection_reason": selection_reason,
        "selection_reasons": selected_selection_reasons,
        "evidence_lines": evidence_lines,
        "risk_flags": list(candidate.get("risk_flags") or []),
    }

    gemma_annotation_refs = _clean_list(candidate.get("gemma_annotation_refs"))
    if gemma_annotation_refs:
        item["gemma_annotation_refs"] = gemma_annotation_refs

    if manifest_clip is not None:
        item["manifest_clip_id"] = manifest_clip["clip_id"]
        item["manifest_clip_start"] = manifest_clip["start"]
        item["manifest_clip_end"] = manifest_clip["end"]

    return item


def select_vision_shortlist(
    triage_candidates: list[dict],
    manifest_clips: list[dict],
    *,
    vision_budget: int,
    sentinel_ratio: float = 0.05,
) -> list[dict]:
    normalized_candidates = [
        normalize_triage_candidate(candidate, _as_int(candidate.get("start"), 0), _as_int(candidate.get("end"), _as_int(candidate.get("start"), 0) + 1))
        for candidate in (triage_candidates or [])
        if isinstance(candidate, dict)
    ]
    normalized_candidates = sorted(normalized_candidates, key=_score_key)

    normalized_manifest = [
        _normalize_manifest_clip(clip, idx)
        for idx, clip in enumerate(manifest_clips or [])
        if isinstance(clip, dict)
    ]
    normalized_manifest = sorted(normalized_manifest, key=lambda clip: (clip["start"], clip["end"], clip["clip_id"]))

    budget = max(0, _as_int(vision_budget, 0))
    if budget == 0:
        return []

    sentinel_target_count = 0
    if normalized_manifest:
        sentinel_target_count = max(1, int(round(budget * max(0.0, _as_float(sentinel_ratio, 0.0)))))
        sentinel_target_count = min(sentinel_target_count, len(normalized_manifest))

    selected: List[Dict[str, Any]] = []
    seen_starts = set()

    def add_item(candidate: Dict[str, Any], selection_reason: str, *, manifest_clip: Dict[str, Any] | None = None, synthetic: bool = False) -> bool:
        if len(selected) >= budget:
            return False
        item = _build_selected_item(
            candidate,
            selection_reason,
            manifest_clip=manifest_clip,
            source_candidate_id=candidate.get("candidate_id") if not synthetic else (manifest_clip["clip_id"] if manifest_clip is not None else candidate.get("candidate_id")),
            synthetic=synthetic,
        )
        start = item["start"]
        if start in seen_starts:
            return False
        seen_starts.add(start)
        selected.append(item)
        return True

    def pick_best(candidates: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        for candidate in candidates:
            if _as_int(candidate.get("start"), 0) in seen_starts:
                continue
            return candidate
        return None

    top_rank_candidate = pick_best(normalized_candidates)
    if top_rank_candidate is not None:
        add_item(top_rank_candidate, _TEXT_TOP_TAG)

    special_lane_specs = [
        "chat_spike",
        "audio_signal",
        "yolo_visual_novelty",
        "gemma_audio_alert_or_laughter",
        "gemma_game_audio_or_non_streamer_voice",
        "gemma_visual_reaction",
        "gemma_visual_payoff",
    ]
    for lane_name in special_lane_specs:
        lane_candidates = [
            candidate
            for candidate in normalized_candidates
            if _derive_lane_reason(candidate, lane_name) is not None
        ]
        lane_candidates = [candidate for candidate in lane_candidates if _as_int(candidate.get("start"), 0) not in seen_starts]
        lane_candidate = pick_best(lane_candidates)
        if lane_candidate is None:
            continue
        lane_reason = _derive_lane_reason(lane_candidate, lane_name) or lane_name
        add_item(lane_candidate, lane_reason)

    if normalized_manifest and len(selected) < budget and sentinel_target_count > 0:
        explicit_sentinel_candidates = [
            candidate
            for candidate in normalized_candidates
            if _derive_lane_reason(candidate, "sentinel_coverage") is not None
            and _as_int(candidate.get("start"), 0) not in seen_starts
        ]
        sentinel_selected = 0
        while explicit_sentinel_candidates and len(selected) < budget and sentinel_selected < sentinel_target_count:
            sentinel_candidate = pick_best(explicit_sentinel_candidates)
            if sentinel_candidate is None:
                break
            if add_item(sentinel_candidate, "sentinel_coverage"):
                sentinel_selected += 1
                explicit_sentinel_candidates = [
                    candidate
                    for candidate in explicit_sentinel_candidates
                    if _as_int(candidate.get("start"), 0) not in seen_starts
                ]
            else:
                break

        if len(selected) < budget and sentinel_selected < sentinel_target_count:
            target_indices: List[int]
            if sentinel_target_count == 1:
                target_indices = [len(normalized_manifest) // 2]
            else:
                target_indices = [
                    round(i * (len(normalized_manifest) - 1) / (sentinel_target_count - 1))
                    for i in range(sentinel_target_count)
                ]

            for idx in target_indices:
                if len(selected) >= budget:
                    break
                if idx < 0 or idx >= len(normalized_manifest):
                    continue
                manifest_clip = normalized_manifest[idx]
                if any(item.get("start") == manifest_clip["start"] for item in selected):
                    continue
                matching_candidate = None
                for candidate in normalized_candidates:
                    if _as_int(candidate.get("start"), 0) == manifest_clip["start"]:
                        matching_candidate = candidate
                        break
                if matching_candidate is not None:
                    add_item(matching_candidate, "sentinel_coverage", manifest_clip=manifest_clip)
                else:
                    synthetic_candidate = {
                        "candidate_id": f"sentinel_{manifest_clip['clip_id']}",
                        "start": manifest_clip["start"],
                        "end": manifest_clip["end"],
                        "suggested_trim_start": manifest_clip["start"],
                        "suggested_trim_end": manifest_clip["end"],
                        "triage_score": 0.0,
                        "triage_confidence": 0.0,
                        "vision_need": "none",
                        "selection_reasons": ["sentinel_coverage"],
                        "evidence_lines": [],
                        "risk_flags": [],
                    }
                    add_item(synthetic_candidate, "sentinel_coverage", manifest_clip=manifest_clip, synthetic=True)

    for candidate in normalized_candidates:
        if len(selected) >= budget:
            break
        if _as_int(candidate.get("start"), 0) in seen_starts:
            continue
        add_item(candidate, _TEXT_TOP_TAG)

    selected = sorted(selected, key=lambda item: (item["start"], item["end"], item.get("source_candidate_id") or ""))
    return selected[:budget]


def _normalize_transcript_segment(segment: Dict[str, Any], index: int) -> Dict[str, Any]:
    start = _as_int(segment.get("start"), 0)
    end = _as_int(segment.get("end"), start + 1)
    if end <= start:
        end = start + 1
    return {
        "start": start,
        "end": end,
        "text": _clean_text(segment.get("text"), f"segment_{index}"),
    }


def _normalize_chat_message(message: Dict[str, Any], index: int) -> Dict[str, Any]:
    timestamp = _as_int(message.get("timestamp"), 0)
    return {
        "timestamp": timestamp,
        "user": _clean_text(message.get("user"), f"user_{index}"),
        "message": _clean_text(message.get("message"), ""),
    }


def _normalize_manifest_for_summary(raw: Dict[str, Any], index: int) -> Dict[str, Any]:
    start = _as_int(raw.get("start"), 0)
    end = _as_int(raw.get("end"), start + 1)
    if end <= start:
        end = start + 1
    return {
        "start": start,
        "end": end,
        "clip_id": _clean_text(raw.get("clip_id") or raw.get("candidate_id"), f"manifest_{index}"),
    }


def summarize_chunk_signals(chunk: dict, manifest_clips: list[dict]) -> dict:
    chunk = chunk if isinstance(chunk, dict) else {}
    chunk_start = _as_int(chunk.get("chunk_start"), 0)
    chunk_end = _as_int(chunk.get("chunk_end"), chunk_start + 1)
    if chunk_end <= chunk_start:
        chunk_end = chunk_start + 1

    transcript_lines = chunk.get("transcript_lines") if isinstance(chunk.get("transcript_lines"), list) else []
    chat_messages = chunk.get("chat_messages") if isinstance(chunk.get("chat_messages"), list) else []

    transcript_count = len(transcript_lines)
    chat_count = len(chat_messages)
    duration_seconds = max(1, chunk_end - chunk_start)
    minutes = duration_seconds / 60.0

    transcript_density_per_min = round(transcript_count / minutes, 6)
    chat_density_per_min = round(chat_count / minutes, 6)
    has_chat_spike = chat_count >= 5 or chat_density_per_min >= 4.0

    normalized_manifest = [
        _normalize_manifest_for_summary(clip, idx)
        for idx, clip in enumerate(manifest_clips or [])
        if isinstance(clip, dict)
    ]

    manifest_candidate_starts = sorted(
        {
            clip["start"]
            for clip in normalized_manifest
            if clip["start"] < chunk_end and clip["end"] > chunk_start
        }
    )

    signal_flags: List[str] = []
    if has_chat_spike:
        signal_flags.append("chat_spike")

    summary = {
        "chunk_start": chunk_start,
        "chunk_end": chunk_end,
        "transcript_count": transcript_count,
        "chat_count": chat_count,
        "transcript_density_per_min": transcript_density_per_min,
        "chat_density_per_min": chat_density_per_min,
        "has_chat_spike": has_chat_spike,
        "manifest_candidate_starts": manifest_candidate_starts,
        "manifest_candidate_count": len(manifest_candidate_starts),
        "signal_flags": signal_flags,
    }

    return summary


def build_triage_chunks(
    transcript_segments: list[dict],
    chat_messages: list[dict],
    *,
    vod_start: int,
    vod_end: int,
    chunk_seconds: int = 600,
    overlap_seconds: int = 60,
) -> list[dict]:
    vod_start = _as_int(vod_start, 0)
    vod_end = _as_int(vod_end, vod_start + 1)
    if vod_end <= vod_start:
        vod_end = vod_start + 1

    chunk_seconds = max(1, _as_int(chunk_seconds, 600))
    overlap_seconds = max(0, _as_int(overlap_seconds, 0))
    step = chunk_seconds - overlap_seconds
    if step <= 0:
        step = chunk_seconds

    transcript_lines = [
        _normalize_transcript_segment(segment, idx)
        for idx, segment in enumerate(transcript_segments or [])
        if isinstance(segment, dict)
    ]
    transcript_lines = sorted(transcript_lines, key=lambda line: (line["start"], line["end"], line["text"]))

    normalized_chat = [
        _normalize_chat_message(message, idx)
        for idx, message in enumerate(chat_messages or [])
        if isinstance(message, dict)
    ]
    normalized_chat = sorted(normalized_chat, key=lambda message: (message["timestamp"], message["user"], message["message"]))

    chunks: List[Dict[str, Any]] = []
    current = vod_start
    chunk_index = 0

    while current < vod_end:
        chunk_end = min(current + chunk_seconds, vod_end)
        if chunk_end <= current:
            chunk_end = current + 1

        chunk_transcript = [
            line
            for line in transcript_lines
            if current <= line["start"] < chunk_end
        ]
        chunk_chat = [
            message
            for message in normalized_chat
            if current <= message["timestamp"] < chunk_end
        ]

        chunk = {
            "chunk_index": chunk_index,
            "chunk_start": current,
            "chunk_end": chunk_end,
            "transcript_lines": chunk_transcript,
            "chat_messages": chunk_chat,
            "manifest_candidate_starts": [],
        }
        chunk["signal_summary"] = summarize_chunk_signals(chunk, [])
        chunks.append(chunk)

        chunk_index += 1
        next_start = current + step
        if next_start <= current:
            next_start = current + 1
        current = next_start

    return chunks


FRAME_INTERVAL_SECONDS = 5


def _coerce_window_bounds(window: Dict[str, Any]) -> Tuple[int, int]:
    start = _as_int(window.get("start"), 0)
    end = _as_int(window.get("end"), start + 1)
    if end <= start:
        end = start + 1
    return start, end


def _frame_time_from_name(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("frame_"):
        return None
    try:
        return int(stem.split("_", 1)[1]) * FRAME_INTERVAL_SECONDS
    except (ValueError, IndexError):
        return None


def _collect_existing_frame_paths(frames_dir: str | Path) -> List[Tuple[int, str]]:
    directory = Path(frames_dir)
    if not directory.exists():
        return []
    collected: List[Tuple[int, str]] = []
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        timestamp = _frame_time_from_name(entry)
        if timestamp is None:
            continue
        collected.append((timestamp, str(entry)))
    return sorted(collected, key=lambda item: (item[0], item[1]))


def _pick_nearest_frames(targets: List[int], available: List[Tuple[int, str]], limit: int) -> List[str]:
    if limit <= 0 or not available:
        return []
    picked: List[str] = []
    seen = set()
    for target in targets:
        best_path = None
        best_key = None
        for timestamp, path in available:
            distance = abs(timestamp - target)
            key = (distance, timestamp, path)
            if best_key is None or key < best_key:
                best_key = key
                best_path = path
        if best_path and best_path not in seen:
            seen.add(best_path)
            picked.append(best_path)
        if len(picked) >= limit:
            break
    if len(picked) < limit:
        for _, path in available:
            if path in seen:
                continue
            picked.append(path)
            seen.add(path)
            if len(picked) >= limit:
                break
    return picked[:limit]


def build_gemma_annotation_windows(
    triage_chunks: list[dict],
    manifest_clips: list[dict],
    *,
    window_seconds: int = 30,
    stride_seconds: int = 30,
    max_windows: int = 0,
) -> list[dict]:
    window_seconds = max(1, _as_int(window_seconds, 30))
    stride_seconds = max(1, _as_int(stride_seconds, window_seconds))
    max_windows = max(0, _as_int(max_windows, 0))

    normalized_manifest = [
        _normalize_manifest_clip(clip, idx)
        for idx, clip in enumerate(manifest_clips or [])
        if isinstance(clip, dict)
    ]
    normalized_manifest = sorted(normalized_manifest, key=lambda clip: (clip["start"], clip["end"], clip["clip_id"]))

    windows: List[Dict[str, Any]] = []
    seen = set()

    def add_window(start: int, end: int, *, source: str, clip_id: str | None = None, chunk_index: int | None = None, manifest_clip_id: str | None = None, signal_flags: List[str] | None = None) -> None:
        start, end = _normalize_window(start, end, start, max(start + 1, end))
        key = (start, end, source, clip_id or "", manifest_clip_id or "", chunk_index if chunk_index is not None else -1)
        if key in seen:
            return
        seen.add(key)
        windows.append({
            "window_id": f"gemma_{start:07d}_{end:07d}",
            "start": start,
            "end": end,
            "source": source,
            "clip_id": clip_id,
            "chunk_index": chunk_index,
            "manifest_clip_id": manifest_clip_id,
            "signal_flags": list(signal_flags or []),
        })

    for clip in normalized_manifest:
        add_window(
            clip["start"],
            min(clip["end"], clip["start"] + window_seconds),
            source="manifest_backed",
            manifest_clip_id=clip["clip_id"],
        )

    for chunk in triage_chunks or []:
        if not isinstance(chunk, dict):
            continue
        summary = chunk.get("signal_summary") if isinstance(chunk.get("signal_summary"), dict) else {}
        flags = summary.get("signal_flags") if isinstance(summary.get("signal_flags"), list) else []
        chunk_start = _as_int(chunk.get("chunk_start"), 0)
        chunk_end = _as_int(chunk.get("chunk_end"), chunk_start + window_seconds)
        chunk_index = chunk.get("chunk_index") if isinstance(chunk.get("chunk_index"), int) else None
        if summary.get("has_chat_spike") or "chat_spike" in flags:
            add_window(
                chunk_start,
                min(chunk_end, chunk_start + window_seconds),
                source="chat_spike",
                chunk_index=chunk_index,
                signal_flags=flags,
            )

    if normalized_manifest and (max_windows == 0 or len(windows) < max_windows):
        target_count = max(1, min(len(normalized_manifest), max(1, len(normalized_manifest) // 3)))
        for idx in range(target_count):
            manifest_clip = normalized_manifest[round(idx * (len(normalized_manifest) - 1) / max(1, target_count - 1))]
            add_window(
                manifest_clip["start"],
                min(manifest_clip["end"], manifest_clip["start"] + window_seconds),
                source="sentinel_coverage",
                manifest_clip_id=manifest_clip["clip_id"],
            )

    windows = sorted(windows, key=lambda item: (item["start"], item["end"], item["source"], item.get("manifest_clip_id") or "", item.get("chunk_index") if item.get("chunk_index") is not None else -1))
    if max_windows > 0:
        windows = windows[:max_windows]
    return windows


def select_gemma_frames_for_window(window: dict, frames_dir: str, frames_per_window: int = 2) -> list[str]:
    start, end = _coerce_window_bounds(window if isinstance(window, dict) else {})
    frames_per_window = max(1, _as_int(frames_per_window, 2))
    available = _collect_existing_frame_paths(frames_dir)
    if not available:
        return []

    midpoint = start + max(1, (end - start) // 2)
    if frames_per_window == 1:
        targets = [midpoint]
    elif frames_per_window == 2:
        targets = [start, midpoint]
    else:
        targets = [start, midpoint, end]
        while len(targets) < frames_per_window:
            targets.append(end)
    return _pick_nearest_frames(targets, available, frames_per_window)


def build_gemma_audio_extract_command(vod_mp4: str, window: dict, output_wav: str) -> list[str]:
    start, end = _coerce_window_bounds(window if isinstance(window, dict) else {})
    duration = min(max(1, end - start), 30)
    return [
        "ffmpeg",
        "-nostats",
        "-y",
        "-ss",
        str(start),
        "-t",
        str(duration),
        "-i",
        str(vod_mp4),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]


def normalize_gemma_annotation(raw: dict, window: dict) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    window_start, window_end = _coerce_window_bounds(window if isinstance(window, dict) else {})

    def clamp_timestamp(value: Any) -> int:
        return _clamp_within_window(value, window_start, window_end, window_start)

    def clamp_confidence(value: Any) -> float:
        return max(0.0, min(1.0, _as_float(value, 0.0)))

    def normalize_event(event: Any, kind: str) -> dict:
        event = event if isinstance(event, dict) else {}
        normalized = {
            "timestamp": clamp_timestamp(event.get("timestamp", window_start)),
            "type": _clean_text(event.get("type"), "unknown").lower(),
            "confidence": round(clamp_confidence(event.get("confidence", 0.0)), 4),
            "evidence": _clean_text(event.get("evidence"), "")[:240],
        }
        if kind == "audio":
            normalized["speaker_guess"] = _clean_text(event.get("speaker_guess"), "unknown").lower()
        return normalized

    parse_ok = raw.get("parse_ok")
    if parse_ok is None:
        parse_ok = raw.get("error") in (None, "")
    parse_ok = bool(parse_ok)

    audio_events = [normalize_event(event, "audio") for event in raw.get("audio_events", []) if isinstance(event, dict)]
    visual_events = [normalize_event(event, "visual") for event in raw.get("visual_events", []) if isinstance(event, dict)]

    risk_flags = _dedupe_preserve_order([str(flag).strip() for flag in (raw.get("risk_flags") or []) if str(flag).strip()])
    clip_relevance_notes = [
        _clean_text(note, "")[:240]
        for note in (raw.get("clip_relevance_notes") or [])
        if _clean_text(note, "")
    ]

    speaker_nuance_raw = raw.get("speaker_nuance") if isinstance(raw.get("speaker_nuance"), dict) else {}
    emotion_nuance_raw = raw.get("emotion_nuance") if isinstance(raw.get("emotion_nuance"), dict) else {}
    speaker_nuance = {
        "primary_speaker": _clean_text(speaker_nuance_raw.get("primary_speaker"), "unknown"),
        "streamer_led_likelihood": clamp_confidence(speaker_nuance_raw.get("streamer_led_likelihood", 0.0)),
        "non_streamer_voice_present": bool(speaker_nuance_raw.get("non_streamer_voice_present", False)),
        "non_streamer_voice_type": _clean_text(speaker_nuance_raw.get("non_streamer_voice_type"), "unknown"),
    }
    emotion_nuance = {
        "streamer_affect": _clean_text(emotion_nuance_raw.get("streamer_affect"), "unknown"),
        "organic_reaction_likelihood": clamp_confidence(emotion_nuance_raw.get("organic_reaction_likelihood", 0.0)),
        "transactional_alert_likelihood": clamp_confidence(emotion_nuance_raw.get("transactional_alert_likelihood", 0.0)),
        "evidence": _clean_text(emotion_nuance_raw.get("evidence"), "")[:240],
    }

    return {
        "window_id": _clean_text(raw.get("window_id"), f"gemma_{window_start:07d}_{window_end:07d}"),
        "start": window_start,
        "end": window_end,
        "source_refs": raw.get("source_refs") if isinstance(raw.get("source_refs"), dict) else {
            "transcript_segment_ids": [],
            "chat_message_ids": [],
            "frame_paths": [],
            "audio_path": "",
        },
        "audio_events": audio_events,
        "visual_events": visual_events,
        "speaker_nuance": speaker_nuance,
        "emotion_nuance": emotion_nuance,
        "clip_relevance_notes": clip_relevance_notes,
        "risk_flags": risk_flags,
        "parse_ok": parse_ok,
        "error": raw.get("error"),
    }


def merge_gemma_annotations_into_chunk(chunk: dict, annotations: list[dict]) -> dict:
    chunk = dict(chunk) if isinstance(chunk, dict) else {}
    normalized = [ann for ann in (annotations or []) if isinstance(ann, dict)]
    chunk["gemma_annotations"] = normalized
    chunk["gemma_annotation_refs"] = _dedupe_preserve_order(
        ann.get("window_id", "") for ann in normalized if ann.get("window_id")
    )
    chunk["gemma_evidence_lines"] = _dedupe_preserve_order(
        note
        for ann in normalized
        for note in (ann.get("clip_relevance_notes") or [])
        if note
    )
    chunk["gemma_signal_summary"] = summarize_gemma_signals_for_triage(normalized)
    return chunk


def summarize_gemma_signals_for_triage(annotations: list[dict]) -> dict:
    normalized = [ann for ann in (annotations or []) if isinstance(ann, dict)]
    audio_counts: Dict[str, int] = {}
    visual_counts: Dict[str, int] = {}
    risk_counts: Dict[str, int] = {}
    streamer_led = []
    transactional = []
    organic = []
    evidence_lines: List[str] = []

    for ann in normalized:
        for event in ann.get("audio_events", []) or []:
            event_type = _clean_text(event.get("type"), "unknown")
            audio_counts[event_type] = audio_counts.get(event_type, 0) + 1
            if event_type in _GEMMA_AUDIO_SIGNAL_TYPES:
                evidence_lines.append(f"audio:{event_type}@{event.get('timestamp', 0)}")
        for event in ann.get("visual_events", []) or []:
            event_type = _clean_text(event.get("type"), "unknown")
            visual_counts[event_type] = visual_counts.get(event_type, 0) + 1
            if event_type in _GEMMA_SENTINEL_TYPES:
                evidence_lines.append(f"visual:{event_type}@{event.get('timestamp', 0)}")
        for flag in ann.get("risk_flags", []) or []:
            text = _clean_text(flag, "")
            if text:
                risk_counts[text] = risk_counts.get(text, 0) + 1
        speaker = ann.get("speaker_nuance") if isinstance(ann.get("speaker_nuance"), dict) else {}
        emotion = ann.get("emotion_nuance") if isinstance(ann.get("emotion_nuance"), dict) else {}
        streamer_led.append(_as_float(speaker.get("streamer_led_likelihood"), 0.0))
        transactional.append(_as_float(emotion.get("transactional_alert_likelihood"), 0.0))
        organic.append(_as_float(emotion.get("organic_reaction_likelihood"), 0.0))

    return {
        "annotation_count": len(normalized),
        "parse_failures": sum(1 for ann in normalized if not ann.get("parse_ok", True)),
        "audio_counts": audio_counts,
        "visual_counts": visual_counts,
        "risk_counts": risk_counts,
        "streamer_led_likelihood": round(max(streamer_led) if streamer_led else 0.0, 4),
        "transactional_alert_likelihood": round(max(transactional) if transactional else 0.0, 4),
        "organic_reaction_likelihood": round(max(organic) if organic else 0.0, 4),
        "has_audio_alert": any(audio_counts.get(tag, 0) for tag in ("donation_alert", "tts_alert", "laugh")),
        "has_visual_reaction": any(visual_counts.get(tag, 0) for tag in ("laughing", "surprised", "focused", "visual_payoff")),
        "evidence_lines": _dedupe_preserve_order(evidence_lines)[:20],
    }
