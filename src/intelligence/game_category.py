from __future__ import annotations

from typing import Any

from src.intelligence.game_knowledge_types import GameCategorySegment


def resolve_game_segments(vod_meta: dict, overrides: dict | None = None) -> list[GameCategorySegment]:
    overrides = overrides or {}
    if overrides.get("category_segments"):
        return [GameCategorySegment.model_validate(seg) for seg in overrides["category_segments"]]
    if overrides.get("game_name") or overrides.get("game_id"):
        duration = vod_meta.get("duration_seconds")
        return [GameCategorySegment(start=0.0, end=duration, game_name=overrides.get("game_name") or vod_meta.get("game_name") or "unknown", game_id=overrides.get("game_id"), source="manual", confidence=1.0)]
    if vod_meta.get("category_segments"):
        segments = [GameCategorySegment.model_validate(seg) for seg in vod_meta["category_segments"]]
        segments.sort(key=lambda s: s.start)
        return [GameCategorySegment(**{**seg.model_dump(), "end": None if seg.end is not None and seg.end <= seg.start else seg.end}) for seg in segments]
    game_name = vod_meta.get("game_name") or vod_meta.get("category")
    if not game_name:
        return []
    duration = vod_meta.get("duration_seconds")
    return [GameCategorySegment(start=0.0, end=duration, game_name=game_name, game_id=vod_meta.get("game_id"), source="vod_meta", confidence=1.0)]


def resolve_game_for_timestamp(segments: list[GameCategorySegment], seconds: float) -> GameCategorySegment | None:
    for segment in sorted(segments, key=lambda s: s.start):
        if segment.end is None:
            if seconds >= segment.start:
                return segment
        elif segment.start <= seconds < segment.end:
            return segment
    return None
