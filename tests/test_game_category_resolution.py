from src.intelligence.game_category import resolve_game_for_timestamp, resolve_game_segments


def test_manual_override_wins():
    vod_meta = {"duration_seconds": 100, "game_name": "Old Game"}
    overrides = {"game_name": "New Game", "game_id": "123"}
    segments = resolve_game_segments(vod_meta, overrides=overrides)
    assert len(segments) == 1
    assert segments[0].game_name == "New Game"
    assert segments[0].game_id == "123"


def test_timestamped_segments_resolved_correctly():
    vod_meta = {
        "duration_seconds": 100,
        "category_segments": [
            {"start": 0, "end": 40, "game_name": "Game A"},
            {"start": 40, "end": 100, "game_name": "Game B"},
        ],
    }
    segments = resolve_game_segments(vod_meta)
    assert resolve_game_for_timestamp(segments, 10).game_name == "Game A"
    assert resolve_game_for_timestamp(segments, 50).game_name == "Game B"


def test_single_game_name_applies_to_whole_vod():
    vod_meta = {"duration_seconds": 100, "game_name": "Game A"}
    segments = resolve_game_segments(vod_meta)
    assert len(segments) == 1
    assert segments[0].start == 0.0
    assert segments[0].end == 100


def test_missing_game_returns_empty_and_none():
    segments = resolve_game_segments({"duration_seconds": 100})
    assert segments == []
    assert resolve_game_for_timestamp([], 10) is None
