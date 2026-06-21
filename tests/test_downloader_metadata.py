from src.preprocessing.downloader import _parse_metadata


def test_parse_metadata_maps_game_fields():
    stdout = '{"title":"VOD","duration":120,"uploader":"streamer","game":"Valorant","game_id":"123","categories":["Valorant","FPS"],"ext":"mp4"}\n'
    meta = _parse_metadata(stdout, "https://www.twitch.tv/videos/1")
    assert meta.game_name == "Valorant"
    assert meta.game_id == "123"
    assert meta.categories == ["Valorant", "FPS"]
    assert meta.category_segments[0].game_name == "Valorant"
    assert meta.category_segments[0].source == "yt_dlp"


def test_parse_metadata_fallback_preserves_defaults():
    meta = _parse_metadata("", "https://www.twitch.tv/videos/1")
    assert meta.id == "1"
    assert meta.categories == []
    assert meta.category_segments == []
