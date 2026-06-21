from datetime import datetime, timezone

from src.intelligence.game_knowledge_store import (
    is_profile_fresh,
    load_game_profile,
    save_game_profile,
    slugify_game_name,
)
from src.intelligence.game_knowledge_types import GameKnowledgeProfile


def test_slugify_game_name():
    assert slugify_game_name("7 Days to Die") == "7-days-to-die"


def test_save_load_roundtrip(tmp_path):
    profile = GameKnowledgeProfile(
        game_name="7 Days to Die",
        slug="7-days-to-die",
        generated_at="2026-01-01T00:00:00Z",
        refreshed_at="2026-01-02T00:00:00Z",
    )
    path = save_game_profile(profile, tmp_path)
    assert path.name == "profile.json"
    assert (tmp_path / "index.json").exists()
    loaded = load_game_profile(None, "7 Days to Die", tmp_path)
    assert loaded is not None
    assert loaded.game_name == "7 Days to Die"


def test_profile_saved_with_game_id_can_load_by_slug_or_id(tmp_path):
    profile = GameKnowledgeProfile(
        game_id="123",
        game_name="Fresh Game",
        slug="fresh-game",
        source_updated_at=100,
        generated_at="2026-01-01T00:00:00Z",
        refreshed_at="2026-01-02T00:00:00Z",
    )

    save_game_profile(profile, tmp_path)

    by_slug = load_game_profile(None, "Fresh Game", tmp_path)
    by_id = load_game_profile("123", "Fresh Game", tmp_path)
    assert by_slug is not None
    assert by_id is not None
    assert by_slug.game_id == "123"
    assert by_id.slug == "fresh-game"


def test_is_profile_fresh_rules():
    profile = GameKnowledgeProfile(
        game_name="Game",
        slug="game",
        source_updated_at=100,
        generated_at="2026-01-01T00:00:00Z",
        refreshed_at="2026-01-02T00:00:00Z",
    )
    now = datetime(2026, 1, 10, tzinfo=timezone.utc)
    assert is_profile_fresh(profile, 100, 7, now) is True
    assert is_profile_fresh(profile, 101, 7, now) is False
    profile.source_updated_at = None
    assert is_profile_fresh(profile, None, 7, now) is False
