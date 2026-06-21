from src.intelligence.game_knowledge_types import GameCategorySegment, GameKnowledgeProfile


def test_game_category_segment_defaults():
    segment = GameCategorySegment(game_name="Minecraft")
    assert segment.start == 0.0
    assert segment.end is None
    assert segment.game_id is None
    assert segment.game_name == "Minecraft"
    assert segment.source == "unknown"
    assert segment.confidence == 1.0


def test_game_knowledge_profile_defaults():
    profile = GameKnowledgeProfile(
        game_name="Minecraft",
        slug="minecraft",
        generated_at="2026-01-01T00:00:00Z",
        refreshed_at="2026-01-01T00:00:00Z",
    )
    assert profile.schema_version == 1
    assert profile.aliases == []
    assert profile.summary == ""
    assert profile.genres == []
    assert profile.core_mechanics == []
    assert profile.sources == []
    assert profile.spoiler_sensitive is False
