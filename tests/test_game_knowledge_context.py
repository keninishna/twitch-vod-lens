from src.intelligence.game_knowledge_context import render_game_knowledge_context
from src.intelligence.game_knowledge_types import GameCategorySegment, GameKnowledgeProfile


def test_renderer_includes_active_game_and_caution():
    profile = GameKnowledgeProfile(
        game_name="Minecraft",
        slug="minecraft",
        generated_at="2026-01-01T00:00:00Z",
        refreshed_at="2026-01-01T00:00:00Z",
    )
    segment = GameCategorySegment(game_name="Minecraft")
    text = render_game_knowledge_context(profile, segment, max_chars=2000)
    assert "GAME CONTEXT (evidence-backed, advisory)" in text
    assert "Minecraft" in text
    assert "actual transcript/chat/visual evidence wins" in text


def test_renderer_unavailable_and_caps():
    text = render_game_knowledge_context(None, None, max_chars=80)
    assert "unavailable" in text.lower()
    assert len(text) <= 80
