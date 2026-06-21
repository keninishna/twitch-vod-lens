from src.synthesis import qwen_clip_analyzer_progressive as mod
from src.intelligence.game_knowledge_types import GameKnowledgeProfile


def test_build_game_context_state_writes_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "ENABLE_GAME_KNOWLEDGE", True)
    monkeypatch.setattr(mod, "GAME_KNOWLEDGE_ROOT", tmp_path / "knowledge")
    monkeypatch.setattr(mod, "GAME_NAME_OVERRIDE", None)
    monkeypatch.setattr(mod, "GAME_ID_OVERRIDE", None)
    monkeypatch.setattr(mod, "GAME_CATEGORY_TIMELINE", None)

    profile = GameKnowledgeProfile(
        game_name="7 Days to Die",
        slug="7-days-to-die",
        summary="Survival crafting game.",
        generated_at="2026-01-01T00:00:00Z",
        refreshed_at="2026-01-01T00:00:00Z",
    )

    def fake_gather(game_name, root, client_id=None, client_secret=None, refresh_days=30):
        return profile, {"cache_status": "refreshed", "warnings": [], "path": "x"}

    monkeypatch.setattr(mod, "gather_game_knowledge", fake_gather)

    state = mod._build_game_context_state(
        {"game_name": "7 Days to Die", "duration_seconds": 120},
        tmp_path,
    )

    artifact = tmp_path / f"game_context_{mod.VOD_ID}.json"
    assert artifact.exists()
    assert state["enabled"] is True
    assert state["segments"]
    assert state["profiles_by_game_name"]["7 Days to Die"]["cache_status"] == "refreshed"


def test_render_game_context_for_seconds_and_run_context(tmp_path, monkeypatch):
    profile = GameKnowledgeProfile(
        game_name="7 Days to Die",
        slug="7-days-to-die",
        summary="Survival crafting game.",
        core_objective="survive",
        clip_eval_hints=["caution"],
        generated_at="2026-01-01T00:00:00Z",
        refreshed_at="2026-01-01T00:00:00Z",
    )
    state = {
        "enabled": True,
        "segments": [{"start": 0, "end": 120, "game_name": "7 Days to Die", "game_id": None, "source": "manual", "confidence": 1.0}],
        "profiles_by_game_name": {"7 Days to Die": {"profile": profile, "cache_status": "fresh", "warnings": []}},
        "warnings": [],
        "artifact_path": str(tmp_path / "game_context.json"),
    }

    text = mod._render_game_context_for_seconds(state, 42)
    run_text = mod._render_game_run_context(state)

    assert "7 Days to Die" in text
    assert "Caution" in text or "caution" in text.lower()
    assert "7 Days to Die" in run_text
    assert "cache_status=fresh" in run_text


def test_render_game_context_disabled(monkeypatch):
    assert "disabled" in mod._render_game_context_for_seconds({"enabled": False}, 0).lower()
    assert "unavailable" in mod._render_game_run_context({"enabled": False}).lower()


def test_analysis_prompt_includes_game_context_rules():
    prompt = mod.ANALYSIS_PROMPT.format(
        clip_title="test clip",
        start=1,
        end=2,
        streamer_profile_context="PROFILE",
        game_knowledge_context="GAME CONTEXT",
        phase1_title_research_summary="TITLE SUMMARY",
        phase1_title_examples="TITLE EXAMPLES",
        transcript="transcript",
        chat_messages="chat",
        yolo_objects="none",
        fast_pass_evidence_context="evidence",
        batch_context="batch",
        platform_guide="guide",
    )
    assert "GAME-CONTEXT RULES" in prompt
    assert "GAME CONTEXT" in prompt
