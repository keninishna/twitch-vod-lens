from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.intelligence.game_knowledge_fetch import (
    build_profile_from_sources,
    fetch_igdb_game,
    fetch_twitch_app_token,
    fetch_wikipedia_summary,
    gather_game_knowledge,
    igdb_request,
)
from src.intelligence.game_knowledge_store import load_game_profile


class FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def _next(self):
        assert self.responses, "no more fake responses"
        return self.responses.pop(0)

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append(("post", url, data, headers, timeout))
        return self._next()

    def get(self, url, timeout=None):
        self.calls.append(("get", url, timeout))
        return self._next()


def test_fetch_twitch_app_token_posts_expected_params():
    session = FakeSession([FakeResponse(200, {"access_token": "abc123"})])
    token = fetch_twitch_app_token("cid", "secret", session=session)
    assert token == "abc123"
    method, url, data, headers, timeout = session.calls[0]
    assert method == "post"
    assert "id.twitch.tv/oauth2/token" in url
    assert data["grant_type"] == "client_credentials"
    assert data["client_id"] == "cid"
    assert data["client_secret"] == "secret"


def test_igdb_request_sends_expected_headers_and_body():
    session = FakeSession([FakeResponse(200, [{"id": 1}])])
    result = igdb_request("games", "fields name; limit 1;", "cid", "token", session=session)
    assert result == [{"id": 1}]
    method, url, data, headers, timeout = session.calls[0]
    assert method == "post"
    assert url.endswith("/games")
    assert data == "fields name; limit 1;"
    assert headers["Client-ID"] == "cid"
    assert headers["Authorization"] == "Bearer token"


def test_fetch_igdb_game_prefers_exact_normalized_match_over_first_fuzzy_match(monkeypatch):
    monkeypatch.setattr(
        "src.intelligence.game_knowledge_fetch.igdb_request",
        lambda *a, **k: [{"name": "7 Days to Die Alpha"}, {"name": "7 Days to Die"}],
    )
    game = fetch_igdb_game("7 Days to Die", "cid", "token")
    assert game["name"] == "7 Days to Die"


def test_fetch_wikipedia_summary_tries_video_game_then_fallback():
    session = FakeSession([
        FakeResponse(404, {}),
        FakeResponse(200, {"title": "7 Days to Die", "extract": "Survival game.", "content_urls": {"desktop": {"page": "https://example.test/wiki"}}}),
    ])
    summary = fetch_wikipedia_summary("7 Days to Die", session=session)
    assert summary["title"] == "7 Days to Die"
    assert summary["extract"] == "Survival game."
    assert len([call for call in session.calls if call[0] == "get"]) == 2


def test_build_profile_from_sources_maps_fields_and_sources():
    profile = build_profile_from_sources(
        "7 Days to Die",
        igdb_game={
            "id": 99,
            "name": "7 Days to Die",
            "summary": "Survive the zombie apocalypse.",
            "genres": [{"name": "Survival"}],
            "keywords": [{"name": "zombies"}],
            "themes": [{"name": "Horror"}],
            "game_modes": [{"name": "Single player"}],
            "franchises": [{"name": "7 Days to Die"}],
            "updated_at": 12345,
        },
        wiki_summary={"title": "7 Days to Die", "extract": "Wiki extract", "url": "https://example.test"},
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert profile.game_id == "99"
    assert profile.slug == "7-days-to-die"
    assert profile.genres == ["Survival"]
    assert "zombies" in profile.common_terms
    assert "Horror" in profile.common_terms
    assert profile.source_updated_at == 12345
    assert any(src.source_type == "igdb" for src in profile.sources)
    assert any(src.source_type == "wikipedia" for src in profile.sources)


def test_gather_game_knowledge_no_creds_creates_and_saves_minimal_profile(tmp_path):
    profile, meta = gather_game_knowledge("Mystery Game", tmp_path, session=FakeSession([FakeResponse(404, {})]))
    assert meta["cache_status"] == "minimal_profile"
    loaded = load_game_profile(None, "Mystery Game", tmp_path)
    assert loaded is not None
    assert loaded.game_name == "Mystery Game"
    assert profile.slug == "mystery-game"


def test_gather_game_knowledge_uses_cached_profile_when_no_creds_and_cache_exists(tmp_path):
    existing = build_profile_from_sources("Cache Game", wiki_summary={"title": "Cache Game", "extract": "Cached"})
    from src.intelligence.game_knowledge_store import save_game_profile
    save_game_profile(existing, tmp_path)
    profile, meta = gather_game_knowledge("Cache Game", tmp_path)
    assert meta["cache_status"] == "cache_hit_no_freshness_probe"
    assert profile.game_name == "Cache Game"


def test_gather_game_knowledge_with_unchanged_igdb_updated_at_skips_refresh(tmp_path, monkeypatch):
    existing = build_profile_from_sources(
        "Fresh Game",
        igdb_game={"id": 1, "name": "Fresh Game", "updated_at": 100, "summary": "Old"},
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    from src.intelligence.game_knowledge_store import save_game_profile
    save_game_profile(existing, tmp_path)

    class EmptyIGDBSession(FakeSession):
        pass

    monkeypatch.setattr("src.intelligence.game_knowledge_fetch.fetch_twitch_app_token", lambda *a, **k: "token")
    monkeypatch.setattr("src.intelligence.game_knowledge_fetch.fetch_igdb_game", lambda *a, **k: {"id": 1, "name": "Fresh Game", "updated_at": 100})
    monkeypatch.setattr("src.intelligence.game_knowledge_fetch.fetch_wikipedia_summary", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not call wikipedia")))
    profile, meta = gather_game_knowledge("Fresh Game", tmp_path, client_id="cid", client_secret="sec", session=EmptyIGDBSession([]), now=datetime(2026, 1, 2, tzinfo=timezone.utc))
    assert meta["cache_status"] == "cache_hit_fresh"
    assert profile.game_name == "Fresh Game"


def test_cli_json_works(monkeypatch, tmp_path, capsys):
    from scripts import update_game_knowledge as cli

    fake_profile = build_profile_from_sources("CLI Game", wiki_summary={"title": "CLI Game", "extract": "x"})
    monkeypatch.setattr(cli, "gather_game_knowledge", lambda *a, **k: (fake_profile, {"cache_status": "refreshed", "warnings": [], "path": "/tmp/x"}))
    monkeypatch.setattr("sys.argv", ["update_game_knowledge.py", "--game-name", "CLI Game", "--root", str(tmp_path), "--json"])
    assert cli.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["game_name"] == "CLI Game"
    assert out["cache_status"] == "refreshed"
