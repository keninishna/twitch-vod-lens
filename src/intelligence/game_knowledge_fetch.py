from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from src.intelligence.game_knowledge_store import (
    is_profile_fresh,
    load_game_profile,
    save_game_profile,
    slugify_game_name,
)
from src.intelligence.game_knowledge_types import GameKnowledgeProfile, GameKnowledgeSource

TWITCH_TOKEN_URL = "https://id.twitch.tv/oauth2/token"
IGDB_API_BASE = "https://api.igdb.com/v4"
WIKIPEDIA_SUMMARY_BASE = "https://en.wikipedia.org/api/rest_v1/page/summary"


def _session(session: Any | None = None) -> Any:
    return session or requests.Session()


def _now_iso(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_name(name: str) -> str:
    return " ".join(name.lower().replace("_", " ").split())


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _extract_names(items: Any) -> list[str]:
    result: list[str] = []
    for item in items or []:
        if isinstance(item, dict):
            if isinstance(item.get("name"), str):
                result.append(item["name"])
            company = item.get("company")
            if isinstance(company, dict) and isinstance(company.get("name"), str):
                result.append(company["name"])
    return _dedupe(result)


def fetch_twitch_app_token(client_id: str, client_secret: str, session: Any | None = None) -> str:
    resp = _session(session).post(
        TWITCH_TOKEN_URL,
        data={"client_id": client_id, "client_secret": client_secret, "grant_type": "client_credentials"},
        timeout=30,
    )
    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(f"Failed to fetch Twitch app token: HTTP {resp.status_code}")
    data = resp.json() if hasattr(resp, "json") else {}
    token = data.get("access_token")
    if not token:
        raise RuntimeError("Twitch token response missing access_token")
    return token


def igdb_request(endpoint: str, body: str, client_id: str, access_token: str, session: Any | None = None) -> list[dict]:
    resp = _session(session).post(
        f"{IGDB_API_BASE}/{endpoint}",
        data=body,
        headers={"Client-ID": client_id, "Authorization": f"Bearer {access_token}", "Accept": "application/json"},
        timeout=30,
    )
    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(f"IGDB request failed for {endpoint}: HTTP {resp.status_code}")
    data = resp.json() if hasattr(resp, "json") else []
    return data if isinstance(data, list) else []


def fetch_igdb_game(game_name: str, client_id: str, access_token: str, session: Any | None = None) -> dict | None:
    query = f'fields name,summary,storyline,genres.name,themes.name,keywords.name,game_modes.name,franchises.name,involved_companies.company.name,updated_at,first_release_date; search "{game_name}"; limit 10;'
    results = igdb_request("games", query, client_id, access_token, session=session)
    if not results:
        return None
    target = _normalize_name(game_name)
    for item in results:
        if _normalize_name(str(item.get("name", ""))) == target:
            return item
    return results[0]


def fetch_wikipedia_summary(game_name: str, session: Any | None = None) -> dict | None:
    sess = _session(session)
    for title in (f"{game_name} (video game)", game_name):
        url = f"{WIKIPEDIA_SUMMARY_BASE}/{quote(title, safe='')}"
        resp = sess.get(url, timeout=30)
        if resp.status_code == 404:
            continue
        if resp.status_code < 200 or resp.status_code >= 300:
            raise RuntimeError(f"Wikipedia summary request failed for {title}: HTTP {resp.status_code}")
        data = resp.json() if hasattr(resp, "json") else {}
        if data.get("type") == "standard" or data.get("extract"):
            return {"title": data.get("title", title), "extract": data.get("extract", ""), "url": data.get("content_urls", {}).get("desktop", {}).get("page")}
    return None


def build_profile_from_sources(game_name: str, igdb_game: dict | None = None, wiki_summary: dict | None = None, now: datetime | None = None) -> GameKnowledgeProfile:
    now_iso = _now_iso(now)
    slug = slugify_game_name(game_name)
    summary = ""
    source_updated_at = None
    genres: list[str] = []
    common_terms: list[str] = []
    sources: list[GameKnowledgeSource] = []
    game_id = str(igdb_game["id"]) if igdb_game and igdb_game.get("id") is not None else None
    if igdb_game:
        summary = igdb_game.get("summary") or igdb_game.get("storyline") or summary
        genres = _extract_names(igdb_game.get("genres"))
        common_terms = _dedupe(_extract_names(igdb_game.get("keywords")) + _extract_names(igdb_game.get("themes")) + _extract_names(igdb_game.get("game_modes")) + _extract_names(igdb_game.get("franchises")))
        source_updated_at = igdb_game.get("updated_at")
        sources.append(GameKnowledgeSource(source_type="igdb", title=igdb_game.get("name", game_name), source_id=game_id, updated_at=source_updated_at, retrieved_at=now_iso))
    if wiki_summary:
        if not summary:
            summary = wiki_summary.get("extract", "") or ""
        sources.append(GameKnowledgeSource(source_type="wikipedia", title=wiki_summary.get("title", game_name), url=wiki_summary.get("url"), retrieved_at=now_iso))
    text = summary.strip()
    core_objective = ""
    if text:
        core_objective = text[:140].rsplit(" ", 1)[0].rstrip(".,;: ") if len(text) > 140 else text.rstrip(".,;: ")
    clip_eval_hints = _dedupe([f"Look for moments related to {genre.lower()} gameplay." for genre in genres[:2]] + [f"Watch for references to {term.lower()}." for term in common_terms[:3]])
    if not clip_eval_hints:
        clip_eval_hints = ["Use only directly observed on-screen evidence."]
    return GameKnowledgeProfile(
        game_id=game_id,
        game_name=game_name,
        slug=slug,
        summary=summary,
        genres=genres,
        core_objective=core_objective,
        common_terms=common_terms,
        notable_entities=[],
        clip_eval_hints=clip_eval_hints,
        source_updated_at=source_updated_at,
        sources=sources,
        generated_at=now_iso,
        refreshed_at=now_iso,
    )


def _minimal_profile(game_name: str, now: datetime | None = None) -> GameKnowledgeProfile:
    now_iso = _now_iso(now)
    return GameKnowledgeProfile(
        game_name=game_name,
        slug=slugify_game_name(game_name),
        summary="",
        genres=[],
        core_objective="Identify the game from direct on-screen evidence.",
        common_terms=[],
        notable_entities=[],
        clip_eval_hints=["Use only directly observed on-screen evidence."],
        sources=[GameKnowledgeSource(source_type="manual", title=game_name, retrieved_at=now_iso)],
        generated_at=now_iso,
        refreshed_at=now_iso,
    )


def gather_game_knowledge(game_name: str, root: Path, client_id: str | None = None, client_secret: str | None = None, force_refresh: bool = False, refresh_days: int = 30, session: Any | None = None, now: datetime | None = None) -> tuple[GameKnowledgeProfile, dict]:
    now = now or datetime.now(timezone.utc)
    warnings: list[str] = []
    cached = load_game_profile(None, game_name, root)
    creds = bool(client_id and client_secret)
    igdb_game = None
    wiki_summary = None
    if creds:
        token = fetch_twitch_app_token(client_id, client_secret, session=session)
        igdb_game = fetch_igdb_game(game_name, client_id, token, session=session)
        if cached and not force_refresh and is_profile_fresh(cached, igdb_game.get("updated_at") if igdb_game else None, refresh_days, now):
            return cached, {"cache_status": "cache_hit_fresh", "warnings": [], "source_booleans": {"cached": True, "igdb": True, "wikipedia": False}}
    else:
        if cached and not force_refresh:
            return cached, {"cache_status": "cache_hit_no_freshness_probe", "warnings": [], "source_booleans": {"cached": True, "igdb": False, "wikipedia": False}}
        warnings.append("No IGDB credentials provided.")
    try:
        wiki_summary = fetch_wikipedia_summary(game_name, session=session)
    except Exception as exc:
        warnings.append(str(exc))
    if igdb_game or wiki_summary:
        profile = build_profile_from_sources(game_name, igdb_game=igdb_game, wiki_summary=wiki_summary, now=now)
        path = save_game_profile(profile, root)
        return profile, {"cache_status": "refreshed", "warnings": warnings, "path": str(path), "source_booleans": {"cached": bool(cached), "igdb": bool(igdb_game), "wikipedia": bool(wiki_summary)}}
    if cached is not None:
        warnings.append("Falling back to stale cached profile.")
        return cached, {"cache_status": "stale_fallback", "warnings": warnings, "source_booleans": {"cached": True, "igdb": False, "wikipedia": False}}
    profile = _minimal_profile(game_name, now=now)
    path = save_game_profile(profile, root)
    return profile, {"cache_status": "minimal_profile", "warnings": warnings, "path": str(path), "source_booleans": {"cached": False, "igdb": False, "wikipedia": False}}


__all__ = [
    "build_profile_from_sources",
    "fetch_igdb_game",
    "fetch_twitch_app_token",
    "fetch_wikipedia_summary",
    "gather_game_knowledge",
    "igdb_request",
]
