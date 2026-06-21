from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from src.intelligence.game_knowledge_types import GameKnowledgeProfile


def slugify_game_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "game"


def game_profile_dir(game_id: str | None, game_name: str, root: Path) -> Path:
    return root / (game_id or slugify_game_name(game_name))


def load_game_profile(game_id: str | None, game_name: str, root: Path) -> GameKnowledgeProfile | None:
    path = game_profile_dir(game_id, game_name, root) / "profile.json"
    if not path.exists():
        index = root / "index.json"
        if index.exists():
            data = json.loads(index.read_text())
            lookup_keys = [game_id, slugify_game_name(game_name), game_name]
            for key in lookup_keys:
                if not key:
                    continue
                entry = data.get(str(key))
                if isinstance(entry, dict) and entry.get("path"):
                    path = Path(entry["path"])
                    break
    if not path.exists():
        return None
    return GameKnowledgeProfile.model_validate_json(path.read_text())


def save_game_profile(profile: GameKnowledgeProfile, root: Path) -> Path:
    directory = game_profile_dir(profile.game_id, profile.game_name, root)
    directory.mkdir(parents=True, exist_ok=True)
    profile_path = directory / "profile.json"
    profile_path.write_text(profile.model_dump_json(indent=2))
    index_path = root / "index.json"
    index = {}
    if index_path.exists():
        index = json.loads(index_path.read_text())
    entry = {
        "slug": profile.slug,
        "game_id": profile.game_id,
        "game_name": profile.game_name,
        "refreshed_at": profile.refreshed_at,
        "path": str(profile_path),
    }
    index[profile.slug] = entry
    if profile.game_id:
        index[str(profile.game_id)] = entry
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
    return profile_path


def is_profile_fresh(profile: GameKnowledgeProfile, upstream_updated_at: int | None, refresh_days: int, now: datetime) -> bool:
    if upstream_updated_at is not None and profile.source_updated_at is not None:
        return upstream_updated_at <= profile.source_updated_at
    try:
        refreshed_at = datetime.fromisoformat(profile.refreshed_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    if refreshed_at.tzinfo is None:
        refreshed_at = refreshed_at.replace(tzinfo=timezone.utc)
    return (now - refreshed_at).days <= refresh_days
