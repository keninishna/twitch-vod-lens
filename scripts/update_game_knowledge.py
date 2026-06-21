#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.intelligence.game_knowledge_fetch import gather_game_knowledge


def main() -> int:
    parser = argparse.ArgumentParser(description="Update cached game knowledge")
    parser.add_argument("--game-name", required=True)
    parser.add_argument("--root", default="data/game_knowledge")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--refresh-days", type=int, default=30)
    parser.add_argument("--client-id", default=os.getenv("TWITCH_CLIENT_ID"))
    parser.add_argument("--client-secret", default=os.getenv("TWITCH_CLIENT_SECRET"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    profile, meta = gather_game_knowledge(
        args.game_name,
        Path(args.root),
        client_id=args.client_id,
        client_secret=args.client_secret,
        force_refresh=args.force_refresh,
        refresh_days=args.refresh_days,
    )
    path = meta.get("path") or str(Path(args.root) / (profile.game_id or profile.slug) / "profile.json")
    if args.json:
        print(json.dumps({"game_name": profile.game_name, "slug": profile.slug, "cache_status": meta.get("cache_status"), "path": path, "warnings": meta.get("warnings", [])}, indent=2))
    else:
        print(f"Game knowledge updated: {profile.game_name} -> {path} (status={meta.get('cache_status')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
