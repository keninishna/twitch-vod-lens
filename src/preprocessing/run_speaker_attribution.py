"""CLI: generate speaker_attribution_<VOD_ID>.json."""

from __future__ import annotations

# Direct-script bootstrap: avoid local src/preprocessing/types.py shadowing stdlib `types`.
if __package__ in (None, ""):
    import os
    import sys

    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    if _THIS_DIR in sys.path:
        sys.path.remove(_THIS_DIR)
    _REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

import argparse
from pathlib import Path

from src.intelligence.streamer_store import load_persistent_voice_profiles, resolve_streamer_id
from src.preprocessing.speaker_attribution import generate_speaker_attribution
from src.preprocessing.speaker_profiles import load_profiles


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate speaker attribution artifact")
    parser.add_argument("--vod-id", required=True)
    parser.add_argument("--vod-media", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--chat", type=Path)
    parser.add_argument("--profiles-dir", type=Path)
    parser.add_argument("--streamer-id", default=None)
    parser.add_argument("--profile-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--require-speaker-id", action="store_true")
    args = parser.parse_args()

    try:
        runtime_profiles = []
        if args.profiles_dir is not None:
            runtime_profiles.extend(load_profiles(args.profiles_dir))

        if args.profile_root is not None:
            if args.streamer_id:
                sid = resolve_streamer_id({}, override=args.streamer_id)
                runtime_profiles.extend(load_persistent_voice_profiles(sid, args.profile_root))
            else:
                print("WARN: --profile-root was provided without --streamer-id; skipping persistent voice profile load")

        result = generate_speaker_attribution(
            vod_id=args.vod_id,
            vod_media=args.vod_media,
            transcript_path=args.transcript,
            chat_path=args.chat,
            profiles_dir=args.profiles_dir,
            profiles=runtime_profiles or None,
            output_path=args.output,
            hf_token=args.hf_token,
            require_speaker_id=args.require_speaker_id,
        )
        print(
            f"speaker attribution saved: {args.output} "
            f"(segments={len(result.segments)}, clusters={len(result.speaker_clusters)})"
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
