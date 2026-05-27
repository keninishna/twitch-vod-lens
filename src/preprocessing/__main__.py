"""
VOD Lens — Main CLI Entry Point

Command-line interface for running the preprocessing pipeline.
"""

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
import json
import subprocess
import sys
from pathlib import Path


def _legacy_preprocess_fallback(url: str, output: Path | None) -> int:
    """Fallback to root-level preprocess.py when modern pipeline import fails."""
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [sys.executable, "preprocess.py", url]
    proc = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout[-1000:])
        print(proc.stderr[-1000:], file=sys.stderr)
        return proc.returncode

    vod_id = url.rstrip("/").split("/")[-1]
    legacy_out = repo_root / "output" / vod_id
    if output:
        # Best effort: expose where output lives; actual schema conversion happens in prepare_phase4.py
        print(f"Legacy preprocessing completed at: {legacy_out}")
        print(f"Requested output path: {output} (not written by legacy fallback)")
    else:
        print(f"Legacy preprocessing completed at: {legacy_out}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="VOD Lens Preprocessing Pipeline",
    )
    parser.add_argument("url", nargs="?", help="Twitch VOD URL")
    parser.add_argument(
        "--audio", "-a",
        help="Path to existing audio file (skip download)",
    )
    parser.add_argument(
        "--model", "-m",
        default="large-v3",
        help="Whisper model size (default: large-v3)",
    )
    parser.add_argument(
        "--language", "-l",
        default="en",
        help="Language code (default: en)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--workdir", "-w",
        type=Path,
        help="Working directory for temp files",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=12.0,
        help="Scene detection threshold (default: 12.0)",
    )

    args = parser.parse_args()

    if not args.url and not args.audio:
        parser.print_help()
        sys.exit(1)

    # Lazy import so --help still works even if modern stack is broken.
    try:
        from src.preprocessing.pipeline import run_pipeline, run_pipeline_minimal
    except Exception as import_error:  # noqa: BLE001
        if not args.url:
            print(
                f"Error: modern pipeline unavailable ({import_error}) and no URL for fallback",
                file=sys.stderr,
            )
            sys.exit(1)

        print(
            f"Warning: modern pipeline unavailable ({import_error}); "
            "falling back to preprocess.py",
            file=sys.stderr,
        )
        rc = _legacy_preprocess_fallback(args.url, args.output)
        sys.exit(rc)

    try:
        if args.audio:
            result = run_pipeline_minimal(
                audio_path=Path(args.audio),
                url=args.url,
                workdir=args.workdir,
                model_size=args.model,
                language=args.language,
            )
        else:
            result = run_pipeline(
                url=args.url,
                workdir=args.workdir,
                model_size=args.model,
                language=args.language,
                scene_threshold=args.threshold,
                output_path=args.output,
            )

        if args.output:
            print(f"Output saved to: {args.output}")
        else:
            print(json.dumps(result.model_dump(), indent=2, default=str))

    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
