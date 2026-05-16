"""
VOD Lens — Main CLI Entry Point

Command-line interface for running the preprocessing pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.preprocessing.pipeline import run_pipeline, run_pipeline_minimal


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

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
