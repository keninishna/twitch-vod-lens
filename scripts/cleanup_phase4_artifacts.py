#!/usr/bin/env python3
"""Remove phase4 artifacts to free disk space after pipeline runs.

Three cleanup modes:

- **intermediate**: deletes per-step temp artifacts (audio batch I/O,
  fast-pass debug outputs). Keeps raw VOD, frames, fusion, manifest.
  Safe to run mid-pipeline before extraction.

- **post-extraction** (default): deletes raw VOD, frames, and all
  intermediate artifacts. Requires ``qwen_vision_progressive.json``
  to exist. Designed to be run after clip extraction/upload.

- **aggressive**: additionally removes small reproducibility inputs
  (fusion, manifest, transcript, scenes, chat, YOLO, speaker
  attribution). Only safe when you're done with the VOD.

Usage:

    python scripts/cleanup_phase4_artifacts.py \\
        --vod-id 2778478641 \\
        --mode post-extraction \\
        --dry-run

    python scripts/cleanup_phase4_artifacts.py \\
        --vod-id 2778478641 \\
        --mode post-extraction \\
        --write-report

    python scripts/cleanup_phase4_artifacts.py \\
        --vod-id 2778478641 \\
        --mode intermediate \\
        --keep-raw --keep-frames
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.artifacts.cleanup import (
    CleanupMode,
    build_cleanup_plan,
    execute_cleanup_plan,
)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove phase4 artifacts to free disk space after pipeline runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --vod-id 2778478641 --dry-run\n"
            "  %(prog)s --vod-id 2778478641 --mode aggressive --write-report\n"
            "  %(prog)s --vod-id 2778478641 --mode intermediate --keep-raw --keep-frames\n"
        ),
    )
    parser.add_argument(
        "--vod-id",
        required=True,
        help="VOD ID (used for directory name validation and artifact filenames)",
    )
    parser.add_argument(
        "--phase4-dir",
        type=Path,
        default=None,
        help=(
            "Phase4 directory (default: vods/phase4_<VOD_ID> relative to repo root)"
        ),
    )
    parser.add_argument(
        "--mode",
        default="post-extraction",
        choices=[m.value for m in CleanupMode],
        help="Cleanup aggressiveness (default: post-extraction)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without actually deleting anything",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow cleanup even if phase4 dir name doesn't match expected pattern",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep raw VOD mp4 file",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep extracted frame images",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write a cleanup_report_<VOD_ID>.json to the phase4 dir",
    )
    return parser


def main() -> int:
    parser = _build_cli()
    args = parser.parse_args()

    phase4_dir: Path = args.phase4_dir or (
        _REPO_ROOT / "vods" / f"phase4_{args.vod_id}"
    )
    phase4_dir = phase4_dir.resolve()

    if not phase4_dir.exists():
        print(f"Error: phase4 directory not found: {phase4_dir}", file=sys.stderr)
        return 1

    # Build cleanup plan
    plan = build_cleanup_plan(
        phase4_dir=phase4_dir,
        vod_id=args.vod_id,
        mode=args.mode,
        include_raw=not args.keep_raw,
        include_frames=not args.keep_frames,
    )

    if not plan:
        print(
            f"Nothing to clean up. Either all targets are already gone, or\n"
            f"  mode={args.mode} requires qwen_vision_progressive.json and it\n"
            f"  was not found in {phase4_dir}.",
            file=sys.stderr,
        )
        return 0

    if args.dry_run:
        print(f"[DRY RUN] Would clean {phase4_dir} (mode={args.mode})")
        print(f"[DRY RUN] {len(plan)} target(s):")
        for t in plan:
            _show_target(t)
        print()
    else:
        print(f"Cleaning {phase4_dir} (mode={args.mode}) ...")

    result = execute_cleanup_plan(
        plan,
        phase4_dir=phase4_dir,
        dry_run=args.dry_run,
        mode=args.mode,
    )

    if result.deleted:
        print(f"  Deleted {len(result.deleted)} items, freed {_fmt_bytes(result.bytes_freed)}")
    if result.skipped:
        print(f"  Skipped {len(result.skipped)} items (protected / not found)")
        for s in result.skipped:
            print(f"    - {s.path}: {s.reason}")

    if args.dry_run:
        # In dry-run, report what *would* be freed
        print(f"  Would free: {_fmt_bytes(result.bytes_freed)}")
    else:
        print(f"  Freed: {_fmt_bytes(result.bytes_freed)}")

    # Write report if requested (only on actual runs)
    if args.write_report and not args.dry_run:
        report_path = phase4_dir / f"cleanup_report_{args.vod_id}.json"
        import json

        report_path.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"  Report: {report_path}")

    return 0


def _show_target(target) -> None:
    if target.kind == "glob":
        print(f"    {target.glob_pattern} ({target.reason})")
    else:
        print(f"    {target.path.relative_to(target.path.parents[2] if target.path.parents else target.path)} ({target.reason})")


def _fmt_bytes(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"


if __name__ == "__main__":
    raise SystemExit(main())