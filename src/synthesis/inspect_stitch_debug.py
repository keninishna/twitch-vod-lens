#!/usr/bin/env python3
"""Inspect Stage 1.5 stitch-debug output from qwen_vision_progressive.json.

Usage:
  python src/synthesis/inspect_stitch_debug.py \
    --json phase4_<VOD_ID>/qwen_vision_progressive.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_pair(d: Dict[str, Any]) -> str:
    left = d.get("left_candidate_id", "?")
    right = d.get("right_candidate_id", "?")
    gap = d.get("gap_seconds", "?")
    score = d.get("score", "?")
    reasons = ", ".join(d.get("reasons") or [])
    return f"{left:>12} -> {right:<12} gap={str(gap):>3}s score={str(score):>2} reasons=[{reasons}]"


def _primary_reason(d: Dict[str, Any]) -> str:
    reasons = d.get("reasons") or []
    return reasons[0] if reasons else "none"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Stage 1.5 stitch-debug decisions")
    parser.add_argument(
        "--json",
        default="qwen_vision_progressive.json",
        help="Path to pipeline output JSON (default: qwen_vision_progressive.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="How many rows to print in each section (default: 20)",
    )
    args = parser.parse_args()

    path = Path(args.json)
    if not path.exists():
        raise SystemExit(f"ERROR: JSON file not found: {path}")

    payload = _load_json(path)
    decisions: List[Dict[str, Any]] = list(payload.get("stage1_5_stitch_debug") or [])
    stitched: List[Dict[str, Any]] = list(payload.get("stage1_5_stitched") or [])

    if not decisions:
        print("No stage1_5_stitch_debug decisions found in this JSON.")
        print("Tip: run pipeline with updated Stage 1.5 debug wiring.")
        return

    pair_decisions = [d for d in decisions if "left_window" in d and "right_window" in d]
    merged = [d for d in pair_decisions if d.get("merged") is True]
    rejected = [d for d in pair_decisions if d.get("merged") is False]

    print("=" * 80)
    print("Stage 1.5 Stitch Debug Summary")
    print("=" * 80)
    print(f"JSON path: {path}")
    print(f"Pair evaluations: {len(pair_decisions)}")
    print(f"Merged pairs:     {len(merged)}")
    print(f"Rejected pairs:   {len(rejected)}")
    print(f"Stitched arcs:    {len(stitched)}")

    merged_reason_counts = Counter(_primary_reason(d) for d in merged)
    rejected_reason_counts = Counter(_primary_reason(d) for d in rejected)

    print("\nTop merge reason categories:")
    for reason, count in merged_reason_counts.most_common(8):
        print(f"  {reason:<40} {count}")

    print("\nTop reject reason categories:")
    for reason, count in rejected_reason_counts.most_common(8):
        print(f"  {reason:<40} {count}")

    print("\nHighest-scoring merged edges:")
    for d in sorted(merged, key=lambda x: (x.get("score", 0), -x.get("gap_seconds", 0)), reverse=True)[: args.limit]:
        print("  " + _fmt_pair(d))

    print("\nNear-miss rejected edges (score >= 3):")
    near_miss = [d for d in rejected if (d.get("score") or 0) >= 3]
    for d in sorted(near_miss, key=lambda x: (x.get("score", 0), -x.get("gap_seconds", 0)), reverse=True)[: args.limit]:
        print("  " + _fmt_pair(d))

    print("\nStitched arc composition:")
    for arc in stitched[: args.limit]:
        sid = arc.get("stitched_id")
        start = arc.get("start")
        end = arc.get("end")
        src = arc.get("source_candidate_ids") or []
        reasons = ", ".join(arc.get("merge_reasons") or [])
        print(f"  {sid}: {start}-{end}s | sources={len(src)} {src} | reasons=[{reasons}]")


if __name__ == "__main__":
    main()
