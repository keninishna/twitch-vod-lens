#!/usr/bin/env python3
"""Compare two qwen_vision_progressive.json outputs for regression checks.

Usage:
  python src/synthesis/evaluate_clip_run.py --baseline old.json --candidate new.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import string
from pathlib import Path


def _norm_title(title: str) -> str:
    tbl = str.maketrans("", "", string.punctuation)
    return " ".join((title or "").lower().translate(tbl).split())


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _extract_selected(payload: dict) -> list[dict]:
    final_ranking = payload.get("final_ranking") or {}
    selected = final_ranking.get("final_selected_clips") or []
    if isinstance(selected, list):
        return selected
    return []


def compute_metrics(payload: dict) -> dict:
    selected = _extract_selected(payload)

    scores = [_safe_float(c.get("score"), 0.0) for c in selected]
    widths = [
        int(_safe_float(c.get("suggested_trim_end"), 0) - _safe_float(c.get("suggested_trim_start"), 0))
        for c in selected
    ]

    titles = [str(c.get("clip_point") or "") for c in selected if str(c.get("clip_point") or "").strip()]
    norm_titles = [_norm_title(t) for t in titles]
    unique_norm_titles = set(norm_titles)
    duplicate_title_count = max(0, len(norm_titles) - len(unique_norm_titles))

    trim_sources = [str(c.get("trim_source") or "") for c in selected]
    rms_count = sum(1 for src in trim_sources if src == "rms_fallback")

    return {
        "selected_count": len(selected),
        "score_mean": round(statistics.mean(scores), 4) if scores else 0.0,
        "score_min": round(min(scores), 4) if scores else 0.0,
        "score_max": round(max(scores), 4) if scores else 0.0,
        "trim_width_mean": round(statistics.mean(widths), 4) if widths else 0.0,
        "trim_width_median": round(statistics.median(widths), 4) if widths else 0.0,
        "trim_width_min": min(widths) if widths else 0,
        "trim_width_max": max(widths) if widths else 0,
        "duplicate_title_count": duplicate_title_count,
        "duplicate_title_rate": round((duplicate_title_count / len(norm_titles)) if norm_titles else 0.0, 4),
        "rms_fallback_count": rms_count,
        "rms_fallback_rate": round((rms_count / len(selected)) if selected else 0.0, 4),
    }


def print_comparison(base: dict, cand: dict) -> None:
    keys = [
        "selected_count",
        "score_mean",
        "score_min",
        "score_max",
        "trim_width_mean",
        "trim_width_median",
        "trim_width_min",
        "trim_width_max",
        "duplicate_title_count",
        "duplicate_title_rate",
        "rms_fallback_count",
        "rms_fallback_rate",
    ]

    print("metric,baseline,candidate,delta")
    for k in keys:
        b = base.get(k, 0)
        c = cand.get(k, 0)
        try:
            d = round(float(c) - float(b), 4)
        except Exception:
            d = "n/a"
        print(f"{k},{b},{c},{d}")



def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Path to baseline qwen_vision_progressive.json")
    parser.add_argument("--candidate", required=True, help="Path to candidate qwen_vision_progressive.json")
    args = parser.parse_args()

    baseline = load_json(Path(args.baseline))
    candidate = load_json(Path(args.candidate))

    base_metrics = compute_metrics(baseline)
    cand_metrics = compute_metrics(candidate)

    print_comparison(base_metrics, cand_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
