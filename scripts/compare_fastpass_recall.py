#!/usr/bin/env python3
"""Compare baseline clip recall against fast-pass artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _clip_key(item: dict) -> tuple[int, int]:
    return (int(item.get("start", 0)), int(item.get("end", 0)))


def _iter_dict_items(payload: Any, field: str) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    if field == "final_ranking":
        payload = payload.get("final_ranking") or {}
        items = payload.get("final_selected_clips") or []
    else:
        items = payload.get(field) or []
    return [item for item in items if isinstance(item, dict)]


def _collect_keys(payload: dict, field: str) -> set[tuple[int, int]]:
    return {_clip_key(item) for item in _iter_dict_items(payload, field)}


def _coverage(baseline: set[tuple[int, int]], candidate: set[tuple[int, int]]) -> float:
    return round((len(baseline & candidate) / len(baseline)) if baseline else 0.0, 4)


def _value_from_paths(payload: Any, path: tuple[str, ...]) -> Any:
    cur = payload
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _first_number(payload: dict, paths: list[tuple[str, ...]]) -> float | None:
    for path in paths:
        value = _value_from_paths(payload, path)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _count_images(payload: dict) -> int | None:
    candidates = [
        ("stats", "images"),
        ("stats", "image_count"),
        ("stats", "total_images"),
        ("image_count",),
    ]
    value = _first_number(payload, candidates)
    if value is not None:
        return int(value)
    total = 0
    seen = False
    for window in _iter_dict_items(payload, "windows"):
        refs = None
        if isinstance(window.get("source_refs"), dict):
            refs = window["source_refs"].get("frame_paths")
        if refs is None:
            refs = window.get("frame_paths")
        if isinstance(refs, list):
            total += len(refs)
            seen = True
    return total if seen else None


def _count_calls(payload: dict) -> int | None:
    value = _first_number(
        payload,
        [
            ("stats", "total_windows"),
            ("stats", "total_calls"),
            ("stats", "calls"),
            ("call_count",),
        ],
    )
    if value is not None:
        return int(value)
    if isinstance(payload.get("windows"), list):
        return sum(1 for item in payload["windows"] if isinstance(item, dict))
    return None


def _runtime_seconds(payload: dict) -> float | None:
    return _first_number(
        payload,
        [
            ("fast_pass", "runtime_delta_seconds"),
            ("fast_pass", "runtime_seconds"),
            ("stats", "wall_clock_seconds"),
            ("runtime_seconds",),
        ],
    )


def _nearest_explanations(
    missed_clip: tuple[int, int],
    gemma_windows: set[tuple[int, int]],
    triage_candidates: set[tuple[int, int]],
    shortlist_candidates: set[tuple[int, int]],
) -> dict[str, Any]:
    def nearest(candidates: set[tuple[int, int]]) -> dict[str, int] | None:
        if not candidates:
            return None
        s, e = missed_clip
        best = min(
            candidates,
            key=lambda c: (
                abs(c[0] - s) + abs(c[1] - e),
                abs(c[0] - s),
                abs(c[1] - e),
                -c[0],
                -c[1],
            ),
        )
        return {"start": best[0], "end": best[1]}

    return {
        "start": missed_clip[0],
        "end": missed_clip[1],
        "nearest_gemma_window": nearest(gemma_windows),
        "nearest_triage_candidate": nearest(triage_candidates),
        "nearest_shortlist_item": nearest(shortlist_candidates),
    }


def _delta(baseline: int | None, fastpass: int | None) -> int | None:
    if baseline is None or fastpass is None:
        return None
    return fastpass - baseline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--fastpass", required=True)
    ap.add_argument("--gemma", required=True)
    ap.add_argument("--triage", required=True)
    ap.add_argument("--shortlist", required=True)
    args = ap.parse_args()

    baseline = _load(args.baseline)
    fastpass = _load(args.fastpass)
    gemma = _load(args.gemma)
    triage = _load(args.triage)
    shortlist = _load(args.shortlist)

    baseline_final = _collect_keys(baseline, "final_ranking")
    fastpass_final = _collect_keys(fastpass, "final_ranking")
    gemma_windows = {_clip_key(w) for w in _iter_dict_items(gemma, "windows")}
    triage_candidates = {_clip_key(c) for c in _iter_dict_items(triage, "candidates") or _iter_dict_items(triage, "triage_candidates") or _iter_dict_items(triage, "results")}
    shortlist_candidates = {_clip_key(c) for c in _iter_dict_items(shortlist, "vision_shortlist") or _iter_dict_items(shortlist, "shortlist") or _iter_dict_items(shortlist, "results")}

    baseline_runtime = _runtime_seconds(baseline)
    fastpass_runtime = _runtime_seconds(fastpass)
    baseline_calls = _count_calls(baseline)
    fastpass_calls = _count_calls(fastpass)
    baseline_images = _count_images(baseline)
    fastpass_images = _count_images(fastpass)

    missed = sorted(baseline_final - fastpass_final)
    summary = {
        "baseline_final_selected_count": len(baseline_final),
        "gemma_coverage": _coverage(baseline_final, gemma_windows),
        "triage_coverage": _coverage(baseline_final, triage_candidates),
        "shortlist_coverage": _coverage(baseline_final, shortlist_candidates),
        "final_selected_coverage": _coverage(baseline_final, fastpass_final),
        "runtime_delta_seconds": _delta(int(baseline_runtime) if baseline_runtime is not None else None, int(fastpass_runtime) if fastpass_runtime is not None else None),
        "call_delta": _delta(baseline_calls, fastpass_calls),
        "image_delta": _delta(baseline_images, fastpass_images),
        "missed_baseline_clips": [
            _nearest_explanations(clip, gemma_windows, triage_candidates, shortlist_candidates)
            for clip in missed
        ],
    }

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
