import json

from scripts.compare_fastpass_recall import (
    _count_calls,
    _count_images,
    _delta,
    _nearest_explanations,
    _runtime_seconds,
)


def test_runtime_call_and_image_helpers_prefer_explicit_stats_and_fallbacks():
    payload = {
        "fast_pass": {"runtime_seconds": 12.5},
        "stats": {"total_windows": 7},
        "windows": [
            {"source_refs": {"frame_paths": ["a.jpg", "b.jpg"]}},
            {"source_refs": {"frame_paths": ["c.jpg"]}},
        ],
    }

    assert _runtime_seconds(payload) == 12.5
    assert _count_calls(payload) == 7
    assert _count_images(payload) == 3
    assert _delta(10, 4) == -6
    assert _delta(None, 4) is None


def test_runtime_and_image_helpers_fall_back_to_window_counts_and_nested_runtime():
    payload = {
        "fast_pass": {"runtime_delta_seconds": 3.25},
        "windows": [{"frame_paths": ["a.jpg"]}, {"frame_paths": ["b.jpg", "c.jpg"]}],
    }

    assert _runtime_seconds(payload) == 3.25
    assert _count_calls(payload) == 2
    assert _count_images(payload) == 3


def test_nearest_explanations_attach_closest_gemma_triage_and_shortlist_items():
    explanation = _nearest_explanations(
        (100, 160),
        gemma_windows={(96, 156), (300, 360)},
        triage_candidates={(90, 150), (110, 170)},
        shortlist_candidates={(105, 155), (200, 260)},
    )

    assert explanation["start"] == 100
    assert explanation["end"] == 160
    assert explanation["nearest_gemma_window"] == {"start": 96, "end": 156}
    assert explanation["nearest_triage_candidate"] == {"start": 110, "end": 170}
    assert explanation["nearest_shortlist_item"] == {"start": 105, "end": 155}


def test_nearest_explanations_handles_empty_candidates():
    explanation = _nearest_explanations((1, 2), set(), set(), set())

    assert explanation["nearest_gemma_window"] is None
    assert explanation["nearest_triage_candidate"] is None
    assert explanation["nearest_shortlist_item"] is None
