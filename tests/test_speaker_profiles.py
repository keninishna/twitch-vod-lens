from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.preprocessing.speaker_enroll import enroll_profile, parse_segments
from src.preprocessing.speaker_profiles import (
    average_embeddings,
    load_profiles,
    load_profiles_from_paths,
    save_profile,
)


def test_average_embeddings_normalizes_and_averages():
    out = average_embeddings([[1.0, 0.0], [0.0, 1.0]])
    # normalized mean of unit basis vectors => [sqrt(2)/2, sqrt(2)/2]
    assert out[0] == pytest.approx(0.7071, abs=1e-3)
    assert out[1] == pytest.approx(0.7071, abs=1e-3)


def test_save_and_load_profiles_roundtrip(tmp_path):
    profile = {
        "profile_id": "streamer_skitch",
        "display_name": "Skitch",
        "embedding": [0.1, 0.2],
    }
    saved = save_profile(profile, tmp_path)
    assert saved.exists()

    loaded = load_profiles(tmp_path)
    assert len(loaded) == 1
    assert loaded[0]["profile_id"] == "streamer_skitch"


def test_load_profiles_from_paths_dedupes_and_skips_invalid(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    bad = tmp_path / "bad.json"
    missing = tmp_path / "missing.json"

    a.write_text(json.dumps({"profile_id": "p1", "embedding": [1.0, 0.0]}), encoding="utf-8")
    b.write_text(json.dumps({"profile_id": "p2", "embedding": [0.0, 1.0]}), encoding="utf-8")
    bad.write_text("not-json", encoding="utf-8")

    loaded = load_profiles_from_paths([a, b, a, bad, missing])

    assert len(loaded) == 2
    assert {x["profile_id"] for x in loaded} == {"p1", "p2"}


def test_parse_segments_parses_comma_ranges():
    assert parse_segments("30-90,300-360") == [(30.0, 90.0), (300.0, 360.0)]


def test_enroll_profile_uses_extraction_and_embedding(monkeypatch, tmp_path):
    calls = {"extract": [], "embed": []}

    def fake_extract_wav(input_media, output_wav, start=None, end=None, sample_rate=16000):
        calls["extract"].append((start, end, sample_rate))
        output_wav.write_text("fakewav")
        return output_wav

    def fake_compute_embedding(wav_path, device="auto"):
        calls["embed"].append((str(wav_path), device))
        # Different vectors per segment index to test averaging path.
        return [1.0, 0.0] if "segment_000" in str(wav_path) else [0.0, 1.0]

    monkeypatch.setattr("src.preprocessing.speaker_enroll.extract_wav", fake_extract_wav)
    monkeypatch.setattr("src.preprocessing.speaker_enroll.compute_embedding", fake_compute_embedding)

    audio = tmp_path / "input.mp4"
    audio.write_text("dummy")

    out = enroll_profile(
        profile_id="streamer_skitch",
        display_name="Skitch",
        role="streamer",
        audio=audio,
        segments=[(10.0, 20.0), (30.0, 50.0)],
        output_dir=tmp_path / "profiles",
        device="cpu",
    )

    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["profile_id"] == "streamer_skitch"
    assert payload["embedding_dim"] == 2
    assert len(payload["created_from"]) == 2
    assert calls["extract"] == [(10.0, 20.0, 16000), (30.0, 50.0, 16000)]
    assert len(calls["embed"]) == 2
