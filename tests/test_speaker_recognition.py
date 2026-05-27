from __future__ import annotations

import pytest

from src.preprocessing.speaker_recognition import (
    aggregate_cluster_embeddings,
    cosine_similarity,
    recognize_speaker_clusters,
)
from src.preprocessing.types import SpeakerTurn


def test_cosine_similarity_deterministic_vectors():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_recognize_speaker_clusters_thresholding(monkeypatch, tmp_path):
    turns = [
        SpeakerTurn(start=0.0, end=3.0, speaker_label="SPEAKER_00"),
        SpeakerTurn(start=5.0, end=8.0, speaker_label="SPEAKER_01"),
    ]

    def fake_aggregate(**_kwargs):
        return {"SPEAKER_00": [1.0, 0.0], "SPEAKER_01": [0.0, 1.0]}

    monkeypatch.setattr("src.preprocessing.speaker_recognition.aggregate_cluster_embeddings", fake_aggregate)

    profiles = [
        {
            "profile_id": "streamer_skitch",
            "role": "streamer",
            "embedding": [1.0, 0.0],
            "thresholds": {"accept_similarity": 0.72, "high_confidence_similarity": 0.80},
        }
    ]

    out = recognize_speaker_clusters(
        audio_path=tmp_path / "audio.wav",
        speaker_turns=turns,
        profiles=profiles,
        output_dir=tmp_path / "tmp",
    )

    assert out["SPEAKER_00"].identity == "streamer"
    assert out["SPEAKER_00"].confidence >= 0.85
    assert out["SPEAKER_01"].identity == "unknown"


def test_aggregate_cluster_embeddings_sampling_limits(monkeypatch, tmp_path):
    turns = [
        SpeakerTurn(start=0.0, end=2.0, speaker_label="SPEAKER_00"),
        SpeakerTurn(start=2.0, end=4.0, speaker_label="SPEAKER_00"),
        SpeakerTurn(start=4.0, end=6.0, speaker_label="SPEAKER_00"),
        SpeakerTurn(start=10.0, end=12.0, speaker_label="SPEAKER_01"),
    ]

    calls = []

    def fake_extract(input_media, output_wav, start=None, end=None, sample_rate=16000):
        output_wav.write_text("wav")
        calls.append((start, end))
        return output_wav

    def fake_embedding(wav_path, device="auto"):
        # Encode label in vector by filename.
        if "SPEAKER_00" in str(wav_path):
            return [1.0, 0.0]
        return [0.0, 1.0]

    monkeypatch.setattr("src.preprocessing.speaker_recognition.extract_wav", fake_extract)
    monkeypatch.setattr("src.preprocessing.speaker_recognition.compute_embedding", fake_embedding)

    out = aggregate_cluster_embeddings(
        audio_path=tmp_path / "in.mp4",
        speaker_turns=turns,
        output_dir=tmp_path / "wavs",
        max_total_seconds_per_speaker=4.0,
        max_turns_per_speaker=2,
    )

    assert set(out.keys()) == {"SPEAKER_00", "SPEAKER_01"}
    # SPEAKER_00 should be capped at 2 turns / 4 seconds.
    speaker_00_calls = [c for c in calls if c[0] in {0.0, 2.0, 4.0}]
    assert len(speaker_00_calls) <= 2
