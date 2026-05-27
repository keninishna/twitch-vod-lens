from __future__ import annotations

import json

import pytest

from src.preprocessing.speaker_attribution import generate_speaker_attribution
from src.preprocessing.types import SpeakerNameCandidate, SpeakerRecognitionResult, SpeakerTurn


@pytest.fixture
def sample_files(tmp_path):
    vod_media = tmp_path / "vod.mp4"
    vod_media.write_text("fake")

    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 10.0, "end": 12.0, "text": "hey Skitch"},
                    {"start": 13.0, "end": 15.0, "text": "yeah thanks"},
                ]
            }
        )
    )

    chat = tmp_path / "chat.json"
    chat.write_text(json.dumps({"messages": [{"timestamp": 10.5, "user": "viewer", "message": "yo"}]}))

    return vod_media, transcript, chat


def test_generate_speaker_attribution_happy_path(monkeypatch, tmp_path, sample_files):
    vod_media, transcript, chat = sample_files

    monkeypatch.setattr("src.preprocessing.speaker_attribution.extract_wav", lambda *a, **k: a[1])
    monkeypatch.setattr(
        "src.preprocessing.speaker_attribution.diarize_audio",
        lambda *_a, **_k: [
            SpeakerTurn(start=10.0, end=12.0, speaker_label="SPEAKER_00"),
            SpeakerTurn(start=13.0, end=15.0, speaker_label="SPEAKER_01"),
        ],
    )
    monkeypatch.setattr(
        "src.preprocessing.speaker_attribution.load_profiles",
        lambda _p: [{"profile_id": "streamer_skitch", "role": "streamer", "embedding": [1.0, 0.0]}],
    )
    monkeypatch.setattr(
        "src.preprocessing.speaker_attribution.recognize_speaker_clusters",
        lambda **_k: {
            "SPEAKER_00": SpeakerRecognitionResult(
                identity="streamer",
                confidence=0.9,
                cosine_similarity=0.88,
                profile_id="streamer_skitch",
            ),
            "SPEAKER_01": SpeakerRecognitionResult(
                identity="unknown",
                confidence=0.0,
                cosine_similarity=0.2,
                profile_id=None,
            ),
        },
    )
    monkeypatch.setattr(
        "src.preprocessing.speaker_attribution.infer_names_heuristic",
        lambda *_a, **_k: {
            "SPEAKER_01": [
                SpeakerNameCandidate(name="Skitch", confidence=0.72, evidence=["addressed then responded"])
            ]
        },
    )

    out_path = tmp_path / "speaker_attribution_2776101332.json"
    result = generate_speaker_attribution(
        vod_id="2776101332",
        vod_media=vod_media,
        transcript_path=transcript,
        chat_path=chat,
        profiles_dir=tmp_path / "profiles",
        output_path=out_path,
    )

    assert out_path.exists()
    assert result.vod_id == "2776101332"
    assert len(result.segments) == 2
    assert result.segments[0].recognition is not None
    assert result.speaker_clusters["SPEAKER_00"].primary_identity == "streamer"


def test_missing_profiles_still_outputs_unknown_identities(monkeypatch, tmp_path, sample_files):
    vod_media, transcript, chat = sample_files

    monkeypatch.setattr("src.preprocessing.speaker_attribution.extract_wav", lambda *a, **k: a[1])
    monkeypatch.setattr(
        "src.preprocessing.speaker_attribution.diarize_audio",
        lambda *_a, **_k: [SpeakerTurn(start=0.0, end=2.0, speaker_label="SPEAKER_00")],
    )
    monkeypatch.setattr("src.preprocessing.speaker_attribution.load_profiles", lambda _p: [])
    monkeypatch.setattr("src.preprocessing.speaker_attribution.infer_names_heuristic", lambda *_a, **_k: {})

    out_path = tmp_path / "speaker_attribution.json"
    result = generate_speaker_attribution(
        vod_id="v1",
        vod_media=vod_media,
        transcript_path=transcript,
        chat_path=chat,
        profiles_dir=tmp_path / "profiles",
        output_path=out_path,
    )

    assert out_path.exists()
    assert result.speaker_clusters["SPEAKER_00"].primary_identity == "unknown"


def test_nonblocking_mode_writes_minimal_artifact_when_backend_fails(monkeypatch, tmp_path, sample_files):
    vod_media, transcript, chat = sample_files

    monkeypatch.setattr("src.preprocessing.speaker_attribution.extract_wav", lambda *a, **k: a[1])
    monkeypatch.setattr(
        "src.preprocessing.speaker_attribution.diarize_audio",
        lambda *_a, **_k: (_ for _ in ()).throw(ImportError("No module named pyannote.audio")),
    )

    out_path = tmp_path / "speaker_attribution.json"
    result = generate_speaker_attribution(
        vod_id="v2",
        vod_media=vod_media,
        transcript_path=transcript,
        chat_path=chat,
        output_path=out_path,
        require_speaker_id=False,
    )

    assert out_path.exists()
    assert result.backend.get("diarization") == "unavailable"
    assert result.segments == []


def test_strict_mode_raises_actionable_error_when_backend_fails(monkeypatch, sample_files):
    vod_media, transcript, chat = sample_files

    monkeypatch.setattr("src.preprocessing.speaker_attribution.extract_wav", lambda *a, **k: a[1])
    monkeypatch.setattr(
        "src.preprocessing.speaker_attribution.diarize_audio",
        lambda *_a, **_k: (_ for _ in ()).throw(ImportError("No module named pyannote.audio")),
    )

    with pytest.raises(RuntimeError) as exc:
        generate_speaker_attribution(
            vod_id="v3",
            vod_media=vod_media,
            transcript_path=transcript,
            chat_path=chat,
            require_speaker_id=True,
        )

    assert "requirements-speakerid.txt" in str(exc.value)


def test_generate_speaker_attribution_uses_explicit_profiles_without_loading_dir(monkeypatch, sample_files):
    vod_media, transcript, chat = sample_files

    monkeypatch.setattr("src.preprocessing.speaker_attribution.extract_wav", lambda *a, **k: a[1])
    monkeypatch.setattr(
        "src.preprocessing.speaker_attribution.diarize_audio",
        lambda *_a, **_k: [SpeakerTurn(start=10.0, end=12.0, speaker_label="SPEAKER_00")],
    )
    monkeypatch.setattr(
        "src.preprocessing.speaker_attribution.load_profiles",
        lambda _p: (_ for _ in ()).throw(AssertionError("load_profiles should not be called")),
    )
    monkeypatch.setattr(
        "src.preprocessing.speaker_attribution.recognize_speaker_clusters",
        lambda **_k: {
            "SPEAKER_00": SpeakerRecognitionResult(
                identity="streamer",
                confidence=0.9,
                cosine_similarity=0.88,
                profile_id="streamer_skitch",
            )
        },
    )
    monkeypatch.setattr("src.preprocessing.speaker_attribution.infer_names_heuristic", lambda *_a, **_k: {})

    result = generate_speaker_attribution(
        vod_id="v4",
        vod_media=vod_media,
        transcript_path=transcript,
        chat_path=chat,
        profiles=[{"profile_id": "streamer_skitch", "role": "streamer", "embedding": [1.0, 0.0]}],
    )

    assert result.speaker_clusters["SPEAKER_00"].primary_identity == "streamer"
