import pytest
from pydantic import ValidationError

from src.preprocessing.types import SpeakerAttributionResult


def test_speaker_attribution_result_accepts_minimal_valid_payload():
    payload = {
        "vod_id": "2776101332",
        "audio_path": "vods/phase4_2776101332/raw/2776101332.mp4",
        "backend": {
            "diarization": "pyannote/speaker-diarization-community-1",
            "embedding": "speechbrain/spkrec-ecapa-voxceleb",
            "name_inference": "heuristic+qwen",
        },
        "segments": [
            {
                "start": 12.34,
                "end": 16.78,
                "speaker_label": "SPEAKER_00",
                "exclusive": True,
                "recognition": {
                    "identity": "streamer",
                    "profile_id": "streamer_skitch",
                    "confidence": 0.91,
                    "cosine_similarity": 0.84,
                },
                "inferred_name": None,
            }
        ],
        "speaker_clusters": {
            "SPEAKER_00": {
                "total_speech_seconds": 4.44,
                "segment_count": 1,
                "primary_identity": "streamer",
                "primary_identity_confidence": 0.91,
                "candidate_names": [],
            }
        },
        "clip_speaker_stats": {
            "120-240": {
                "primary_speaker_label": "SPEAKER_00",
                "primary_speaker_identity": "streamer",
                "primary_speaker_name": "Skitch",
                "streamer_speaking_seconds": 18.2,
                "streamer_speaking_ratio": 0.31,
                "streamer_speaking_confidence": 0.89,
                "off_streamer_voice_detected": True,
                "dominant_non_streamer_label": "SPEAKER_01",
                "dominant_non_streamer_name": "Guest/unknown",
            }
        },
    }

    result = SpeakerAttributionResult.model_validate(payload)
    assert result.vod_id == "2776101332"
    assert result.segments[0].recognition is not None
    assert result.clip_speaker_stats["120-240"].streamer_speaking_ratio == pytest.approx(0.31)


def test_speaker_turn_rejects_invalid_time_ranges():
    invalid_payload = {
        "vod_id": "2776101332",
        "audio_path": "vods/phase4_2776101332/raw/2776101332.mp4",
        "backend": {},
        "segments": [
            {
                "start": 16.78,
                "end": 16.78,
                "speaker_label": "SPEAKER_00",
            }
        ],
        "speaker_clusters": {},
        "clip_speaker_stats": {},
    }

    with pytest.raises(ValidationError):
        SpeakerAttributionResult.model_validate(invalid_payload)
