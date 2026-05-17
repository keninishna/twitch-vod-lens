from src.synthesis.audio_normalization import normalize_audio_result


def test_normalize_audio_result_extracts_structured_flags():
    raw = {
        "analysis": """
        - Long silence gap before punchline
        - Streamer laughs hard after reading chat message
        - Confidence: 0.82
        """,
        "extraction_time_seconds": 3.2,
        "inference_time_seconds": 11.5,
    }

    out = normalize_audio_result(raw)

    assert out["dead_air_detected"] is True
    assert out["laughter_detected"] is True
    assert out["confidence"] == 0.82
    assert out["key_events"]


def test_normalize_audio_result_defaults_when_missing_confidence():
    raw = {"analysis": "mostly music, no speech for long stretches"}
    out = normalize_audio_result(raw)

    assert out["music_only"] is True
    assert out["confidence"] == 0.5
