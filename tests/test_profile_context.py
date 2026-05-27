from __future__ import annotations

from src.intelligence.profile_context import render_streamer_profile_context
from src.intelligence.types import StreamerProfile


def test_render_streamer_profile_context_filters_low_confidence_and_missing_evidence() -> None:
    profile = StreamerProfile.model_validate(
        {
            "streamer_id": "skitch",
            "voice_profiles": [
                {
                    "profile_id": "streamer_ok",
                    "path": "data/streamer_intelligence/skitch/voice_profiles/streamer_ok.json",
                    "role": "streamer",
                    "confidence": 0.90,
                    "evidence_refs": ["vod:1@10-20"],
                },
                {
                    "profile_id": "streamer_low",
                    "path": "data/streamer_intelligence/skitch/voice_profiles/streamer_low.json",
                    "role": "streamer",
                    "confidence": 0.40,
                    "evidence_refs": ["vod:1@20-30"],
                },
            ],
            "inside_jokes": [
                {
                    "key": "good_bit",
                    "description": "Recurring joke",
                    "confidence": 0.75,
                    "evidence_refs": ["vod:1@100-120"],
                },
                {
                    "key": "bad_bit",
                    "description": "No refs",
                    "confidence": 0.80,
                    "evidence_refs": ["vod:1@130-140"],
                },
            ],
        }
    )

    rendered = render_streamer_profile_context(profile, max_chars=2000)

    assert "streamer_ok" in rendered
    assert "streamer_low" not in rendered
    assert "good_bit" in rendered
    assert "conflict rule: current VOD evidence overrides profile context" in rendered


def test_render_streamer_profile_context_respects_max_chars() -> None:
    profile = StreamerProfile.model_validate(
        {
            "streamer_id": "skitch",
            "personality_traits": [
                {
                    "trait": "long",
                    "description": "x" * 1000,
                    "confidence": 0.9,
                    "evidence_refs": ["vod:1@1-2"],
                }
            ],
        }
    )

    rendered = render_streamer_profile_context(profile, max_chars=180)

    assert len(rendered) <= 180
    assert rendered.endswith("…")
