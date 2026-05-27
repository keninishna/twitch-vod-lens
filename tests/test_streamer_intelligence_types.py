from __future__ import annotations

from pydantic import ValidationError
import pytest

from src.intelligence.types import ProfileUpdateProposal, StreamerObservation, StreamerProfile


def test_streamer_profile_accepts_evidence_backed_durable_claims() -> None:
    profile = StreamerProfile.model_validate(
        {
            "streamer_id": "skitch",
            "voice_profiles": [
                {
                    "profile_id": "streamer_skitch_v1",
                    "path": "data/streamer_intelligence/skitch/voice_profiles/streamer_skitch_v1.json",
                    "role": "streamer",
                    "confidence": 0.92,
                    "evidence_refs": ["vod:2776101332@30-90"],
                }
            ],
            "inside_jokes": [
                {
                    "key": "abrasive_dono_alert",
                    "description": "A loud donation alert used as a recurring community bit.",
                    "confidence": 0.86,
                    "evidence_refs": ["vod:2776101332@145-190"],
                }
            ],
        }
    )

    assert profile.streamer_id == "skitch"
    assert profile.voice_profiles[0].confidence == pytest.approx(0.92)
    assert profile.inside_jokes[0].evidence_refs


def test_durable_claim_rejects_invalid_confidence_and_missing_evidence_refs() -> None:
    with pytest.raises(ValidationError):
        StreamerProfile.model_validate(
            {
                "streamer_id": "skitch",
                "inside_jokes": [
                    {
                        "key": "bad_fact",
                        "description": "No evidence refs should fail",
                        "confidence": 1.2,
                        "evidence_refs": [],
                    }
                ],
            }
        )


def test_streamer_observation_requires_contract_fields_and_valid_range() -> None:
    valid = StreamerObservation.model_validate(
        {
            "vod_id": "2776101332",
            "timestamp_start": 120.0,
            "timestamp_end": 147.5,
            "type": "inside_joke",
            "claim": "Donation alert is treated as an inside joke by regulars.",
            "evidence": ["chat:line:2345", "transcript:segment:101"],
            "source": "chat",
            "confidence": 0.84,
            "evidence_refs": ["vod:2776101332@120-147.5"],
        }
    )

    assert valid.vod_id == "2776101332"

    with pytest.raises(ValidationError):
        StreamerObservation.model_validate(
            {
                "vod_id": "2776101332",
                "timestamp_start": 150.0,
                "timestamp_end": 149.0,
                "type": "inside_joke",
                "claim": "Invalid range",
                "evidence": ["transcript:segment:99"],
                "source": "transcript",
                "confidence": 0.7,
                "evidence_refs": ["vod:2776101332@149-150"],
            }
        )


def test_profile_update_proposal_carries_candidate_observations() -> None:
    proposal = ProfileUpdateProposal.model_validate(
        {
            "vod_id": "2776101332",
            "streamer_id": "skitch",
            "promote_mode": "propose",
            "candidate_observations": [
                {
                    "vod_id": "2776101332",
                    "timestamp_start": 300.0,
                    "timestamp_end": 345.0,
                    "type": "content_pattern",
                    "claim": "Strong clips often include setup + payoff within 30-45s.",
                    "evidence": ["clip:300-345", "ranking:score:8.0"],
                    "source": "llm_summary",
                    "confidence": 0.81,
                    "evidence_refs": ["vod:2776101332@300-345"],
                }
            ],
        }
    )

    assert proposal.promote_mode == "propose"
    assert len(proposal.candidate_observations) == 1
