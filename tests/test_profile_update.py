from __future__ import annotations

from src.intelligence.profile_update import (
    apply_profile_update_auto,
    build_profile_update_proposal,
    dedupe_observations,
    is_sensitive_claim,
    normalize_claim,
    partition_observations_for_merge,
)
from src.intelligence.types import ObservationType, StreamerObservation


def _obs(
    *,
    claim: str,
    confidence: float,
    evidence: list[str],
    obs_type: ObservationType = "inside_joke",
) -> StreamerObservation:
    return StreamerObservation(
        vod_id="2776101332",
        timestamp_start=10.0,
        timestamp_end=20.0,
        type=obs_type,
        claim=claim,
        evidence=evidence,
        source="llm_summary",
        confidence=confidence,
        evidence_refs=["vod:2776101332@10-20"],
    )


def test_normalize_claim_and_sensitivity() -> None:
    raw = "  (Recurring  Donation  Alert  Bit!!!)  "
    assert normalize_claim(raw) == "recurring donation alert bit"

    assert is_sensitive_claim("mentions legal name in chat") is True
    assert is_sensitive_claim("funny recurring donation alert bit") is False


def test_dedupe_observations_keeps_highest_confidence() -> None:
    low = _obs(claim="Recurring alert bit", confidence=0.70, evidence=["a", "b"])
    high = _obs(claim="  recurring alert bit!!! ", confidence=0.90, evidence=["a"])
    other = _obs(claim="Different claim", confidence=0.80, evidence=["a", "b"])

    deduped = dedupe_observations([low, high, other])

    assert len(deduped) == 2
    kept = [o for o in deduped if normalize_claim(o.claim) == "recurring alert bit"][0]
    assert kept.confidence == 0.90


def test_partition_observations_for_merge_thresholds() -> None:
    auto = _obs(claim="good claim", confidence=0.85, evidence=["line1", "line2"])
    queued = _obs(claim="maybe claim", confidence=0.70, evidence=["line1"])
    rejected_low = _obs(claim="weak claim", confidence=0.40, evidence=["line1", "line2"])
    rejected_sensitive = _obs(claim="contains medical details", confidence=0.95, evidence=["line1", "line2"])

    auto_accept, queue, reject = partition_observations_for_merge(
        [auto, queued, rejected_low, rejected_sensitive]
    )

    assert len(auto_accept) == 1
    assert auto_accept[0].claim == "good claim"
    assert len(queue) == 1
    assert queue[0].claim == "maybe claim"
    assert len(reject) == 2


def test_build_profile_update_proposal_generates_candidates() -> None:
    clips = [
        {
            "start": 120,
            "end": 240,
            "suggested_trim_start": 130,
            "suggested_trim_end": 165,
            "score": 8,
            "clip_point": "she explains the chaotic donation alert",
            "inside_joke": "abrasive donation alert is a recurring community bit",
            "intelligence_report": {
                "narrative_arc": "setup with new chatter confusion, then payoff with explanation"
            },
        },
        {
            "start": 300,
            "end": 360,
            "score": 4,
            "clip_point": "no candidate fields here",
            "intelligence_report": {},
        },
    ]

    proposal = build_profile_update_proposal(
        vod_id="2776101332",
        streamer_id="skitch",
        final_selected_clips=clips,
        mode="auto",
    )

    assert proposal.streamer_id == "skitch"
    assert proposal.vod_id == "2776101332"
    assert proposal.promote_mode == "auto"
    assert len(proposal.candidate_observations) == 2

    types = sorted([o.type for o in proposal.candidate_observations])
    assert types == ["content_pattern", "inside_joke"]


def test_build_profile_update_proposal_carries_streamer_identity_metadata() -> None:
    proposal = build_profile_update_proposal(
        vod_id="2776101332",
        streamer_id="asyajade",
        final_selected_clips=[],
        mode="propose",
        streamer_id_source="override",
        metadata_streamer_id="lostgirls27",
        override_streamer_id="asyajade",
        mismatch_warning="streamer_id override mismatch",
    )

    assert proposal.streamer_id_source == "override"
    assert proposal.metadata_streamer_id == "lostgirls27"
    assert proposal.override_streamer_id == "asyajade"
    assert proposal.mismatch_warning == "streamer_id override mismatch"


def test_apply_profile_update_auto_merges_high_confidence_observations() -> None:
    proposal = build_profile_update_proposal(
        vod_id="2776101332",
        streamer_id="skitch",
        final_selected_clips=[
            {
                "start": 120,
                "end": 180,
                "score": 9,
                "inside_joke": "abrasive donation alert is a recurring community bit",
                "intelligence_report": {
                    "narrative_arc": "new chatter asks, streamer explains the recurring alert joke"
                },
            }
        ],
        mode="auto",
    )

    from src.intelligence.types import StreamerProfile

    profile = StreamerProfile(streamer_id="skitch")
    updated, accepted, queued, rejected = apply_profile_update_auto(profile, proposal)

    assert len(accepted) == 2
    assert len(queued) == 0
    assert len(rejected) == 0
    assert len(updated.inside_jokes) == 1
    assert len(updated.content_patterns) == 1
