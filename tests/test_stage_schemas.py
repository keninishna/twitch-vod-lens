import pytest

from src.synthesis.schemas.clip_intelligence_stages import (
    ClipContext,
    DiscoveryCandidate,
    FinalSelectedClip,
    ScoredCandidate,
    StitchedCandidate,
    StageContractValidationError,
    validate_stage_payload,
)


def test_discovery_candidate_rejects_title_fields_in_stage1():
    payload = {
        "candidate_id": "cand_120",
        "start": 120,
        "end": 180,
        "narrative_type": "inside_joke_explainer",
        "trigger": "Donation alert fires",
        "payoff": "Streamer explains the inside joke to new chatter",
        "evidence_lines": [
            "[126.2s] streamer: wait you don't know that sound?",
            "[133.8s] streamer: chat this is an old meme from last year",
        ],
        "confidence": 0.86,
        "clip_point": "should not be allowed in discovery",
    }

    with pytest.raises(StageContractValidationError):
        validate_stage_payload("discovery", payload)


def test_stitched_candidate_requires_provenance_fields():
    payload = {
        "stitched_id": "stitched_1",
        "start": 600,
        "end": 715,
        "narrative_type": "story_arc",
        "trigger": "chatter mentions meeting McLaren owner",
        "payoff": "streamer reads and reacts with context",
        "evidence_lines": ["[702.1s] streamer reads chat message aloud"],
        "confidence": 0.79,
        "source_candidate_ids": ["cand_600", "cand_660"],
        "source_windows": [[580, 700], [640, 760]],
        "merge_reasons": ["temporal_gap<=20", "shared_entity: mclaren"],
    }

    model = validate_stage_payload("stitched", payload)
    assert isinstance(model, StitchedCandidate)
    assert model.source_candidate_ids == ["cand_600", "cand_660"]


def test_scored_candidate_enforces_required_scoring_fields():
    payload = {
        "candidate_id": "cand_998",
        "start": 998,
        "end": 1118,
        "final_score": 8,
        "raw_score": 9,
        "eligible_for_final": True,
        "penalty_trace": [{"code": "duration_ok", "points": 0}],
        "hard_gates": [],
        "rejection_reasons": [],
        "trim_source": "qwen",
    }

    model = validate_stage_payload("scored", payload)
    assert isinstance(model, ScoredCandidate)
    assert model.final_score == 8
    assert model.eligible_for_final is True


def test_clip_context_rejects_invalid_dead_air_ratio():
    payload = {
        "clip_start": 200,
        "clip_end": 320,
        "transcript_lines": [{"start": 205.0, "end": 208.1, "text": "hello chat"}],
        "chat_messages": [{"timestamp": 210.0, "user": "viewer1", "message": "LMAO"}],
        "chat_read_flags": [],
        "dead_air_gaps": [{"start": 230.0, "end": 248.0, "duration": 18.0}],
        "total_dead_air_seconds": 18.0,
        "dead_air_ratio": 1.2,
        "objects_detected": ["person"],
    }

    with pytest.raises(StageContractValidationError):
        validate_stage_payload("context", payload)


def test_final_selected_clip_requires_intelligence_report_shape():
    payload = {
        "rank": 1,
        "clip_id": "cand_998",
        "start": 998,
        "end": 1118,
        "suggested_trim_start": 1010,
        "suggested_trim_end": 1050,
        "trim_source": "qwen",
        "score": 8,
        "raw_score": 9,
        "normalized_score": 8,
        "clip_point": "The moment she realizes chat was right",
        "narrative_type": "chat_reveal",
        "platform_scores": {
            "tiktok": 7,
            "shorts": 7,
            "twitter": 8,
            "twitch": 9,
            "reels": 7,
        },
        "platform_recommendations": ["twitter", "twitch"],
        "intelligence_report": {
            "why_selected": "Clear setup and payoff with self-contained context",
            "narrative_arc": "chat setup -> streamer confusion -> realization",
            "evidence": ["line 1", "line 2"],
            "trim_rationale": "Starts at question, ends right after punchline",
            "duration_fit": "40s optimal",
            "platform_fit": "Strong for X and Twitch community context",
            "risks": ["inside joke may need subtitle"],
            "streamer_feedback": "High personality moment",
        },
    }

    model = validate_stage_payload("final_selected", payload)
    assert isinstance(model, FinalSelectedClip)
    assert model.intelligence_report.why_selected.startswith("Clear setup")


def test_unknown_stage_raises_contract_error():
    with pytest.raises(StageContractValidationError):
        validate_stage_payload("unknown-stage", {"x": 1})


def test_module_exports_models_for_direct_instantiation():
    model = DiscoveryCandidate(
        candidate_id="cand_42",
        start=42,
        end=102,
        narrative_type="story",
        trigger="question",
        payoff="answer",
        evidence_lines=["[45.0] what happened?", "[62.0] here's what happened"],
        confidence=0.7,
    )
    assert model.end > model.start

    ctx = ClipContext(
        clip_start=42,
        clip_end=162,
        transcript_lines=[],
        chat_messages=[],
        chat_read_flags=[],
        dead_air_gaps=[],
        total_dead_air_seconds=0.0,
        dead_air_ratio=0.0,
        objects_detected=[],
    )
    assert ctx.dead_air_ratio == 0.0
