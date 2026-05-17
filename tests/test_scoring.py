from src.synthesis.scoring import normalize_clip_analysis


def _base_candidate(start=100, end=220):
    return {
        "stitched_id": f"stitched_{start}_{end}_1",
        "start": start,
        "end": end,
        "narrative_type": "chat_reveal",
        "trigger": "question from chat",
        "payoff": "streamer explains inside joke",
        "evidence_lines": ["[120s] you don't know this sound?"],
        "confidence": 0.8,
        "source_candidate_ids": [f"cand_{start}"],
        "source_windows": [[start, end]],
        "merge_reasons": ["single_candidate"],
    }


def _base_analysis(score=9):
    return {
        "clip_worthiness": score,
        "has_narrative_payoff": True,
        "requires_context": False,
        "transactional_reaction": False,
        "suggested_trim_start": 120,
        "suggested_trim_end": 160,
        "platform_scores": {
            "tiktok": 7,
            "shorts": 7,
            "twitter": 8,
            "twitch": 8,
            "reels": 7,
        },
        "platform_recommendations": ["twitter", "twitch"],
        "reason": "clear setup and payoff",
    }


def _base_context():
    return {
        "clip_start": 100,
        "clip_end": 220,
        "transcript_lines": [],
        "chat_messages": [],
        "chat_read_flags": [],
        "dead_air_gaps": [],
        "total_dead_air_seconds": 0.0,
        "dead_air_ratio": 0.0,
        "objects_detected": [],
    }


def test_normalize_clip_analysis_happy_path_eligible_for_final():
    scored = normalize_clip_analysis(_base_candidate(), _base_analysis(score=9), _base_context())

    assert scored["candidate_id"] == "stitched_100_220_1"
    assert scored["raw_score"] == 9.0
    assert scored["final_score"] == 9.0
    assert scored["eligible_for_final"] is True
    assert scored["rejection_reasons"] == []
    assert scored["trim_source"] == "qwen"


def test_dead_air_gap_over_10_applies_minus5_and_cap5():
    context = _base_context()
    context["dead_air_gaps"] = [{"start": 130, "end": 145, "duration": 15}]

    scored = normalize_clip_analysis(_base_candidate(), _base_analysis(score=9), context)

    assert scored["raw_score"] == 9.0
    assert scored["final_score"] <= 5.0
    assert any(p["code"] == "dead_air_single_gap_gt_10" and p["points"] == 5 for p in scored["penalty_trace"])


def test_dead_air_inside_trim_forces_reject():
    context = _base_context()
    context["dead_air_gaps"] = [{"start": 130, "end": 136, "duration": 6}]
    analysis = _base_analysis(score=9)
    analysis["suggested_trim_start"] = 120
    analysis["suggested_trim_end"] = 150

    scored = normalize_clip_analysis(_base_candidate(), analysis, context)

    assert scored["eligible_for_final"] is False
    assert "dead_air_inside_trim" in scored["rejection_reasons"]
    assert scored["final_score"] <= 3.0


def test_platform_recommendations_must_be_subset_of_scores_ge_6():
    analysis = _base_analysis(score=9)
    analysis["platform_scores"]["twitter"] = 5
    analysis["platform_recommendations"] = ["twitter", "twitch"]

    scored = normalize_clip_analysis(_base_candidate(), analysis, _base_context())

    assert "platform_recommendation_invalid" in scored["rejection_reasons"]
    assert scored["eligible_for_final"] is False


def test_full_unresolved_120_window_is_capped_and_rejected_without_justification():
    candidate = _base_candidate(start=500, end=620)
    analysis = _base_analysis(score=9)
    analysis["suggested_trim_start"] = 500
    analysis["suggested_trim_end"] = 620
    analysis["trim_justification"] = ""

    scored = normalize_clip_analysis(candidate, analysis, _base_context())

    assert scored["final_score"] <= 5.0
    assert "full_window_unresolved" in scored["rejection_reasons"]
    assert scored["eligible_for_final"] is False


def test_audio_context_adds_non_selector_penalty_signal():
    audio = {
        "dead_air_detected": True,
        "music_only": False,
        "confidence": 0.8,
    }

    scored = normalize_clip_analysis(
        _base_candidate(),
        _base_analysis(score=9),
        _base_context(),
        audio=audio,
    )

    assert any(p["code"] == "audio_dead_air_signal" for p in scored["penalty_trace"])
    assert scored["final_score"] == 8.0
