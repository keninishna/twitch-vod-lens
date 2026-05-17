from src.synthesis.title_dedup import (
    finalize_stage3_candidates,
    is_near_duplicate_title,
    normalize_title,
)


def test_normalize_title_canonicalizes_case_punctuation_and_spacing():
    assert normalize_title("  The moment she REALIZED chat was right!!! ") == "the moment she realized chat was right"


def test_is_near_duplicate_title_detects_small_wording_variants():
    a = "The moment she realized chat was right"
    b = "The moment she realizes chat was right"
    assert is_near_duplicate_title(a, b) is True


def test_finalize_stage3_candidates_suppresses_duplicate_titles_and_ranks_by_score():
    scored = [
        {
            "candidate_id": "stitched_100_220_1",
            "start": 100,
            "end": 220,
            "final_score": 9.0,
            "raw_score": 9.5,
            "eligible_for_final": True,
            "penalty_trace": [],
            "hard_gates": [],
            "rejection_reasons": [],
            "trim_source": "qwen",
        },
        {
            "candidate_id": "stitched_222_320_2",
            "start": 222,
            "end": 320,
            "final_score": 8.8,
            "raw_score": 9.0,
            "eligible_for_final": True,
            "penalty_trace": [],
            "hard_gates": [],
            "rejection_reasons": [],
            "trim_source": "qwen",
        },
    ]

    stitched = [
        {
            "stitched_id": "stitched_100_220_1",
            "start": 100,
            "end": 220,
            "narrative_type": "chat_reveal",
            "trigger": "chat asks about alert",
            "payoff": "streamer explains inside joke",
            "evidence_lines": ["[120s] ..."],
            "confidence": 0.8,
            "source_candidate_ids": ["cand_100"],
            "source_windows": [[100, 220]],
            "merge_reasons": ["single_candidate"],
        },
        {
            "stitched_id": "stitched_222_320_2",
            "start": 222,
            "end": 320,
            "narrative_type": "chat_reveal",
            "trigger": "another chat question",
            "payoff": "same explanation repeated",
            "evidence_lines": ["[250s] ..."],
            "confidence": 0.78,
            "source_candidate_ids": ["cand_222"],
            "source_windows": [[222, 320]],
            "merge_reasons": ["single_candidate"],
        },
    ]

    analysis_by_candidate = {
        "stitched_100_220_1": {
            "clip_point": "The moment she realizes chat was right",
            "suggested_trim_start": 130,
            "suggested_trim_end": 170,
            "platform_scores": {"twitch": 9, "twitter": 8},
            "platform_recommendations": ["twitch", "twitter"],
        },
        "stitched_222_320_2": {
            "clip_point": "The moment she realized chat was right",
            "suggested_trim_start": 245,
            "suggested_trim_end": 280,
            "platform_scores": {"twitch": 8, "twitter": 8},
            "platform_recommendations": ["twitch", "twitter"],
        },
    }

    final = finalize_stage3_candidates(
        scored_candidates=scored,
        stitched_candidates=stitched,
        analysis_by_candidate=analysis_by_candidate,
        min_score=8.0,
    )

    assert len(final) == 1
    assert final[0]["rank"] == 1
    assert final[0]["clip_id"] == "stitched_100_220_1"
    assert final[0]["clip_point"].lower().startswith("the moment")
