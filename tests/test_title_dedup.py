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


def test_finalize_stage3_candidates_fallback_selects_top_n_when_no_clip_meets_gate():
    scored = [
        {
            "candidate_id": "stitched_100_220_1",
            "start": 100,
            "end": 220,
            "final_score": 7.5,
            "raw_score": 8.0,
            "eligible_for_final": False,
            "penalty_trace": [],
            "hard_gates": [],
            "rejection_reasons": ["below_score_threshold"],
            "trim_source": "qwen",
        },
        {
            "candidate_id": "stitched_222_320_2",
            "start": 222,
            "end": 320,
            "final_score": 6.8,
            "raw_score": 7.2,
            "eligible_for_final": False,
            "penalty_trace": [],
            "hard_gates": [],
            "rejection_reasons": ["below_score_threshold"],
            "trim_source": "qwen",
        },
        {
            "candidate_id": "stitched_330_440_3",
            "start": 330,
            "end": 440,
            "final_score": 5.9,
            "raw_score": 6.1,
            "eligible_for_final": False,
            "penalty_trace": [],
            "hard_gates": [],
            "rejection_reasons": ["below_score_threshold"],
            "trim_source": "qwen",
        },
        {
            "candidate_id": "stitched_450_560_4",
            "start": 450,
            "end": 560,
            "final_score": 4.1,
            "raw_score": 4.8,
            "eligible_for_final": False,
            "penalty_trace": [],
            "hard_gates": [],
            "rejection_reasons": ["below_score_threshold"],
            "trim_source": "qwen",
        },
    ]

    stitched = [
        {
            "stitched_id": "stitched_100_220_1",
            "start": 100,
            "end": 220,
            "narrative_type": "chat_reveal",
            "trigger": "question",
            "payoff": "answer",
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
            "narrative_type": "chat_banter",
            "trigger": "question",
            "payoff": "answer",
            "evidence_lines": ["[250s] ..."],
            "confidence": 0.78,
            "source_candidate_ids": ["cand_222"],
            "source_windows": [[222, 320]],
            "merge_reasons": ["single_candidate"],
        },
        {
            "stitched_id": "stitched_330_440_3",
            "start": 330,
            "end": 440,
            "narrative_type": "storytelling",
            "trigger": "story start",
            "payoff": "story end",
            "evidence_lines": ["[350s] ..."],
            "confidence": 0.74,
            "source_candidate_ids": ["cand_330"],
            "source_windows": [[330, 440]],
            "merge_reasons": ["single_candidate"],
        },
        {
            "stitched_id": "stitched_450_560_4",
            "start": 450,
            "end": 560,
            "narrative_type": "ambient",
            "trigger": "none",
            "payoff": "none",
            "evidence_lines": ["[470s] ..."],
            "confidence": 0.55,
            "source_candidate_ids": ["cand_450"],
            "source_windows": [[450, 560]],
            "merge_reasons": ["single_candidate"],
        },
    ]

    analysis_by_candidate = {
        "stitched_100_220_1": {
            "clip_point": "She breaks down the streak mechanic",
            "suggested_trim_start": 130,
            "suggested_trim_end": 170,
            "platform_scores": {"twitch": 7},
            "platform_recommendations": ["twitch"],
        },
        "stitched_222_320_2": {
            "clip_point": "Chat pushes her into a quick explainer",
            "suggested_trim_start": 245,
            "suggested_trim_end": 280,
            "platform_scores": {"twitch": 7},
            "platform_recommendations": ["twitch"],
        },
        "stitched_330_440_3": {
            "clip_point": "Gym story lands with a clean payoff",
            "suggested_trim_start": 350,
            "suggested_trim_end": 390,
            "platform_scores": {"twitch": 6},
            "platform_recommendations": ["twitch"],
        },
        "stitched_450_560_4": {
            "clip_point": "Low-energy ambient segment",
            "suggested_trim_start": 470,
            "suggested_trim_end": 510,
            "platform_scores": {"twitch": 5},
            "platform_recommendations": [],
        },
    }

    final = finalize_stage3_candidates(
        scored_candidates=scored,
        stitched_candidates=stitched,
        analysis_by_candidate=analysis_by_candidate,
        min_score=8.0,
        fallback_top_n_when_empty=3,
    )

    assert len(final) == 3
    assert [c["clip_id"] for c in final] == [
        "stitched_100_220_1",
        "stitched_222_320_2",
        "stitched_330_440_3",
    ]


def test_finalize_stage3_candidates_chat_title_removes_redundant_chat_message_phrase():
    scored = [
        {
            "candidate_id": "stitched_638_1118_1",
            "start": 638,
            "end": 1118,
            "final_score": 7.0,
            "raw_score": 7.0,
            "eligible_for_final": False,
            "penalty_trace": [],
            "hard_gates": [],
            "rejection_reasons": ["below_score_threshold"],
            "trim_source": "qwen",
        }
    ]

    stitched = [
        {
            "stitched_id": "stitched_638_1118_1",
            "start": 638,
            "end": 1118,
            "narrative_type": "chat_reveal",
            "trigger": "chat message from 'lost mine sadly' about a streak",
            "payoff": "streamer explains the streak context",
            "evidence_lines": ["[700s] ..."],
            "confidence": 0.75,
            "source_candidate_ids": ["cand_638"],
            "source_windows": [[638, 1118]],
            "merge_reasons": ["single_candidate"],
        }
    ]

    analysis_by_candidate = {
        "stitched_638_1118_1": {
            "clip_point": "Streamer reads a chat message about chat message from 'lost mine sadly' about a streak",
            "suggested_trim_start": 638,
            "suggested_trim_end": 758,
            "platform_scores": {"twitch": 6},
            "platform_recommendations": ["twitch"],
        }
    }

    final = finalize_stage3_candidates(
        scored_candidates=scored,
        stitched_candidates=stitched,
        analysis_by_candidate=analysis_by_candidate,
        min_score=8.0,
        fallback_top_n_when_empty=1,
    )

    assert len(final) == 1
    assert final[0]["clip_point"] == "What happens when chat drops a message about a streak?"


def test_finalize_stage3_candidates_preserves_speaker_attribution_fields_in_output():
    scored = [
        {
            "candidate_id": "stitched_100_220_1",
            "start": 100,
            "end": 220,
            "final_score": 8.1,
            "raw_score": 8.3,
            "eligible_for_final": True,
            "penalty_trace": [],
            "hard_gates": [],
            "rejection_reasons": [],
            "trim_source": "qwen",
        }
    ]

    stitched = [
        {
            "stitched_id": "stitched_100_220_1",
            "start": 100,
            "end": 220,
            "narrative_type": "chat_banter",
            "trigger": "guest asks a sharp question",
            "payoff": "streamer answers and laughs",
            "evidence_lines": ["[130s] guest question", "[148s] streamer response"],
            "confidence": 0.86,
            "source_candidate_ids": ["cand_100"],
            "source_windows": [[100, 220]],
            "merge_reasons": ["single_candidate"],
        }
    ]

    analysis_by_candidate = {
        "stitched_100_220_1": {
            "clip_point": "Streamer answers a guest's wild question",
            "suggested_trim_start": 128,
            "suggested_trim_end": 168,
            "platform_scores": {"twitch": 8},
            "platform_recommendations": ["twitch"],
            "speaker_attribution": {
                "primary_speaker_identity": "guest",
                "primary_speaker_name": "SkitchFriend",
                "streamer_speaking_ratio": 0.22,
                "streamer_speaking_confidence": 0.77,
                "off_streamer_voice_detected": True,
                "evidence": ["SPEAKER_01 dominates first half"],
            },
        }
    }

    final = finalize_stage3_candidates(
        scored_candidates=scored,
        stitched_candidates=stitched,
        analysis_by_candidate=analysis_by_candidate,
        min_score=7.0,
    )

    assert len(final) == 1
    sa = final[0]["speaker_attribution"]
    assert sa["primary_speaker_identity"] == "guest"
    assert sa["primary_speaker_name"] == "SkitchFriend"
    assert sa["off_streamer_voice_detected"] is True
