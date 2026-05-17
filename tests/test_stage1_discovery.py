from src.synthesis.stage1_discovery import (
    build_discovery_batch_context,
    map_analysis_to_discovery,
)


def test_map_analysis_to_discovery_outputs_required_fields_only():
    clip = {"start": 998, "end": 1118, "title": "clip title"}
    analysis = {
        "narrative_type": "chat_reveal",
        "narrative_arc": "chat setup -> confusion -> payoff",
        "clip_worthiness": 8,
        "reason": "Good standalone story arc.",
        "clip_point": "The moment she realizes chat was right",
        "platform_recommendations": ["twitter", "twitch"],
        "platform_scores": {"twitter": 8, "twitch": 9},
    }

    out = map_analysis_to_discovery(clip, analysis)

    assert out["candidate_id"] == "cand_998"
    assert out["start"] == 998
    assert out["end"] == 1118
    assert out["narrative_type"] == "chat_reveal"
    assert "trigger" in out and out["trigger"]
    assert "payoff" in out and out["payoff"]
    assert isinstance(out["evidence_lines"], list) and out["evidence_lines"]
    assert 0 <= out["confidence"] <= 1

    # Stage 1 discovery-only: no title/platform finals in payload
    assert "clip_point" not in out
    assert "platform_recommendations" not in out
    assert "platform_scores" not in out


def test_build_discovery_batch_context_never_mentions_titles_or_platform_recs():
    all_results = [
        {
            "start": 998,
            "discovery": {
                "narrative_type": "chat_reveal",
                "trigger": "viewer message appears",
                "payoff": "streamer clarifies inside joke",
                "confidence": 0.82,
                "evidence_lines": ["[1002s] ..."],
            },
            "analysis": {
                "clip_point": "This should never leak into stage1 context",
                "platform_recommendations": ["twitter"],
            },
        }
    ]

    context = build_discovery_batch_context(all_results, total=8, batch_idx=1)

    assert "clip_point" not in context
    assert "title_given" not in context
    assert "platform_recommendations" not in context
    assert "chat_reveal" in context
    assert "confidence=0.82" in context
