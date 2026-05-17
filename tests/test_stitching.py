from src.synthesis.stitching import stitch_discoveries


def test_stitch_discoveries_merges_adjacent_story_arc_with_provenance():
    discoveries = [
        {
            "candidate_id": "cand_100",
            "start": 100,
            "end": 180,
            "narrative_type": "chat_reveal",
            "trigger": "Donation alert fires and chat laughs",
            "payoff": "Streamer explains the inside joke",
            "evidence_lines": ["[118s] wait you don't know this sound?"],
            "confidence": 0.72,
        },
        {
            "candidate_id": "cand_182",
            "start": 182,
            "end": 260,
            "narrative_type": "chat_reveal",
            "trigger": "New chatter asks what the alert means",
            "payoff": "Streamer gives full background of the meme",
            "evidence_lines": ["[204s] okay context: this is from last year"],
            "confidence": 0.81,
        },
    ]

    stitched = stitch_discoveries(discoveries, max_gap_seconds=20)

    assert len(stitched) == 1
    m = stitched[0]
    assert m["stitched_id"] == "stitched_100_260_1"
    assert m["start"] == 100
    assert m["end"] == 260
    assert m["confidence"] == 0.81
    assert m["source_candidate_ids"] == ["cand_100", "cand_182"]
    assert m["source_windows"] == [[100, 180], [182, 260]]
    assert "temporal_gap<=20" in m["merge_reasons"]
    assert "narrative_type_match:chat_reveal" in m["merge_reasons"]


def test_stitch_discoveries_does_not_merge_when_gap_too_large():
    discoveries = [
        {
            "candidate_id": "cand_100",
            "start": 100,
            "end": 180,
            "narrative_type": "story_arc",
            "trigger": "story starts",
            "payoff": "beat one",
            "evidence_lines": ["[120s] ..."],
            "confidence": 0.75,
        },
        {
            "candidate_id": "cand_240",
            "start": 240,
            "end": 300,
            "narrative_type": "story_arc",
            "trigger": "new unrelated segment",
            "payoff": "different payoff",
            "evidence_lines": ["[260s] ..."],
            "confidence": 0.70,
        },
    ]

    stitched = stitch_discoveries(discoveries, max_gap_seconds=20)

    assert len(stitched) == 2
    assert stitched[0]["source_candidate_ids"] == ["cand_100"]
    assert stitched[1]["source_candidate_ids"] == ["cand_240"]


def test_stitch_discoveries_can_merge_on_shared_tokens_even_if_narrative_type_differs():
    discoveries = [
        {
            "candidate_id": "cand_600",
            "start": 600,
            "end": 680,
            "narrative_type": "chat_story",
            "trigger": "Chatter mentions meeting McLaren owner",
            "payoff": "Streamer starts reading details",
            "evidence_lines": ["[640s] meeting F1 McLaren owner"],
            "confidence": 0.66,
        },
        {
            "candidate_id": "cand_682",
            "start": 682,
            "end": 740,
            "narrative_type": "reaction",
            "trigger": "Streamer reads McLaren owner message aloud",
            "payoff": "Chat explodes and she doubles down",
            "evidence_lines": ["[705s] no way, McLaren owner?"],
            "confidence": 0.78,
        },
    ]

    stitched = stitch_discoveries(discoveries, max_gap_seconds=20)

    assert len(stitched) == 1
    reasons = stitched[0]["merge_reasons"]
    assert "temporal_gap<=20" in reasons
    assert any(r.startswith("shared_tokens:") for r in reasons)


def test_stitch_discoveries_can_bridge_across_noisy_middle_segment():
    discoveries = [
        {
            "candidate_id": "cand_100",
            "start": 100,
            "end": 130,
            "narrative_type": "chat_reveal",
            "trigger": "Chat asks about the F1 owner story",
            "payoff": "Streamer starts context setup",
            "evidence_lines": ["[112s] wait, you met who?"],
            "confidence": 0.74,
        },
        {
            "candidate_id": "cand_140",
            "start": 140,
            "end": 150,
            "narrative_type": "filler",
            "trigger": "Quick unrelated thank-you",
            "payoff": "Moves on",
            "evidence_lines": ["[145s] thanks for the sub"],
            "confidence": 0.41,
        },
        {
            "candidate_id": "cand_165",
            "start": 165,
            "end": 195,
            "narrative_type": "chat_reveal",
            "trigger": "Returns to the same F1 owner thread",
            "payoff": "Punchline lands in chat",
            "evidence_lines": ["[178s] the same McLaren owner message again"],
            "confidence": 0.83,
        },
    ]

    debug_decisions = []
    stitched = stitch_discoveries(
        discoveries,
        max_gap_seconds=20,
        min_shared_tokens=2,
        max_bridge_gap_seconds=45,
        debug_decisions=debug_decisions,
    )

    # cand_100 and cand_165 should stitch despite a noisy middle candidate.
    assert len(stitched) == 2
    merged_group = next(c for c in stitched if c["source_candidate_ids"] != ["cand_140"])
    assert merged_group["source_candidate_ids"] == ["cand_100", "cand_165"]
    assert "temporal_bridge_gap<=45" in merged_group["merge_reasons"]

    # Debug trace should capture both failed and successful pair decisions.
    assert any(
        d.get("left_candidate_id") == "cand_100" and d.get("right_candidate_id") == "cand_140" and not d.get("merged")
        for d in debug_decisions
    )
    assert any(
        d.get("left_candidate_id") == "cand_100" and d.get("right_candidate_id") == "cand_165" and d.get("merged")
        for d in debug_decisions
    )
