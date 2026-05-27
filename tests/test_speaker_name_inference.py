from __future__ import annotations

from src.preprocessing.speaker_name_inference import (
    build_qwen_name_resolution_prompt,
    extract_name_mentions,
    infer_names_heuristic,
    merge_name_candidates,
)


def test_extract_name_mentions_finds_intro_and_addressed_names():
    names = extract_name_mentions("hey Skitch, what do you think? I'm Nova")
    lowered = {n.lower() for n in names}
    assert "skitch" in lowered
    assert "nova" in lowered


def test_infer_names_heuristic_address_then_response_assigns_responder():
    diarized = [
        {"start": 10.0, "speaker_label": "SPEAKER_00", "text": "hey Skitch"},
        {"start": 12.0, "speaker_label": "SPEAKER_01", "text": "yeah thanks"},
    ]

    out = infer_names_heuristic(diarized)
    assert "SPEAKER_01" in out
    assert out["SPEAKER_01"][0].name == "Skitch"


def test_infer_names_heuristic_self_intro_high_confidence():
    diarized = [{"start": 20.0, "speaker_label": "SPEAKER_01", "text": "I'm Skitch"}]

    out = infer_names_heuristic(diarized)
    assert out["SPEAKER_01"][0].name == "Skitch"
    assert out["SPEAKER_01"][0].confidence >= 0.9


def test_infer_names_heuristic_streamer_chat_greeting_no_response_no_assignment():
    diarized = [{"start": 30.0, "speaker_label": "SPEAKER_00", "text": "hey bob hey alice hey everyone"}]
    chat_messages = [
        {"timestamp": 29.0, "user": "bob", "message": "lol"},
        {"timestamp": 29.5, "user": "alice", "message": "hi"},
    ]

    out = infer_names_heuristic(diarized, chat_messages=chat_messages)
    assert out == {}


def test_infer_names_heuristic_multiple_names_lowers_confidence():
    diarized = [
        {"start": 40.0, "speaker_label": "SPEAKER_00", "text": "hey Bob and hi Alice"},
        {"start": 41.0, "speaker_label": "SPEAKER_02", "text": "sup"},
    ]

    out = infer_names_heuristic(diarized)
    # At least one inferred candidate on responder with lowered confidence.
    assert "SPEAKER_02" in out
    assert out["SPEAKER_02"][0].confidence <= 0.72


def test_name_also_chat_user_with_voice_response_keeps_assignment_with_evidence_note():
    diarized = [
        {"start": 50.0, "speaker_label": "SPEAKER_00", "text": "hey Bob"},
        {"start": 52.0, "speaker_label": "SPEAKER_03", "text": "yeah"},
    ]
    chat_messages = [{"timestamp": 49.5, "user": "bob", "message": "Pog"}]

    out = infer_names_heuristic(diarized, chat_messages=chat_messages)
    assert out["SPEAKER_03"][0].name == "Bob"
    assert "chat usernames" in " ".join(out["SPEAKER_03"][0].evidence)


def test_merge_name_candidates_dedupes_by_name_and_keeps_max_confidence():
    heuristic = {
        "SPEAKER_01": [{"name": "Skitch", "confidence": 0.7, "evidence": ["a"]}],
    }
    qwen = {
        "speaker_name_candidates": [
            {
                "speaker_label": "SPEAKER_01",
                "name": "Skitch",
                "confidence": 0.82,
                "evidence": ["b"],
            }
        ]
    }

    out = merge_name_candidates(heuristic, qwen)
    assert out["SPEAKER_01"][0].confidence == 0.82
    assert set(out["SPEAKER_01"][0].evidence) == {"a", "b"}


def test_build_qwen_name_resolution_prompt_contains_json_contract():
    prompt = build_qwen_name_resolution_prompt(
        diarized_transcript=[{"start": 0.0, "speaker_label": "SPEAKER_00", "text": "hey Skitch"}],
        chat_messages=[{"timestamp": 0.5, "user": "viewer", "message": "yo"}],
    )
    assert '"speaker_name_candidates"' in prompt
    assert "Output ONLY valid JSON" in prompt
