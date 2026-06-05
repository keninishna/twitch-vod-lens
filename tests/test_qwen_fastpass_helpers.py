from pathlib import Path

from src.synthesis import qwen_clip_analyzer_progressive as mod


def test_analysis_prompt_accepts_fast_pass_evidence_context():
    evidence = mod._build_fast_pass_evidence_block(
        {
            "trigger": "donation alert fires",
            "payoff": "streamer explains the inside joke",
            "evidence_lines": ["[110s] wait you don't know this sound?"],
            "risk_flags": ["possibly_transactional"],
            "gemma_annotation_refs": ["gemma_0000110_0000140"],
        }
    )

    prompt = mod.ANALYSIS_PROMPT.format(
        clip_title="test clip",
        start=100,
        end=160,
        streamer_profile_context="PROFILE",
        phase1_title_research_summary="TITLE SUMMARY",
        phase1_title_examples="TITLE EXAMPLES",
        transcript="transcript",
        chat_messages="chat",
        yolo_objects="none",
        fast_pass_evidence_context=evidence,
        batch_context="batch",
        platform_guide="guide",
    )

    assert "trigger: donation alert fires" in prompt
    assert "gemma_annotation_refs" in prompt
    assert "possibly_transactional" in prompt


def test_sample_clip_frames_fast_pass_uses_trim_aware_nearest_frames(tmp_path, monkeypatch):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for frame_name in ["frame_000021.jpg", "frame_000023.jpg", "frame_000025.jpg"]:
        (frames_dir / frame_name).write_bytes(b"jpeg")

    monkeypatch.setattr(mod, "FRAMES_DIR", frames_dir)

    sampled = mod.sample_clip_frames(
        {
            "start": 100,
            "end": 140,
            "suggested_trim_start": 105,
            "suggested_trim_end": 125,
        },
        count=3,
        fast_pass=True,
    )

    assert [ts for _, ts in sampled] == [105, 115, 125]


def test_run_fast_pass_text_triage_uses_text_only_qwen_payload(monkeypatch):
    payloads = []

    def fake_qwen_call(payload, timeout=180):
        payloads.append(payload)
        return {
            "candidates": [
                {
                    "candidate_id": "triage_100",
                    "start": 100,
                    "end": 150,
                    "suggested_trim_start": 104,
                    "suggested_trim_end": 145,
                    "narrative_type": "storytelling",
                    "trigger": "new chatter asks about the alert",
                    "payoff": "streamer explains the meme",
                    "evidence_lines": ["transcript 104s: what is that sound?"],
                    "risk_flags": ["visual_context_required"],
                    "triage_score": 8.2,
                    "triage_confidence": 0.76,
                    "vision_need": "critical",
                    "selection_reasons": ["gemma_visual_reaction"],
                    "gemma_annotation_refs": ["gemma_0000100_0000130"],
                }
            ]
        }

    monkeypatch.setattr(mod, "qwen_call", fake_qwen_call)

    triage_chunks = [
        {
            "chunk_start": 100,
            "chunk_end": 160,
            "transcript_lines": [{"start": 104, "end": 106, "text": "what is that sound"}],
            "chat_messages": [{"timestamp": 105, "user": "viewer", "message": "what does that alert mean"}],
        }
    ]
    gemma_artifact = {
        "windows": [
            {
                "window_id": "gemma_0000100_0000130",
                "start": 100,
                "end": 130,
                "audio_events": [{"timestamp": 108, "type": "donation_alert", "confidence": 0.9, "evidence": "alert sound"}],
                "visual_events": [{"timestamp": 115, "type": "visual_payoff", "confidence": 0.8, "evidence": "streamer laughs"}],
                "risk_flags": ["visual_context_required"],
                "speaker_nuance": {"streamer_led_likelihood": 0.8},
                "emotion_nuance": {"transactional_alert_likelihood": 0.3, "organic_reaction_likelihood": 0.7},
                "clip_relevance_notes": ["laugh then explanation arc"],
                "parse_ok": True,
                "error": None,
            }
        ]
    }

    candidates, stats = mod._run_fast_pass_text_triage(
        triage_chunks=triage_chunks,
        gemma_artifact=gemma_artifact,
        mode="gemma-enriched",
    )

    assert stats["qwen_text_calls"] == 1
    assert isinstance(payloads[0]["messages"][0]["content"], str)
    assert "GEMMA_EVIDENCE" in payloads[0]["messages"][0]["content"]
    assert candidates[0]["candidate_id"] == "triage_100"
    assert candidates[0]["start"] == 100
    assert candidates[0]["end"] == 150
    assert candidates[0]["vision_need"] == "critical"
