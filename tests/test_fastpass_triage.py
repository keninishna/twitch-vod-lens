from pathlib import Path

from src.synthesis.fastpass_triage import (
    build_gemma_annotation_windows,
    build_gemma_audio_extract_command,
    build_triage_chunks,
    compute_vision_budget,
    merge_gemma_annotations_into_chunk,
    normalize_gemma_annotation,
    normalize_triage_candidate,
    select_gemma_frames_for_window,
    select_vision_shortlist,
    summarize_chunk_signals,
    summarize_gemma_signals_for_triage,
)


def test_compute_vision_budget_clamps_to_min_max_and_available_candidates():
    assert compute_vision_budget(200, 0.2, 25, 50) == 40
    assert compute_vision_budget(300, 0.4, 25, 50) == 50
    assert compute_vision_budget(10, 0.2, 25, 50) == 10
    assert compute_vision_budget(0, 0.2, 25, 50) == 0


def test_normalize_triage_candidate_fills_safe_defaults_and_repairs_window():
    out = normalize_triage_candidate(
        {
            "start": "bad",
            "end": None,
            "triage_score": "not-a-number",
            "triage_confidence": None,
            "evidence_lines": "not-a-list",
            "risk_flags": None,
            "selection_reasons": "not-a-list",
        },
        fallback_start=1234,
        fallback_end=1294,
    )

    assert out["candidate_id"] == "triage_1234_1294"
    assert out["start"] == 1234
    assert out["end"] == 1294
    assert out["suggested_trim_start"] == 1234
    assert out["suggested_trim_end"] == 1294
    assert out["narrative_type"] == "other"
    assert out["trigger"]
    assert out["payoff"]
    assert out["evidence_lines"]
    assert out["risk_flags"] == []
    assert out["selection_reasons"] == []
    assert out["triage_score"] == 0.0
    assert out["triage_confidence"] == 0.0
    assert out["vision_need"] == "none"


def test_select_vision_shortlist_includes_special_lanes_and_dedupes_by_start():
    triage_candidates = [
        {
            "candidate_id": "triage_100_top",
            "start": 100,
            "end": 160,
            "suggested_trim_start": 104,
            "suggested_trim_end": 148,
            "triage_score": 9.0,
            "triage_confidence": 0.96,
            "vision_need": "verify_expression",
            "selection_reasons": ["text_top_rank"],
            "evidence_lines": ["[100s] setup -> payoff"],
        },
        {
            "candidate_id": "triage_100_dup",
            "start": 100,
            "end": 158,
            "suggested_trim_start": 105,
            "suggested_trim_end": 147,
            "triage_score": 7.0,
            "triage_confidence": 0.61,
            "vision_need": "verify_expression",
            "selection_reasons": ["chat_spike"],
            "evidence_lines": ["[100s] duplicate lane coverage"],
        },
        {
            "candidate_id": "triage_200_chat",
            "start": 200,
            "end": 260,
            "suggested_trim_start": 204,
            "suggested_trim_end": 248,
            "triage_score": 8.5,
            "triage_confidence": 0.9,
            "vision_need": "verify_expression",
            "selection_reasons": ["chat_spike"],
            "evidence_lines": ["[200s] chat spikes with story payoff"],
        },
        {
            "candidate_id": "triage_300_audio",
            "start": 300,
            "end": 360,
            "suggested_trim_start": 304,
            "suggested_trim_end": 348,
            "triage_score": 8.2,
            "triage_confidence": 0.88,
            "vision_need": "critical",
            "selection_reasons": ["audio_signal"],
            "evidence_lines": ["[300s] loud reaction plus setup"],
        },
        {
            "candidate_id": "triage_400_yolo",
            "start": 400,
            "end": 460,
            "suggested_trim_start": 404,
            "suggested_trim_end": 448,
            "triage_score": 8.1,
            "triage_confidence": 0.87,
            "vision_need": "verify_scene",
            "selection_reasons": ["yolo_visual_novelty"],
            "evidence_lines": ["[400s] visual-only oddity"],
        },
        {
            "candidate_id": "triage_500_sentinel",
            "start": 500,
            "end": 560,
            "suggested_trim_start": 504,
            "suggested_trim_end": 548,
            "triage_score": 7.9,
            "triage_confidence": 0.85,
            "vision_need": "none",
            "selection_reasons": ["sentinel_coverage"],
            "evidence_lines": ["[500s] even coverage anchor"],
        },
    ]

    manifest_clips = [
        {"clip_id": "clip_100", "start": 100, "end": 160},
        {"clip_id": "clip_200", "start": 200, "end": 260},
        {"clip_id": "clip_300", "start": 300, "end": 360},
        {"clip_id": "clip_400", "start": 400, "end": 460},
        {"clip_id": "clip_500", "start": 500, "end": 560},
    ]

    shortlist = select_vision_shortlist(
        triage_candidates,
        manifest_clips,
        vision_budget=5,
        sentinel_ratio=0.2,
    )

    starts = [item["start"] for item in shortlist]
    assert starts == sorted(starts)
    assert len(starts) == len(set(starts))
    assert starts == [100, 200, 300, 400, 500]

    by_start = {item["start"]: item for item in shortlist}
    assert by_start[100]["source_candidate_id"] == "triage_100_top"
    assert by_start[100]["selection_reason"] == "text_top_rank"
    assert by_start[200]["selection_reason"] == "chat_spike"
    assert by_start[300]["selection_reason"] == "audio_signal"
    assert by_start[400]["selection_reason"] == "yolo_visual_novelty"
    assert by_start[500]["selection_reason"] == "sentinel_coverage"

    for item in shortlist:
        assert item["evidence_lines"]
        assert item["suggested_trim_end"] > item["suggested_trim_start"]
        assert 0.0 <= item["triage_confidence"] <= 1.0
        assert item["vision_need"] in {"none", "verify_expression", "verify_scene", "critical"}


def test_select_vision_shortlist_includes_gemma_rescue_lanes_and_annotation_refs():
    triage_candidates = [
        {
            "candidate_id": "triage_100_gemma_audio",
            "start": 100,
            "end": 160,
            "suggested_trim_start": 102,
            "suggested_trim_end": 150,
            "triage_score": 9.5,
            "triage_confidence": 0.98,
            "vision_need": "critical",
            "selection_reasons": ["gemma_audio_alert_or_laughter"],
            "evidence_lines": ["[100s] donation alert reaction"],
            "gemma_annotation_refs": ["gemma_0000100_0000160"],
        },
        {
            "candidate_id": "triage_100_lower_dup",
            "start": 100,
            "end": 165,
            "suggested_trim_start": 103,
            "suggested_trim_end": 151,
            "triage_score": 7.0,
            "triage_confidence": 0.5,
            "vision_need": "critical",
            "selection_reasons": ["gemma_visual_reaction"],
            "evidence_lines": ["[100s] duplicate rescue lane"],
            "gemma_annotation_refs": ["gemma_0000100_0000160"],
        },
        {
            "candidate_id": "triage_220_gemma_game_audio",
            "start": 220,
            "end": 280,
            "suggested_trim_start": 224,
            "suggested_trim_end": 272,
            "triage_score": 8.8,
            "triage_confidence": 0.93,
            "vision_need": "verify_expression",
            "selection_reasons": ["gemma_game_audio_or_non_streamer_voice"],
            "evidence_lines": ["[220s] non-streamer voice over game audio"],
            "gemma_annotation_refs": ["gemma_0000220_0000280"],
        },
        {
            "candidate_id": "triage_320_gemma_visual_reaction",
            "start": 320,
            "end": 380,
            "suggested_trim_start": 324,
            "suggested_trim_end": 372,
            "triage_score": 8.4,
            "triage_confidence": 0.9,
            "vision_need": "verify_expression",
            "selection_reasons": ["gemma_visual_reaction"],
            "evidence_lines": ["[320s] streamer reaction visible"],
            "gemma_annotation_refs": ["gemma_0000320_0000380"],
        },
        {
            "candidate_id": "triage_420_gemma_visual_payoff",
            "start": 420,
            "end": 480,
            "suggested_trim_start": 424,
            "suggested_trim_end": 472,
            "triage_score": 8.2,
            "triage_confidence": 0.89,
            "vision_need": "verify_scene",
            "selection_reasons": ["gemma_visual_payoff"],
            "evidence_lines": ["[420s] visual payoff lands"],
            "gemma_annotation_refs": ["gemma_0000420_0000480"],
        },
    ]

    shortlist = select_vision_shortlist(triage_candidates, [], vision_budget=4, sentinel_ratio=0.0)

    starts = [item["start"] for item in shortlist]
    assert starts == [100, 220, 320, 420]
    assert len(starts) == len(set(starts))
    assert [item["selection_reason"] for item in shortlist] == [
        "text_top_rank",
        "gemma_game_audio_or_non_streamer_voice",
        "gemma_visual_reaction",
        "gemma_visual_payoff",
    ]
    assert "gemma_audio_alert_or_laughter" in shortlist[0]["selection_reasons"]
    assert shortlist[0]["source_candidate_id"] == "triage_100_gemma_audio"
    assert shortlist[0]["gemma_annotation_refs"] == ["gemma_0000100_0000160"]
    assert shortlist[1]["gemma_annotation_refs"] == ["gemma_0000220_0000280"]
    assert shortlist[2]["gemma_annotation_refs"] == ["gemma_0000320_0000380"]
    assert shortlist[3]["gemma_annotation_refs"] == ["gemma_0000420_0000480"]


def test_build_triage_chunks_overlaps_and_includes_expected_lines():
    transcript_segments = [
        {"start": 10, "end": 20, "text": "intro"},
        {"start": 550, "end": 560, "text": "setup"},
        {"start": 610, "end": 620, "text": "payoff"},
        {"start": 1180, "end": 1190, "text": "finale"},
    ]
    chat_messages = [
        {"timestamp": 15, "user": "a", "message": "hello"},
        {"timestamp": 615, "user": "b", "message": "mid"},
        {"timestamp": 1200, "user": "c", "message": "last"},
    ]

    chunks = build_triage_chunks(
        transcript_segments,
        chat_messages,
        vod_start=0,
        vod_end=1300,
        chunk_seconds=600,
        overlap_seconds=60,
    )

    assert [(chunk["chunk_start"], chunk["chunk_end"]) for chunk in chunks] == [
        (0, 600),
        (540, 1140),
        (1080, 1300),
    ]
    assert [line["text"] for line in chunks[0]["transcript_lines"]] == ["intro", "setup"]
    assert [msg["message"] for msg in chunks[0]["chat_messages"]] == ["hello"]
    assert [line["text"] for line in chunks[1]["transcript_lines"]] == ["setup", "payoff"]
    assert [msg["message"] for msg in chunks[1]["chat_messages"]] == ["mid"]
    assert [line["text"] for line in chunks[2]["transcript_lines"]] == ["finale"]
    assert [msg["message"] for msg in chunks[2]["chat_messages"]] == ["last"]
    assert chunks[0]["signal_summary"]["chat_count"] == 1
    assert chunks[1]["signal_summary"]["chat_count"] == 1


def test_build_triage_chunks_handles_invalid_overlap_without_stalling():
    chunks = build_triage_chunks(
        [],
        [],
        vod_start=0,
        vod_end=130,
        chunk_seconds=60,
        overlap_seconds=60,
    )

    assert [(chunk["chunk_start"], chunk["chunk_end"]) for chunk in chunks] == [
        (0, 60),
        (60, 120),
        (120, 130),
    ]


def test_build_triage_chunks_yields_chunks_even_without_text_or_chat():
    chunks = build_triage_chunks(
        [],
        [],
        vod_start=100,
        vod_end=250,
        chunk_seconds=100,
        overlap_seconds=20,
    )

    assert chunks
    assert chunks[0]["chunk_start"] == 100
    assert chunks[-1]["chunk_end"] == 250
    for chunk in chunks:
        assert chunk["transcript_lines"] == []
        assert chunk["chat_messages"] == []
        assert "signal_summary" in chunk


def test_summarize_chunk_signals_reports_density_chat_count_and_manifest_coverage():
    chunk = {
        "chunk_start": 540,
        "chunk_end": 600,
        "transcript_lines": [
            {"start": 550, "end": 560, "text": "setup"},
            {"start": 570, "end": 580, "text": "payoff"},
        ],
        "chat_messages": [
            {"timestamp": 545, "user": "a", "message": "one"},
            {"timestamp": 550, "user": "b", "message": "two"},
            {"timestamp": 555, "user": "c", "message": "three"},
            {"timestamp": 560, "user": "d", "message": "four"},
            {"timestamp": 565, "user": "e", "message": "five"},
            {"timestamp": 570, "user": "f", "message": "six"},
        ],
        "manifest_candidate_starts": [],
    }
    manifest_clips = [
        {"clip_id": "clip_530", "start": 530, "end": 590},
        {"clip_id": "clip_700", "start": 700, "end": 760},
        {"clip_id": "clip_1120", "start": 1120, "end": 1180},
    ]

    summary = summarize_chunk_signals(chunk, manifest_clips)

    assert summary["chunk_start"] == 540
    assert summary["chunk_end"] == 600
    assert summary["transcript_count"] == 2
    assert summary["chat_count"] == 6
    assert summary["manifest_candidate_starts"] == [530]
    assert summary["transcript_density_per_min"] > 0
    assert summary["chat_density_per_min"] > 0
    assert summary["has_chat_spike"] is True



def test_build_gemma_annotation_windows_prefers_manifest_and_chat_spike_windows_deterministically():
    triage_chunks = [
        {
            "chunk_index": 0,
            "chunk_start": 0,
            "chunk_end": 600,
            "signal_summary": {"has_chat_spike": False, "signal_flags": []},
        },
        {
            "chunk_index": 1,
            "chunk_start": 600,
            "chunk_end": 1200,
            "signal_summary": {"has_chat_spike": True, "signal_flags": ["chat_spike"]},
        },
    ]
    manifest_clips = [
        {"clip_id": "clip_a", "start": 120, "end": 210},
        {"clip_id": "clip_b", "start": 900, "end": 975},
        {"clip_id": "clip_c", "start": 1500, "end": 1590},
    ]

    windows = build_gemma_annotation_windows(triage_chunks, manifest_clips, window_seconds=30, stride_seconds=30, max_windows=0)

    assert [w["start"] for w in windows] == sorted(w["start"] for w in windows)
    assert any(w["source"] == "manifest_backed" and w["manifest_clip_id"] == "clip_a" for w in windows)
    assert any(w["source"] == "chat_spike" and w["chunk_index"] == 1 for w in windows)
    assert any(w["source"] == "sentinel_coverage" for w in windows)

    capped = build_gemma_annotation_windows(triage_chunks, manifest_clips, window_seconds=30, stride_seconds=30, max_windows=2)
    assert len(capped) == 2
    assert capped[0]["start"] <= capped[1]["start"]



def test_select_gemma_frames_for_window_uses_nearest_existing_frames(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for idx in (20, 24, 30, 36):
        (frames_dir / f"frame_{idx:06d}.jpg").write_text("x")

    window = {"start": 100, "end": 130}
    frames = select_gemma_frames_for_window(window, str(frames_dir), frames_per_window=3)

    assert len(frames) == 3
    assert all(Path(fp).exists() for fp in frames)
    assert frames[0].endswith("frame_000020.jpg")
    assert frames[1].endswith("frame_000024.jpg")



def test_build_gemma_audio_extract_command_uses_ffmpeg_and_caps_duration():
    cmd = build_gemma_audio_extract_command("/vods/raw.mp4", {"start": 100, "end": 150}, "/tmp/out.wav")
    assert cmd[0] == "ffmpeg"
    assert "-ar" in cmd and cmd[cmd.index("-ar") + 1] == "16000"
    assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "1"
    assert "-c:a" in cmd and cmd[cmd.index("-c:a") + 1] == "pcm_s16le"
    assert cmd[-1] == "/tmp/out.wav"
    assert cmd[cmd.index("-t") + 1] == "30"



def test_normalize_gemma_annotation_clamps_fields_and_preserves_evidence():
    raw = {
        "window_id": "bad",
        "start": 90,
        "end": 160,
        "audio_events": [
            {"timestamp": 80, "type": "TTS_ALERT", "confidence": 2.0, "evidence": "  alert  ", "speaker_guess": "Streamer"},
            {"timestamp": 200, "type": "laugh", "confidence": -1, "evidence": "laugh"},
        ],
        "visual_events": [{"timestamp": 300, "type": "laughing", "confidence": 0.9, "evidence": "big laugh"}],
        "speaker_nuance": {"primary_speaker": "streamer", "streamer_led_likelihood": 1.5, "non_streamer_voice_present": 1, "non_streamer_voice_type": "TTS"},
        "emotion_nuance": {"streamer_affect": "amused", "organic_reaction_likelihood": 0.8, "transactional_alert_likelihood": 0.6, "evidence": "nice"},
        "risk_flags": ["possible_alert_reaction", "possible_alert_reaction", " game_audio_dominant "],
        "clip_relevance_notes": [" note one ", ""],
        "error": None,
    }
    normalized = normalize_gemma_annotation(raw, {"start": 100, "end": 130})

    assert normalized["parse_ok"] is True
    assert normalized["start"] == 100
    assert normalized["end"] == 130
    assert normalized["audio_events"][0]["timestamp"] == 100
    assert normalized["audio_events"][0]["confidence"] == 1.0
    assert normalized["audio_events"][0]["type"] == "tts_alert"
    assert normalized["audio_events"][0]["speaker_guess"] == "streamer"
    assert normalized["audio_events"][1]["confidence"] == 0.0
    assert normalized["visual_events"][0]["timestamp"] == 130
    assert normalized["speaker_nuance"]["streamer_led_likelihood"] == 1.0
    assert normalized["speaker_nuance"]["non_streamer_voice_present"] is True
    assert normalized["emotion_nuance"]["transactional_alert_likelihood"] == 0.6
    assert normalized["risk_flags"] == ["possible_alert_reaction", "game_audio_dominant"]
    assert normalized["clip_relevance_notes"] == ["note one"]



def test_merge_and_summarize_gemma_signals_keep_evidence_as_evidence():
    annotations = [
        {
            "window_id": "gemma_0000100_0000130",
            "audio_events": [{"timestamp": 110, "type": "donation_alert", "confidence": 0.7, "evidence": "alert"}],
            "visual_events": [{"timestamp": 120, "type": "visual_payoff", "confidence": 0.8, "evidence": "payoff"}],
            "speaker_nuance": {"streamer_led_likelihood": 0.9},
            "emotion_nuance": {"transactional_alert_likelihood": 0.4, "organic_reaction_likelihood": 0.6},
            "risk_flags": ["speaker_uncertain"],
            "clip_relevance_notes": ["setup -> payoff"],
            "parse_ok": True,
        },
        {
            "window_id": "gemma_0000130_0000160",
            "audio_events": [{"timestamp": 140, "type": "game_audio", "confidence": 0.6, "evidence": "game"}],
            "visual_events": [{"timestamp": 145, "type": "laughing", "confidence": 0.5, "evidence": "laugh"}],
            "speaker_nuance": {"streamer_led_likelihood": 0.2},
            "emotion_nuance": {"transactional_alert_likelihood": 0.1, "organic_reaction_likelihood": 0.9},
            "risk_flags": ["game_audio_dominant"],
            "clip_relevance_notes": ["reaction"],
            "parse_ok": False,
        },
    ]
    summary = summarize_gemma_signals_for_triage(annotations)
    merged = merge_gemma_annotations_into_chunk({"chunk_start": 100}, annotations)

    assert summary["annotation_count"] == 2
    assert summary["parse_failures"] == 1
    assert summary["has_audio_alert"] is True
    assert summary["has_visual_reaction"] is True
    assert summary["streamer_led_likelihood"] == 0.9
    assert summary["transactional_alert_likelihood"] == 0.4
    assert merged["gemma_annotation_refs"] == ["gemma_0000100_0000130", "gemma_0000130_0000160"]
    assert merged["gemma_signal_summary"]["annotation_count"] == 2
    assert merged["gemma_signal_summary"]["parse_failures"] == 1
    assert merged["gemma_evidence_lines"] == ["setup -> payoff", "reaction"]
