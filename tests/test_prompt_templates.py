from src.synthesis.qwen_clip_analyzer_progressive import (
    ANALYSIS_PROMPT,
    FINAL_SYNTHESIS_PROMPT,
    PROVISIONAL_SYNTHESIS_PROMPT,
)


def test_analysis_prompt_renders_single_brace_json_schema():
    rendered = ANALYSIS_PROMPT.format(
        clip_title="test clip",
        start=0,
        end=120,
        transcript="hello",
        chat_messages="none",
        yolo_objects="person",
        fast_pass_evidence_context="",
        batch_context="none",
        streamer_profile_context="profile context",
        game_knowledge_context="game context",
        phase1_title_research_summary="summary",
        phase1_title_examples="examples",
        platform_guide="guide",
    )

    assert "return valid JSON only:\n{" in rendered
    assert "{{" not in rendered
    assert "}}" not in rendered
    assert '"failure_modes": [' in rendered
    assert '"speaker_framing_assessment": {' in rendered
    assert "Do NOT assume deterministic speaker penalties/hard gates exist in Python" in rendered


def test_provisional_prompt_renders_single_brace_json_schema():
    rendered = PROVISIONAL_SYNTHESIS_PROMPT.format(
        total_clips=3,
        vod_title="VOD",
        streamer="streamer",
        complete_log="log",
        audio_context="audio",
        game_run_context="game run context",
        vod_id="123",
    )

    assert "Return valid JSON only:\n{" in rendered
    assert "{{" not in rendered
    assert "}}" not in rendered


def test_final_prompt_renders_single_brace_json_schema():
    rendered = FINAL_SYNTHESIS_PROMPT.format(
        vod_title="VOD",
        streamer="streamer",
        complete_log="log",
        audio_context="audio",
        game_run_context="game run context",
        total_clips=3,
        vod_id="123",
        frames_requested_count=0,
        platform_guide="guide",
    )

    assert "Return valid JSON only:\n{" in rendered
    assert "{{" not in rendered
    assert "}}" not in rendered
    assert "SPEAKER-FRAMING RULE" in rendered
    assert "do not assume deterministic speaker-specific penalties/gates" in rendered
