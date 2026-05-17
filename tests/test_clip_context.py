from src.synthesis.clip_context import build_clip_context, render_prompt_context


def test_build_clip_context_detects_dead_air_and_ratio():
    transcript_segments = [
        {"start": 90.0, "end": 95.0, "text": "hey chat"},
        {"start": 112.0, "end": 116.0, "text": "welcome back"},
        {"start": 118.0, "end": 122.0, "text": "let's go"},
    ]
    chat_messages = []

    ctx = build_clip_context(
        seconds=110,
        transcript_segments=transcript_segments,
        chat_messages=chat_messages,
        window=20,
    )

    assert ctx["dead_air_gaps"], "Expected a dead-air gap between 95 and 112"
    assert ctx["total_dead_air_seconds"] == 17.0
    assert round(ctx["dead_air_ratio"], 3) == round(17.0 / 40.0, 3)


def test_build_clip_context_detects_chat_read_flags():
    transcript_segments = [
        {
            "start": 740.0,
            "end": 745.0,
            "text": "someone in chat said they met the F1 McLaren owner",
        },
    ]
    chat_messages = [
        {
            "timestamp": 742.0,
            "user": "Buchaanan",
            "message": "I met the F1 McLaren owner at an event last night and it was insane",
        },
    ]

    ctx = build_clip_context(
        seconds=742,
        transcript_segments=transcript_segments,
        chat_messages=chat_messages,
        window=30,
    )

    assert len(ctx["chat_read_flags"]) == 1
    assert ctx["chat_read_flags"][0]["user"] == "Buchaanan"


def test_render_prompt_context_includes_dead_air_and_chat_read_annotations():
    transcript_segments = [
        {"start": 90.0, "end": 95.0, "text": "hey chat"},
        {"start": 112.0, "end": 116.0, "text": "welcome back"},
        {
            "start": 118.0,
            "end": 122.0,
            "text": "someone in chat said they met the F1 McLaren owner",
        },
    ]
    chat_messages = [
        {
            "timestamp": 119.0,
            "user": "Buchaanan",
            "message": "I met the F1 McLaren owner at an event last night and it was insane",
        },
    ]

    ctx = build_clip_context(
        seconds=110,
        transcript_segments=transcript_segments,
        chat_messages=chat_messages,
        window=20,
    )
    transcript_text, chat_text = render_prompt_context(ctx)

    assert "⚠️ DEAD AIR DETECTED" in transcript_text
    assert "⚠️ CHAT-READ FLAGS" in transcript_text
    assert "@Buchaanan" in transcript_text
    assert "@Buchaanan" in chat_text
