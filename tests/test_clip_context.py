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


def test_build_clip_context_includes_speaker_stats_and_warning_for_non_streamer_primary():
    transcript_segments = [
        {"start": 100.0, "end": 103.0, "text": "guest starts talking"},
        {"start": 104.0, "end": 108.0, "text": "streamer replies briefly"},
    ]
    speaker_attribution = {
        "segments": [
            {
                "start": 99.0,
                "end": 106.0,
                "speaker_label": "SPEAKER_01",
                "recognition": {"identity": "guest", "confidence": 0.88},
                "inferred_name": "SkitchFriend",
            },
            {
                "start": 106.0,
                "end": 108.0,
                "speaker_label": "SPEAKER_00",
                "recognition": {"identity": "streamer", "confidence": 0.74},
                "inferred_name": "Skitch",
            },
        ],
        "speaker_clusters": {
            "SPEAKER_01": {
                "candidate_names": [
                    {
                        "name": "SkitchFriend",
                        "confidence": 0.8,
                        "evidence": ["hey skitchfriend"],
                    }
                ]
            }
        },
    }

    ctx = build_clip_context(
        seconds=105,
        transcript_segments=transcript_segments,
        chat_messages=[],
        window=10,
        speaker_attribution=speaker_attribution,
    )

    assert ctx["primary_speaker_identity"] == "guest"
    assert ctx["primary_speaker_name"] == "SkitchFriend"
    assert ctx["off_streamer_voice_detected"] is True
    assert ctx["speaker_turns"]

    transcript_text, _ = render_prompt_context(ctx)
    assert "⚠️ SPEAKER ATTRIBUTION" in transcript_text
