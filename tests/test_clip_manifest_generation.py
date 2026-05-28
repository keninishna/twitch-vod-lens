from src.preprocessing.clip_manifest import generate_clip_manifest


def test_yolo_positive_window_ranks_higher_than_equal_no_object_window():
    fusion = {
        "vod_meta": {"duration_seconds": 12},
        "transcript": {
            "segments": [
                {"start": 0, "end": 6, "text": "window one speech"},
                {"start": 6, "end": 12, "text": "window two speech"},
            ]
        },
        "timeline": [
            {"timestamp": 1, "chat_intensity": 0.6},
            {"timestamp": 7, "chat_intensity": 0.6},
        ],
    }

    yolo_detections = {
        2: ["person"],
    }

    manifest = generate_clip_manifest(
        fusion,
        vod_id="vod-1",
        vod_title="Synthetic VOD",
        streamer="tester",
        window_seconds=6,
        step_seconds=6,
        yolo_frames=yolo_detections,
    )

    clips = manifest["clips"]
    assert len(clips) == 2
    assert clips[0]["objects_detected"] == ["person"]
    assert clips[1]["objects_detected"] == []
    assert clips[0]["score"] > clips[1]["score"]


def test_empty_window_low_score_and_required_keys():
    fusion = {
        "vod_meta": {"duration_seconds": 10},
        "transcript": {"segments": []},
        "timeline": [],
    }

    manifest = generate_clip_manifest(
        fusion,
        vod_id="vod-2",
        vod_title="Empty Signals",
        streamer="tester",
        window_seconds=10,
        step_seconds=10,
    )

    assert manifest["clips"], "Expected at least one generated clip"
    clip = manifest["clips"][0]

    assert clip["score"] <= 0.5

    required_keys = {
        "start",
        "end",
        "title",
        "score",
        "objects_detected",
        "summary",
        "has_speech",
        "chat_intensity",
        "label",
    }
    assert required_keys.issubset(clip.keys())


def test_generated_records_include_required_keys():
    fusion = {
        "vod_meta": {"duration_seconds": 15},
        "transcript": {
            "segments": [
                {"start": 0, "end": 5, "text": "intro and setup"},
                {"start": 5, "end": 10, "text": "mid game action"},
                {"start": 10, "end": 15, "text": "closing reaction"},
            ]
        },
        "timeline": [
            {"timestamp": 2, "chat_intensity": 0.25},
            {"timestamp": 7, "chat_intensity": 0.45},
            {"timestamp": 12, "chat_intensity": 0.35},
        ],
    }

    manifest = generate_clip_manifest(
        fusion,
        vod_id="vod-3",
        vod_title="Required Keys",
        streamer="tester",
        window_seconds=5,
        step_seconds=5,
        yolo_frames={2: ["keyboard"], 12: ["monitor"]},
    )

    required_keys = {
        "start",
        "end",
        "title",
        "score",
        "objects_detected",
        "summary",
        "has_speech",
        "chat_intensity",
        "label",
    }

    assert manifest["clips"], "Expected one or more generated clips"
    for clip in manifest["clips"]:
        assert required_keys.issubset(clip.keys())
