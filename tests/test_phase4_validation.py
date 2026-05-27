import json

from src.preprocessing.validate_phase4_inputs import (
    resolve_streamer_identity_for_phase4,
    validate_phase4_dir,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _minimal_fusion(vod_id: str):
    return {
        "vod_meta": {
            "id": vod_id,
            "title": "Test",
            "duration_seconds": 240,
            "url": f"https://www.twitch.tv/videos/{vod_id}",
            "streamer": "tester",
        },
        "transcript": {"segments": [{"start": 0, "end": 2, "text": "hi"}]},
        "chat": {"messages": []},
        "timeline": [{"timestamp": 1, "chat_intensity": 0.1}],
    }


def _minimal_manifest(vod_id: str):
    return {
        "vod_id": vod_id,
        "vod_title": "Test",
        "streamer": "tester",
        "duration_seconds": 240.0,
        "clips": [
            {
                "start": 0.0,
                "end": 120.0,
                "title": "Candidate 0s-120s",
                "score": 6.5,
                "objects_detected": [],
                "summary": "preview",
                "has_speech": True,
                "chat_intensity": 0.1,
                "label": "window_candidate",
            }
        ],
        "total_clips": 1,
    }


def _minimal_speaker_attribution(vod_id: str):
    return {
        "vod_id": vod_id,
        "audio_path": f"/tmp/{vod_id}.mp4",
        "backend": {"diarization": "unavailable"},
        "segments": [],
        "speaker_clusters": {},
        "clip_speaker_stats": {},
    }


def test_validate_phase4_dir_passes_with_minimal_valid_contract(tmp_path):
    vod_id = "123456789"
    phase4 = tmp_path / "vods" / f"phase4_{vod_id}"
    (phase4 / "raw").mkdir(parents=True)
    (phase4 / "frames").mkdir(parents=True)

    (phase4 / "raw" / f"{vod_id}.mp4").write_bytes(b"not-really-mp4")
    (phase4 / "frames" / "frame_000001.jpg").write_bytes(b"jpg")
    _write_json(phase4 / f"fusion_result_{vod_id}.json", _minimal_fusion(vod_id))
    _write_json(phase4 / "clip_manifest.json", _minimal_manifest(vod_id))

    result = validate_phase4_dir(vod_id, phase4, min_frames=1)
    assert result.ok is True
    assert result.errors == []


def test_validate_phase4_dir_fails_on_manifest_vod_id_mismatch(tmp_path):
    vod_id = "123456789"
    phase4 = tmp_path / "vods" / f"phase4_{vod_id}"
    (phase4 / "raw").mkdir(parents=True)
    (phase4 / "frames").mkdir(parents=True)

    (phase4 / "raw" / f"{vod_id}.mp4").write_bytes(b"not-really-mp4")
    (phase4 / "frames" / "frame_000001.jpg").write_bytes(b"jpg")
    _write_json(phase4 / f"fusion_result_{vod_id}.json", _minimal_fusion(vod_id))

    bad_manifest = _minimal_manifest(vod_id)
    bad_manifest["vod_id"] = "DIFFERENT"
    _write_json(phase4 / "clip_manifest.json", bad_manifest)

    result = validate_phase4_dir(vod_id, phase4, min_frames=1)
    assert result.ok is False
    assert any("vod_id mismatch" in err for err in result.errors)


def test_validate_phase4_dir_fails_on_missing_frames(tmp_path):
    vod_id = "123456789"
    phase4 = tmp_path / "vods" / f"phase4_{vod_id}"
    (phase4 / "raw").mkdir(parents=True)

    (phase4 / "raw" / f"{vod_id}.mp4").write_bytes(b"not-really-mp4")
    _write_json(phase4 / f"fusion_result_{vod_id}.json", _minimal_fusion(vod_id))
    _write_json(phase4 / "clip_manifest.json", _minimal_manifest(vod_id))

    result = validate_phase4_dir(vod_id, phase4, min_frames=1)
    assert result.ok is False
    assert any("frames" in err for err in result.errors)


def test_validate_phase4_dir_require_speaker_id_fails_when_missing(tmp_path):
    vod_id = "123456789"
    phase4 = tmp_path / "vods" / f"phase4_{vod_id}"
    (phase4 / "raw").mkdir(parents=True)
    (phase4 / "frames").mkdir(parents=True)

    (phase4 / "raw" / f"{vod_id}.mp4").write_bytes(b"not-really-mp4")
    (phase4 / "frames" / "frame_000001.jpg").write_bytes(b"jpg")
    _write_json(phase4 / f"fusion_result_{vod_id}.json", _minimal_fusion(vod_id))
    _write_json(phase4 / "clip_manifest.json", _minimal_manifest(vod_id))

    result = validate_phase4_dir(vod_id, phase4, min_frames=1, require_speaker_id=True)
    assert result.ok is False
    assert any("missing required speaker attribution artifact" in err for err in result.errors)


def test_validate_phase4_dir_require_speaker_id_passes_with_valid_artifact(tmp_path):
    vod_id = "123456789"
    phase4 = tmp_path / "vods" / f"phase4_{vod_id}"
    (phase4 / "raw").mkdir(parents=True)
    (phase4 / "frames").mkdir(parents=True)

    (phase4 / "raw" / f"{vod_id}.mp4").write_bytes(b"not-really-mp4")
    (phase4 / "frames" / "frame_000001.jpg").write_bytes(b"jpg")
    _write_json(phase4 / f"fusion_result_{vod_id}.json", _minimal_fusion(vod_id))
    _write_json(phase4 / "clip_manifest.json", _minimal_manifest(vod_id))
    _write_json(
        phase4 / f"speaker_attribution_{vod_id}.json",
        _minimal_speaker_attribution(vod_id),
    )

    result = validate_phase4_dir(vod_id, phase4, min_frames=1, require_speaker_id=True)
    assert result.ok is True
    assert any("speaker attribution:" in line for line in result.info)


def test_resolve_streamer_identity_for_phase4_uses_metadata_by_default(tmp_path):
    vod_id = "123456789"
    phase4 = tmp_path / "vods" / f"phase4_{vod_id}"
    phase4.mkdir(parents=True)
    _write_json(phase4 / f"fusion_result_{vod_id}.json", _minimal_fusion(vod_id))

    identity, parse_err = resolve_streamer_identity_for_phase4(phase4, vod_id, override=None)

    assert parse_err is None
    assert identity["streamer_id"] == "tester"
    assert identity["source"] == "metadata"


def test_resolve_streamer_identity_for_phase4_reports_override_mismatch(tmp_path):
    vod_id = "123456789"
    phase4 = tmp_path / "vods" / f"phase4_{vod_id}"
    phase4.mkdir(parents=True)
    _write_json(phase4 / f"fusion_result_{vod_id}.json", _minimal_fusion(vod_id))

    identity, parse_err = resolve_streamer_identity_for_phase4(
        phase4,
        vod_id,
        override="other_streamer",
    )

    assert parse_err is None
    assert identity["streamer_id"] == "other_streamer"
    assert identity["source"] == "override"
    assert identity["metadata_streamer_id"] == "tester"
    assert identity["override_mismatch"] is True
    assert identity["warning"]
