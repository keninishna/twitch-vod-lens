from pathlib import Path

import pytest

from src.synthesis.extract_and_upload_clips import resolve_raw_vod_path


def test_resolve_raw_vod_path_prefers_explicit_path(tmp_path: Path):
    vod_id = "123456789"
    phase4_dir = tmp_path / f"phase4_{vod_id}"
    (phase4_dir / "raw").mkdir(parents=True)

    explicit_path = tmp_path / "explicit_vod.mp4"
    explicit_path.write_bytes(b"explicit")

    phase4_candidate = phase4_dir / "raw" / f"{vod_id}.mp4"
    phase4_candidate.write_bytes(b"phase4")

    resolved = resolve_raw_vod_path(
        vod_id=vod_id,
        phase4_dir=phase4_dir,
        explicit_path=explicit_path,
    )

    assert resolved == explicit_path.resolve()


def test_resolve_raw_vod_path_uses_phase4_raw_fallback(tmp_path: Path):
    vod_id = "987654321"
    phase4_dir = tmp_path / f"phase4_{vod_id}"
    phase4_candidate = phase4_dir / "raw" / f"{vod_id}.mp4"
    phase4_candidate.parent.mkdir(parents=True)
    phase4_candidate.write_bytes(b"phase4-only")

    resolved = resolve_raw_vod_path(
        vod_id=vod_id,
        phase4_dir=phase4_dir,
    )

    assert resolved == phase4_candidate.resolve()


def test_resolve_raw_vod_path_uses_fusion_metadata_source_video(tmp_path: Path):
    vod_id = "1122334455"
    phase4_dir = tmp_path / f"phase4_{vod_id}"
    phase4_dir.mkdir(parents=True)

    metadata_source = tmp_path / "absolute_metadata_source.mp4"
    metadata_source.write_bytes(b"metadata")

    fusion_data = {
        "vod_meta": {
            "source_video": str(metadata_source.resolve()),
        }
    }

    resolved = resolve_raw_vod_path(
        vod_id=vod_id,
        phase4_dir=phase4_dir,
        fusion_data=fusion_data,
    )

    assert resolved == metadata_source.resolve()


def test_resolve_raw_vod_path_missing_raises_actionable_error(tmp_path: Path):
    phase4_dir = tmp_path / "phase4_missing"
    phase4_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_raw_vod_path(
            vod_id=None,
            phase4_dir=phase4_dir,
            explicit_path=None,
            fusion_data=None,
        )

    msg = str(exc_info.value)
    assert "Remediation" in msg
    assert "--vod" in msg
    assert "Checked" in msg
