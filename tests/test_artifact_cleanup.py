"""Tests for the phase4 artifact cleanup planner and executor."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.artifacts.cleanup import (
    CleanupMode,
    CleanupTarget,
    build_cleanup_plan,
    execute_cleanup_plan,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def phase4_dir(tmp_path: Path) -> Path:
    """Create a realistic phase4 directory with sample artifacts."""
    d = tmp_path / "phase4_123"
    d.mkdir()
    _write(d / "qwen_vision_progressive.json", {"vod_id": "123"})
    _write(d / f"fusion_result_123.json", {"vod_meta": {"id": "123"}})
    _write(d / "clip_manifest.json", {"vod_id": "123"})
    _write(d / "transcript.json", {"segments": []})
    _write(d / "scenes.json", [])
    _write(d / "chat.json", {"messages": []})
    _write(d / "yolo_detections.json", {})
    _write(d / f"speaker_attribution_123.json", {"segments": []})
    _write(d / f"profile_update_proposal_123.json", {"proposal": True})
    _write(d / "gemma_multimodal_annotations.json", {"windows": []})
    _write(d / "text_triage_candidates.json", [])
    _write(d / "vision_shortlist.json", [])
    _write(d / "audio_batch_input.json", {})
    _write(d / "audio_batch_output.json", {})
    (d / "raw").mkdir()
    _write(d / "raw" / "123.mp4", b"x" * 10240)
    (d / "frames").mkdir()
    _write(d / "frames" / "frame_000001.jpg", b"y" * 512)
    _write(d / "frames" / "frame_000002.jpg", b"y" * 512)
    return d


@pytest.fixture
def phase4_no_final(phase4_dir: Path) -> Path:
    """Like phase4_dir but missing qwen_vision_progressive.json."""
    (phase4_dir / "qwen_vision_progressive.json").unlink()
    return phase4_dir


def _write(path, content=None):
    if content is None:
        content = b"{}"
    if isinstance(content, (dict, list)):
        content = json.dumps(content, indent=2).encode()
    elif isinstance(content, str):
        content = content.encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


# ---------------------------------------------------------------------------
# build_cleanup_plan
# ---------------------------------------------------------------------------


class TestBuildCleanupPlan:
    def test_intermediate_mode_includes_audio_and_fastpass(self, phase4_dir):
        plan = build_cleanup_plan(phase4_dir, "123")
        files = {str(t.path.name) for t in plan}
        assert "audio_batch_input.json" in files
        assert "audio_batch_output.json" in files
        assert "gemma_multimodal_annotations.json" in files
        assert "text_triage_candidates.json" in files
        assert "vision_shortlist.json" in files

    def test_post_extraction_includes_raw_and_frames(self, phase4_dir):
        plan = build_cleanup_plan(phase4_dir, "123", mode="post-extraction")
        names = {str(t.path.name) for t in plan}
        assert "123.mp4" in names
        assert "frames" in {str(t.path.name) for t in plan if t.kind == "dir"}
        assert "audio_batch_input.json" in names

    def test_post_extraction_excludes_qwen_final(self, phase4_dir):
        plan = build_cleanup_plan(phase4_dir, "123", mode="post-extraction")
        names = {str(t.path.name) for t in plan}
        assert "qwen_vision_progressive.json" not in names

    def test_post_extraction_excludes_profile_update(self, phase4_dir):
        plan = build_cleanup_plan(phase4_dir, "123", mode="post-extraction")
        names = {str(t.path.name) for t in plan}
        assert not any(n.startswith("profile_update_proposal_") for n in names)

    def test_post_extraction_returns_empty_when_final_missing(self, phase4_no_final):
        plan = build_cleanup_plan(phase4_no_final, "123", mode="post-extraction")
        assert len(plan) == 0

    def test_aggressive_includes_small_inputs(self, phase4_dir):
        plan = build_cleanup_plan(phase4_dir, "123", mode="aggressive")
        names = {str(t.path.name) for t in plan}
        assert f"fusion_result_123.json" in names
        assert "clip_manifest.json" in names
        assert "transcript.json" in names
        assert "yolo_detections.json" in names
        assert f"speaker_attribution_123.json" in names

    def test_intermediate_excludes_raw_and_frames(self, phase4_dir):
        plan = build_cleanup_plan(phase4_dir, "123", mode="intermediate")
        names = {str(t.path.name) for t in plan}
        assert "123.mp4" not in names
        assert "frames" not in {str(t.path.name) for t in plan}

    def test_raw_frames_can_be_excluded(self, phase4_dir):
        plan = build_cleanup_plan(
            phase4_dir, "123", mode="post-extraction", include_raw=False, include_frames=False
        )
        names = {str(t.path.name) for t in plan}
        assert "123.mp4" not in names
        assert "frames" not in {str(t.path.name) for t in plan}

    def test_raises_on_invalid_mode(self, phase4_dir):
        with pytest.raises(ValueError):
            build_cleanup_plan(phase4_dir, "123", mode="invalid")


# ---------------------------------------------------------------------------
# execute_cleanup_plan — intermediate mode
# ---------------------------------------------------------------------------


class TestExecuteIntermediate:
    def test_deletes_intermediate_artifacts(self, phase4_dir):
        plan = build_cleanup_plan(phase4_dir, "123", mode="intermediate")
        n_before = sum(1 for _ in phase4_dir.rglob("*") if _.is_file())
        result = execute_cleanup_plan(plan, phase4_dir=phase4_dir)
        n_after = sum(1 for _ in phase4_dir.rglob("*") if _.is_file())
        assert result.bytes_freed > 0
        assert n_after < n_before
        # qwen and profile still there
        assert (phase4_dir / "qwen_vision_progressive.json").exists()
        assert (phase4_dir / "profile_update_proposal_123.json").exists()
        # intermediates gone
        assert not (phase4_dir / "gemma_multimodal_annotations.json").exists()
        assert not (phase4_dir / "audio_batch_input.json").exists()

    def test_intermediate_keeps_raw_frames_fusion(self, phase4_dir):
        plan = build_cleanup_plan(phase4_dir, "123", mode="intermediate")
        execute_cleanup_plan(plan, phase4_dir=phase4_dir)
        assert (phase4_dir / "raw" / "123.mp4").exists()
        assert (phase4_dir / "frames" / "frame_000001.jpg").exists()
        assert (phase4_dir / "fusion_result_123.json").exists()


# ---------------------------------------------------------------------------
# execute_cleanup_plan — post-extraction mode
# ---------------------------------------------------------------------------


class TestExecutePostExtraction:
    def test_deletes_post_extraction_artifacts(self, phase4_dir):
        plan = build_cleanup_plan(phase4_dir, "123", mode="post-extraction")
        n_before = sum(1 for _ in phase4_dir.rglob("*") if _.is_file())
        result = execute_cleanup_plan(plan, phase4_dir=phase4_dir)
        n_after = sum(1 for _ in phase4_dir.rglob("*") if _.is_file())
        assert result.bytes_freed > 0
        assert n_after < n_before
        # raw VOD + frames gone
        assert not (phase4_dir / "raw" / "123.mp4").exists()
        assert not (phase4_dir / "frames").exists()
        # intermediates gone too
        assert not (phase4_dir / "gemma_multimodal_annotations.json").exists()
        # final output remains
        assert (phase4_dir / "qwen_vision_progressive.json").exists()

    def test_post_extraction_keeps_qwen_and_profile_update(self, phase4_dir):
        plan = build_cleanup_plan(phase4_dir, "123", mode="post-extraction")
        execute_cleanup_plan(plan, phase4_dir=phase4_dir)
        assert (phase4_dir / "qwen_vision_progressive.json").exists()
        assert (phase4_dir / "profile_update_proposal_123.json").exists()

    def test_post_extraction_keeps_fusion_manifest(self, phase4_dir):
        plan = build_cleanup_plan(phase4_dir, "123", mode="post-extraction")
        execute_cleanup_plan(plan, phase4_dir=phase4_dir)
        assert (phase4_dir / "fusion_result_123.json").exists()
        assert (phase4_dir / "clip_manifest.json").exists()
        assert (phase4_dir / "transcript.json").exists()


# ---------------------------------------------------------------------------
# execute_cleanup_plan — aggressive mode
# ---------------------------------------------------------------------------


class TestExecuteAggressive:
    def test_aggressive_deletes_small_inputs(self, phase4_dir):
        plan = build_cleanup_plan(phase4_dir, "123", mode="aggressive")
        n_before = sum(1 for _ in phase4_dir.rglob("*") if _.is_file())
        result = execute_cleanup_plan(plan, phase4_dir=phase4_dir)
        n_after = sum(1 for _ in phase4_dir.rglob("*") if _.is_file())
        assert result.bytes_freed > 0
        assert n_after < n_before
        assert not (phase4_dir / "fusion_result_123.json").exists()
        assert not (phase4_dir / "clip_manifest.json").exists()
        assert not (phase4_dir / "transcript.json").exists()
        assert not (phase4_dir / "yolo_detections.json").exists()
        # But never final output
        assert (phase4_dir / "qwen_vision_progressive.json").exists()

    def test_aggressive_keeps_profile_update(self, phase4_dir):
        plan = build_cleanup_plan(phase4_dir, "123", mode="aggressive")
        execute_cleanup_plan(plan, phase4_dir=phase4_dir)
        assert (phase4_dir / "profile_update_proposal_123.json").exists()


# ---------------------------------------------------------------------------
# Safety invariants
# ---------------------------------------------------------------------------


class TestSafety:
    def test_never_deletes_qwen_final(self, phase4_dir):
        plan = build_cleanup_plan(phase4_dir, "123", mode="aggressive")
        # Manually insert a target for qwen (simulate rogue caller)
        rogue = CleanupTarget(
            path=phase4_dir / "qwen_vision_progressive.json",
            kind="file",
            reason="rogue",
            required_mode="aggressive",
        )
        result = execute_cleanup_plan([rogue], phase4_dir=phase4_dir)
        assert len(result.deleted) == 0
        assert len(result.skipped) == 1
        assert "protected" in result.skipped[0].reason
        assert (phase4_dir / "qwen_vision_progressive.json").exists()

    def test_never_deletes_outside_phase4(self, phase4_dir):
        outside = phase4_dir.parent / "outside.txt"
        _write(outside, b"danger")
        rogue = CleanupTarget(
            path=outside,
            kind="file",
            reason="rogue",
            required_mode="aggressive",
        )
        result = execute_cleanup_plan([rogue], phase4_dir=phase4_dir)
        assert len(result.deleted) == 0
        assert outside.exists()

    def test_skips_symlinks(self, phase4_dir):
        outside = phase4_dir.parent / "symlink_target.txt"
        _write(outside, b"sensitive")
        link = phase4_dir / "linked.txt"
        link.symlink_to(outside)
        rogue = CleanupTarget(
            path=link,
            kind="file",
            reason="rogue",
            required_mode="intermediate",
        )
        result = execute_cleanup_plan([rogue], phase4_dir=phase4_dir)
        assert len(result.deleted) == 0
        assert outside.exists()

    def test_dry_run_does_not_delete(self, phase4_dir):
        plan = build_cleanup_plan(phase4_dir, "123", mode="post-extraction")
        n_before = sum(1 for _ in phase4_dir.rglob("*") if _.is_file())
        result = execute_cleanup_plan(plan, phase4_dir=phase4_dir, dry_run=True)
        n_after = sum(1 for _ in phase4_dir.rglob("*") if _.is_file())
        assert n_after == n_before
        assert result.dry_run is True
        assert result.bytes_freed > 0  # reports what would be freed

    def test_dry_run_marks_as_dry(self, phase4_dir):
        plan = build_cleanup_plan(phase4_dir, "123", mode="post-extraction")
        result = execute_cleanup_plan(plan, phase4_dir=phase4_dir, dry_run=True)
        assert result.dry_run


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


class TestResultShape:
    def test_result_to_dict(self, phase4_dir):
        plan = build_cleanup_plan(phase4_dir, "123", mode="intermediate")
        result = execute_cleanup_plan(plan, phase4_dir=phase4_dir)
        d = result.to_dict()
        assert d["dry_run"] is False
        assert "phase4_dir" in d
        assert "deleted" in d
        assert "skipped" in d
        assert isinstance(d["bytes_freed"], int)
        assert d["bytes_freed"] > 0

    def test_result_skipped_missing_files(self, phase4_dir):
        # Remove an intermediate artifact then plan for it
        (phase4_dir / "gemma_multimodal_annotations.json").unlink()
        plan = build_cleanup_plan(phase4_dir, "123", mode="intermediate")
        result = execute_cleanup_plan(plan, phase4_dir=phase4_dir)
        assert len(result.skipped) > 0
        assert any("not found" in s.reason for s in result.skipped)

    def test_result_reports_correct_mode(self, phase4_dir):
        plan = build_cleanup_plan(phase4_dir, "123", mode="post-extraction")
        result = execute_cleanup_plan(plan, phase4_dir=phase4_dir, mode="post-extraction")
        assert result.mode == "post-extraction"
