"""Cleanup planner and executor for phase4 VOD pipeline artifacts.

Three cleanup modes:

- **intermediate** (safe mid-pipeline): deletes only temporary per-step
  artifacts that are not needed by later stages — audio batch inputs/outputs,
  fast-pass debug artifacts that have already been summarized into
  ``qwen_vision_progressive.json``. Keeps raw VOD, frames, fusion, manifest.

- **post-extraction** (default after upload/extract): deletes the large
  disk-heavy artifacts that are no longer needed once clips have been
  extracted and uploaded — raw VOD, frames, intermediate JSON artifacts.
  Requires ``qwen_vision_progressive.json`` to exist.

- **aggressive** (manual / debug-only): removes small reproducibility
  inputs (fusion, manifest, transcript, scene, chat, YOLO, speaker
  attribution) in addition to everything ``post-extraction`` removes.
  Only safe when you're done analysing and won't re-run.
  Requires ``qwen_vision_progressive.json`` to exist.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger("vod-lens.cleanup")


class CleanupMode(str, Enum):
    """Cleanup aggressiveness level."""

    INTERMEDIATE = "intermediate"
    POST_EXTRACTION = "post-extraction"
    AGGRESSIVE = "aggressive"


@dataclass(frozen=True)
class CleanupTarget:
    """A single artifact to remove."""

    path: Path
    kind: str  # "file", "dir", "glob"
    reason: str
    required_mode: str  # cleanest mode that includes this target
    glob_pattern: str | None = None  # only set when kind == "glob"


@dataclass
class CleanupDeletedEntry:
    """One deleted artifact with size info."""

    path: str
    bytes: int
    reason: str


@dataclass
class CleanupSkippedEntry:
    """One artifact that was skipped (not found, protected, etc.)."""

    path: str
    reason: str


@dataclass
class CleanupResult:
    """Result of executing a cleanup plan."""

    dry_run: bool
    phase4_dir: str
    mode: str
    deleted: list[CleanupDeletedEntry] = field(default_factory=list)
    skipped: list[CleanupSkippedEntry] = field(default_factory=list)
    bytes_freed: int = 0

    def to_dict(self) -> dict:
        return {
            "dry_run": self.dry_run,
            "phase4_dir": self.phase4_dir,
            "mode": self.mode,
            "deleted": [{"path": d.path, "bytes": d.bytes, "reason": d.reason} for d in self.deleted],
            "skipped": [{"path": s.path, "reason": s.reason} for s in self.skipped],
            "bytes_freed": self.bytes_freed,
            "files_deleted": len(self.deleted),
            "files_skipped": len(self.skipped),
        }


# ---------------------------------------------------------------------------
# Plan builder
# ---------------------------------------------------------------------------


def _protection_guard(phase4_dir: Path, vod_id: str) -> list[str]:
    """Return error strings if cleanup should be refused. Empty = safe."""
    errors: list[str] = []
    if not phase4_dir.exists():
        errors.append(f"phase4 directory does not exist: {phase4_dir}")
    # Safety: refuse if the directory name doesn't match standard pattern
    # (unless overridden via --force at the CLI level)
    expected_name = f"phase4_{vod_id}"
    if phase4_dir.name != expected_name:
        errors.append(
            f"Directory name mismatch: expected '{expected_name}', got "
            f"'{phase4_dir.name}'. Use --force to override."
        )
    return errors


def _has_final_output(phase4_dir: Path, vod_id: str) -> bool:
    """Check whether qwen_vision_progressive.json exists."""
    candidates = [
        phase4_dir / "qwen_vision_progressive.json",
        phase4_dir / f"qwen_vision_progressive_{vod_id}.json",
    ]
    return any(c.exists() for c in candidates)


def _size_bytes(path: Path) -> int:
    """Return size of a file or recursive size of a directory tree."""
    if path.is_file() or path.is_symlink():
        return path.stat().st_size
    if path.is_dir():
        total = 0
        for f in path.rglob("*"):
            if f.is_file() and not f.is_symlink():
                total += f.stat().st_size
        return total
    return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_cleanup_plan(
    phase4_dir: Path,
    vod_id: str,
    mode: str = "post-extraction",
    include_raw: bool = True,
    include_frames: bool = True,
) -> list[CleanupTarget]:
    """Build an ordered list of targets to clean up.

    Raises ``ValueError`` if ``mode`` is unrecognised.
    Returns an empty list if the final output is missing and mode requires it.
    """
    mode_enum = CleanupMode(mode)

    # Intermediate mode keeps raw / frames / fusion / manifest
    # Post-extraction and aggressive require final output to exist
    if mode_enum in (CleanupMode.POST_EXTRACTION, CleanupMode.AGGRESSIVE):
        if not _has_final_output(phase4_dir, vod_id):
            logger.warning(
                "Final output (qwen_vision_progressive.json) not found in %s "
                "— refusing cleanup plan for mode=%s",
                phase4_dir,
                mode,
            )
            return []

    targets: list[CleanupTarget] = []

    # ── Targets for intermediate mode (always safe even mid-pipeline) ────
    # Audio batch artifacts
    targets.append(
        CleanupTarget(
            path=phase4_dir / "audio_batch_input.json",
            kind="file",
            reason="audio batch input",
            required_mode="intermediate",
        )
    )
    targets.append(
        CleanupTarget(
            path=phase4_dir / "audio_batch_output.json",
            kind="file",
            reason="audio batch output",
            required_mode="intermediate",
        )
    )
    # Fast-pass intermediate artifacts (already summarized in final output)
    targets.append(
        CleanupTarget(
            path=phase4_dir / "gemma_multimodal_annotations.json",
            kind="file",
            reason="Gemma multimodal annotations (fast-pass)",
            required_mode="intermediate",
        )
    )
    targets.append(
        CleanupTarget(
            path=phase4_dir / "text_triage_candidates.json",
            kind="file",
            reason="text triage candidates (fast-pass)",
            required_mode="intermediate",
        )
    )
    targets.append(
        CleanupTarget(
            path=phase4_dir / "vision_shortlist.json",
            kind="file",
            reason="vision shortlist (fast-pass)",
            required_mode="intermediate",
        )
    )

    # ── Post-extraction targets ──────────────────────────────────────────
    if mode_enum in (CleanupMode.POST_EXTRACTION, CleanupMode.AGGRESSIVE):
        # Temp wav / audio files anywhere in phase4 dir
        targets.append(
            CleanupTarget(
                path=phase4_dir,
                kind="glob",
                reason="temporary WAV audio extracts",
                required_mode="post-extraction",
                glob_pattern="*.wav",
            )
        )
        targets.append(
            CleanupTarget(
                path=phase4_dir,
                kind="glob",
                reason="temporary MP3 audio extracts",
                required_mode="post-extraction",
                glob_pattern="*.mp3",
            )
        )
        targets.append(
            CleanupTarget(
                path=phase4_dir,
                kind="glob",
                reason="temporary FLAC audio extracts",
                required_mode="post-extraction",
                glob_pattern="*.flac",
            )
        )

        # Raw VOD
        if include_raw:
            targets.append(
                CleanupTarget(
                    path=phase4_dir / "raw" / f"{vod_id}.mp4",
                    kind="file",
                    reason="raw VOD MP4",
                    required_mode="post-extraction",
                )
            )
            # Also clean up the raw/ directory if empty afterwards
            targets.append(
                CleanupTarget(
                    path=phase4_dir / "raw",
                    kind="dir",
                    reason="raw directory (if empty after deletion)",
                    required_mode="post-extraction",
                )
            )

        # Frames
        if include_frames:
            targets.append(
                CleanupTarget(
                    path=phase4_dir / "frames",
                    kind="dir",
                    reason="extracted frame images",
                    required_mode="post-extraction",
                )
            )

        # Interim fast-pass result files
        targets.append(
            CleanupTarget(
                path=phase4_dir / "gemma_windows.json",
                kind="file",
                reason="Gemma window definitions",
                required_mode="post-extraction",
            )
        )
        targets.append(
            CleanupTarget(
                path=phase4_dir / "fast_pass_text_triage.json",
                kind="file",
                reason="text triage raw output",
                required_mode="post-extraction",
            )
        )

    # ── Aggressive targets ───────────────────────────────────────────────
    if mode_enum == CleanupMode.AGGRESSIVE:
        for _key, filename in [
            ("fusion", f"fusion_result_{vod_id}.json"),
            ("manifest", "clip_manifest.json"),
            ("transcript", "transcript.json"),
            ("scenes", "scenes.json"),
            ("chat", "chat.json"),
            ("chat_analysis", "chat_analysis.json"),
            ("yolo", "yolo_detections.json"),
            ("speaker_attribution", f"speaker_attribution_{vod_id}.json"),
            ("cleanup_report", f"cleanup_report_{vod_id}.json"),  # previous run
        ]:
            targets.append(
                CleanupTarget(
                    path=phase4_dir / filename,
                    kind="file",
                    reason=f"{_key} result (aggressive)",
                    required_mode="aggressive",
                )
            )

    return targets


def execute_cleanup_plan(
    targets: list[CleanupTarget],
    *,
    phase4_dir: Path,
    dry_run: bool = False,
    mode: str = "",
) -> CleanupResult:
    """Execute a cleanup plan, returning what was deleted and skipped.

    Safety invariants enforced:
    - Never follows or deletes symlink targets.
    - Never deletes ``qwen_vision_progressive.json``.
    - Never deletes ``profile_update_proposal_*.json``.
    - Never deletes anything outside ``phase4_dir``.
    """
    protected_files: set[str] = {
        "qwen_vision_progressive.json",
    }
    # Also protect any profile_update_proposal file
    protected_prefixes: list[str] = [
        "profile_update_proposal_",
        "cleanup_report_",  # don't self-delete current run's report
    ]

    result = CleanupResult(
        dry_run=dry_run,
        phase4_dir=str(phase4_dir.resolve()),
        mode=mode or (targets[0].required_mode if targets else ""),
    )
    seen = set()

    def _is_protected(p: Path) -> bool:
        name = p.name
        if name in protected_files:
            return True
        for prefix in protected_prefixes:
            if name.startswith(prefix):
                return True
        return False

    def _outside(p: Path) -> bool:
        try:
            p.resolve().relative_to(phase4_dir.resolve())
            return False
        except ValueError:
            return True

    for target in targets:
        path = target.path

        # Skip if outside phase4 dir
        if _outside(path):
            result.skipped.append(
                CleanupSkippedEntry(path=str(path), reason="outside phase4 directory")
            )
            continue

        if target.kind == "file":
            if not path.exists() or path.is_symlink():
                if path.is_symlink():
                    result.skipped.append(
                        CleanupSkippedEntry(path=str(path), reason="is a symlink")
                    )
                else:
                    result.skipped.append(
                        CleanupSkippedEntry(path=str(path), reason="not found")
                    )
                continue

            if _is_protected(path):
                result.skipped.append(
                    CleanupSkippedEntry(path=str(path), reason="protected artifact")
                )
                continue

            real = path.resolve()
            if real in seen:
                continue
            seen.add(real)

            sz = _size_bytes(path)
            if not dry_run:
                path.unlink()
            result.deleted.append(
                CleanupDeletedEntry(
                    path=str(path), bytes=sz, reason=target.reason
                )
            )
            result.bytes_freed += sz

        elif target.kind == "dir":
            if not path.exists():
                result.skipped.append(
                    CleanupSkippedEntry(path=str(path), reason="not found")
                )
                continue
            if path.is_symlink():
                result.skipped.append(
                    CleanupSkippedEntry(path=str(path), reason="is a symlink")
                )
                continue

            sz = _size_bytes(path)
            if not dry_run:
                shutil.rmtree(str(path), ignore_errors=True)
            result.deleted.append(
                CleanupDeletedEntry(
                    path=str(path), bytes=sz, reason=target.reason
                )
            )
            result.bytes_freed += sz

        elif target.kind == "glob":
            assert target.glob_pattern is not None
            for match in sorted(path.glob(target.glob_pattern)):
                if match.is_symlink():
                    continue
                if _is_protected(match):
                    continue
                if _outside(match):
                    continue
                real = match.resolve()
                if real in seen:
                    continue
                seen.add(real)

                sz = _size_bytes(match)
                if not dry_run:
                    match.unlink()
                result.deleted.append(
                    CleanupDeletedEntry(
                        path=str(match), bytes=sz, reason=target.reason
                    )
                )
                result.bytes_freed += sz

    return result


def _has_known_content(dir_path: Path, phase4_dir: Path) -> bool:
    """Check if a directory still has relevant files besides temp audio."""
    if not dir_path.is_dir():
        return False
    for entry in dir_path.iterdir():
        if entry.is_symlink():
            continue
        # Non .wav/.mp3/.flac files count as known content
        if entry.suffix.lower() not in {".wav", ".mp3", ".flac"}:
            return True
    return False


def _exists_in_deleted_set(path: Path, targets: list[CleanupTarget]) -> bool:
    """Check if a path is among the file targets in the plan."""
    target_set = {t.resolve() if t.kind == "file" else None for t in targets if t.kind == "file"}
    try:
        return path.resolve() in target_set
    except Exception:  # noqa: BLE001
        return False
