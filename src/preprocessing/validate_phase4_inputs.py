#!/usr/bin/env python3
"""Validate Phase-4 inputs required by the clip-intelligence pipeline.

Contract checked:
  vods/phase4_<VOD_ID>/
    raw/<VOD_ID>.mp4
    fusion_result_<VOD_ID>.json
    clip_manifest.json
    frames/frame_*.jpg

Usage:
  PYTHONPATH=. python3 src/preprocessing/validate_phase4_inputs.py --vod-id 2776101332
"""

from __future__ import annotations

# Direct-script bootstrap: avoid local src/preprocessing/types.py shadowing stdlib `types`.
if __package__ in (None, ""):
    import os
    import sys

    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    if _THIS_DIR in sys.path:
        sys.path.remove(_THIS_DIR)
    _REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.intelligence.streamer_store import resolve_streamer_id_context
from src.preprocessing.types import SpeakerAttributionResult


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)


REQUIRED_CLIP_KEYS = {
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


def _load_json(path: Path) -> tuple[dict[str, Any] | list[Any] | None, str | None]:
    try:
        return json.loads(path.read_text()), None
    except Exception as exc:  # noqa: BLE001 - validator should report, not crash
        return None, f"{path}: failed to parse JSON ({exc})"


def _extract_vod_id_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def resolve_streamer_identity_for_phase4(
    phase4_dir: Path,
    vod_id: str,
    override: str | None,
) -> tuple[dict[str, Any], str | None]:
    """Resolve streamer identity from fusion metadata + optional override.

    Returns:
      (streamer_identity_context, parse_error_or_none)
    """

    vod_meta: dict[str, Any] = {}
    fusion_path = phase4_dir / f"fusion_result_{vod_id}.json"
    if not fusion_path.exists():
        return resolve_streamer_id_context(vod_meta, override=override), None

    fusion_json, err = _load_json(fusion_path)
    if err:
        return resolve_streamer_id_context(vod_meta, override=override), err

    if isinstance(fusion_json, dict) and isinstance(fusion_json.get("vod_meta"), dict):
        vod_meta = fusion_json.get("vod_meta", {})

    return resolve_streamer_id_context(vod_meta, override=override), None


def _validate_manifest(manifest: dict[str, Any], vod_id: str, result: ValidationResult) -> None:
    mvod = str(manifest.get("vod_id", "")).strip()
    if mvod and mvod != vod_id:
        result.errors.append(
            f"clip_manifest.json vod_id mismatch: manifest={mvod} expected={vod_id}"
        )

    clips = manifest.get("clips")
    if not isinstance(clips, list):
        result.errors.append("clip_manifest.json missing 'clips' list")
        return

    if not clips:
        result.errors.append("clip_manifest.json has empty 'clips' list")
        return

    for idx, clip in enumerate(clips):
        if not isinstance(clip, dict):
            result.errors.append(f"clip_manifest.json clips[{idx}] is not an object")
            continue

        missing = REQUIRED_CLIP_KEYS.difference(clip.keys())
        if missing:
            result.errors.append(
                f"clip_manifest.json clips[{idx}] missing keys: {sorted(missing)}"
            )

        try:
            start = float(clip.get("start"))
            end = float(clip.get("end"))
            if end <= start:
                result.errors.append(
                    f"clip_manifest.json clips[{idx}] invalid range: start={start} end={end}"
                )
        except Exception:  # noqa: BLE001
            result.errors.append(
                f"clip_manifest.json clips[{idx}] has non-numeric start/end"
            )


def _validate_fusion(fusion: dict[str, Any], vod_id: str, result: ValidationResult) -> None:
    vod_meta = fusion.get("vod_meta")
    if not isinstance(vod_meta, dict):
        result.errors.append("fusion_result JSON missing object 'vod_meta'")
        return

    fvid = str(vod_meta.get("id", "")).strip()
    if fvid and fvid != vod_id:
        result.errors.append(
            f"fusion_result vod_meta.id mismatch: fusion={fvid} expected={vod_id}"
        )

    transcript = fusion.get("transcript", {})
    if not isinstance(transcript, dict):
        result.errors.append("fusion_result transcript is not an object")
    else:
        segs = transcript.get("segments")
        if not isinstance(segs, list):
            result.errors.append("fusion_result transcript.segments missing list")

    chat = fusion.get("chat", {})
    if not isinstance(chat, dict):
        result.errors.append("fusion_result chat is not an object")
    else:
        msgs = chat.get("messages")
        if not isinstance(msgs, list):
            result.warnings.append("fusion_result chat.messages missing list")


def _validate_speaker_attribution(
    speaker_path: Path,
    vod_id: str,
    result: ValidationResult,
) -> None:
    speaker_json, err = _load_json(speaker_path)
    if err:
        result.errors.append(err)
        return
    if not isinstance(speaker_json, dict):
        result.errors.append(f"speaker attribution root must be object: {speaker_path}")
        return

    try:
        artifact = SpeakerAttributionResult.model_validate(speaker_json)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"speaker attribution schema invalid: {speaker_path} ({exc})")
        return

    if artifact.vod_id != vod_id:
        result.errors.append(
            f"speaker attribution vod_id mismatch: artifact={artifact.vod_id} expected={vod_id}"
        )

    result.info.append(
        "speaker attribution: "
        f"segments={len(artifact.segments)}, clusters={len(artifact.speaker_clusters)}"
    )


def validate_phase4_dir(
    vod_id: str,
    phase4_dir: Path,
    *,
    min_frames: int = 1,
    require_speaker_id: bool = False,
    speaker_artifact_path: Path | None = None,
) -> ValidationResult:
    result = ValidationResult(ok=False)

    result.info.append(f"Validating: {phase4_dir}")

    if not phase4_dir.exists():
        result.errors.append(f"missing directory: {phase4_dir}")
        return result

    if phase4_dir.is_symlink():
        result.errors.append(
            f"phase4 directory is a symlink ({phase4_dir} -> {phase4_dir.resolve()}); "
            "this is disallowed to prevent cross-VOD data reuse"
        )

    raw_mp4 = phase4_dir / "raw" / f"{vod_id}.mp4"
    if not raw_mp4.exists():
        result.errors.append(f"missing raw MP4: {raw_mp4}")
    elif raw_mp4.stat().st_size <= 0:
        result.errors.append(f"raw MP4 is empty: {raw_mp4}")

    fusion_path = phase4_dir / f"fusion_result_{vod_id}.json"
    if not fusion_path.exists():
        result.errors.append(f"missing fusion file: {fusion_path}")
    else:
        fusion_json, err = _load_json(fusion_path)
        if err:
            result.errors.append(err)
        elif isinstance(fusion_json, dict):
            _validate_fusion(fusion_json, vod_id, result)
        else:
            result.errors.append(f"fusion file root must be object: {fusion_path}")

    manifest_path = phase4_dir / "clip_manifest.json"
    if not manifest_path.exists():
        result.errors.append(f"missing clip manifest: {manifest_path}")
    else:
        manifest_json, err = _load_json(manifest_path)
        if err:
            result.errors.append(err)
        elif isinstance(manifest_json, dict):
            _validate_manifest(manifest_json, vod_id, result)
        else:
            result.errors.append(f"manifest root must be object: {manifest_path}")

    frames_dir = phase4_dir / "frames"
    if not frames_dir.exists():
        result.errors.append(f"missing frames directory: {frames_dir}")
    else:
        frame_count = sum(1 for _ in frames_dir.glob("*.jpg"))
        result.info.append(f"frames: {frame_count}")
        if frame_count < min_frames:
            result.errors.append(
                f"insufficient frames: found {frame_count}, require >= {min_frames}"
            )

    speaker_path = speaker_artifact_path or (phase4_dir / f"speaker_attribution_{vod_id}.json")
    if speaker_path.exists():
        _validate_speaker_attribution(speaker_path, vod_id, result)
    elif require_speaker_id:
        result.errors.append(
            f"missing required speaker attribution artifact: {speaker_path}"
        )
    else:
        result.info.append("speaker attribution: not present (optional)")

    result.ok = not result.errors
    return result


def _default_phase4_dir(repo_root: Path, vod_id: str) -> Path:
    return repo_root / "vods" / f"phase4_{vod_id}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase-4 pipeline inputs")
    parser.add_argument("--vod-id", required=True, help="Twitch VOD ID")
    parser.add_argument(
        "--phase4-dir",
        type=Path,
        help="Path to phase4 dir (default: <repo>/vods/phase4_<vod_id>)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repo root for default phase4-dir resolution (default: cwd)",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=1,
        help="Minimum required frame JPG count (default: 1)",
    )
    parser.add_argument(
        "--require-speaker-id",
        action="store_true",
        help="Require speaker_attribution_<VOD_ID>.json and validate schema",
    )
    parser.add_argument(
        "--speaker-artifact-path",
        type=Path,
        default=None,
        help="Optional explicit path for speaker attribution artifact",
    )
    parser.add_argument(
        "--streamer-id",
        default=None,
        help="Optional streamer ID override for persistent profile checks",
    )
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=Path("data/streamer_intelligence"),
        help="Persistent streamer intelligence root",
    )
    parser.add_argument(
        "--enable-persistent-intelligence",
        action="store_true",
        help="Validate that persistent profile root/streamer profile path is resolvable",
    )

    args = parser.parse_args()
    phase4_dir = args.phase4_dir or _default_phase4_dir(args.repo_root, args.vod_id)

    result = validate_phase4_dir(
        args.vod_id,
        phase4_dir,
        min_frames=args.min_frames,
        require_speaker_id=args.require_speaker_id,
        speaker_artifact_path=args.speaker_artifact_path,
    )

    if args.enable_persistent_intelligence:
        streamer_identity, parse_err = resolve_streamer_identity_for_phase4(
            phase4_dir,
            args.vod_id,
            args.streamer_id,
        )
        if parse_err:
            result.warnings.append(
                "unable to parse fusion metadata for streamer-id resolution: "
                f"{parse_err}"
            )
        streamer_id = streamer_identity["streamer_id"]
        profile_root = args.profile_root.resolve()
        profile_path = profile_root / streamer_id / "profile.json"
        result.info.append(
            "streamer_id resolution: "
            f"streamer_id={streamer_id} source={streamer_identity['source']} "
            f"metadata={streamer_identity['metadata_streamer_id']} "
            f"override={streamer_identity['override_streamer_id']}"
        )
        if streamer_identity.get("warning"):
            result.warnings.append(streamer_identity["warning"])
        if profile_root.exists():
            result.info.append(f"persistent profile root: {profile_root}")
        else:
            result.warnings.append(f"persistent profile root does not exist yet: {profile_root}")

        if profile_path.exists():
            result.info.append(f"persistent profile found: {profile_path}")
        else:
            result.warnings.append(f"persistent profile missing (will be auto-created on first load): {profile_path}")

    for line in result.info:
        print(f"INFO: {line}")
    for line in result.warnings:
        print(f"WARN: {line}")
    for line in result.errors:
        print(f"ERROR: {line}")

    if result.ok:
        print("OK: phase4 inputs are valid")
        return 0

    print("FAIL: phase4 inputs are invalid")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
