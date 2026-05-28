#!/usr/bin/env python3
"""Prepare a full phase4_<VOD_ID> directory for clip-intelligence runs.

This script orchestrates:
1) raw VOD download
2) preprocessing pipeline (fusion_result_<VOD_ID>.json)
3) deterministic clip_manifest generation
4) frame extraction (5s cadence by default)
5) phase4 contract validation

Usage:
  PYTHONPATH=. python3 src/preprocessing/prepare_phase4.py \
    --url https://www.twitch.tv/videos/2776101332 \
    --vod-id 2776101332
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
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.intelligence.streamer_store import (
    load_persistent_voice_profiles,
    load_streamer_profile,
    resolve_streamer_id_context,
)
from src.preprocessing.clip_manifest import generate_clip_manifest, load_yolo_detections
from src.preprocessing.speaker_attribution import generate_speaker_attribution
from src.preprocessing.speaker_profiles import load_profiles
from src.preprocessing.validate_phase4_inputs import validate_phase4_dir


@dataclass
class CmdResult:
    returncode: int
    stdout: str
    stderr: str


def _run(cmd: list[str], *, cwd: Path, timeout: int = 3600) -> CmdResult:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return CmdResult(proc.returncode, proc.stdout, proc.stderr)


def _extract_vod_id(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text())


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _numeric(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return default


def _compute_duration_seconds(fusion: dict[str, Any]) -> float:
    vod_meta = fusion.get("vod_meta", {}) if isinstance(fusion, dict) else {}
    dur = _numeric(vod_meta.get("duration_seconds"), 0.0)
    if dur > 0:
        return dur

    transcript = fusion.get("transcript", {}) if isinstance(fusion, dict) else {}
    segments = transcript.get("segments", []) if isinstance(transcript, dict) else []
    max_seg = 0.0
    for seg in segments:
        if isinstance(seg, dict):
            max_seg = max(max_seg, _numeric(seg.get("end"), 0.0))

    timeline = fusion.get("timeline", []) if isinstance(fusion, dict) else []
    max_t = 0.0
    for row in timeline:
        if isinstance(row, dict):
            max_t = max(max_t, _numeric(row.get("timestamp"), 0.0))

    return max(max_seg, max_t)


def _windowed_clip_manifest(
    fusion: dict[str, Any],
    *,
    vod_id: str,
    vod_title: str,
    streamer: str,
    window_seconds: int,
    step_seconds: int,
) -> dict[str, Any]:
    duration = _compute_duration_seconds(fusion)
    if duration <= 0:
        raise RuntimeError("fusion result has no detectable duration")

    transcript = fusion.get("transcript", {}) if isinstance(fusion, dict) else {}
    segments = transcript.get("segments", []) if isinstance(transcript, dict) else []
    timeline = fusion.get("timeline", []) if isinstance(fusion, dict) else []

    def seg_overlap(seg: dict[str, Any], start: float, end: float) -> float:
        ss = _numeric(seg.get("start"))
        se = _numeric(seg.get("end"))
        if se <= ss:
            return 0.0
        return max(0.0, min(end, se) - max(start, ss))

    clips: list[dict[str, Any]] = []
    w = max(10, int(window_seconds))
    step = max(1, int(step_seconds))

    upper = max(0, int(math.ceil(duration)) - w)
    starts = list(range(0, upper + 1, step))
    if starts[-1] != upper:
        starts.append(upper)

    for start in starts:
        end = min(start + w, int(math.ceil(duration)))

        # speech coverage in seconds
        speech_seconds = 0.0
        speech_lines: list[str] = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            ov = seg_overlap(seg, start, end)
            if ov > 0:
                speech_seconds += ov
                text = str(seg.get("text", "")).strip()
                if text:
                    speech_lines.append(text)

        speech_ratio = speech_seconds / max(1.0, (end - start))
        has_speech = speech_seconds > 2.0

        # chat intensity from timeline (if present)
        chat_values: list[float] = []
        for row in timeline:
            if not isinstance(row, dict):
                continue
            ts = _numeric(row.get("timestamp"), -1)
            if start <= ts <= end:
                chat_values.append(_numeric(row.get("chat_intensity"), 0.0))
        chat_intensity = sum(chat_values) / len(chat_values) if chat_values else 0.0

        # deterministic score in 0..10
        score = 0.0
        score += min(6.0, speech_ratio * 8.0)      # max +6
        score += min(3.0, chat_intensity * 2.0)    # max +3
        if 25 <= (end - start) <= 90:
            score += 1.0                            # max +1
        score = round(max(0.0, min(10.0, score)), 2)

        preview = " ".join(speech_lines)[:220]
        clip = {
            "start": float(start),
            "end": float(end),
            "title": f"Candidate {start}s-{end}s",
            "score": score,
            "objects_detected": [],
            "summary": preview or "No transcript preview",
            "has_speech": bool(has_speech),
            "chat_intensity": round(chat_intensity, 3),
            "label": "window_candidate",
        }
        clips.append(clip)

    return {
        "vod_id": vod_id,
        "vod_title": vod_title,
        "streamer": streamer,
        "duration_seconds": float(duration),
        "clips": clips,
        "total_clips": len(clips),
    }


def _download_raw_vod(url: str, raw_mp4_path: Path, *, repo_root: Path, overwrite: bool) -> None:
    if raw_mp4_path.exists() and raw_mp4_path.stat().st_size > 0 and not overwrite:
        print(f"[prepare_phase4] raw mp4 exists; skipping download: {raw_mp4_path}")
        return

    raw_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "-f",
        "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        str(raw_mp4_path),
        url,
    ]
    print(f"[prepare_phase4] downloading raw VOD -> {raw_mp4_path}")
    res = _run(cmd, cwd=repo_root, timeout=7200)
    if res.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {res.stderr[-600:]}")
    if not raw_mp4_path.exists() or raw_mp4_path.stat().st_size <= 0:
        raise RuntimeError(f"download finished but raw mp4 missing/empty: {raw_mp4_path}")


def _build_fusion_from_legacy_output(
    *,
    vod_id: str,
    url: str,
    legacy_dir: Path,
    raw_mp4_path: Path,
    fusion_out: Path,
) -> None:
    transcript_path = legacy_dir / "transcript.json"
    scenes_path = legacy_dir / "scenes.json"
    chat_path = legacy_dir / "chat_analysis.json"

    if not transcript_path.exists():
        raise RuntimeError(f"legacy preprocessing missing transcript: {transcript_path}")

    transcript_segments = _load_json(transcript_path)
    if not isinstance(transcript_segments, list):
        raise RuntimeError("legacy transcript.json must be a list")

    scenes = _load_json(scenes_path) if scenes_path.exists() else []
    chat = _load_json(chat_path) if chat_path.exists() else {
        "total_messages": 0,
        "total_emotes": 0,
        "spikes": [],
        "top_emotes": [],
    }

    duration = 0.0
    for seg in transcript_segments:
        if isinstance(seg, dict):
            duration = max(duration, _numeric(seg.get("end"), 0.0))

    spikes = chat.get("spikes", []) if isinstance(chat, dict) else []

    def _chat_intensity_for_range(start: float, end: float) -> float:
        vals: list[float] = []
        for s in spikes:
            if not isinstance(s, dict):
                continue
            ss = _numeric(s.get("start"), -1)
            se = _numeric(s.get("end"), -1)
            if se <= ss:
                continue
            if ss < end and se > start:
                vals.append(_numeric(s.get("intensity"), 0.0))
        return max(vals) if vals else 0.0

    timeline: list[dict[str, Any]] = []
    for idx, seg in enumerate(transcript_segments):
        if not isinstance(seg, dict):
            continue
        s = _numeric(seg.get("start"), 0.0)
        e = _numeric(seg.get("end"), s)
        timeline.append(
            {
                "timestamp": s,
                "transcript": str(seg.get("text", "")),
                "scene_change": False,
                "scene_index": None,
                "chat_intensity": _chat_intensity_for_range(s, e),
                "top_emotes": [],
                "segment_index": idx,
            }
        )

    fusion_payload = {
        "vod_meta": {
            "id": vod_id,
            "title": f"VOD_{vod_id}",
            "duration_seconds": int(duration),
            "url": url,
            "streamer": "unknown",
            "source_video": str(raw_mp4_path),
        },
        "transcript": {
            "segments": transcript_segments,
            "language": "en",
            "duration_seconds": duration,
        },
        "scenes": scenes if isinstance(scenes, list) else [],
        "chat": {
            "messages": [],
            "summary": chat if isinstance(chat, dict) else {},
        },
        "timeline": timeline,
        "processing_time_seconds": 0.0,
        "generated_by": "prepare_phase4_legacy_adapter",
    }

    _save_json(fusion_out, fusion_payload)



def _run_preprocessing(
    vod_id: str,
    url: str,
    phase4_dir: Path,
    fusion_out: Path,
    *,
    repo_root: Path,
    skip: bool,
) -> None:
    if skip and fusion_out.exists():
        print(f"[prepare_phase4] skipping preprocessing; using existing fusion: {fusion_out}")
        return

    # Try modern pipeline first.
    modern_cmd = [
        sys.executable,
        "src/preprocessing/__main__.py",
        url,
        "--workdir",
        str(phase4_dir),
        "--output",
        str(fusion_out),
    ]
    print("[prepare_phase4] running modern preprocessing pipeline ...")
    modern = _run(modern_cmd, cwd=repo_root, timeout=14400)
    if modern.returncode == 0 and fusion_out.exists():
        return

    print("[prepare_phase4] modern preprocessing failed; falling back to legacy preprocess.py")

    legacy_cmd = [sys.executable, "preprocess.py", url]
    legacy = _run(legacy_cmd, cwd=repo_root, timeout=14400)
    if legacy.returncode != 0:
        raise RuntimeError(
            "both preprocessing paths failed\n"
            f"modern stdout tail:\n{modern.stdout[-1200:]}\n"
            f"modern stderr tail:\n{modern.stderr[-1200:]}\n"
            f"legacy stdout tail:\n{legacy.stdout[-1200:]}\n"
            f"legacy stderr tail:\n{legacy.stderr[-1200:]}"
        )

    legacy_dir = repo_root / "output" / vod_id
    if not legacy_dir.exists():
        raise RuntimeError(f"legacy preprocess succeeded but output dir missing: {legacy_dir}")

    _build_fusion_from_legacy_output(
        vod_id=vod_id,
        url=url,
        legacy_dir=legacy_dir,
        raw_mp4_path=phase4_dir / "raw" / f"{vod_id}.mp4",
        fusion_out=fusion_out,
    )


def _extract_frames(raw_mp4_path: Path, frames_dir: Path, *, interval_s: int, repo_root: Path, overwrite: bool) -> None:
    existing = sum(1 for _ in frames_dir.glob("*.jpg")) if frames_dir.exists() else 0
    if existing > 0 and not overwrite:
        print(f"[prepare_phase4] frames already exist ({existing}); skipping extraction")
        return

    frames_dir.mkdir(parents=True, exist_ok=True)
    # Keep naming compatible with current synthesis loader: frame_000001.jpg
    out_pattern = frames_dir / "frame_%06d.jpg"
    fps_expr = f"1/{max(1, interval_s)}"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_mp4_path),
        "-vf",
        f"fps={fps_expr}",
        str(out_pattern),
    ]
    print(f"[prepare_phase4] extracting frames every {interval_s}s ...")
    res = _run(cmd, cwd=repo_root, timeout=7200)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {res.stderr[-800:]}")



def _find_existing_path(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def _run_speaker_attribution_step(
    *,
    vod_id: str,
    phase4_dir: Path,
    raw_mp4_path: Path,
    fusion_out: Path,
    profiles_dir: Path | None,
    hf_token: str | None,
    require_speaker_id: bool,
    streamer_id: str | None,
    profile_root: Path,
    use_persistent_intelligence: bool,
) -> Path:
    speaker_out = phase4_dir / f"speaker_attribution_{vod_id}.json"

    transcript_path = _find_existing_path(
        [
            phase4_dir / "transcript.json",
            fusion_out,
        ]
    )
    if transcript_path is None:
        raise RuntimeError(
            "speaker attribution requested but transcript source was not found "
            f"(checked: {phase4_dir / 'transcript.json'}, {fusion_out})"
        )

    chat_path = _find_existing_path(
        [
            phase4_dir / "chat.json",
            phase4_dir / "chat_analysis.json",
            fusion_out,
        ]
    )

    runtime_profiles: list[dict[str, Any]] = []
    if profiles_dir is not None:
        runtime_profiles.extend(load_profiles(profiles_dir))
    if use_persistent_intelligence and streamer_id:
        runtime_profiles.extend(load_persistent_voice_profiles(streamer_id, profile_root))

    result = generate_speaker_attribution(
        vod_id=vod_id,
        vod_media=raw_mp4_path,
        transcript_path=transcript_path,
        chat_path=chat_path,
        profiles_dir=None,
        profiles=runtime_profiles or None,
        output_path=speaker_out,
        hf_token=hf_token,
        require_speaker_id=require_speaker_id,
    )

    print(
        "[prepare_phase4] speaker attribution ready: "
        f"{speaker_out} (segments={len(result.segments)}, clusters={len(result.speaker_clusters)})"
    )
    return speaker_out


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare phase4_<VOD_ID> inputs")
    parser.add_argument("--url", required=True, help="Twitch VOD URL")
    parser.add_argument("--vod-id", help="VOD ID (default: extracted from URL)")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repo root (default: cwd)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output phase4 directory (default: <repo>/vods/phase4_<vod_id>)",
    )
    parser.add_argument("--window-seconds", type=int, default=120)
    parser.add_argument("--step-seconds", type=int, default=120)
    parser.add_argument("--frame-interval", type=int, default=5)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-frames", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing artifacts")
    parser.add_argument(
        "--enable-speaker-id",
        action="store_true",
        help="Generate speaker_attribution_<VOD_ID>.json after phase4 artifacts are prepared",
    )
    parser.add_argument(
        "--speaker-profiles-dir",
        type=Path,
        default=None,
        help="Optional speaker profile directory used by SpeakerID recognition",
    )
    parser.add_argument(
        "--require-speaker-id",
        action="store_true",
        help="Fail the run if speaker attribution cannot be generated or validated",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional HuggingFace token override for pyannote diarization",
    )
    parser.add_argument(
        "--streamer-id",
        default=None,
        help="Override streamer ID for persistent intelligence profile resolution",
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
        help="Enable persistent streamer intelligence profile load/validation",
    )
    parser.add_argument(
        "--update-streamer-profile",
        action="store_true",
        help="Reserved for synthesis stage; accepted here for CLI consistency",
    )
    parser.add_argument(
        "--profile-update-mode",
        choices=["propose", "auto", "off"],
        default="propose",
        help="Reserved for synthesis stage; accepted here for CLI consistency",
    )

    args = parser.parse_args()

    vod_id = args.vod_id or _extract_vod_id(args.url)
    repo_root = args.repo_root.resolve()
    phase4_dir = (args.out or (repo_root / "vods" / f"phase4_{vod_id}")).resolve()
    raw_mp4_path = phase4_dir / "raw" / f"{vod_id}.mp4"
    fusion_out = phase4_dir / f"fusion_result_{vod_id}.json"
    manifest_out = phase4_dir / "clip_manifest.json"
    frames_dir = phase4_dir / "frames"

    phase4_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not args.skip_download:
            _download_raw_vod(args.url, raw_mp4_path, repo_root=repo_root, overwrite=args.overwrite)
        else:
            print("[prepare_phase4] skip-download enabled")

        _run_preprocessing(
            vod_id,
            args.url,
            phase4_dir,
            fusion_out,
            repo_root=repo_root,
            skip=args.skip_preprocess,
        )

        fusion = _load_json(fusion_out)
        if not isinstance(fusion, dict):
            raise RuntimeError(f"fusion output must be JSON object: {fusion_out}")

        vod_meta = fusion.get("vod_meta", {}) if isinstance(fusion.get("vod_meta"), dict) else {}
        vod_title = str(vod_meta.get("title") or f"VOD {vod_id}")
        streamer = str(vod_meta.get("streamer") or vod_meta.get("uploader") or "unknown")

        profile_root = args.profile_root.resolve()
        streamer_identity = resolve_streamer_id_context(vod_meta, args.streamer_id)
        streamer_id = streamer_identity["streamer_id"]
        print(
            "[prepare_phase4] resolved streamer_id: "
            f"{streamer_id} (source={streamer_identity['source']}, "
            f"metadata={streamer_identity['metadata_streamer_id']}, "
            f"override={streamer_identity['override_streamer_id']})"
        )
        if streamer_identity.get("warning"):
            print(f"WARN: {streamer_identity['warning']}")
        if args.enable_persistent_intelligence:
            profile = load_streamer_profile(streamer_id=streamer_id, root=profile_root)
            print(
                "[prepare_phase4] persistent profile ready: "
                f"streamer_id={streamer_id} path={profile_root / streamer_id / 'profile.json'} "
                f"(voice_profiles={len(profile.voice_profiles)})"
            )

        if args.update_streamer_profile:
            print(
                "[prepare_phase4] note: --update-streamer-profile is applied during synthesis, "
                f"mode={args.profile_update_mode}"
            )

        yolo_path = phase4_dir / "yolo_detections.json"
        yolo_frames = load_yolo_detections(yolo_path)

        manifest = generate_clip_manifest(
            fusion,
            vod_id=vod_id,
            vod_title=vod_title,
            streamer=streamer,
            window_seconds=args.window_seconds,
            step_seconds=args.step_seconds,
            yolo_frames=yolo_frames,
        )
        _save_json(manifest_out, manifest)
        print(f"[prepare_phase4] wrote manifest: {manifest_out} (clips={manifest['total_clips']})")

        if not args.skip_frames:
            if not raw_mp4_path.exists():
                raise RuntimeError(
                    f"raw mp4 missing for frame extraction: {raw_mp4_path}"
                )
            _extract_frames(
                raw_mp4_path,
                frames_dir,
                interval_s=args.frame_interval,
                repo_root=repo_root,
                overwrite=args.overwrite,
            )
        else:
            print("[prepare_phase4] skip-frames enabled")

        speaker_artifact_path = None
        if args.enable_speaker_id or args.require_speaker_id:
            speaker_profiles_dir = (
                args.speaker_profiles_dir.resolve()
                if args.speaker_profiles_dir is not None
                else None
            )
            speaker_artifact_path = _run_speaker_attribution_step(
                vod_id=vod_id,
                phase4_dir=phase4_dir,
                raw_mp4_path=raw_mp4_path,
                fusion_out=fusion_out,
                profiles_dir=speaker_profiles_dir,
                hf_token=args.hf_token,
                require_speaker_id=args.require_speaker_id,
                streamer_id=streamer_id,
                profile_root=profile_root,
                use_persistent_intelligence=args.enable_persistent_intelligence,
            )

        check = validate_phase4_dir(
            vod_id,
            phase4_dir,
            min_frames=1,
            require_speaker_id=args.require_speaker_id,
            speaker_artifact_path=speaker_artifact_path,
        )
        for line in check.info:
            print(f"INFO: {line}")
        for line in check.warnings:
            print(f"WARN: {line}")
        for line in check.errors:
            print(f"ERROR: {line}")

        if not check.ok:
            print("[prepare_phase4] FAIL: phase4 validation failed")
            return 1

        print("[prepare_phase4] OK: phase4 directory is ready")
        return 0

    except Exception as exc:  # noqa: BLE001
        print(f"[prepare_phase4] ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
