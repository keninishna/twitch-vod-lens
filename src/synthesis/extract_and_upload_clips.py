#!/usr/bin/env python3
"""
extract_and_upload_clips.py

End-to-end clip extraction and Nextcloud upload for the VOD Lens pipeline.

Reads qwen_vision_progressive.json output, filters high-scoring clips,
extracts them with browser-compatible ffmpeg settings, uploads to Nextcloud
via WebDAV, and creates public share links via the OCS API.

Usage:
    python extract_and_upload_clips.py --json path/to/qwen_vision_progressive.json \
        [--vod path/to/raw_vod.mp4] --min-score 7 --output-dir ./clips

If --vod is omitted, the script auto-detects the raw MP4 from common phase4 paths.

Environment:
    NEXTCLOUD_USER       (default: john)
    NEXTCLOUD_PASSWORD   (default: NextcloudFan!2025)
    NEXTCLOUD_BASE_URL   (default: https://files.washingtondcspirit.com)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_BASE_URL = "https://files.washingtondcspirit.com"
DEFAULT_USER = "john"
DEFAULT_PASSWORD = "NextcloudFan!2025"


def log(msg: str):
    print(msg, flush=True)


def run(cmd: list[str] | str, timeout: int = 300, check: bool = True) -> str:
    """Run a shell command and return stdout."""
    kwargs = {
        "shell": isinstance(cmd, str),
        "capture_output": True,
        "text": True,
        "timeout": timeout,
    }
    result = subprocess.run(cmd, **kwargs)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}):\n{result.stderr}"
        )
    return result.stdout


def load_pipeline_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_clips(data: dict, min_score: int) -> list[dict]:
    """Return final_selected_clips with score >= min_score."""
    clips = data.get("final_ranking", {}).get("final_selected_clips", [])
    return [c for c in clips if c.get("score", 0) >= min_score]


def _extract_vod_id(data: dict, json_path: Path) -> str | None:
    """Best-effort VOD ID extraction from output JSON/path."""
    v = data.get("vod_id")
    if isinstance(v, str) and v.strip():
        return v.strip()

    # phase4_<VOD_ID> directory naming
    parent = json_path.parent.name
    if parent.startswith("phase4_") and len(parent) > len("phase4_"):
        return parent[len("phase4_"):]

    # fallback: find first 8+ digit token in filename/path
    m = re.search(r"(\d{8,})", str(json_path))
    if m:
        return m.group(1)

    return None


def _metadata_raw_vod_candidates(fusion_data: dict | None, phase4_dir: Path) -> list[Path]:
    if not isinstance(fusion_data, dict):
        return []

    vod_meta = fusion_data.get("vod_meta") if isinstance(fusion_data.get("vod_meta"), dict) else {}

    raw_values: list[str] = []
    for key in ("source_video", "raw_vod_path", "vod_path"):
        value = vod_meta.get(key)
        if isinstance(value, str) and value.strip():
            raw_values.append(value.strip())

    top_level = fusion_data.get("raw_vod_path")
    if isinstance(top_level, str) and top_level.strip():
        raw_values.append(top_level.strip())

    candidates: list[Path] = []
    for raw in raw_values:
        p = Path(raw).expanduser()
        if p.is_absolute():
            candidates.append(p.resolve())
        else:
            candidates.append((phase4_dir / p).resolve())
    return candidates


def resolve_raw_vod_path(
    vod_id: str | None,
    phase4_dir: Path,
    explicit_path: Path | None = None,
    fusion_data: dict | None = None,
) -> Path:
    """Resolve raw MP4 path in canonical order for Task 28."""
    checked: list[Path] = []

    # 1) explicit --vod path
    if explicit_path is not None:
        explicit_candidate = explicit_path.expanduser().resolve()
        checked.append(explicit_candidate)
        if explicit_candidate.exists():
            return explicit_candidate

    # 2) phase4_<VOD_ID>/raw/<VOD_ID>.mp4
    if vod_id:
        phase4_candidate = (phase4_dir / "raw" / f"{vod_id}.mp4").resolve()
        checked.append(phase4_candidate)
        if phase4_candidate.exists():
            return phase4_candidate

    # 3) raw path from fusion metadata
    for candidate in _metadata_raw_vod_candidates(fusion_data, phase4_dir):
        checked.append(candidate)
        if candidate.exists():
            return candidate

    # 4) legacy repo-level raw/<VOD_ID>.mp4
    if vod_id:
        repo_root = Path(__file__).resolve().parents[2]
        legacy_candidate = (repo_root / "raw" / f"{vod_id}.mp4").resolve()
        checked.append(legacy_candidate)
        if legacy_candidate.exists():
            log(f"WARN: using legacy repo-level raw path: {legacy_candidate}")
            return legacy_candidate

    searched = "\n".join(f"  - {p}" for p in checked) if checked else "  - (no candidates)"
    raise FileNotFoundError(
        "Could not resolve raw VOD mp4.\n"
        "Checked:\n"
        f"{searched}\n"
        "Remediation: pass an explicit file with --vod /absolute/path/to/<VOD_ID>.mp4"
    )


def extract_clip(
    vod_path: str,
    start: float,
    end: float,
    output_path: str,
    scale: str = "854:480",
) -> None:
    """
    Extract a clip with browser-safe H.264 settings.

    Do NOT use -c copy — raw Twitch VODs are 852x480 (non-mod16) and break
    some browser decoders. Re-encode to 854x480 with +faststart.
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", vod_path,
        "-c:v", "libx264",
        "-profile:v", "main",
        "-level", "3.1",
        "-preset", "medium",
        "-crf", "18",
        "-vf", f"scale={scale}",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]
    log(f"  Extracting {start}s-{end}s → {output_path}")
    run(cmd, timeout=600)


def upload_webdav(local_path: str, remote_name: str) -> None:
    """Upload a file to Nextcloud root via WebDAV PUT."""
    user = os.environ.get("NEXTCLOUD_USER", DEFAULT_USER)
    pw = os.environ.get("NEXTCLOUD_PASSWORD", DEFAULT_PASSWORD)
    base = os.environ.get("NEXTCLOUD_BASE_URL", DEFAULT_BASE_URL).rstrip("/")

    url = f"{base}/remote.php/dav/files/{user}/{remote_name}"

    # If file exists, delete first to avoid stale-cache breakage
    run(["curl", "-s", "-u", f"{user}:{pw}", "-X", "DELETE", url], check=False)

    log(f"  Uploading {local_path} → {url}")
    run(
        [
            "curl", "-s", "-u", f"{user}:{pw}",
            "-X", "PUT", url,
            "--upload-file", local_path,
            "-w", "\nHTTP:%{http_code}\n",
        ],
        timeout=300,
    )


def create_share(remote_name: str) -> str:
    """Create a public read-only share and return the share URL."""
    user = os.environ.get("NEXTCLOUD_USER", DEFAULT_USER)
    pw = os.environ.get("NEXTCLOUD_PASSWORD", DEFAULT_PASSWORD)
    base = os.environ.get("NEXTCLOUD_BASE_URL", DEFAULT_BASE_URL).rstrip("/")

    api_url = f"{base}/ocs/v2.php/apps/files_sharing/api/v1/shares"

    log(f"  Creating share for /{remote_name}")
    resp = run(
        [
            "curl", "-s", "-u", f"{user}:{pw}",
            "-X", "POST", api_url,
            "-H", "OCS-APIRequest: true",
            "-d", f"path=/{remote_name}",
            "-d", "shareType=3",
            "-d", "permissions=1",
        ],
        timeout=60,
    )

    # Parse XML for <url>...
    match = re.search(r"<url>([^<]+)</url>", resp)
    if not match:
        raise RuntimeError(f"Could not parse share URL from response:\n{resp[:500]}")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract high-scoring clips and upload to Nextcloud"
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Path to qwen_vision_progressive.json",
    )
    parser.add_argument(
        "--vod",
        help="Path to raw VOD mp4 (optional; auto-detected if omitted)",
    )
    parser.add_argument(
        "--vod-id",
        help="Override VOD ID used for raw VOD/fusion path resolution",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=7,
        help="Minimum clip score to extract (default: 7)",
    )
    parser.add_argument(
        "--output-dir",
        default="./clips",
        help="Directory to store extracted clips (default: ./clips)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without extracting or uploading",
    )
    args = parser.parse_args()

    json_path = Path(args.json).resolve()
    out_dir = Path(args.output_dir)

    if not json_path.exists():
        print(f"Error: JSON not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    log(f"Loading pipeline output: {json_path}")
    data = load_pipeline_json(str(json_path))

    inferred_vod_id = _extract_vod_id(data, json_path)
    vod_id = args.vod_id.strip() if isinstance(args.vod_id, str) and args.vod_id.strip() else inferred_vod_id
    phase4_dir = json_path.parent
    explicit_vod = Path(args.vod) if args.vod else None

    fusion_data: dict | None = None
    if vod_id:
        fusion_path = phase4_dir / f"fusion_result_{vod_id}.json"
        if fusion_path.exists():
            try:
                with open(fusion_path, "r", encoding="utf-8") as f:
                    parsed = json.load(f)
                if isinstance(parsed, dict):
                    fusion_data = parsed
                else:
                    log(f"WARN: fusion metadata is not a JSON object: {fusion_path}")
            except (OSError, json.JSONDecodeError) as exc:
                log(f"WARN: could not parse fusion metadata {fusion_path}: {exc}")

    try:
        vod_path = resolve_raw_vod_path(
            vod_id=vod_id,
            phase4_dir=phase4_dir,
            explicit_path=explicit_vod,
            fusion_data=fusion_data,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        log(f"[DRY RUN] Resolved VOD path: {vod_path} (exists={vod_path.exists()})")

    if not vod_path.exists():
        print(f"Error: VOD not found: {vod_path}", file=sys.stderr)
        sys.exit(1)

    log(f"Using VOD file: {vod_path}")

    clips = filter_clips(data, args.min_score)
    if not clips:
        log(f"No clips scored >= {args.min_score}. Exiting.")
        sys.exit(0)

    log(f"Found {len(clips)} clip(s) with score >= {args.min_score}")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for clip in clips:
        rank = clip.get("rank", "?")
        start = clip.get("suggested_trim_start", clip.get("start", 0))
        end = clip.get("suggested_trim_end", clip.get("end", 0))
        score = clip.get("score", 0)
        title = clip.get("clip_point", "untitled")

        safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")[:40]
        filename = f"clip_rank{rank}_{safe_title}_{int(start)}s.mp4"
        local_path = out_dir / filename

        log(f"\n[Rank {rank}] {title}")
        log(f"  Score: {score} | Trim: {start}s-{end}s ({end-start:.0f}s)")

        if args.dry_run:
            log(f"  [DRY RUN] Would extract to {local_path}")
            log(f"  [DRY RUN] Would upload as {filename}")
            continue

        extract_clip(str(vod_path), start, end, str(local_path))
        upload_webdav(str(local_path), filename)
        share_url = create_share(filename)

        log(f"  ✅ Share link: {share_url}")
        results.append(
            {
                "rank": rank,
                "start": start,
                "end": end,
                "score": score,
                "title": title,
                "filename": filename,
                "share_url": share_url,
            }
        )

    if results and not args.dry_run:
        manifest = out_dir / "share_links.json"
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        log(f"\nManifest saved: {manifest}")

    log("\nDone.")


if __name__ == "__main__":
    main()
