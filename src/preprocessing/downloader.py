"""
VOD Lens — VOD Downloader Module

Downloads VOD audio from Twitch URLs using yt-dlp.
Produces: audio file (16kHz mono WAV) + VodMeta metadata.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

from src.preprocessing.types import VodMeta


def download_vod(
    url: str,
    output_dir: Path,
    audio_only: bool = True,
    format_spec: Optional[str] = None,
) -> tuple[Path, VodMeta]:
    """
    Download a Twitch VOD. Returns (audio_path, metadata).

    Args:
        url: Twitch VOD URL (e.g. https://www.twitch.tv/videos/123456789)
        output_dir: Directory to save downloaded files
        audio_only: If True, download audio only (smaller, faster)
        format_spec: yt-dlp format string override

    Returns:
        Tuple of (path to audio WAV file, VodMeta object)

    Raises:
        RuntimeError: If yt-dlp fails
        FileNotFoundError: If output file is missing
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine format
    if format_spec:
        fmt = format_spec
    elif audio_only:
        fmt = "bestaudio/best"
    else:
        # Twitch VODs often expose muxed ladder formats (e.g. 720p60-1) rather than
        # separate bestvideo+bestaudio tracks; requesting separate tracks can fail.
        fmt = "best[height<=720]/best"

    output_template = str(output_dir / "vod_input.%(ext)s")

    # Step 1: Download
    cmd = [
        "yt-dlp",
        "-f", fmt,
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--print-json",  # outputs metadata JSON to stdout
        "--no-playlist",
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (exit {result.returncode}): {result.stderr[:500]}"
        )

    # Parse metadata from --print-json (last JSON line of stdout)
    meta = _parse_metadata(result.stdout, url)

    if audio_only:
        # Step 2: Convert to 16kHz mono WAV for Whisper
        input_file = output_dir / f"vod_input.{meta.format or 'mp4'}"
        if not input_file.exists():
            # yt-dlp may output with different extension
            input_file = _find_downloaded_file(output_dir, "vod_input")

        audio_path = output_dir / "audio.wav"
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", str(input_file),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(audio_path),
        ]
        subprocess.run(ffmpeg_cmd, capture_output=True, check=True, timeout=300)

        # Clean up the original download to save space
        input_file.unlink(missing_ok=True)

        return audio_path, meta

    # Not audio-only - return the video file
    video_file = _find_downloaded_file(output_dir, "vod_input")
    return video_file, meta


def extract_vod_id(url: str) -> str:
    """Extract Twitch VOD ID from a URL."""
    parts = url.rstrip("/").split("/")
    return parts[-1]


def _parse_metadata(stdout: str, url: str) -> VodMeta:
    """Parse yt-dlp JSON metadata into VodMeta."""
    # yt-dlp --print-json outputs one JSON line per video
    for line in stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    else:
        # Fallback: minimal metadata from URL
        return VodMeta(
            id=extract_vod_id(url),
            title=f"VOD_{extract_vod_id(url)}",
            duration_seconds=0,
            url=url,
            streamer="unknown",
        )

    return VodMeta(
        id=extract_vod_id(url),
        title=data.get("title", f"VOD_{extract_vod_id(url)}"),
        duration_seconds=int(data.get("duration", 0)),
        url=url,
        streamer=data.get("uploader", "unknown"),
        resolution=data.get("resolution"),
        fps=data.get("fps"),
        format=data.get("ext", "mp4"),
    )


def _find_downloaded_file(directory: Path, prefix: str) -> Path:
    """Find a file starting with prefix in the directory."""
    for f in directory.iterdir():
        if f.name.startswith(prefix) and f.is_file():
            return f
    # Try with glob
    matches = list(directory.glob(f"{prefix}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No file starting with '{prefix}' in {directory}")


def cleanup(output_dir: Path) -> None:
    """Remove all downloaded files from the output directory."""
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)
