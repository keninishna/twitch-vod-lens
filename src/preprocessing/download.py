"""VOD downloader module.

Downloads a Twitch VOD via yt-dlp. Produces audio-only MP3 for
Whisper and low-res MP4 for scene detection.
"""

import json
import subprocess
from pathlib import Path


def download_vod(vod_url: str, output_dir: Path) -> dict:
    """Download VOD audio + low-res video.

    Args:
        vod_url: Full Twitch VOD URL (e.g. https://www.twitch.tv/videos/123456789)
        output_dir: Directory to write downloads into.

    Returns:
        dict: VOD metadata (title, game, duration, etc.)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get metadata first
    result = subprocess.run(
        ["yt-dlp", "--dump-json", vod_url],
        capture_output=True, text=True, check=True,
    )
    meta = json.loads(result.stdout)

    # Twitch metadata id is often prefixed (e.g. v2782308109), but downstream
    # legacy pipeline expects numeric VOD ID from URL for filenames.
    vid = vod_url.strip("/").split("/")[-1]

    # Download audio for Whisper
    subprocess.run(
        ["yt-dlp", "-x", "--audio-format", "mp3",
         "-o", f"{output_dir}/{vid}.%(ext)s", vod_url],
        check=True,
        capture_output=True,
    )

    # Download low-res video for scene detection (480p)
    subprocess.run(
        ["yt-dlp", "-f", "best[height<=480]/best",
         "-o", f"{output_dir}/{vid}_video.%(ext)s", vod_url],
        check=True,
        capture_output=True,
    )

    # Write metadata
    meta_path = output_dir / "vod_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta
