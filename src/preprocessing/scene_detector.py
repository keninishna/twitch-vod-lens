"""
VOD Lens — Scene Detection Module

Detects scene changes in a VOD using PySceneDetect content-aware
analysis. Downscales to 480p for performance on long VODs.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Optional

from src.preprocessing.types import SceneBoundary, SceneClip


def detect_scenes(
    video_path: Path,
    threshold: float = 12.0,
    downscale: Optional[str] = "480p",
    method: str = "content",
) -> list[SceneClip]:
    """
    Detect scene changes in a VOD video.

    Args:
        video_path: Path to video file (MP4)
        threshold: Detection threshold (lower = more sensitive, default 12.0)
        downscale: Downscale resolution for faster processing (None = full res)
        method: Detection method ("content" for content-aware, "adaptive" for adaptive)

    Returns:
        List of SceneClip objects between detected boundaries

    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If detection fails
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_path = Path(video_path)
    start_time = time.time()

    # Use scenedetect CLI
    stats_path = video_path.parent / "scenes_stats.csv"
    scene_list_path = video_path.parent / "scenes.json"

    cmd = [
        "scenedetect",
        "--input", str(video_path),
        "--output", str(video_path.parent),
    ]

    if downscale:
        cmd.extend(["--downscale", downscale])

    if method == "content":
        cmd.extend(["detect-content", "--threshold", str(threshold)])
    elif method == "adaptive":
        cmd.extend(["detect-adaptive", "--threshold", str(threshold)])
    else:
        cmd.extend(["detect-content", "--threshold", str(threshold)])

    cmd.extend([
        "list-scenes",
        "--output", str(video_path.parent),
        "--filename", "scenes",
    ])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    if result.returncode != 0:
        raise RuntimeError(
            f"PySceneDetect failed (exit {result.returncode}): {result.stderr[:500]}"
        )

    # Parse the scene list CSV/JSON output
    scenes = _parse_scene_list(video_path.parent)

    # Clean up stats file
    stats_path.unlink(missing_ok=True)

    elapsed = time.time() - start_time

    return scenes


def _parse_scene_list(output_dir: Path) -> list[SceneClip]:
    """Parse scene list from scenedetect output."""
    # Try scene list CSV first
    csv_files = list(output_dir.glob("*scenes*.csv"))
    if csv_files:
        return _parse_csv_scenes(csv_files[0])

    # Fall back to parsing scenedetect stdout
    return []


def _parse_csv_scenes(csv_path: Path) -> list[SceneClip]:
    """Parse scene list CSV into SceneClip objects."""
    scenes: list[SceneClip] = []
    import csv

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                start = float(row.get("Start Timecode (seconds)", row.get("Start (s)", 0)))
                end = float(row.get("End Timecode (seconds)", row.get("End (s)", 0)))
            except (ValueError, TypeError):
                continue

            scenes.append(
                SceneClip(
                    index=i,
                    start=start,
                    end=end,
                    duration=end - start,
                )
            )

    return scenes


def detect_with_ffmpeg(video_path: Path, scene_threshold: float = 0.3) -> list[SceneClip]:
    """
    Alternative scene detection using ffmpeg's scene detection filter.
    Faster than PySceneDetect but less accurate.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-filter:v", f"select='gt(scene,{scene_threshold})',showinfo",
        "-f", "null",
        "-",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    scenes: list[SceneClip] = []
    import re

    pattern = r"pts_time:([\d.]+)"
    times = [float(m) for m in re.findall(pattern, result.stderr)]

    # Convert timestamps to SceneClips
    for i, ts in enumerate(times):
        end = times[i + 1] if i + 1 < len(times) else ts + 300  # default 5min
        scenes.append(
            SceneClip(
                index=i,
                start=ts,
                end=end,
                duration=end - ts,
            )
        )

    return scenes
