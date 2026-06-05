"""
VOD Lens — Scene Detection Module

Detects scene changes in a VOD using the PySceneDetect Python API.
This avoids depending on the `scenedetect` CLI being available on PATH,
which is brittle on user-managed WSL installs where the package may be
installed but only exposed under ~/.local/bin.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Optional

from src.preprocessing.types import SceneClip


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
        downscale: Kept for compatibility. When truthy, enables PySceneDetect's
            automatic downscaling. When None/False, disables auto-downscale.
        method: Detection method ("content" or "adaptive")

    Returns:
        List of SceneClip objects between detected boundaries

    Raises:
        FileNotFoundError: If video file doesn't exist
        ImportError: If scenedetect is not installed
        RuntimeError: If detection fails
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        from scenedetect import AdaptiveDetector, ContentDetector, SceneManager, open_video
    except ImportError as exc:
        raise ImportError(
            "PySceneDetect is required. Install: pip install scenedetect"
        ) from exc

    video_path = Path(video_path)
    start_time = time.time()

    try:
        video = open_video(str(video_path))
        scene_manager = SceneManager()

        # `downscale` used to be passed to the CLI as a hint like "480p".
        # In the Python API we preserve the intent by toggling PySceneDetect's
        # automatic downscaling instead of requiring any external binary/path setup.
        if hasattr(scene_manager, "auto_downscale"):
            scene_manager.auto_downscale = bool(downscale)
        if not downscale and hasattr(scene_manager, "downscale"):
            scene_manager.downscale = 1

        if method == "adaptive":
            detector = AdaptiveDetector(adaptive_threshold=threshold)
        else:
            detector = ContentDetector(threshold=threshold)

        scene_manager.add_detector(detector)
        scene_manager.detect_scenes(video=video, show_progress=False)
        scene_list = scene_manager.get_scene_list(start_in_scene=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"PySceneDetect failed: {exc}") from exc
    finally:
        # Backends vary; close/release when available.
        try:
            if "video" in locals():
                if hasattr(video, "release"):
                    video.release()
                elif hasattr(video, "reset"):
                    video.reset()
        except Exception:
            pass

    scenes = _scene_list_to_clips(scene_list)

    elapsed = time.time() - start_time
    _ = elapsed  # retained for parity/debuggability if timing logs are added later
    return scenes


def _scene_list_to_clips(scene_list) -> list[SceneClip]:
    """Convert PySceneDetect scene tuples into SceneClip contracts."""
    scenes: list[SceneClip] = []
    for i, (start, end) in enumerate(scene_list):
        start_s = float(start.get_seconds())
        end_s = float(end.get_seconds())
        scenes.append(
            SceneClip(
                index=i,
                start=start_s,
                end=end_s,
                duration=max(0.0, end_s - start_s),
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
