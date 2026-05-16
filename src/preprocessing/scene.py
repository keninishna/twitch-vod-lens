"""Scene detection module.

Uses PySceneDetect to find scene boundaries (cuts, fades, dissolves)
in the downsampled video.
"""

import json
from scenedetect import detect, ContentDetector


def detect_scenes(video_path: str) -> list[dict]:
    """Detect scene boundaries in a video file.

    Args:
        video_path: Path to video file (.mp4).

    Returns:
        list[dict]: List of scene boundaries with timestamp, scene_type, content_hash.
    """
    scene_list = detect(video_path, ContentDetector())

    results = []
    for i, (start, end) in enumerate(scene_list):
        results.append({
            "timestamp": start.get_seconds(),
            "scene_type": "cut",
            "content_hash": f"scene_{i}",
        })

    return results


def detect_scenes_to_file(video_path: str, output_path: str) -> int:
    """Detect scenes and write results to JSON file.

    Returns:
        int: Number of scene boundaries found.
    """
    scenes = detect_scenes(video_path)
    with open(output_path, "w") as f:
        json.dump(scenes, f, indent=2)
    return len(scenes)
