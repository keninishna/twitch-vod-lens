"""Chat analysis module.

Downloads VOD chat via yt-dlp embedded chat or Twitch API,
analyzes for activity spikes, emotes, and notable messages.
"""

import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np


def download_chat(vod_id: str, output_path: str) -> list[dict]:
    """Download chat for a VOD using yt-dlp's built-in chat download.

    Falls back to TwitchDownloaderCLI if available.

    Args:
        vod_id: Twitch VOD ID.
        output_path: Path to save raw chat JSON.

    Returns:
        list[dict]: Chat messages with content_offset_seconds and message data.
    """
    # Try yt-dlp with --write-chat first
    result = subprocess.run(
        ["yt-dlp", "--write-chat", "--skip-download",
         "-o", output_path.replace(".json", ""),
         f"https://www.twitch.tv/videos/{vod_id}"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        chat_file = Path(str(output_path).replace(".json", "") + ".chat.json")
        if chat_file.exists():
            return json.loads(chat_file.read_text())

    # Fallback: try TwitchDownloaderCLI
    try:
        subprocess.run(
            ["TwitchDownloaderCLI", "chatdownload",
             "--id", vod_id, "--output", output_path,
             "--embed-images", "--timestamp-format", "Relative"],
            check=True, capture_output=True, text=True,
        )
        with open(output_path) as f:
            data = json.load(f)
        return data.get("comments", [])
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # If all fails, return empty
    return []


def analyze_chat(vod_id: str) -> dict:
    """Download and analyze chat for a VOD.

    Returns:
        dict: {
            "total_messages": int,
            "total_emotes": int,
            "spikes": list[dict],
            "top_emotes": list[dict]
        }
    """
    chat_path = f"/tmp/{vod_id}_chat.json"
    messages = download_chat(vod_id, chat_path)

    if not messages:
        return {
            "total_messages": 0,
            "total_emotes": 0,
            "spikes": [],
            "top_emotes": [],
        }

    # Compute message density in 10-second windows
    window = 10  # seconds
    max_time = max(m["content_offset_seconds"] for m in messages) if messages else 0
    bins = int(max_time / window) + 1

    counts = [0] * bins
    all_emotes = []

    for msg in messages:
        idx = int(msg["content_offset_seconds"] / window)
        counts[idx] += 1
        # Count emotes from message fragments
        fragments = msg.get("message", {}).get("fragments", [])
        for frag in fragments:
            if frag.get("emote"):
                all_emotes.append(frag["emote"]["id"])

    # Find spikes using z-score
    mean = np.mean(counts)
    std = np.std(counts)
    spikes = []
    for i, count in enumerate(counts):
        if std > 0:
            z = (count - mean) / std
            if z > 2.0:  # 2 standard deviations = significant spike
                spikes.append({
                    "start": i * window,
                    "end": (i + 1) * window,
                    "intensity": float(z),
                    "message_count": count,
                })

    # Count unique emote occurrences
    emote_counts = Counter(all_emotes)
    top_emotes = [{"id": k, "count": v}
                  for k, v in emote_counts.most_common(10)]

    return {
        "total_messages": len(messages),
        "total_emotes": len(all_emotes),
        "spikes": spikes,
        "top_emotes": top_emotes,
    }


def analyze_chat_to_file(vod_id: str, output_path: str) -> int:
    """Analyze chat and write results to JSON file.

    Returns:
        int: Number of spikes found.
    """
    result = analyze_chat(vod_id)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    return len(result["spikes"])
