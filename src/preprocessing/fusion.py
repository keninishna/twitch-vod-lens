"""Signal fusion engine.

Takes transcript, scenes, and chat spikes and computes weighted
interesting-moment scores. Merges overlapping events and ranks by score.
"""

import json


def fuse_signals(transcript_path: str, scenes_path: str,
                 chat_path: str, output_path: str) -> int:
    """Fuse all signals into scored moments.

    Args:
        transcript_path: Path to transcript.json.
        scenes_path: Path to scenes.json.
        chat_path: Path to chat_analysis.json.
        output_path: Path to write moments.json.

    Returns:
        int: Number of moments found.
    """
    with open(transcript_path) as f:
        transcript = json.load(f)
    with open(scenes_path) as f:
        scenes = json.load(f)
    with open(chat_path) as f:
        chat = json.load(f)

    moments = []

    # Chat spikes are strong signals
    for spike in chat.get("spikes", []):
        moments.append({
            "start": spike["start"],
            "end": spike["end"],
            "score": min(100, spike.get("intensity", 0) * 20),
            "signals": ["chat_spike"],
            "message_count": spike.get("message_count", 0),
        })

    # Scene changes that coincide with voice excitement
    for seg in transcript:
        text = seg.get("text", "")
        excitement = 0

        # ALL CAPS = excitement
        if text.isupper() and len(text) > 5:
            excitement += 20

        # Repeated punctuation = excitement
        if any(c * 3 in text.lower() for c in "!?."):
            excitement += 15

        # Short utterances = reactions (e.g. "LETS GO", "NO WAY")
        if len(text.split()) < 5:
            excitement += 10

        if excitement > 25:
            moments.append({
                "start": seg["start"],
                "end": seg["end"],
                "score": min(100, 50 + excitement),
                "signals": ["voice_excitement"],
                "text": text,
            })

    # Merge overlapping moments (within 5 seconds)
    moments.sort(key=lambda m: m["start"])
    merged = []
    for m in moments:
        if merged and m["start"] < merged[-1]["end"] + 5:
            merged[-1]["end"] = max(merged[-1]["end"], m["end"])
            merged[-1]["score"] = max(merged[-1]["score"], m["score"])
            # Merge signal lists
            merged[-1]["signals"] = list(set(merged[-1]["signals"] + m["signals"]))
        else:
            merged.append(m)

    # Sort by score descending
    merged.sort(key=lambda m: m["score"], reverse=True)

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    return len(merged)
