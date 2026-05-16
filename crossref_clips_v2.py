#!/usr/bin/env python3
"""Cross-reference YOLO detections with fusion JSON to find visually-verified clip candidates.
This version works with the actual fusion JSON structure from VOD Lens."""

import json
import os
import sys
from collections import defaultdict

VOD_ID = "phase4_2770929139"
BASE = os.path.expanduser(f"~/twitch-vod-analyzer/vods/{VOD_ID}/")
YOLO_PATH = os.path.join(BASE, "yolo_detections.json")
FUSION_PATH = os.path.join(BASE, "fusion_result.json")
OUTPUT_PATH = os.path.join(BASE, "clip_candidates.json")

# Object categories we track for a lofi study stream
FOOD_DRINK = {"bowl", "banana", "apple", "sandwich", "orange", "carrot",
              "hot dog", "pizza", "donut", "cake", "cup", "bottle", "dining table"}
PERSON = {"person"}
DEVICES = {"cell phone", "laptop", "tv", "mouse", "remote", "keyboard", "book"}
PETS = {"cat", "dog", "bird", "teddy bear"}
INTERESTING = {"vase", "clock", "scissors", "hair drier", "toothbrush"}


def categorize_detections(detections):
    """Categorize a frame's detections into meaningful groups for a study stream."""
    cats = set()
    for d in detections:
        cn = d["class_name"].lower()
        if cn in FOOD_DRINK:
            cats.add("food/drink")
        if cn in PERSON:
            cats.add("person")
        if cn in DEVICES:
            cats.add("device")
        if cn in PETS:
            cats.add("pet")
        if cn in INTERESTING:
            cats.add("object")
    return cats


def main():
    with open(YOLO_PATH) as f:
        yolo_data = json.load(f)

    with open(FUSION_PATH) as f:
        fusion = json.load(f)

    # Get transcript segments
    transcript_segs = fusion.get("transcript", {}).get("segments", [])
    scenes = fusion.get("scenes", [])
    timeline = fusion.get("timeline", [])
    chat = fusion.get("chat", {})

    print(f"YOLO: {yolo_data['total_frames']} frames, {yolo_data['total_objects']} objects")
    print(f"Transcript segments: {len(transcript_segs)}")
    print(f"Scenes: {len(scenes)}")
    print(f"Timeline entries: {len(timeline)}")

    # Build frame -> categories lookup (timestamp in seconds)
    frame_map = {}
    for fname, fdata in yolo_data["results"].items():
        ts = fdata["timestamp_sec"]
        cats = categorize_detections(fdata["detections"])
        frame_map[ts] = {
            "categories": cats,
            "num_objects": fdata["num_objects"],
            "frame": fname
        }

    # Build scene-level visual summary
    scene_visuals = {}
    for scene in scenes:
        s_start = int(scene["start"])
        s_end = int(scene["end"])
        cats = set()
        obj_count = 0
        frames_seen = 0
        for ts in range(s_start, s_end, 5):
            if ts in frame_map:
                cats.update(frame_map[ts]["categories"])
                obj_count += frame_map[ts]["num_objects"]
                frames_seen += 1
        scene_visuals[scene["index"]] = {
            "categories": cats,
            "total_objects": obj_count,
            "frames_with_objects": frames_seen,
            "duration": scene["end"] - scene["start"],
            "label": scene.get("label", ""),
            "start": s_start,
            "end": s_end
        }

    # Build chat activity index: for each timestamp, find chat activity level
    # Timeline entries contain chat_intensity
    chat_intensity = {}
    for entry in timeline:
        ts = int(entry["timestamp"])
        chat_intensity[ts] = {
            "intensity": entry.get("chat_intensity", 0),
            "text": entry.get("transcript", "")
        }

    # Score clip candidates
    candidates = []
    MIN_CLIP_DURATION = 30
    MAX_CLIP_DURATION = 120

    for scene in scenes:
        s_start = int(scene["start"])
        s_end = int(scene["end"])
        s_label = scene.get("label", "unknown")

        # Skip very short scenes or very long quiet scenes
        duration = s_end - s_start
        if duration < MIN_CLIP_DURATION:
            continue

        # Get visual info for this scene
        vis = scene_visuals.get(scene["index"], {})
        categories = vis.get("categories", set())
        obj_count = vis.get("total_objects", 0)

        # Get chat intensity
        chat_peaks = [v for t, v in chat_intensity.items()
                      if s_start <= t <= s_end and v["intensity"] > 0]
        avg_chat = sum(c["intensity"] for c in chat_peaks) / max(len(chat_peaks), 1)

        # Get transcript text in this scene
        scene_text = " ".join([
            seg["text"] for seg in transcript_segs
            if s_start <= seg.get("start", 0) <= s_end
        ])

        # Scoring
        score = 0.0

        # Visual alignment score
        has_person = "person" in categories
        has_food = "food/drink" in categories
        has_device = "device" in categories
        has_pet = "pet" in categories
        objects_present = len(categories)

        if has_person and s_label in ("interaction", "intro", "outro"):
            score += 4.0  # Streamer visible during interactive moments
        if has_food:
            score += 3.0  # Food/drink visible
        if has_pet:
            score += 5.0  # Pet on stream = great clip
        if objects_present >= 2:
            score += 2.0  # Multiple objects = visually interesting
        if obj_count > 20:
            score += 2.0  # High object density

        # Chat activity bonus
        if avg_chat > 0.5:
            score += 3.0
        elif avg_chat > 0.1:
            score += 1.0

        # Scene label bonus
        label_score = {"interaction": 3, "intro": 2, "outro": 2, "highlight": 4,
                       "funny": 4, "quiet": -1}.get(s_label, 0)
        score += label_score

        # Speaking bonus - if the streamer is actually talking
        if scene_text.strip():
            score += 2.0

        # Duration bonus: prefer 30-90s clips
        if 30 <= duration <= 90:
            score += 1.0

        candidates.append({
            "scene_index": scene["index"],
            "start": s_start,
            "end": s_end,
            "duration": duration,
            "label": s_label,
            "score": round(score, 1),
            "visual_categories": list(categories),
            "objects_detected": obj_count,
            "has_person": has_person,
            "has_food": has_food,
            "has_pet": has_pet,
            "has_device": has_device,
            "avg_chat_intensity": round(avg_chat, 2),
            "speaking": bool(scene_text.strip()),
            "text_preview": scene_text[:150] if scene_text else ""
        })

    # Sort by score descending, take top 10
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top_10 = candidates[:10]

    print(f"\nAnalyzed {len(candidates)} scene candidates, top 10 selected.")
    print(f"\n{'#':>3} {'Score':>6} {'Duration':>9} {'Label':>12} {'Visual':>22} {'Chat':>6} Speak  Summary")
    print("=" * 110)
    for i, c in enumerate(top_10, 1):
        vis = ", ".join(c["visual_categories"][:3]) if c["visual_categories"] else "empty"
        summary = c["text_preview"][:50] if c["text_preview"] else c["label"]
        print(f"{i:>3} {c['score']:>6.1f} {c['start']:>5}-{c['end']:<4}s "
              f"{c['label']:>12} {vis:<22} {c['avg_chat_intensity']:>6.2f} "
              f"{'Y' if c['speaking'] else 'N':>4}  {summary}")

    # Output clip candidates for extraction
    result = {
        "vod_id": VOD_ID,
        "total_scenes_analyzed": len(scenes),
        "total_candidates": len(candidates),
        "top_candidates": []
    }

    for i, c in enumerate(top_10, 1):
        clip_start = max(c["start"], 0)
        clip_duration = min(max(c["duration"], 30), 120)
        clip_end = clip_start + clip_duration

        result["top_candidates"].append({
            "rank": i,
            "score": c["score"],
            "time_range": f"*{clip_start}-{clip_end}",
            "start_sec": clip_start,
            "end_sec": clip_end,
            "duration_sec": clip_end - clip_start,
            "label": c["label"],
            "visual_content": ", ".join(c["visual_categories"]),
            "chat_intensity": c["avg_chat_intensity"],
            "has_speaking": c["speaking"],
            "summary": c["text_preview"][:200] if c["text_preview"] else c["label"]
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved clip candidates to: {OUTPUT_PATH}")
    return result


if __name__ == "__main__":
    main()
