#!/usr/bin/env python3
"""Cross-reference YOLO detections with fusion JSON to find visually-verified clip candidates."""

import json
import os
from collections import defaultdict

VOD_ID = "phase4_2770929139"
BASE = os.path.expanduser(f"~/twitch-vod-analyzer/vods/{VOD_ID}/")
YOLO_PATH = os.path.join(BASE, "yolo_detections.json")
FUSION_PATH = os.path.join(BASE, "fusion_result.json")
OUTPUT_PATH = os.path.join(BASE, "clip_candidates.json")

# Object categories we track
FOOD_CLASSES = {"bowl", "banana", "apple", "sandwich", "orange", "carrot",
                "hot dog", "pizza", "donut", "cake", "cup", "bottle"}
PERSON_CLASSES = {"person"}
DEVICE_CLASSES = {"cell phone", "laptop", "tv/monitor", "book", "mouse", "remote"}
ANIMAL_CLASSES = {"cat", "dog", "bird", "teddy bear", "horse"}


def classify_objects(detections):
    """Categorize a frame's detections into groups."""
    cats = set()
    for d in detections:
        cn = d["class_name"].lower()
        if cn in FOOD_CLASSES:
            cats.add("food")
        if cn in PERSON_CLASSES:
            cats.add("person")
        if cn in DEVICE_CLASSES:
            cats.add("device/screen")
        if cn in ANIMAL_CLASSES:
            cats.add("animal")
    return cats


def frame_to_segment(timestamp_sec, segments):
    """Find which transcript segment a frame timestamp falls into."""
    for seg in segments:
        start = seg.get("start", 0)
        end = seg.get("end", start + 30)
        if start <= timestamp_sec < end:
            return seg
    return None


def compute_visual_score(categories, segment):
    """Score how well visual content matches transcript topic."""
    score = 0.0
    topics = set()
    if "topic" in segment:
        t = segment.get("topic", "").lower()
        topics.add(t)

    # Match transcript topics with visual categories
    topic_text = " ".join(topics).lower()

    if "food" in categories and any(w in topic_text for w in ["food", "eat", "snack", "cook", "meal", "dinner", "lunch", "breakfast", "drink", "coffee", "tea"]):
        score += 3.0
    if "person" in categories and any(w in topic_text for w in ["person", "people", "face", "someone", "i", "me", "we", "our", "my"]):
        score += 2.0
    if "device/screen" in categories and any(w in topic_text for w in ["screen", "computer", "phone", "laptop", "code", "type", "watch", "video"]):
        score += 2.0
    if "animal" in categories and any(w in topic_text for w in ["pet", "dog", "cat", "animal", "cute"]):
        score += 3.0

    # Boost for high emotion segments (more interesting clips)
    emotion = segment.get("emotion", "").lower()
    if emotion in ["happy", "excited", "funny", "surprised"] and "person" in categories:
        score += 2.0
    if emotion in ["sad", "frustrated", "angry"] and "person" in categories:
        score += 1.5  # dramatic moments

    # Boost for novelty/interesting content
    if segment.get("novelty", False):
        score += 1.0

    return score


def main():
    print(f"Loading YOLO detections from {YOLO_PATH}")
    with open(YOLO_PATH) as f:
        yolo_data = json.load(f)

    print(f"Loading fusion results from {FUSION_PATH}")
    with open(FUSION_PATH) as f:
        fusion = json.load(f)

    segments = fusion.get("segments", [])
    if not segments:
        segments = fusion.get("transcript_segments", [])

    print(f"YOLO: {yolo_data['total_frames']} frames, {yolo_data['total_objects']} objects")
    print(f"Fusion: {len(segments)} segments")

    # Build frame -> categories lookup
    frame_data = {}
    for fname, fdata in yolo_data["results"].items():
        ts = fdata["timestamp_sec"]
        cats = classify_objects(fdata["detections"])
        frame_data[ts] = {
            "categories": cats,
            "num_objects": fdata["num_objects"],
            "frame": fname
        }

    # Score each segment
    scored_segments = []
    for seg in segments:
        start = seg.get("start", 0)
        end = seg.get("end", start + 30)

        # Collect all frames in this segment's time range
        seg_frames = []
        seg_categories = set()
        for ts in range(int(start), int(end), 5):
            if ts in frame_data:
                fd = frame_data[ts]
                seg_frames.append(fd)
                seg_categories.update(fd["categories"])

        if not seg_frames:
            continue

        visual_score = compute_visual_score(seg_categories, seg)

        # Base clip quality score: emotion + visual alignment
        base_score = len(seg_frames) / 6.0  # length factor (max 1.0 for full 30s segment)
        emotion_weight = {"happy": 3, "excited": 4, "funny": 5, "surprised": 3,
                          "sad": 2, "frustrated": 2, "angry": 2, "neutral": 1}.get(
            seg.get("emotion", "neutral").lower(), 1)

        total_score = visual_score + (emotion_weight * 0.5) + base_score

        scored_segments.append({
            "start": start,
            "end": end,
            "topic": seg.get("topic", ""),
            "emotion": seg.get("emotion", ""),
            "summary": seg.get("summary", seg.get("text", ""))[:200],
            "visual_categories": list(seg_categories),
            "visual_score": round(visual_score, 1),
            "emotion_weight": emotion_weight,
            "total_score": round(total_score, 1),
            "frames_sampled": len(seg_frames),
            "objects_in_segment": sum(f["num_objects"] for f in seg_frames)
        })

    # Sort by total score, descending
    scored_segments.sort(key=lambda x: x["total_score"], reverse=True)

    # Select top 10
    top_10 = scored_segments[:10]

    # Format output
    candidates = {
        "vod_id": VOD_ID,
        "total_segments_scored": len(scored_segments),
        "top_candidates": []
    }

    for i, seg in enumerate(top_10, 1):
        # Ensure clip boundaries are sensible (min 30s, max 120s)
        clip_duration = min(max(seg["end"] - seg["start"], 30), 120)
        clip_start = max(seg["start"], 0)
        clip_end = clip_start + clip_duration

        # Output format for yt-dlp --download-sections
        candidate = {
            "rank": i,
            "score": seg["total_score"],
            "time_range": f"*{int(clip_start)}-{int(clip_end)}",
            "start_sec": int(clip_start),
            "end_sec": int(clip_end),
            "duration_sec": int(clip_end - clip_start),
            "topic": seg["topic"],
            "emotion": seg["emotion"],
            "summary": seg["summary"],
            "visual_content": ", ".join(seg["visual_categories"]),
            "visual_score": seg["visual_score"],
            "objects_detected": seg["objects_in_segment"]
        }
        candidates["top_candidates"].append(candidate)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(candidates, f, indent=2)

    print(f"\nTop 10 Clip Candidates:")
    print(f"{'#':>3} {'Score':>6} {'Time':>12} {'Emotion':>10} {'Visual':>20} {'Topic'}")
    print("-" * 100)
    for c in candidates["top_candidates"]:
        print(f"{c['rank']:>3} {c['score']:>6.1f} {c['time_range']:>12} "
              f"{c['emotion']:>10} {c['visual_content']:>20} {c['topic'][:40]}")

    print(f"\nSaved to: {OUTPUT_PATH}")
    return candidates


if __name__ == "__main__":
    main()
