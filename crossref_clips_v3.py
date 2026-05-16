#!/usr/bin/env python3
"""Cross-reference YOLO detections with fusion JSON - v3 fixed for non-multiple-of-5 timestamps."""

import json
import os

VOD_ID = "phase4_2770929139"
BASE = os.path.expanduser(f"~/twitch-vod-analyzer/vods/{VOD_ID}/")
YOLO_PATH = os.path.join(BASE, "yolo_detections.json")
FUSION_PATH = os.path.join(BASE, "fusion_result.json")
OUTPUT_PATH = os.path.join(BASE, "clip_candidates.json")

FOOD_DRINK = {"bowl", "banana", "apple", "sandwich", "orange", "carrot",
              "hot dog", "pizza", "donut", "cake", "cup", "bottle", "dining table"}
PERSON = {"person"}
DEVICES = {"cell phone", "laptop", "tv", "mouse", "remote", "keyboard", "book", "tv/monitor"}
PETS = {"cat", "dog", "bird", "teddy bear"}
INTERESTING = {"vase", "clock", "scissors", "hair drier", "toothbrush"}


def categorize(detections):
    cats = set()
    for d in detections:
        cn = d["class_name"].lower()
        if cn in FOOD_DRINK: cats.add("food/drink")
        if cn in PERSON:     cats.add("person")
        if cn in DEVICES:    cats.add("device")
        if cn in PETS:       cats.add("pet")
        if cn in INTERESTING: cats.add("object")
    return cats


def main():
    with open(YOLO_PATH) as f:
        yolo_data = json.load(f)
    with open(FUSION_PATH) as f:
        fusion = json.load(f)

    transcript_segs = fusion.get("transcript", {}).get("segments", [])
    scenes = fusion.get("scenes", [])
    timeline = fusion.get("timeline", [])
    chat = fusion.get("chat", {})

    # Build frame lookup keyed by timestamp rounded to nearest 5
    frame_map = {}
    for fname, fdata in yolo_data["results"].items():
        ts = fdata["timestamp_sec"]
        frame_map[ts] = {
            "categories": categorize(fdata["detections"]),
            "num_objects": fdata["num_objects"],
            "frame": fname
        }

    # Build chat intensity lookup
    chat_intensity = {}
    for entry in timeline:
        ts = int(entry["timestamp"])
        chat_intensity[ts] = {
            "intensity": entry.get("chat_intensity", 0),
            "text": entry.get("transcript", "")
        }

    candidates = []

    for scene in scenes:
        s_start = int(scene["start"])
        s_end = int(scene["end"])
        s_label = scene.get("label", "unknown")
        duration = s_end - s_start

        if duration < 30:
            continue

        # Collect visual data, rounding to nearest 5s
        cats = set()
        obj_count = 0
        frames_seen = 0
        for ts in range(s_start - (s_start % 5), s_end, 5):
            if ts in frame_map:
                cats.update(frame_map[ts]["categories"])
                obj_count += frame_map[ts]["num_objects"]
                frames_seen += 1

        # Chat intensity
        chat_peaks = [v for t, v in chat_intensity.items()
                      if s_start <= t <= s_end and v["intensity"] > 0]
        avg_chat = sum(c["intensity"] for c in chat_peaks) / max(len(chat_peaks), 1)

        # Transcript text
        scene_text = " ".join([
            seg["text"] for seg in transcript_segs
            if s_start <= seg.get("start", 0) <= s_end
        ])

        # === SCORING ===
        score = 0.0

        has_person = "person" in cats
        has_food = "food/drink" in cats
        has_device = "device" in cats
        has_pet = "pet" in cats

        # Person visible + interaction = great clip
        if has_person:
            score += 2.0
            if s_label in ("interaction", "intro", "outro"):
                score += 2.0

        # Food/drink visible = interesting
        if has_food:
            score += 4.0

        # Device usage (phone, laptop)
        if has_device:
            score += 2.0

        # Pet = automatic winner
        if has_pet:
            score += 6.0

        # Chat activity
        if avg_chat > 0.5:
            score += 3.0
        elif avg_chat > 0.1:
            score += 1.5

        # Scene label
        label_bonus = {"interaction": 1, "intro": 3, "outro": 3, "highlight": 5,
                       "funny": 5, "quiet": 0}.get(s_label, 0)
        score += label_bonus

        # Streamer speaking
        if scene_text.strip():
            score += 1.5

        # Duration sweet spot
        if 30 <= duration <= 120:
            score += 1.0

        candidates.append({
            "scene_index": scene["index"],
            "start": s_start,
            "end": s_end,
            "duration": duration,
            "label": s_label,
            "score": round(score, 1),
            "visual_categories": list(cats),
            "objects_detected": obj_count,
            "has_person": has_person,
            "has_food": has_food,
            "has_pet": has_pet,
            "has_device": has_device,
            "avg_chat_intensity": round(avg_chat, 2),
            "has_speech": bool(scene_text.strip()),
            "text_preview": scene_text[:200] if scene_text else ""
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[:10]

    print(f"\nAnalyzed {len(candidates)} candidates.")
    print(f"{'#':>3} {'Score':>6}  {'Time':>12}  {'Label':>12}  {'Visual':<30} {'Chat':>5}  Text")
    print("=" * 90)
    for i, c in enumerate(top, 1):
        vis = ", ".join(c["visual_categories"][:4]) if c["visual_categories"] else "(none)"
        txt = c["text_preview"][:60] if c["text_preview"] else "-"
        print(f"{i:>3} {c['score']:>6.1f}  {c['start']:>5}-{c['end']:<4}s  "
              f"{c['label']:>12}  {vis:<30} {c['avg_chat_intensity']:>5.2f}  {txt}")

    # Output
    result = {
        "vod_id": VOD_ID,
        "total_scenes": len(scenes),
        "candidates_analyzed": len(candidates),
        "top_candidates": []
    }
    for i, c in enumerate(top, 1):
        clip_start = max(c["start"], 0)
        clip_dur = min(max(c["duration"], 30), 120)
        clip_end = clip_start + clip_dur
        result["top_candidates"].append({
            "rank": i,
            "score": c["score"],
            "time_range": f"*{clip_start}-{clip_end}",
            "start_sec": clip_start,
            "end_sec": clip_end,
            "duration_sec": clip_end - clip_start,
            "label": c["label"],
            "visual_content": ", ".join(c["visual_categories"]) or "(none)",
            "chat_intensity": c["avg_chat_intensity"],
            "has_speech": c["has_speech"],
            "summary": c["text_preview"][:200] or c["label"]
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved: {OUTPUT_PATH}")
    return result


if __name__ == "__main__":
    main()
