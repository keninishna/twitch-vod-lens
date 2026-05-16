#!/usr/bin/env python3
"""Run YOLOv11 inference on all frames, save per-frame detections as JSON."""

import json
import os
import sys
import glob
from collections import defaultdict
from pathlib import Path
import time

# Import ultralytics
from ultralytics import YOLO

FRAMES_DIR = os.path.expanduser("~/twitch-vod-analyzer/vods/phase4_2770929139/frames/")
OUTPUT_DIR = os.path.expanduser("~/twitch-vod-analyzer/vods/phase4_2770929139/")
FUSION_PATH = os.path.expanduser("~/twitch-vod-analyzer/vods/phase4_2770929139/fusion_result.json")
MODEL = "yolo11x.pt"  # largest YOLO11 for best accuracy

# Target objects we care about for clip selection
TARGET_CLASSES = {
    0: "person",
    39: "bottle",
    41: "cup",
    47: "tv/monitor",
    56: "chair",
    62: "tv/monitor",
    63: "laptop",
    64: "mouse",
    65: "remote",
    67: "cell phone",
    72: "book",
    73: "book",
    74: "clock",
    76: "vase",
    77: "scissors",
    78: "teddy bear",
    79: "hair drier",
    80: "toothbrush",
    # Food-related
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
}

conf_threshold = 0.35  # confidence threshold

def main():
    print(f"CUDA available: {__import__('torch').cuda.is_available()}")
    if __import__('torch').cuda.is_available():
        print(f"GPU: {__import__('torch').cuda.get_device_name(0)}")

    # Load model
    print(f"Loading {MODEL}...")
    model = YOLO(MODEL)
    print("Model loaded.")

    # Get all frames sorted by timestamp
    frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
    print(f"Found {len(frames)} frames to process.")

    # Process in batches
    batch_size = 32
    all_results = {}
    total_objects = 0
    frame_count_processed = 0
    start_time = time.time()

    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        results = model(batch, conf=conf_threshold, device=0, verbose=False, imgsz=640)

        for j, result in enumerate(results):
            frame_path = batch[j]
            frame_name = os.path.basename(frame_path)
            # Extract timestamp from filename (frame_XXXX.jpg) -> seconds
            frame_num = int(frame_name.replace("frame_", "").replace(".jpg", ""))
            timestamp_sec = frame_num * 5  # 5-second intervals

            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    detections.append({
                        "class_id": cls_id,
                        "class_name": result.names[cls_id],
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2]
                    })
                    total_objects += 1

            all_results[frame_name] = {
                "timestamp_sec": timestamp_sec,
                "detections": detections,
                "num_objects": len(detections)
            }
            frame_count_processed += 1

        elapsed = time.time() - start_time
        fps = (i + len(batch)) / elapsed if elapsed > 0 else 0
        print(f"  Processed {frame_count_processed}/{len(frames)} frames, {total_objects} objects found, {fps:.1f} fps")

    # Save results
    output = {
        "model": MODEL,
        "confidence_threshold": conf_threshold,
        "total_frames": frame_count_processed,
        "total_objects": total_objects,
        "results": all_results
    }

    out_path = os.path.join(OUTPUT_DIR, "yolo_detections.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nDone! {frame_count_processed} frames processed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Total objects detected: {total_objects}")
    print(f"Results saved to: {out_path}")

if __name__ == "__main__":
    main()
