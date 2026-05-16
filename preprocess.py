#!/usr/bin/env python3
"""Phase 1 orchestrator — VOD preprocessing pipeline.

Usage:
    python preprocess.py https://www.twitch.tv/videos/123456789

Runs download -> transcribe -> scene detect -> chat analyze -> fusion
on the given Twitch VOD URL. All outputs go to ./output/<vod_id>/
"""

import json
import sys
import time
from pathlib import Path

# Ensure we can import the preprocessing package
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.preprocessing.download import download_vod
from src.preprocessing.transcribe import transcribe_to_file
from src.preprocessing.scene import detect_scenes_to_file
from src.preprocessing.chat import analyze_chat_to_file
from src.preprocessing.fusion import fuse_signals


def run(vod_url: str) -> None:
    """Run the full preprocessing pipeline on a VOD URL."""
    start = time.time()

    # Extract VOD ID from URL
    vod_id = vod_url.strip("/").split("/")[-1]
    output_dir = Path(f"./output/{vod_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    print("Downloading VOD...")
    meta = download_vod(vod_url, output_dir)
    if not meta:
        print("  ERROR: Failed to download VOD")
        return
    print(f"  Title: {meta.get('title', 'N/A')}")
    print(f"  Game: {meta.get('game', 'N/A')}")
    print(f"  Duration: {meta.get('duration', 0)} seconds")

    # Step 2: Transcribe
    audio_path = str(output_dir / f"{vod_id}.mp3")
    transcript_path = str(output_dir / "transcript.json")
    print("Transcribing audio (faster-whisper large-v3, CUDA)...")
    n_segments = transcribe_to_file(audio_path, transcript_path)
    print(f"  {n_segments} transcript segments -> transcript.json")

    # Step 3: Detect scenes
    video_path = str(output_dir / f"{vod_id}_video.mp4")
    scenes_path = str(output_dir / "scenes.json")
    print("Detecting scene boundaries...")
    n_scenes = detect_scenes_to_file(video_path, scenes_path)
    print(f"  {n_scenes} scene boundaries -> scenes.json")

    # Step 4: Analyze chat
    chat_path = str(output_dir / "chat_analysis.json")
    print("Analyzing chat...")
    n_spikes = analyze_chat_to_file(vod_id, chat_path)
    print(f"  {n_spikes} chat spikes -> chat_analysis.json")

    # Step 5: Fuse signals
    moments_path = str(output_dir / "moments.json")
    print("Fusing signals into scored moments...")
    n_moments = fuse_signals(transcript_path, scenes_path, chat_path, moments_path)
    print(f"  {n_moments} scored moments -> moments.json")

    # Summary
    elapsed = time.time() - start
    print(f"\nDone! {n_moments} moments found in {elapsed/60:.1f} minutes.")

    # Print top 5 moments
    with open(moments_path) as f:
        moments = json.load(f)
    print("\nTop 5 moments:")
    for i, m in enumerate(moments[:5], 1):
        duration = m["end"] - m["start"]
        print(f"  {i}. Score {m['score']:.0f} @ {m['start']:.0f}s-{m['end']:.0f}s "
              f"({duration:.0f}s) signals={m['signals']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <twitch_vod_url>")
        sys.exit(1)
    run(sys.argv[1])
