#!/usr/bin/env python3
"""Phase 1 integration test harness.

Tests each module in the preprocessing pipeline with known test data.
Run on the WSL2 machine with the RTX 5090.

Usage:
    python tests/test_pipeline.py                  # Quick smoke test
    python tests/test_pipeline.py --vod <url>      # Test with real VOD
    python tests/test_pipeline.py --full           # Full stress test
"""

import json
import os
import shutil
import subprocess
import sys
import time

import pytest

# Ensure we can import the packages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set HF cache to writable location
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/hf"))


def test_transcribe():
    """Test whisper transcription with the test speech file."""
    pytest.importorskip("faster_whisper")
    from src.preprocessing.transcribe import transcribe

    audio = "output/e2e_test/test_speech.mp3"
    if not os.path.exists(audio):
        pytest.skip("test audio not found")

    results = transcribe(audio, model_size="base")
    assert len(results) > 0, "No transcript segments produced"
    print(f"  transcript: {len(results)} segments")

    # Verify structure
    for seg in results:
        assert "start" in seg
        assert "end" in seg
        assert "text" in seg
        assert "confidence" in seg


def test_scene_detect():
    """Test scene detection with the test video."""
    pytest.importorskip("scenedetect")
    from src.preprocessing.scene import detect_scenes

    video = "output/e2e_test/test_speech_h264.mp4"
    if not os.path.exists(video):
        pytest.skip("test video not found")

    results = detect_scenes(video)
    print(f"  scenes: {len(results)} boundaries")


def test_chat_analysis():
    """Test chat analysis with a mock chat file."""
    pytest.importorskip("numpy")
    from src.preprocessing.chat import analyze_chat

    # Create a synthetic chat for testing
    vod_id = "test_chat_123"
    result = analyze_chat(vod_id)
    assert "total_messages" in result
    assert "spikes" in result
    assert "top_emotes" in result
    print(f"  chat: {result['total_messages']} messages, {len(result['spikes'])} spikes")


def test_fusion():
    """Test fusion with known inputs."""
    from src.preprocessing.fusion import fuse_signals

    transcript = [{"start": 0, "end": 2, "text": "LETS GO", "confidence": 0.9}]
    scenes = [{"timestamp": 1.0, "scene_type": "cut", "content_hash": "s0"}]
    chat = {
        "total_messages": 100,
        "total_emotes": 30,
        "spikes": [{"start": 0, "end": 5, "intensity": 3.0, "message_count": 25}],
        "top_emotes": [{"id": "PogChamp", "count": 5}],
    }

    os.makedirs("output/test_fusion", exist_ok=True)
    for name, data in [("transcript.json", transcript), ("scenes.json", scenes),
                       ("chat_analysis.json", chat)]:
        with open(f"output/test_fusion/{name}", "w") as f:
            json.dump(data, f)

    n = fuse_signals(
        "output/test_fusion/transcript.json",
        "output/test_fusion/scenes.json",
        "output/test_fusion/chat_analysis.json",
        "output/test_fusion/moments.json",
    )

    assert n > 0, "No moments produced"
    with open("output/test_fusion/moments.json") as f:
        moments = json.load(f)
    assert moments[0]["score"] > 0, "Score should be > 0"
    assert "chat_spike" in moments[0]["signals"] or "voice_excitement" in moments[0]["signals"]
    print(f"  fusion: {n} moments, top score={moments[0]['score']}")


def test_e2e():
    """End-to-end test: download VOD from YouTube and run pipeline."""
    pytest.importorskip("faster_whisper")

    if not shutil.which("yt-dlp"):
        pytest.skip("yt-dlp not found")
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not found")

    os.makedirs("output/e2e_test", exist_ok=True)

    url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    print(f"  Downloading: {url}")

    # Download audio
    subprocess.run([
        "yt-dlp", "-x", "--audio-format", "mp3",
        "-o", "output/e2e_test/test_speech.%(ext)s", url,
    ], check=True, capture_output=True)

    # Run full pipeline
    from src.preprocessing.transcribe import transcribe_to_file
    from src.preprocessing.fusion import fuse_signals

    n_seg = transcribe_to_file("output/e2e_test/test_speech.mp3",
                                "output/e2e_test/transcript.json", "base")
    print(f"  transcript: {n_seg} segments")

    # Mock chat (no Twitch chat data for YouTube)
    with open("output/e2e_test/chat_analysis.json", "w") as f:
        json.dump({"total_messages": 0, "total_emotes": 0, "spikes": [], "top_emotes": []}, f)

    # Empty scenes
    with open("output/e2e_test/scenes.json", "w") as f:
        json.dump([], f)

    n_moments = fuse_signals(
        "output/e2e_test/transcript.json",
        "output/e2e_test/scenes.json",
        "output/e2e_test/chat_analysis.json",
        "output/e2e_test/moments.json",
    )
    print(f"  fusion: {n_moments} moments")


if __name__ == "__main__":
    import shutil

    tests = [
        ("Transcribe", test_transcribe),
        ("Scene Detect", test_scene_detect),
        ("Chat Analysis", test_chat_analysis),
        ("Fusion", test_fusion),
    ]

    if "--vod" in sys.argv:
        idx = sys.argv.index("--vod")
        url = sys.argv[idx + 1]
        print(f"\nEnd-to-end test with VOD: {url}")
        tests.append(("E2E VOD", lambda: test_e2e_vod(url)))

    if "--full" in sys.argv:
        tests.append(("E2E", test_e2e))

    print("\nVOD Lens Phase 1 Test Results")
    print("=" * 40)
    all_pass = True
    for name, fn in tests:
        start = time.time()
        try:
            result = fn()
            elapsed = time.time() - start
            status = "PASS" if result == "PASS" else result
            print(f"  [{status:5s}] {name} ({elapsed:.1f}s)")
            if status not in ("PASS", "SKIP"):
                all_pass = False
        except Exception as e:
            elapsed = time.time() - start
            print(f"  [FAIL] {name} ({elapsed:.1f}s): {e}")
            all_pass = False

    print("=" * 40)
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    sys.exit(0 if all_pass else 1)
