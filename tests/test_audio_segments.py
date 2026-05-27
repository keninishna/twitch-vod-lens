from __future__ import annotations

import subprocess
from pathlib import Path

from src.preprocessing.audio_segments import extract_turn_wavs, extract_wav
from src.preprocessing.types import SpeakerTurn


def test_extract_wav_builds_expected_ffmpeg_command(monkeypatch, tmp_path):
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, check, capture_output, text):
        captured["cmd"] = cmd

        class Result:
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)

    output_path = tmp_path / "sample.wav"
    result = extract_wav(
        input_media=Path("input.mp4"),
        output_wav=output_path,
        start=12.5,
        end=30.0,
    )

    assert result == output_path
    assert captured["cmd"] == [
        "ffmpeg",
        "-y",
        "-ss",
        "12.5",
        "-to",
        "30.0",
        "-i",
        "input.mp4",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(output_path),
    ]


def test_extract_turn_wavs_filters_short_turns(monkeypatch, tmp_path):
    calls: list[tuple[float, float]] = []

    def fake_extract_wav(input_media, output_wav, start=None, end=None, sample_rate=16000):
        calls.append((start, end))
        return output_wav

    monkeypatch.setattr("src.preprocessing.audio_segments.extract_wav", fake_extract_wav)

    turns = [
        SpeakerTurn(start=0.0, end=0.9, speaker_label="SPEAKER_00"),
        SpeakerTurn(start=1.0, end=4.0, speaker_label="SPEAKER_00"),
        SpeakerTurn(start=10.0, end=12.0, speaker_label="SPEAKER_01"),
    ]

    outputs = extract_turn_wavs(
        input_media=Path("input.mp4"),
        turns=turns,
        output_dir=tmp_path,
        min_duration=1.5,
    )

    assert len(outputs) == 2
    assert calls == [(1.0, 4.0), (10.0, 12.0)]
