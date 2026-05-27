"""Audio extraction helpers for speaker-ID preprocessing."""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.preprocessing.types import SpeakerTurn


def extract_wav(
    input_media: Path,
    output_wav: Path,
    start: float | None = None,
    end: float | None = None,
    sample_rate: int = 16000,
) -> Path:
    """Extract a mono WAV segment from media using ffmpeg."""

    cmd: list[str] = ["ffmpeg", "-y"]
    if start is not None:
        cmd.extend(["-ss", str(start)])
    if end is not None:
        cmd.extend(["-to", str(end)])

    cmd.extend(
        [
            "-i",
            str(input_media),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "wav",
            str(output_wav),
        ]
    )

    output_wav.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed extracting WAV: {exc.stderr}") from exc

    return output_wav


def extract_turn_wavs(
    input_media: Path,
    turns: list[SpeakerTurn],
    output_dir: Path,
    min_duration: float = 1.5,
) -> list[Path]:
    """Extract per-turn wav clips from diarization turns."""

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    for idx, turn in enumerate(turns):
        duration = turn.end - turn.start
        if duration < min_duration:
            continue

        safe_label = turn.speaker_label.replace("/", "_")
        output_path = output_dir / (
            f"turn_{idx:04d}_{safe_label}_{int(turn.start * 1000)}_{int(turn.end * 1000)}.wav"
        )

        extract_wav(
            input_media=input_media,
            output_wav=output_path,
            start=turn.start,
            end=turn.end,
        )
        extracted.append(output_path)

    return extracted
