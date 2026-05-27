"""Pyannote diarization backend wrapper for speaker attribution."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

from src.preprocessing.types import SpeakerTurn


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    return "cpu"


def _normalize_non_overlapping(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    if not turns:
        return []

    ordered = sorted(turns, key=lambda t: (t.start, t.end))
    normalized: list[SpeakerTurn] = []

    for turn in ordered:
        start = turn.start
        end = turn.end

        if normalized and start < normalized[-1].end:
            start = normalized[-1].end
        if end <= start:
            continue

        normalized.append(
            SpeakerTurn(
                start=start,
                end=end,
                speaker_label=turn.speaker_label,
                exclusive=turn.exclusive,
            )
        )

    return normalized


def diarize_audio(
    audio_path: Path,
    hf_token: str | None = None,
    model_id: str = "pyannote/speaker-diarization-community-1",
    device: str = "auto",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[SpeakerTurn]:
    """Run pyannote diarization and return normalized speaker turns."""

    pyannote_audio = importlib.import_module("pyannote.audio")
    pipeline_cls = pyannote_audio.Pipeline

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    pipeline = pipeline_cls.from_pretrained(model_id, use_auth_token=token)

    resolved_device = _resolve_device(device)
    try:
        import torch

        pipeline.to(torch.device(resolved_device))
    except Exception:
        # CPU fallback or fake test pipelines that do not implement `.to(...)`.
        pass

    kwargs = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    output = pipeline(str(audio_path), **kwargs)
    diarization = (
        getattr(output, "exclusive_speaker_diarization", None)
        or getattr(output, "speaker_diarization", None)
        or output
    )

    turns: list[SpeakerTurn] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(
            SpeakerTurn(
                start=float(turn.start),
                end=float(turn.end),
                speaker_label=str(speaker),
                exclusive=hasattr(output, "exclusive_speaker_diarization"),
            )
        )

    return _normalize_non_overlapping(turns)
