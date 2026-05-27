"""
VOD Lens — Whisper Transcription Module

Transcribes audio using faster-whisper with CUDA acceleration
on the RTX 5090. Produces segment-level timestamps with word-level
precision.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from src.preprocessing.types import TranscriptResult, TranscriptSegment, WordTiming


def transcribe(
    audio_path: Path,
    model_size: str = "large-v3",
    language: str = "en",
    device: str = "auto",
    compute_type: str = "float16",
    word_timestamps: bool = True,
    beam_size: int = 5,
) -> TranscriptResult:
    """
    Transcribe audio file using faster-whisper.

    Args:
        audio_path: Path to 16kHz mono WAV file
        model_size: Whisper model size (tiny, base, small, medium, large-v3)
        language: Language code (en, fr, etc.) or None for auto-detect
        device: "auto", "cuda", or "cpu"
        compute_type: float16 for GPU, int8_float16 for GPU with less VRAM
        word_timestamps: Enable word-level timestamps
        beam_size: Beam search size (higher = better but slower)

    Returns:
        TranscriptResult with segments and word timings

    Raises:
        ImportError: If faster-whisper is not installed
        FileNotFoundError: If audio file doesn't exist
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "faster-whisper is required. Install: pip install faster-whisper"
        )

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Auto-detect device
    if device == "auto":
        device = _detect_device()

    start_time = time.time()

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root=None,  # use default cache
    )

    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        vad_filter=True,  # filter out non-speech
    )

    # Process segments
    transcript_segments: list[TranscriptSegment] = []
    all_word_timings: list[WordTiming] = []

    for seg in segments:
        transcript_segments.append(
            TranscriptSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
                confidence=seg.avg_logprob if hasattr(seg, "avg_logprob") else 1.0,
            )
        )

        if word_timestamps and hasattr(seg, "words"):
            for word in seg.words:
                all_word_timings.append(
                    WordTiming(
                        word=word.word.strip(),
                        start=word.start,
                        end=word.end,
                        confidence=word.probability if hasattr(word, "probability") else 1.0,
                    )
                )

    elapsed = time.time() - start_time
    duration = info.duration if hasattr(info, "duration") else 0.0

    return TranscriptResult(
        segments=transcript_segments,
        language=info.language if hasattr(info, "language") else language,
        duration_seconds=duration,
        word_timings=all_word_timings if word_timestamps else None,
    )


def _detect_device() -> str:
    """Detect if CUDA GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
