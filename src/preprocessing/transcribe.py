"""Whisper transcription module.

Runs faster-whisper large-v3 on the extracted audio. Produces timestamped
transcript with segment-level timestamps.
"""

import json
from faster_whisper import WhisperModel


def transcribe(audio_path: str, model_size: str = "large-v3") -> list[dict]:
    """Transcribe audio using faster-whisper.

    Args:
        audio_path: Path to audio file (.mp3 or .wav).
        model_size: Whisper model size (default: large-v3).

    Returns:
        list[dict]: List of transcript segments with start, end, text, confidence.
    """
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    segments, info = model.transcribe(audio_path, beam_size=5)

    results = []
    for seg in segments:
        results.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "confidence": seg.avg_logprob,
        })

    return results


def transcribe_to_file(audio_path: str, output_path: str,
                       model_size: str = "large-v3") -> int:
    """Transcribe audio and write results to JSON file.

    Returns:
        int: Number of segments produced.
    """
    segments = transcribe(audio_path, model_size)
    with open(output_path, "w") as f:
        json.dump(segments, f, indent=2)
    return len(segments)
