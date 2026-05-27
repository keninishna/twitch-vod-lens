"""Speaker attribution artifact orchestration.

Builds per-VOD speaker attribution outputs by combining:
- diarization turns
- transcript alignment
- optional profile-based recognition
- heuristic name inference
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from src.preprocessing.audio_segments import extract_wav
from src.preprocessing.speaker_alignment import assign_speakers_to_transcript
from src.preprocessing.speaker_diarization import diarize_audio
from src.preprocessing.speaker_name_inference import infer_names_heuristic
from src.preprocessing.speaker_profiles import load_profiles
from src.preprocessing.speaker_recognition import recognize_speaker_clusters
from src.preprocessing.types import (
    ClipSpeakerStats,
    SpeakerAttributionResult,
    SpeakerClusterSummary,
    SpeakerNameCandidate,
    SpeakerRecognitionResult,
    SpeakerTurn,
)


DIARIZATION_BACKEND = "pyannote/speaker-diarization-community-1"
EMBEDDING_BACKEND = "speechbrain/spkrec-ecapa-voxceleb"
NAME_INFERENCE_BACKEND = "heuristic+qwen"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_transcript_segments(transcript_path: Path) -> list[dict[str, Any]]:
    raw = _load_json(transcript_path)

    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]

    if isinstance(raw, dict):
        # transcript.json from transcriber
        if isinstance(raw.get("segments"), list):
            return [x for x in raw["segments"] if isinstance(x, dict)]

        # fusion_result transcript envelope
        transcript = raw.get("transcript")
        if isinstance(transcript, dict) and isinstance(transcript.get("segments"), list):
            return [x for x in transcript["segments"] if isinstance(x, dict)]

    raise RuntimeError(f"Unsupported transcript schema in {transcript_path}")


def _load_chat_messages(chat_path: Path | None) -> list[dict[str, Any]]:
    if chat_path is None or not chat_path.exists():
        return []

    raw = _load_json(chat_path)
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]

    if isinstance(raw, dict):
        if isinstance(raw.get("messages"), list):
            return [x for x in raw["messages"] if isinstance(x, dict)]

        chat = raw.get("chat")
        if isinstance(chat, dict) and isinstance(chat.get("messages"), list):
            return [x for x in chat["messages"] if isinstance(x, dict)]

    return []


def _to_candidate_map(
    raw_candidates: dict[str, list[SpeakerNameCandidate | dict[str, Any]]],
) -> dict[str, list[SpeakerNameCandidate]]:
    out: dict[str, list[SpeakerNameCandidate]] = {}
    for label, values in raw_candidates.items():
        parsed: list[SpeakerNameCandidate] = []
        for v in values:
            if isinstance(v, SpeakerNameCandidate):
                parsed.append(v)
            elif isinstance(v, dict):
                parsed.append(SpeakerNameCandidate.model_validate(v))
        out[label] = parsed
    return out


def _unknown_recognition_map(turns: list[SpeakerTurn]) -> dict[str, SpeakerRecognitionResult]:
    return {
        label: SpeakerRecognitionResult(
            identity="unknown",
            confidence=0.0,
            cosine_similarity=None,
            profile_id=None,
        )
        for label in {t.speaker_label for t in turns}
    }


def _summarize_clusters(
    turns: list[SpeakerTurn],
    recognition_map: dict[str, SpeakerRecognitionResult],
    name_candidates: dict[str, list[SpeakerNameCandidate]],
) -> dict[str, SpeakerClusterSummary]:
    by_label: dict[str, list[SpeakerTurn]] = {}
    for turn in turns:
        by_label.setdefault(turn.speaker_label, []).append(turn)

    summaries: dict[str, SpeakerClusterSummary] = {}
    for label, label_turns in by_label.items():
        total_speech_seconds = sum(max(0.0, t.end - t.start) for t in label_turns)
        rec = recognition_map.get(label) or SpeakerRecognitionResult(identity="unknown", confidence=0.0)

        summaries[label] = SpeakerClusterSummary(
            total_speech_seconds=total_speech_seconds,
            segment_count=len(label_turns),
            primary_identity=rec.identity,
            primary_identity_confidence=rec.confidence,
            candidate_names=name_candidates.get(label, []),
        )

    return summaries


def _build_actionable_error(exc: Exception) -> RuntimeError:
    msg = str(exc)
    lower = msg.lower()

    if "no module named" in lower or "pyannote" in lower or "speechbrain" in lower:
        return RuntimeError(
            "SpeakerID dependency missing. Install optional deps: "
            "pip install -r requirements-speakerid.txt"
        )

    if "token" in lower or "huggingface" in lower or "401" in lower or "403" in lower:
        return RuntimeError(
            "SpeakerID model access failed. Set HF_TOKEN/HUGGINGFACE_TOKEN and accept "
            "pyannote/speaker-diarization-community-1 terms on Hugging Face."
        )

    return RuntimeError(f"SpeakerID failed: {msg}")


def generate_speaker_attribution(
    *,
    vod_id: str,
    vod_media: Path,
    transcript_path: Path,
    chat_path: Path | None = None,
    profiles_dir: Path | None = None,
    profiles: list[dict[str, Any]] | None = None,
    output_path: Path | None = None,
    hf_token: str | None = None,
    require_speaker_id: bool = False,
) -> SpeakerAttributionResult:
    """Generate speaker attribution artifact for a VOD."""

    if not vod_media.exists():
        raise FileNotFoundError(f"VOD media not found: {vod_media}")
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    transcript_segments = _load_transcript_segments(transcript_path)
    chat_messages = _load_chat_messages(chat_path)

    try:
        with tempfile.TemporaryDirectory(prefix=f"speakerid-{vod_id}-") as td:
            tmp_dir = Path(td)
            wav_path = tmp_dir / f"{vod_id}.wav"

            # 1) Extract mono 16k wav
            extract_wav(vod_media, wav_path, sample_rate=16000)

            # 2) Diarize
            turns = diarize_audio(wav_path, hf_token=hf_token)

            # 3) Align to transcript
            aligned_transcript = assign_speakers_to_transcript(transcript_segments, turns)

            # 4) Recognize clusters (if profiles exist)
            runtime_profiles = profiles if profiles is not None else (load_profiles(profiles_dir) if profiles_dir else [])
            if runtime_profiles:
                recognition_map = recognize_speaker_clusters(
                    audio_path=wav_path,
                    speaker_turns=turns,
                    profiles=runtime_profiles,
                    output_dir=tmp_dir / "cluster_samples",
                )
            else:
                recognition_map = _unknown_recognition_map(turns)

            # 5) Name inference
            raw_name_candidates = infer_names_heuristic(
                aligned_transcript,
                chat_messages=chat_messages,
            )
            name_candidates = _to_candidate_map(raw_name_candidates)

            # 6) Enrich turns and summarize
            enriched_turns: list[SpeakerTurn] = []
            for turn in turns:
                top_name = None
                if name_candidates.get(turn.speaker_label):
                    best = name_candidates[turn.speaker_label][0]
                    if best.confidence >= 0.70:
                        top_name = best.name

                enriched_turns.append(
                    SpeakerTurn(
                        start=turn.start,
                        end=turn.end,
                        speaker_label=turn.speaker_label,
                        exclusive=turn.exclusive,
                        recognition=recognition_map.get(turn.speaker_label),
                        inferred_name=top_name,
                    )
                )

            result = SpeakerAttributionResult(
                vod_id=vod_id,
                audio_path=str(vod_media),
                backend={
                    "diarization": DIARIZATION_BACKEND,
                    "embedding": EMBEDDING_BACKEND,
                    "name_inference": NAME_INFERENCE_BACKEND,
                },
                segments=enriched_turns,
                speaker_clusters=_summarize_clusters(
                    enriched_turns,
                    recognition_map,
                    name_candidates,
                ),
                clip_speaker_stats={},
            )

    except Exception as exc:  # noqa: BLE001
        actionable = _build_actionable_error(exc)
        if require_speaker_id:
            raise actionable

        # Non-blocking mode: preserve baseline pipeline by emitting minimal artifact.
        result = SpeakerAttributionResult(
            vod_id=vod_id,
            audio_path=str(vod_media),
            backend={
                "diarization": "unavailable",
                "embedding": "unavailable",
                "name_inference": "unavailable",
                "error": str(actionable),
            },
            segments=[],
            speaker_clusters={},
            clip_speaker_stats={},
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2)

    return result
