"""Speaker recognition against enrolled voice profiles."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.preprocessing.audio_segments import extract_wav
from src.preprocessing.speaker_profiles import compute_embedding
from src.preprocessing.types import SpeakerRecognitionResult, SpeakerTurn


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(dot / (na * nb))


def aggregate_cluster_embeddings(
    audio_path: Path,
    speaker_turns: list[SpeakerTurn],
    output_dir: Path,
    device: str = "auto",
    min_duration: float = 1.5,
    max_total_seconds_per_speaker: float = 120.0,
    max_turns_per_speaker: int = 40,
) -> dict[str, list[float]]:
    """Build one averaged embedding per diarized speaker label."""

    turns_by_label: dict[str, list[SpeakerTurn]] = defaultdict(list)
    for turn in speaker_turns:
        if (turn.end - turn.start) >= min_duration:
            turns_by_label[turn.speaker_label].append(turn)

    output_dir.mkdir(parents=True, exist_ok=True)
    aggregated: dict[str, list[float]] = {}

    for label, turns in turns_by_label.items():
        turns = sorted(turns, key=lambda t: (t.end - t.start), reverse=True)
        selected: list[SpeakerTurn] = []
        acc_seconds = 0.0
        for turn in turns:
            if len(selected) >= max_turns_per_speaker:
                break
            dur = turn.end - turn.start
            if acc_seconds + dur > max_total_seconds_per_speaker and selected:
                break
            selected.append(turn)
            acc_seconds += dur

        vectors: list[list[float]] = []
        for idx, turn in enumerate(selected):
            wav_path = output_dir / f"{label}_turn_{idx:03d}.wav"
            extract_wav(audio_path, wav_path, start=turn.start, end=turn.end, sample_rate=16000)
            vectors.append(compute_embedding(wav_path, device=device))

        if not vectors:
            continue

        dim = len(vectors[0])
        mean_vec = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]
        aggregated[label] = mean_vec

    return aggregated


def _profile_thresholds(profile: dict[str, Any]) -> tuple[float, float]:
    thresholds = profile.get("thresholds") if isinstance(profile.get("thresholds"), dict) else {}
    accept = float(thresholds.get("accept_similarity", 0.72))
    high = float(thresholds.get("high_confidence_similarity", 0.80))
    return accept, high


def recognize_speaker_clusters(
    audio_path: Path,
    speaker_turns: list[SpeakerTurn],
    profiles: list[dict[str, Any]],
    output_dir: Path,
    device: str = "auto",
) -> dict[str, SpeakerRecognitionResult]:
    """Match diarized speaker labels to known profiles using cosine similarity."""

    cluster_embeddings = aggregate_cluster_embeddings(
        audio_path=audio_path,
        speaker_turns=speaker_turns,
        output_dir=output_dir,
        device=device,
    )

    results: dict[str, SpeakerRecognitionResult] = {}

    for label, emb in cluster_embeddings.items():
        best_profile: dict[str, Any] | None = None
        best_score = -1.0
        best_accept = 0.72
        best_high = 0.80

        for profile in profiles:
            p_emb = profile.get("embedding")
            if not isinstance(p_emb, list) or not p_emb:
                continue

            score = cosine_similarity(emb, [float(x) for x in p_emb])
            if score > best_score:
                best_score = score
                best_profile = profile
                best_accept, best_high = _profile_thresholds(profile)

        if best_profile is None or best_score < best_accept:
            results[label] = SpeakerRecognitionResult(
                identity="unknown",
                confidence=0.0,
                cosine_similarity=best_score if best_score >= 0 else None,
                profile_id=None,
            )
            continue

        confidence = min(1.0, max(0.0, (best_score - best_accept) / max(1e-6, (1.0 - best_accept))))
        identity = str(best_profile.get("role") or "unknown")

        # High-confidence bump near profile's high threshold.
        if best_score >= best_high:
            confidence = max(confidence, 0.85)

        results[label] = SpeakerRecognitionResult(
            identity=identity if identity in {"streamer", "guest", "unknown", "chatter", "mixed"} else "unknown",
            confidence=confidence,
            cosine_similarity=best_score,
            profile_id=str(best_profile.get("profile_id")) if best_profile.get("profile_id") else None,
        )

    # Ensure labels with no emb still appear as unknown.
    for label in {t.speaker_label for t in speaker_turns}:
        results.setdefault(
            label,
            SpeakerRecognitionResult(identity="unknown", confidence=0.0, cosine_similarity=None, profile_id=None),
        )

    return results
