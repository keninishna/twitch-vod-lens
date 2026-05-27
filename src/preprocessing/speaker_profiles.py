"""Voice profile utilities for speaker enrollment and recognition."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    return "cpu"


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0:
        return vec
    return [v / norm for v in vec]


def compute_embedding(wav_path: Path, device: str = "auto") -> list[float]:
    """Compute ECAPA speaker embedding for a mono wav file."""

    from speechbrain.inference.speaker import EncoderClassifier
    import torch
    import torchaudio

    if not wav_path.exists():
        raise FileNotFoundError(f"WAV not found: {wav_path}")

    resolved = _resolve_device(device)
    waveform, sample_rate = torchaudio.load(str(wav_path))

    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != 16000:
        resample = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resample(waveform)

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": resolved},
    )

    waveform = waveform.to(resolved)
    with torch.no_grad():
        embedding = classifier.encode_batch(waveform)

    vec = embedding.detach().cpu().squeeze().flatten().tolist()
    return _l2_normalize([float(v) for v in vec])


def average_embeddings(embeddings: list[list[float]]) -> list[float]:
    """Normalize individual vectors, average element-wise, normalize again."""

    if not embeddings:
        raise ValueError("embeddings must be non-empty")

    dims = {len(e) for e in embeddings}
    if len(dims) != 1:
        raise ValueError("all embeddings must have same dimensionality")

    normalized = [_l2_normalize([float(v) for v in emb]) for emb in embeddings]
    dim = len(normalized[0])

    avg = []
    for i in range(dim):
        avg.append(sum(emb[i] for emb in normalized) / len(normalized))

    return _l2_normalize(avg)


def save_profile(profile: dict[str, Any], profile_dir: Path) -> Path:
    """Save profile JSON keyed by profile_id into profile_dir."""

    profile_id = str(profile.get("profile_id", "")).strip()
    if not profile_id:
        raise ValueError("profile must include non-empty profile_id")

    payload = dict(profile)
    payload.setdefault("created_at", datetime.now(UTC).isoformat())

    profile_dir.mkdir(parents=True, exist_ok=True)
    out = profile_dir / f"{profile_id}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_profiles(profile_dir: Path) -> list[dict[str, Any]]:
    """Load all JSON profiles from profile_dir (empty list if directory missing)."""

    if not profile_dir.exists():
        return []

    profiles: list[dict[str, Any]] = []
    for path in sorted(profile_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                profiles.append(data)
        except Exception:
            continue

    return profiles


def load_profiles_from_paths(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load profile JSON objects from explicit file paths.

    - Ignores missing/non-file paths.
    - Ignores malformed JSON payloads.
    - Deduplicates by resolved absolute file path.
    """

    seen: set[str] = set()
    profiles: list[dict[str, Any]] = []

    for raw_path in paths:
        path = Path(raw_path)
        try:
            resolved = str(path.expanduser().resolve())
        except Exception:
            resolved = str(path)

        if resolved in seen:
            continue
        seen.add(resolved)

        if not path.exists() or not path.is_file():
            continue

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if isinstance(data, dict):
            profiles.append(data)

    return profiles
