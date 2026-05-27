"""CLI for enrolling reusable speaker voice profiles from verified segments."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from src.preprocessing.audio_segments import extract_wav
from src.preprocessing.speaker_profiles import average_embeddings, compute_embedding, save_profile


def parse_segments(spec: str) -> list[tuple[float, float]]:
    """Parse comma-separated start-end segment spec (e.g., '30-90,300-360')."""

    segments: list[tuple[float, float]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" not in chunk:
            raise ValueError(f"invalid segment spec: {chunk}")
        left, right = chunk.split("-", 1)
        start = float(left)
        end = float(right)
        if end <= start:
            raise ValueError(f"invalid segment range (end<=start): {chunk}")
        segments.append((start, end))

    if not segments:
        raise ValueError("no valid segments parsed")

    return segments


def enroll_profile(
    profile_id: str,
    display_name: str,
    role: str,
    audio: Path,
    segments: list[tuple[float, float]],
    output_dir: Path,
    device: str = "auto",
    accept_similarity: float = 0.72,
    high_confidence_similarity: float = 0.80,
) -> Path:
    """Enroll a profile from selected source audio segments."""

    if not audio.exists():
        raise FileNotFoundError(f"audio not found: {audio}")

    embeddings: list[list[float]] = []
    provenance: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="speaker-enroll-") as tmp:
        tmpdir = Path(tmp)

        for idx, (start, end) in enumerate(segments):
            out_wav = tmpdir / f"segment_{idx:03d}.wav"
            extract_wav(audio, out_wav, start=start, end=end, sample_rate=16000)
            emb = compute_embedding(out_wav, device=device)
            embeddings.append(emb)
            provenance.append(
                {
                    "start": start,
                    "end": end,
                    "source_audio": str(audio),
                    "notes": "user-specified enrollment segment",
                }
            )

    final_embedding = average_embeddings(embeddings)

    profile = {
        "profile_id": profile_id,
        "display_name": display_name,
        "role": role,
        "embedding_model": "speechbrain/spkrec-ecapa-voxceleb",
        "embedding_dim": len(final_embedding),
        "embedding": final_embedding,
        "created_from": provenance,
        "thresholds": {
            "accept_similarity": float(accept_similarity),
            "high_confidence_similarity": float(high_confidence_similarity),
        },
    }

    return save_profile(profile, output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Enroll a reusable speaker voice profile")
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--role", default="streamer")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--segments", required=True, help="comma-separated ranges: 30-90,300-360")
    parser.add_argument("--output-dir", type=Path, default=Path("data/speaker_profiles"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--accept-similarity", type=float, default=0.72)
    parser.add_argument("--high-confidence-similarity", type=float, default=0.80)
    args = parser.parse_args()

    try:
        parsed_segments = parse_segments(args.segments)
        out = enroll_profile(
            profile_id=args.profile_id,
            display_name=args.display_name,
            role=args.role,
            audio=args.audio,
            segments=parsed_segments,
            output_dir=args.output_dir,
            device=args.device,
            accept_similarity=args.accept_similarity,
            high_confidence_similarity=args.high_confidence_similarity,
        )
        print(f"Saved profile: {out}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
