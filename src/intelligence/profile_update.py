"""Profile update proposal helpers (SpeakerID Phase 04 Task 17)."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Literal

from src.intelligence.types import (
    ContentPattern,
    InsideJoke,
    ObservationType,
    ProfileUpdateProposal,
    StreamerObservation,
    StreamerProfile,
)

_SENSITIVE_KEYWORDS = (
    "medical",
    "diagnosis",
    "address",
    "phone",
    "ssn",
    "doxx",
    "legal name",
)

_STRIP_PUNCT = "\"'.,;:!?()[]{} "


def normalize_claim(text: str) -> str:
    """Normalize free-text claims for dedupe/safety checks."""

    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    return normalized.strip(_STRIP_PUNCT)


def is_sensitive_claim(text: str) -> bool:
    """Detect potentially sensitive claims that should not auto-persist."""

    claim = normalize_claim(text)
    return any(keyword in claim for keyword in _SENSITIVE_KEYWORDS)


def dedupe_observations(observations: list[StreamerObservation]) -> list[StreamerObservation]:
    """Dedupe by (type, normalized claim), keeping strongest evidence entry."""

    best_by_key: dict[tuple[str, str], StreamerObservation] = {}

    for obs in observations:
        key = (obs.type, normalize_claim(obs.claim))
        existing = best_by_key.get(key)
        if existing is None:
            best_by_key[key] = obs
            continue

        if obs.confidence > existing.confidence:
            best_by_key[key] = obs
            continue

        if obs.confidence == existing.confidence and len(obs.evidence) > len(existing.evidence):
            best_by_key[key] = obs

    return list(best_by_key.values())


def partition_observations_for_merge(
    observations: list[StreamerObservation],
    min_auto_confidence: float = 0.80,
) -> tuple[list[StreamerObservation], list[StreamerObservation], list[StreamerObservation]]:
    """Partition observations by merge policy.

    Returns:
        (auto_accept, queue, reject)
    """

    auto_accept: list[StreamerObservation] = []
    queue: list[StreamerObservation] = []
    reject: list[StreamerObservation] = []

    for obs in observations:
        if is_sensitive_claim(obs.claim):
            reject.append(obs)
            continue

        if obs.confidence >= min_auto_confidence and len(obs.evidence) >= 2:
            auto_accept.append(obs)
            continue

        if 0.60 <= obs.confidence < min_auto_confidence:
            queue.append(obs)
            continue

        reject.append(obs)

    return auto_accept, queue, reject


def _clamp_confidence(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _safe_float(value: object, default: float) -> float:
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _clip_window(clip: dict) -> tuple[float, float]:
    start = _safe_float(
        clip.get("suggested_trim_start", clip.get("start", 0.0)),
        0.0,
    )
    end = _safe_float(
        clip.get("suggested_trim_end", clip.get("end", start + 1.0)),
        start + 1.0,
    )

    if end <= start:
        end = start + 1.0

    return start, end


def _clip_title(clip: dict) -> str:
    return str(
        clip.get("clip_point")
        or clip.get("title")
        or clip.get("label")
        or "untitled clip"
    )


def _build_observation(
    *,
    vod_id: str,
    clip: dict,
    observation_type: ObservationType,
    claim: str,
) -> StreamerObservation:
    start, end = _clip_window(clip)
    title = _clip_title(clip)
    score = _safe_float(clip.get("score", 0.0), 0.0)
    confidence = _clamp_confidence(score / 10.0)

    evidence = [
        f"title: {title}",
        f"window: {start:.3f}-{end:.3f}",
    ]
    evidence_refs = [f"vod:{vod_id}@{start:.3f}-{end:.3f}"]

    return StreamerObservation(
        vod_id=vod_id,
        timestamp_start=start,
        timestamp_end=end,
        type=observation_type,
        claim=claim,
        evidence=evidence,
        source="llm_summary",
        confidence=confidence,
        evidence_refs=evidence_refs,
        created_at=datetime.now(timezone.utc),
    )


def build_profile_update_proposal(
    vod_id: str,
    streamer_id: str,
    final_selected_clips: list[dict],
    mode: str = "propose",
    streamer_id_source: str = "fallback",
    metadata_streamer_id: str | None = None,
    override_streamer_id: str | None = None,
    mismatch_warning: str | None = None,
) -> ProfileUpdateProposal:
    """Build deterministic profile-update candidates from final selected clips."""

    observations: list[StreamerObservation] = []

    for clip in final_selected_clips:
        if clip.get("inside_joke"):
            observations.append(
                _build_observation(
                    vod_id=vod_id,
                    clip=clip,
                    observation_type="inside_joke",
                    claim=str(clip.get("inside_joke")),
                )
            )

        narrative_arc = (
            (clip.get("intelligence_report") or {}).get("narrative_arc")
            if isinstance(clip, dict)
            else None
        )
        if narrative_arc:
            observations.append(
                _build_observation(
                    vod_id=vod_id,
                    clip=clip,
                    observation_type="content_pattern",
                    claim=str(narrative_arc),
                )
            )

    observations = dedupe_observations(observations)

    promote_mode: Literal["propose", "auto", "off"] = "propose"
    if mode == "auto":
        promote_mode = "auto"
    elif mode == "off":
        promote_mode = "off"

    return ProfileUpdateProposal(
        vod_id=vod_id,
        streamer_id=streamer_id,
        streamer_id_source=(
            streamer_id_source
            if streamer_id_source in {"metadata", "override", "fallback"}
            else "fallback"
        ),
        metadata_streamer_id=metadata_streamer_id,
        override_streamer_id=override_streamer_id,
        mismatch_warning=mismatch_warning,
        candidate_observations=observations,
        promote_mode=promote_mode,
    )


def _upsert_inside_jokes(
    profile: StreamerProfile,
    observations: list[StreamerObservation],
) -> None:
    existing = {
        normalize_claim(j.key): j
        for j in profile.inside_jokes
    }
    for obs in observations:
        if obs.type != "inside_joke":
            continue
        norm = normalize_claim(obs.claim)
        if not norm:
            continue

        current = existing.get(norm)
        if current is None:
            profile.inside_jokes.append(
                InsideJoke(
                    key=obs.claim,
                    description=obs.claim,
                    context=f"Observed in VOD {obs.vod_id}",
                    confidence=obs.confidence,
                    evidence_refs=list(obs.evidence_refs),
                    created_at=obs.created_at,
                )
            )
            existing[norm] = profile.inside_jokes[-1]
            continue

        if obs.confidence > current.confidence:
            current.confidence = obs.confidence
            current.description = obs.claim
        merged_refs = list(dict.fromkeys([*current.evidence_refs, *obs.evidence_refs]))
        current.evidence_refs = merged_refs
        current.updated_at = datetime.now(timezone.utc)


def _upsert_content_patterns(
    profile: StreamerProfile,
    observations: list[StreamerObservation],
) -> None:
    existing = {
        normalize_claim(p.pattern): p
        for p in profile.content_patterns
    }
    for obs in observations:
        if obs.type != "content_pattern":
            continue
        norm = normalize_claim(obs.claim)
        if not norm:
            continue

        current = existing.get(norm)
        if current is None:
            profile.content_patterns.append(
                ContentPattern(
                    pattern=obs.claim,
                    description=obs.claim,
                    impact="neutral",
                    confidence=obs.confidence,
                    evidence_refs=list(obs.evidence_refs),
                    created_at=obs.created_at,
                )
            )
            existing[norm] = profile.content_patterns[-1]
            continue

        if obs.confidence > current.confidence:
            current.confidence = obs.confidence
            current.description = obs.claim
        merged_refs = list(dict.fromkeys([*current.evidence_refs, *obs.evidence_refs]))
        current.evidence_refs = merged_refs
        current.updated_at = datetime.now(timezone.utc)


def apply_profile_update_auto(
    profile: StreamerProfile,
    proposal: ProfileUpdateProposal,
    min_auto_confidence: float = 0.80,
) -> tuple[StreamerProfile, list[StreamerObservation], list[StreamerObservation], list[StreamerObservation]]:
    """Apply auto-merge policy to a profile update proposal.

    Returns:
      (updated_profile, accepted, queued, rejected)
    """

    deduped = dedupe_observations(proposal.candidate_observations)
    accepted, queued, rejected = partition_observations_for_merge(
        deduped,
        min_auto_confidence=min_auto_confidence,
    )

    updated = profile.model_copy(deep=True)
    _upsert_inside_jokes(updated, accepted)
    _upsert_content_patterns(updated, accepted)
    updated.updated_at = datetime.now(timezone.utc)

    return updated, accepted, queued, rejected
