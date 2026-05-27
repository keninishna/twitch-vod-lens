"""Persistent streamer intelligence type contracts (SpeakerID Phase 04)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


ObservationType = Literal[
    "inside_joke",
    "personality_trait",
    "community_chatter",
    "content_pattern",
    "voice_profile",
    "guest_identity",
    "clip_quality_lesson",
    "other",
]

ObservationSource = Literal[
    "transcript",
    "chat",
    "speaker_attribution",
    "fusion",
    "llm_summary",
    "manual",
]

VoiceRole = Literal["streamer", "guest", "chatter", "unknown"]
PatternImpact = Literal["high_value", "low_value", "neutral"]


class DurableClaim(BaseModel):
    """Shared durability contract for profile facts and observations."""

    confidence: float = Field(ge=0.0, le=1.0)
    evidence_refs: List[str] = Field(default_factory=list, min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None


class VoiceProfileRef(DurableClaim):
    """Reference to a reusable enrolled voice profile artifact."""

    profile_id: str
    path: str
    role: VoiceRole = "unknown"
    display_name: Optional[str] = None


class PersonalityTrait(DurableClaim):
    """Evidence-backed personality or style trait."""

    trait: str
    description: Optional[str] = None


class CommunityChatterSummary(DurableClaim):
    """Summary fact about a recurring chatter/community member."""

    username: str
    aliases: List[str] = Field(default_factory=list)
    role: Optional[str] = None
    message_count: int = Field(default=0, ge=0)
    last_seen_vod_id: Optional[str] = None


class InsideJoke(DurableClaim):
    """A recurring in-joke with compact explanation."""

    key: str
    description: str
    context: Optional[str] = None


class ContentPattern(DurableClaim):
    """Pattern observed in clips/content performance."""

    pattern: str
    description: str
    impact: PatternImpact = "neutral"


class StreamerObservation(DurableClaim):
    """Immutable evidence-backed observation from one VOD."""

    observation_id: Optional[str] = None
    vod_id: str
    timestamp_start: float = Field(ge=0.0)
    timestamp_end: float = Field(gt=0.0)
    type: ObservationType
    claim: str
    evidence: List[str] = Field(default_factory=list, min_length=1)
    source: ObservationSource

    @model_validator(mode="after")
    def _validate_time_range(self) -> "StreamerObservation":
        if self.timestamp_end <= self.timestamp_start:
            raise ValueError("timestamp_end must be greater than timestamp_start")
        return self


class ProfileUpdateProposal(BaseModel):
    """Post-run proposal for promoting new observations into profile state."""

    vod_id: str
    streamer_id: str
    streamer_id_source: Literal["metadata", "override", "fallback"] = "fallback"
    metadata_streamer_id: Optional[str] = None
    override_streamer_id: Optional[str] = None
    mismatch_warning: Optional[str] = None
    candidate_observations: List[StreamerObservation] = Field(default_factory=list)
    promote_mode: Literal["propose", "auto", "off"] = "propose"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StreamerProfile(BaseModel):
    """Persistent per-streamer intelligence profile."""

    streamer_id: str
    profile_version: int = 1
    display_name: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    voice_profiles: List[VoiceProfileRef] = Field(default_factory=list)
    personality_traits: List[PersonalityTrait] = Field(default_factory=list)
    community_chatters: List[CommunityChatterSummary] = Field(default_factory=list)
    inside_jokes: List[InsideJoke] = Field(default_factory=list)
    content_patterns: List[ContentPattern] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
