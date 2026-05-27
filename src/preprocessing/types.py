"""
VOD Lens — Shared Type Contracts

These Pydantic models define the input/output schemas for every
worker module in the preprocessing pipeline. All modules import
from this file to prevent field name drift across workers.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Literal, Optional


class VodMeta(BaseModel):
    """Metadata about a VOD, extracted from yt-dlp or Twitch API."""
    id: str
    title: str
    duration_seconds: int
    url: str
    streamer: str
    resolution: Optional[str] = None
    fps: Optional[float] = None
    format: Optional[str] = None


class TranscriptSegment(BaseModel):
    """A single utterance from Whisper transcription."""
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text: str
    confidence: float = Field(ge=0.0, le=1.0)


class WordTiming(BaseModel):
    """Word-level timing from Whisper."""
    word: str
    start: float
    end: float
    confidence: float = Field(ge=0.0, le=1.0)


class TranscriptResult(BaseModel):
    """Full transcription output."""
    segments: List[TranscriptSegment]
    language: str = "en"
    duration_seconds: float
    word_timings: Optional[List[WordTiming]] = None


class SceneBoundary(BaseModel):
    """A detected scene change boundary."""
    timestamp: float = Field(description="Time of scene change in seconds")
    frame_number: int
    score: float = Field(description="Detection confidence score")


class SceneClip(BaseModel):
    """A segment between two scene boundaries."""
    index: int
    start: float
    end: float
    duration: float
    label: Optional[str] = None


class ChatMessage(BaseModel):
    """A single chat message from a VOD."""
    timestamp: float = Field(description="Time in seconds into the VOD")
    user: str
    message: str
    emotes: Optional[List[str]] = None
    is_subscriber: bool = False
    is_moderator: bool = False


class ChatActivity(BaseModel):
    """Aggregated chat activity over time."""
    window_start: float
    window_end: float
    message_count: int
    unique_users: int
    peak_emote: Optional[str] = None
    peak_emote_count: int = 0


class ChatAnalysis(BaseModel):
    """Full chat analysis output."""
    messages: List[ChatMessage]
    activity: List[ChatActivity]
    total_messages: int
    unique_chatters: int
    total_emotes: int


class FusionTimeline(BaseModel):
    """Unified timeline entry combining transcript, scenes, and chat."""
    timestamp: float
    transcript: Optional[TranscriptSegment] = None
    scene_change: bool = False
    scene_index: Optional[int] = None
    chat_intensity: float = 0.0
    top_emotes: List[str] = []


class FusionResult(BaseModel):
    """Final fused output of the preprocessing pipeline."""
    vod_meta: VodMeta
    transcript: TranscriptResult
    scenes: List[SceneClip]
    chat: ChatAnalysis
    timeline: List[FusionTimeline]
    processing_time_seconds: float


class ClipSuggestion(BaseModel):
    """A suggested clip from the VOD (produced by fusion/synthesis)."""
    start: float
    end: float
    title: str
    reason: str
    tags: List[str]
    platform_scores: dict = Field(
        default_factory=lambda: {
            "tiktok": 0.0,
            "youtube_shorts": 0.0,
            "instagram_reels": 0.0,
        }
    )


SpeakerIdentity = Literal["streamer", "guest", "unknown", "chatter", "mixed"]


class SpeakerRecognitionResult(BaseModel):
    """Voice-profile match output for a speaker turn or cluster."""

    identity: SpeakerIdentity = "unknown"
    confidence: float = Field(ge=0.0, le=1.0)
    cosine_similarity: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    profile_id: Optional[str] = None


class SpeakerNameCandidate(BaseModel):
    """Text-inferred possible speaker name with supporting evidence."""

    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)


class SpeakerTurn(BaseModel):
    """Anonymous diarization turn with optional recognition + name inference."""

    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    speaker_label: str
    exclusive: bool = True
    recognition: Optional[SpeakerRecognitionResult] = None
    inferred_name: Optional[str] = None

    @model_validator(mode="after")
    def _validate_range(self) -> "SpeakerTurn":
        if self.end <= self.start:
            raise ValueError("SpeakerTurn end must be greater than start")
        return self


class SpeakerClusterSummary(BaseModel):
    """Aggregate stats for each diarization speaker label."""

    total_speech_seconds: float = Field(ge=0.0)
    segment_count: int = Field(ge=0)
    primary_identity: SpeakerIdentity = "unknown"
    primary_identity_confidence: float = Field(ge=0.0, le=1.0)
    candidate_names: List[SpeakerNameCandidate] = Field(default_factory=list)


class ClipSpeakerStats(BaseModel):
    """Per-clip speaker composition summary used in synthesis context."""

    primary_speaker_label: str
    primary_speaker_identity: SpeakerIdentity = "unknown"
    primary_speaker_name: Optional[str] = None
    streamer_speaking_seconds: float = Field(ge=0.0)
    streamer_speaking_ratio: float = Field(ge=0.0, le=1.0)
    streamer_speaking_confidence: float = Field(ge=0.0, le=1.0)
    off_streamer_voice_detected: bool = False
    dominant_non_streamer_label: Optional[str] = None
    dominant_non_streamer_name: Optional[str] = None


class SpeakerAttributionResult(BaseModel):
    """Full per-VOD speaker attribution artifact contract."""

    vod_id: str
    audio_path: str
    backend: Dict[str, str] = Field(default_factory=dict)
    segments: List[SpeakerTurn]
    speaker_clusters: Dict[str, SpeakerClusterSummary] = Field(default_factory=dict)
    clip_speaker_stats: Dict[str, ClipSpeakerStats] = Field(default_factory=dict)