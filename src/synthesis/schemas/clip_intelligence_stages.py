"""Stage schema contracts for VOD Lens clip intelligence pipeline.

These models define strict JSON contracts for each stage in the
progressive intelligence refactor:
- discovery
- stitched
- scored
- final_selected
- context
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Type

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class StageContractValidationError(ValueError):
    """Raised when a stage payload fails schema validation."""


class StrictModel(BaseModel):
    """Base model that forbids undeclared fields."""

    model_config = ConfigDict(extra="forbid")


class TranscriptLine(StrictModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def _validate_range(self) -> "TranscriptLine":
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return self


class ChatMessage(StrictModel):
    timestamp: float = Field(ge=0)
    user: str
    message: str


class ChatReadFlag(StrictModel):
    timestamp: float = Field(ge=0)
    user: str
    message: str
    matched_transcript: str


class DeadAirGap(StrictModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    duration: float = Field(ge=0)

    @model_validator(mode="after")
    def _validate_duration(self) -> "DeadAirGap":
        if self.end < self.start:
            raise ValueError("dead air gap end must be >= start")
        return self


class SpeakerTurnLite(StrictModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    speaker_label: str
    identity: Literal["streamer", "guest", "unknown", "chatter", "mixed"] = "unknown"
    inferred_name: str | None = None
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _validate_range(self) -> "SpeakerTurnLite":
        if self.end <= self.start:
            raise ValueError("speaker turn end must be > start")
        return self


class ClipContext(StrictModel):
    clip_start: float = Field(ge=0)
    clip_end: float = Field(gt=0)
    transcript_lines: List[TranscriptLine]
    chat_messages: List[ChatMessage]
    chat_read_flags: List[ChatReadFlag]
    dead_air_gaps: List[DeadAirGap]
    total_dead_air_seconds: float = Field(ge=0)
    dead_air_ratio: float = Field(ge=0, le=1)
    objects_detected: List[str]
    speaker_turns: List[SpeakerTurnLite] = Field(default_factory=list)
    primary_speaker_label: str | None = None
    primary_speaker_identity: Literal["streamer", "guest", "unknown", "chatter", "mixed"] | None = None
    primary_speaker_name: str | None = None
    streamer_speaking_seconds: float = Field(default=0.0, ge=0)
    streamer_speaking_ratio: float = Field(default=0.0, ge=0, le=1)
    streamer_speaking_confidence: float = Field(default=0.0, ge=0, le=1)
    off_streamer_voice_detected: bool = False
    speaker_name_evidence: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_window(self) -> "ClipContext":
        if self.clip_end <= self.clip_start:
            raise ValueError("clip_end must be > clip_start")
        return self


class DiscoveryCandidate(StrictModel):
    candidate_id: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    narrative_type: str
    trigger: str
    payoff: str
    evidence_lines: List[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _validate_window(self) -> "DiscoveryCandidate":
        if self.end <= self.start:
            raise ValueError("end must be > start")
        return self


class StitchedCandidate(StrictModel):
    stitched_id: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    narrative_type: str
    trigger: str
    payoff: str
    evidence_lines: List[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    source_candidate_ids: List[str] = Field(min_length=1)
    source_windows: List[List[float]] = Field(min_length=1)
    merge_reasons: List[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_window(self) -> "StitchedCandidate":
        if self.end <= self.start:
            raise ValueError("end must be > start")
        return self


class PenaltyTraceEntry(StrictModel):
    code: str
    points: float


class HardGateEntry(StrictModel):
    code: str
    action: str


class ScoredCandidate(StrictModel):
    candidate_id: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    final_score: float = Field(ge=0, le=10)
    raw_score: float = Field(ge=0, le=10)
    eligible_for_final: bool
    penalty_trace: List[PenaltyTraceEntry]
    hard_gates: List[HardGateEntry]
    rejection_reasons: List[str]
    trim_source: Literal["qwen", "rms_fallback", "python_corrected"]
    clamped_trim_start: float | None = None
    clamped_trim_end: float | None = None

    @model_validator(mode="after")
    def _validate_window(self) -> "ScoredCandidate":
        if self.end <= self.start:
            raise ValueError("end must be > start")
        return self


class IntelligenceReport(StrictModel):
    why_selected: str
    narrative_arc: str
    evidence: List[str] = Field(min_length=1)
    trim_rationale: str
    duration_fit: str
    platform_fit: str
    risks: List[str]
    streamer_feedback: str


class FinalSpeakerAttribution(StrictModel):
    primary_speaker_identity: Literal["streamer", "guest", "unknown", "chatter", "mixed"] = "unknown"
    primary_speaker_name: str | None = None
    streamer_speaking_ratio: float = Field(ge=0, le=1)
    streamer_speaking_confidence: float = Field(ge=0, le=1)
    off_streamer_voice_detected: bool = False
    evidence: List[str] = Field(default_factory=list)


class FinalSelectedClip(StrictModel):
    rank: int = Field(ge=1)
    clip_id: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    suggested_trim_start: float = Field(ge=0)
    suggested_trim_end: float = Field(gt=0)
    trim_source: Literal["qwen", "rms_fallback", "python_corrected"]
    score: float = Field(ge=0, le=10)
    raw_score: float = Field(ge=0, le=10)
    normalized_score: float = Field(ge=0, le=10)
    title: str
    clip_point: str
    narrative_type: str
    platform_scores: Dict[str, float]
    platform_recommendations: List[str]
    intelligence_report: IntelligenceReport
    speaker_attribution: FinalSpeakerAttribution | None = None

    @model_validator(mode="after")
    def _validate_ranges(self) -> "FinalSelectedClip":
        if self.end <= self.start:
            raise ValueError("end must be > start")
        if self.suggested_trim_end <= self.suggested_trim_start:
            raise ValueError("suggested_trim_end must be > suggested_trim_start")
        if self.suggested_trim_start < self.start or self.suggested_trim_end > self.end:
            raise ValueError("suggested trim range must remain inside candidate range")
        return self


StageModel = Type[StrictModel]

_STAGE_MODEL_MAP: Dict[str, StageModel] = {
    "discovery": DiscoveryCandidate,
    "stitched": StitchedCandidate,
    "scored": ScoredCandidate,
    "final_selected": FinalSelectedClip,
    "context": ClipContext,
}


def validate_stage_payload(stage: str, payload: Dict[str, Any]) -> StrictModel:
    """Validate payload against the named stage contract.

    Raises StageContractValidationError on unknown stages or invalid payloads.
    """

    model_cls = _STAGE_MODEL_MAP.get(stage)
    if model_cls is None:
        raise StageContractValidationError(f"Unknown stage '{stage}'")

    try:
        return model_cls.model_validate(payload)
    except ValidationError as exc:
        raise StageContractValidationError(
            f"Stage '{stage}' payload failed validation: {exc}"
        ) from exc
