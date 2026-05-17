"""Schema contracts for synthesis pipeline stages."""

from .clip_intelligence_stages import (
    ClipContext,
    DiscoveryCandidate,
    FinalSelectedClip,
    ScoredCandidate,
    StageContractValidationError,
    StitchedCandidate,
    validate_stage_payload,
)

__all__ = [
    "ClipContext",
    "DiscoveryCandidate",
    "FinalSelectedClip",
    "ScoredCandidate",
    "StageContractValidationError",
    "StitchedCandidate",
    "validate_stage_payload",
]
