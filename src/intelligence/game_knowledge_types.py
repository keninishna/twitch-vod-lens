from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class GameCategorySegment(BaseModel):
    start: float = 0.0
    end: float | None = None
    game_id: str | None = None
    game_name: str
    source: str = "unknown"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("end")
    @classmethod
    def _validate_end(cls, value: float | None) -> float | None:
        return value


class GameKnowledgeSource(BaseModel):
    source_type: Literal["igdb", "wikipedia", "steam", "official", "web", "manual"]
    title: str
    url: str | None = None
    source_id: str | None = None
    updated_at: int | None = None
    retrieved_at: str
    content_hash: str | None = None


class GameKnowledgeProfile(BaseModel):
    schema_version: int = 1
    game_id: str | None = None
    game_name: str
    slug: str
    aliases: list[str] = Field(default_factory=list)
    summary: str = ""
    genres: list[str] = Field(default_factory=list)
    core_objective: str = ""
    core_mechanics: list[str] = Field(default_factory=list)
    common_terms: list[str] = Field(default_factory=list)
    notable_entities: list[str] = Field(default_factory=list)
    clip_eval_hints: list[str] = Field(default_factory=list)
    spoiler_sensitive: bool = False
    source_updated_at: int | None = None
    sources: list[GameKnowledgeSource] = Field(default_factory=list)
    generated_at: str
    refreshed_at: str
