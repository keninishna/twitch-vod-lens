from __future__ import annotations

from pathlib import Path

from src.preprocessing import pipeline, prepare_phase4
from src.preprocessing.types import (
    ChatActivity,
    ChatAnalysis,
    FusionResult,
    TranscriptResult,
    TranscriptSegment,
    VodMeta,
)


def test_run_pipeline_minimal_uses_typed_fusion_path(tmp_path: Path, monkeypatch) -> None:
    audio_path = tmp_path / "dummy.wav"
    audio_path.write_bytes(b"not-real-audio")

    def _fake_transcribe(*, audio_path: Path, model_size: str, language: str) -> TranscriptResult:
        return TranscriptResult(
            segments=[
                TranscriptSegment(
                    start=0.0,
                    end=2.0,
                    text="hello world",
                    confidence=0.95,
                )
            ],
            language=language,
            duration_seconds=2.0,
        )

    def _fake_download_chat(vod_id: str):
        return []

    def _fake_analyze_chat(messages) -> ChatAnalysis:
        return ChatAnalysis(
            messages=[],
            activity=[
                ChatActivity(
                    window_start=0.0,
                    window_end=10.0,
                    message_count=0,
                    unique_users=0,
                    peak_emote=None,
                    peak_emote_count=0,
                )
            ],
            total_messages=0,
            unique_chatters=0,
            total_emotes=0,
        )

    monkeypatch.setattr(pipeline, "transcribe", _fake_transcribe)
    monkeypatch.setattr(pipeline, "download_chat", _fake_download_chat)
    monkeypatch.setattr(pipeline, "analyze_chat", _fake_analyze_chat)

    result = pipeline.run_pipeline_minimal(
        audio_path=audio_path,
        url=None,
        workdir=tmp_path,
        model_size="tiny",
        language="en",
    )

    assert result.vod_meta.id == "test"
    assert result.transcript.duration_seconds == 2.0
    assert len(result.timeline) == 1
    assert result.timeline[0].transcript is not None
    assert result.timeline[0].chat_intensity == 0.0
