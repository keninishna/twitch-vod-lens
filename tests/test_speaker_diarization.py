from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.preprocessing.speaker_diarization import diarize_audio


class _FakeTurn:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class _FakeDiarization:
    def __init__(self, rows):
        self._rows = rows

    def itertracks(self, yield_label=True):
        return iter(self._rows)


class _FakeOutput:
    def __init__(self, exclusive_rows=None, fallback_rows=None):
        if exclusive_rows is not None:
            self.exclusive_speaker_diarization = _FakeDiarization(exclusive_rows)
        if fallback_rows is not None:
            self.speaker_diarization = _FakeDiarization(fallback_rows)


class _FakePipeline:
    init_args = None
    call_args = None

    @classmethod
    def from_pretrained(cls, model_id, use_auth_token=None):
        cls.init_args = (model_id, use_auth_token)
        return cls()

    def to(self, *_args, **_kwargs):
        return self

    def __call__(self, _audio_path, **kwargs):
        _FakePipeline.call_args = kwargs
        return _FakeOutput(
            exclusive_rows=[
                (_FakeTurn(0.0, 1.0), None, "SPEAKER_01"),
                (_FakeTurn(0.8, 2.0), None, "SPEAKER_00"),
            ]
        )


class _FakePyannoteAudio:
    Pipeline = _FakePipeline


def test_diarize_audio_prefers_exclusive_and_normalizes_overlap(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "token-from-env")
    monkeypatch.setattr(
        "src.preprocessing.speaker_diarization.importlib.import_module",
        lambda _name: _FakePyannoteAudio,
    )

    turns = diarize_audio(
        audio_path=Path("sample.wav"),
        min_speakers=1,
        max_speakers=3,
    )

    assert _FakePipeline.init_args == (
        "pyannote/speaker-diarization-community-1",
        "token-from-env",
    )
    assert _FakePipeline.call_args == {"min_speakers": 1, "max_speakers": 3}

    assert len(turns) == 2
    assert turns[0].speaker_label == "SPEAKER_01"
    assert turns[0].start == pytest.approx(0.0)
    assert turns[0].end == pytest.approx(1.0)
    assert turns[1].speaker_label == "SPEAKER_00"
    assert turns[1].start == pytest.approx(1.0)
    assert turns[1].end == pytest.approx(2.0)


def test_speakerid_integration_import_gate():
    pytest.importorskip("pyannote.audio")
    if os.environ.get("RUN_SPEAKERID_INTEGRATION") != "1":
        pytest.skip("Set RUN_SPEAKERID_INTEGRATION=1 to run speaker-id integration tests")
