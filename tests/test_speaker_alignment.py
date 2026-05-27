from __future__ import annotations

from src.preprocessing.speaker_alignment import assign_speakers_to_transcript, overlap_seconds
from src.preprocessing.types import SpeakerTurn


def test_overlap_seconds_exact_and_partial_and_none():
    assert overlap_seconds(0, 10, 0, 10) == 10
    assert overlap_seconds(0, 10, 8, 12) == 2
    assert overlap_seconds(0, 10, 10, 20) == 0


def test_assign_speakers_to_transcript_exact_overlap():
    transcript = [{"start": 10.0, "end": 14.0, "text": "hello"}]
    turns = [SpeakerTurn(start=9.0, end=14.0, speaker_label="SPEAKER_00")]

    labeled = assign_speakers_to_transcript(transcript, turns)
    assert labeled[0]["speaker_label"] == "SPEAKER_00"
    assert labeled[0]["text"] == "hello"


def test_assign_speakers_to_transcript_partial_overlap_under_threshold_is_unknown():
    transcript = [{"start": 10.0, "end": 20.0, "text": "long segment"}]
    turns = [SpeakerTurn(start=10.0, end=12.0, speaker_label="SPEAKER_00")]

    labeled = assign_speakers_to_transcript(transcript, turns)
    assert labeled[0]["speaker_label"] == "UNKNOWN"


def test_assign_speakers_to_transcript_no_overlap_is_unknown():
    transcript = [{"start": 30.0, "end": 35.0, "text": "no speaker"}]
    turns = [SpeakerTurn(start=0.0, end=5.0, speaker_label="SPEAKER_00")]

    labeled = assign_speakers_to_transcript(transcript, turns)
    assert labeled[0]["speaker_label"] == "UNKNOWN"


def test_assign_speakers_to_transcript_tie_handling_is_stable_with_sorted_turns():
    transcript = [{"start": 10.0, "end": 14.0, "text": "tie case"}]
    turns = [
        SpeakerTurn(start=8.0, end=12.0, speaker_label="SPEAKER_B"),
        SpeakerTurn(start=12.0, end=16.0, speaker_label="SPEAKER_A"),
    ]

    labeled = assign_speakers_to_transcript(transcript, turns)
    assert labeled[0]["speaker_label"] == "SPEAKER_B"
