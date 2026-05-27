# SpeakerID Phase 01 — Foundations, Dependencies, Diarization, and Alignment

> **For Hermes:** Load only this phase doc plus `INDEX.md` unless you need cross-phase details. Use `subagent-driven-development` to execute one task at a time.

**Goal:** Establish optional dependencies, schema contracts, audio extraction, diarization, and transcript alignment.

**Depends on:** Phase 00 contracts.

**Tasks covered:** 1–5.

---

### Task 1: Add optional dependency manifest and ignore local voice profiles

**Objective:** Add speaker-ID dependencies without breaking the existing lightweight runtime.

**Files:**
- Create: `requirements-speakerid.txt`
- Modify: `.gitignore`

**Steps:**
1. Create `requirements-speakerid.txt`:
   ```text
   pyannote.audio>=4.0.0
   speechbrain>=1.0.0
   torchaudio>=2.0.0
   soundfile>=0.12.1
   librosa>=0.10.0
   ```
2. Add to `.gitignore`:
   ```text
   data/speaker_profiles/*.json
   data/speaker_profiles/*.wav
   data/streamer_intelligence/*/voice_profiles/*.json
   data/streamer_intelligence/*/voice_profiles/*.wav
   pretrained_models/
   ```
3. Verification:
   ```bash
   git diff -- requirements-speakerid.txt .gitignore
   ```

### Task 2: Add speaker attribution Pydantic models

**Objective:** Define strict shared contracts before implementation modules import each other.

**Files:**
- Modify: `src/preprocessing/types.py`
- Test: `tests/test_speaker_attribution_types.py`

**Models to add:**
- `SpeakerTurn`
- `SpeakerRecognitionResult`
- `SpeakerNameCandidate`
- `SpeakerClusterSummary`
- `ClipSpeakerStats`
- `SpeakerAttributionResult`

**Important fields:**
- `start`, `end`, `speaker_label`
- `identity`: `streamer | guest | unknown | chatter | mixed`
- `confidence`, `cosine_similarity`, `profile_id`
- `candidate_names` with evidence strings
- `streamer_speaking_seconds`, `streamer_speaking_ratio`, `off_streamer_voice_detected`

**Verification:**
```bash
pytest -q tests/test_speaker_attribution_types.py
```

Expected: schema accepts a minimal valid artifact and rejects invalid time ranges.

### Task 3: Implement audio extraction utilities

**Objective:** Produce clean 16 kHz mono WAV slices for diarization and embedding.

**Files:**
- Create: `src/preprocessing/audio_segments.py`
- Test: `tests/test_audio_segments.py`

**Functions:**
- `extract_wav(input_media: Path, output_wav: Path, start: float | None = None, end: float | None = None, sample_rate=16000) -> Path`
- `extract_turn_wavs(input_media: Path, turns: list[SpeakerTurn], output_dir: Path, min_duration=1.5) -> list[Path]`

**Command pattern:**
```bash
ffmpeg -y -i <input> -vn -ac 1 -ar 16000 -f wav <output.wav>
```

**Verification:** mock `subprocess.run` in unit tests; do not require real ffmpeg in unit tests.

### Task 4: Implement pyannote diarization backend

**Objective:** Generate anonymous speaker turns from VOD audio.

**Files:**
- Create: `src/preprocessing/speaker_diarization.py`
- Test: `tests/test_speaker_diarization.py`

**Public API:**
```python
def diarize_audio(
    audio_path: Path,
    hf_token: str | None = None,
    model_id: str = "pyannote/speaker-diarization-community-1",
    device: str = "auto",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[SpeakerTurn]:
    ...
```

**Implementation notes:**
- Import `pyannote.audio` inside the function so normal tests/imports do not fail when speaker deps are absent.
- Read token from `HF_TOKEN` or `HUGGINGFACE_TOKEN` if not provided.
- Prefer `output.exclusive_speaker_diarization` when available; fall back to `output.speaker_diarization`.
- Return sorted non-overlapping turns.

**Verification:**
- Unit test with fake pyannote output objects.
- Add an integration test marked `pytest.importorskip("pyannote.audio")` and skipped unless `RUN_SPEAKERID_INTEGRATION=1`.

### Task 5: Align diarization turns to transcript segments

**Objective:** Attach speaker labels to transcript segments by timestamp overlap.

**Files:**
- Create: `src/preprocessing/speaker_alignment.py`
- Test: `tests/test_speaker_alignment.py`

**Functions:**
- `overlap_seconds(a_start, a_end, b_start, b_end) -> float`
- `assign_speakers_to_transcript(transcript_segments, speaker_turns) -> list[dict]`

**Rules:**
- Assign the speaker with the largest overlap.
- If overlap ratio is below 0.30, set speaker_label to `UNKNOWN`.
- Preserve original transcript fields.
- If word timings exist later, add word-level speaker labels in a follow-up; segment-level is enough for Phase 1.

**Verification:** tests cover exact overlap, partial overlap, no overlap, and tie handling.
