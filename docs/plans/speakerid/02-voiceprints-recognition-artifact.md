# SpeakerID Phase 02 — Voice Thumbprints, Recognition, Name Inference, and Attribution Artifact

> **For Hermes:** Load only this phase doc plus `INDEX.md` unless you need cross-phase details. Use `subagent-driven-development` to execute one task at a time.

**Goal:** Enroll reusable voice thumbprints, recognize diarized speakers, infer names from dialogue, and produce the per-VOD speaker attribution artifact.

**Depends on:** Phase 01 speaker turns and aligned transcript.

**Tasks covered:** 6–9.

**Status (May 23, 2026):** ✅ Implemented in repo (`speaker_profiles.py`, `speaker_enroll.py`, `speaker_recognition.py`, `speaker_name_inference.py`, `speaker_attribution.py`, `run_speaker_attribution.py`) with focused tests passing.

---

### Task 6: Implement voice thumbprint enrollment

**Objective:** Create reusable speaker profiles from verified streamer voice samples.

**Files:**
- Create: `src/preprocessing/speaker_profiles.py`
- Create: `src/preprocessing/speaker_enroll.py`
- Test: `tests/test_speaker_profiles.py`

**Public APIs:**
```python
def compute_embedding(wav_path: Path, device: str = "auto") -> list[float]: ...
def average_embeddings(embeddings: list[list[float]]) -> list[float]: ...
def save_profile(profile: dict, profile_dir: Path) -> Path: ...
def load_profiles(profile_dir: Path) -> list[dict]: ...
```

**CLI:**
```bash
PYTHONPATH=. python3 -m src.preprocessing.speaker_enroll \
  --profile-id streamer_skitch \
  --display-name Skitch \
  --role streamer \
  --audio vods/phase4_<VOD_ID>/raw/<VOD_ID>.mp4 \
  --segments 30-90,300-360 \
  --output-dir data/speaker_profiles
```

**Implementation notes:**
- Import SpeechBrain inside functions.
- Use ffmpeg utility from Task 3 to extract each enrollment segment.
- Normalize embeddings before averaging.
- Store provenance for every segment used.

### Task 7: Implement speaker recognition against profiles

**Objective:** Map diarized labels to known identities such as `streamer`.

**Files:**
- Create: `src/preprocessing/speaker_recognition.py`
- Test: `tests/test_speaker_recognition.py`

**Functions:**
- `cosine_similarity(a, b) -> float`
- `recognize_speaker_clusters(audio_path, speaker_turns, profiles, output_dir) -> dict[str, SpeakerRecognitionResult]`
- `aggregate_cluster_embeddings(...)`

**Rules:**
- Ignore turns shorter than 1.5s for embeddings.
- For each diarized label, sample up to N seconds / M turns to avoid long runtime, e.g. max 120s total per speaker.
- Accept a profile when similarity >= profile threshold (`default 0.72`).
- Mark high confidence when similarity >= high-confidence threshold (`default 0.80`).
- If no profile passes threshold, identity remains `unknown`.

**Verification:** tests use deterministic vectors for cosine similarity and profile thresholding.

### Task 8: Implement text-based speaker name inference

**Objective:** Infer likely real names for anonymous speaker labels from dialogue context.

**Files:**
- Create: `src/preprocessing/speaker_name_inference.py`
- Test: `tests/test_speaker_name_inference.py`

**Functions:**
- `extract_name_mentions(text: str) -> list[str]`
- `infer_names_heuristic(diarized_transcript, chat_messages=None) -> dict[str, list[SpeakerNameCandidate]]`
- `build_qwen_name_resolution_prompt(...) -> str`
- `merge_name_candidates(heuristic_candidates, qwen_candidates) -> dict`

**Required test cases:**
1. `SPEAKER_00: hey Skitch` followed by `SPEAKER_01: yeah thanks` -> `SPEAKER_01` candidate `Skitch`.
2. `SPEAKER_01: I'm Skitch` -> `SPEAKER_01` candidate `Skitch` high confidence.
3. Streamer greets chat usernames with no voice response -> no assignment.
4. Multiple names in one utterance -> lower confidence / ambiguous.
5. Name is also a chat user in nearby chat -> mark candidate evidence as chat-addressed unless a voice response exists.

**LLM prompt contract:** output only JSON:
```json
{
  "speaker_name_candidates": [
    {
      "speaker_label": "SPEAKER_01",
      "name": "Skitch",
      "confidence": 0.72,
      "evidence": ["..."] ,
      "reasoning_short": "Addressed by name and responded immediately"
    }
  ]
}
```

Use Qwen only as a verifier/resolver; deterministic evidence and confidence remain auditable.

### Task 9: Build artifact orchestrator

**Objective:** Generate the full `speaker_attribution_<VOD_ID>.json` artifact in one command.

**Files:**
- Create: `src/preprocessing/speaker_attribution.py`
- Create: `src/preprocessing/run_speaker_attribution.py`
- Test: `tests/test_speaker_attribution.py`

**CLI:**
```bash
PYTHONPATH=. python3 -m src.preprocessing.run_speaker_attribution \
  --vod-id <VOD_ID> \
  --vod-media vods/phase4_<VOD_ID>/raw/<VOD_ID>.mp4 \
  --transcript vods/phase4_<VOD_ID>/transcript.json \
  --chat vods/phase4_<VOD_ID>/chat.json \
  --profiles-dir data/streamer_intelligence/<STREAMER_ID>/voice_profiles \
  --output vods/phase4_<VOD_ID>/speaker_attribution_<VOD_ID>.json
```

**Pipeline inside command:**
1. Extract 16 kHz mono WAV.
2. Diarize WAV.
3. Align speaker turns to transcript.
4. Recognize clusters against voice profiles.
5. Infer name candidates from diarized transcript and chat.
6. Aggregate cluster summaries.
7. Write artifact.

**Failure behavior:**
- If optional deps/token are missing, fail with actionable message.
- If profiles are missing, still write diarization + name inference with identities as `unknown`.
- Never block baseline preprocessing unless user passes `--require-speaker-id`.
