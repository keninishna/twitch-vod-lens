# SpeakerID Phase 00 — Overview, Architecture, and Data Contracts

> **For Hermes:** Load only this phase doc plus `INDEX.md` unless you need cross-phase details. Use `subagent-driven-development` to execute one task at a time.

**Goal:** Understand the complete system shape without loading implementation task detail.

**Read next:** Phase 01 for implementation start.

---

## Research Summary

### Recommended initial stack

Use a **local, modular two-model stack**:

1. **pyannote.audio `speaker-diarization-community-1`** for diarization.
   - Ingests mono 16 kHz audio and outputs speaker turns.
   - Runs locally with `pyannote.audio` after accepting Hugging Face gated model terms.
   - Provides `exclusive_speaker_diarization`, useful for assigning one speaker per timestamp when aligning with transcript words.
   - Better current open-source default than older pyannote `speaker-diarization-3.1`.

2. **SpeechBrain ECAPA-TDNN `speechbrain/spkrec-ecapa-voxceleb`** for speaker embeddings / voice thumbprints.
   - Trained on VoxCeleb1/2.
   - Produces speaker embeddings and supports same/different speaker verification by cosine distance.
   - Works on 16 kHz mono audio.
   - Good fit for “enroll streamer voice once, then compare every diarized speaker cluster.”

3. **LLM + deterministic heuristics for text-based SpeakerID.**
   - Adobe Research’s 2024 text-based SpeakerID work shows transcript-only name attribution is viable and reports 80.3% precision in their setup.
   - AssemblyAI has a commercial “Speaker Identification” feature that maps generic speaker labels to provided names/roles without voice enrollment, using conversation context. This validates the product direction, but local Qwen should be used first to avoid sending audio externally.

4. **Persistent streamer intelligence store.**
   - Speaker attribution should not be isolated to one VOD. Every run can improve a streamer profile with reusable signals: verified voice profiles, aliases, recurring chatters, named guests, common bits, topic preferences, content boundaries, and title/clip-quality lessons.
   - The store must be **evidence-backed**, not a free-form memory blob. Every durable claim should point to VOD ID, timestamp span, source field, confidence, and last-seen metadata.
   - Treat persistent intelligence as **retrieval context and scoring support**, not ground truth. New VOD evidence can override old profile assumptions when the current VOD contradicts them.

### Alternatives considered

| Option | Pros | Cons | Recommendation |
|---|---|---|---|
| pyannote community-1 + SpeechBrain ECAPA | Local, proven, modular, debuggable | HF token/model gate, new deps | **Phase 1 default** |
| WhisperX diarization | Nice transcript+speaker alignment; already conceptually close to current Whisper pipeline | Current repo uses faster-whisper directly; WhisperX Docker integration may be heavier than adding standalone pyannote | Consider if replacing transcription path later |
| NVIDIA NeMo Sortformer | Modern end-to-end diarization, streaming variant, handles up to 4 speakers | Heavy NeMo dependency; more infra; overkill for first pass | Phase 3 optional backend |
| pyannoteAI Precision-2 / AssemblyAI | Higher accuracy / name inference as service | Paid/cloud/external audio | Optional benchmark, not default |

### Key product distinction

- **Diarization:** assigns anonymous labels like `SPEAKER_00`, `SPEAKER_01`.
- **Voice recognition / thumbprinting:** maps labels to known identities from enrolled voice profiles, e.g. `SPEAKER_00 -> streamer`.
- **Text-based name inference:** maps labels to names from dialogue evidence, e.g. `SPEAKER_01 -> Skitch`, based on utterances like “hey Skitch” followed by that speaker responding.

All three are needed. Diarization alone does **not** know real names.

---

## Target Data Contract

### New artifact

Create one JSON artifact per VOD:

```text
vods/phase4_<VOD_ID>/speaker_attribution_<VOD_ID>.json
```

Suggested schema:

```json
{
  "vod_id": "2776101332",
  "audio_path": "vods/phase4_2776101332/raw/2776101332.mp4",
  "backend": {
    "diarization": "pyannote/speaker-diarization-community-1",
    "embedding": "speechbrain/spkrec-ecapa-voxceleb",
    "name_inference": "heuristic+qwen"
  },
  "segments": [
    {
      "start": 12.34,
      "end": 16.78,
      "speaker_label": "SPEAKER_00",
      "exclusive": true,
      "recognition": {
        "identity": "streamer",
        "profile_id": "streamer_skitch",
        "confidence": 0.91,
        "cosine_similarity": 0.84
      },
      "inferred_name": null
    }
  ],
  "speaker_clusters": {
    "SPEAKER_00": {
      "total_speech_seconds": 4312.1,
      "segment_count": 812,
      "primary_identity": "streamer",
      "primary_identity_confidence": 0.91,
      "candidate_names": []
    },
    "SPEAKER_01": {
      "total_speech_seconds": 380.4,
      "segment_count": 94,
      "primary_identity": "unknown",
      "primary_identity_confidence": 0.42,
      "candidate_names": [
        {
          "name": "Skitch",
          "confidence": 0.72,
          "evidence": ["SPEAKER_00 at 142.1s says 'hey Skitch' and SPEAKER_01 responds at 144.3s"]
        }
      ]
    }
  },
  "clip_speaker_stats": {
    "120-240": {
      "primary_speaker_label": "SPEAKER_00",
      "primary_speaker_identity": "streamer",
      "primary_speaker_name": "Skitch",
      "streamer_speaking_seconds": 18.2,
      "streamer_speaking_ratio": 0.31,
      "streamer_speaking_confidence": 0.89,
      "off_streamer_voice_detected": true,
      "dominant_non_streamer_label": "SPEAKER_01",
      "dominant_non_streamer_name": "Guest/unknown"
    }
  }
}
```

### Voice profile artifact

Store enrolled voice thumbprints outside per-VOD output:

```text
data/speaker_profiles/<profile_id>.json
```

Suggested schema:

```json
{
  "profile_id": "streamer_skitch",
  "display_name": "Skitch",
  "role": "streamer",
  "embedding_model": "speechbrain/spkrec-ecapa-voxceleb",
  "embedding_dim": 192,
  "embedding": [0.0123, -0.0456],
  "created_from": [
    {"vod_id": "2776101332", "start": 30.0, "end": 90.0, "notes": "manually verified streamer solo segment"}
  ],
  "thresholds": {
    "accept_similarity": 0.72,
    "high_confidence_similarity": 0.80
  }
}
```

Do **not** commit real profile JSONs to git unless Ken explicitly wants that. Add `data/speaker_profiles/*.json` to `.gitignore` if the directory is inside the repo.

### Persistent streamer intelligence store

Store cross-VOD streamer intelligence separately from per-VOD outputs and raw speaker profiles:

```text
data/streamer_intelligence/<streamer_id>/profile.json
data/streamer_intelligence/<streamer_id>/observations.jsonl
data/streamer_intelligence/<streamer_id>/voice_profiles/<profile_id>.json
```

Use JSON first because it is reviewable, diffable, and easy for agents to patch. Move to SQLite only if profile size or concurrent writes become painful.

Suggested `profile.json` schema:

```json
{
  "streamer_id": "skitch",
  "display_name": "Skitch",
  "aliases": ["skitch", "Skitch"],
  "profile_version": 1,
  "updated_at": "2026-05-23T00:00:00Z",
  "voice_profiles": [
    {
      "profile_id": "streamer_skitch",
      "role": "streamer",
      "path": "voice_profiles/streamer_skitch.json",
      "embedding_model": "speechbrain/spkrec-ecapa-voxceleb",
      "accept_similarity": 0.72,
      "high_confidence_similarity": 0.80,
      "confidence": 0.91,
      "evidence_refs": ["obs_20260523_0001"]
    }
  ],
  "personality": {
    "traits": [
      {"label": "dry humor", "confidence": 0.72, "evidence_refs": ["obs_..."]}
    ],
    "stream_style": [
      {"label": "chat-driven banter", "confidence": 0.81, "evidence_refs": ["obs_..."]}
    ],
    "title_guidance": [
      {"rule": "prefer factual chat-attributed hooks over generic reaction titles", "confidence": 0.9, "evidence_refs": ["obs_..."]}
    ]
  },
  "community": {
    "common_chatters": [
      {"username": "example_chatter", "seen_vods": 3, "last_seen_vod_id": "2776101332", "roles": ["regular"], "notes": [], "evidence_refs": ["obs_..."]}
    ],
    "known_guests": [
      {"name": "GuestName", "voice_profile_id": null, "confidence": 0.68, "evidence_refs": ["obs_..."]}
    ],
    "inside_jokes": [
      {"label": "abrasive donation alert", "meaning": "community bit around harsh alert sound", "confidence": 0.86, "evidence_refs": ["obs_..."]}
    ]
  },
  "content_patterns": {
    "high_value_clip_patterns": [
      {"pattern": "inside joke explained to new chatter", "confidence": 0.84, "evidence_refs": ["obs_..."]}
    ],
    "low_value_clip_patterns": [
      {"pattern": "transactional donation laugh with no explanation", "confidence": 0.77, "evidence_refs": ["obs_..."]}
    ],
    "recurring_topics": []
  },
  "safety_and_privacy": {
    "do_not_infer": ["private personal details not explicitly stated in VOD"],
    "pii_policy": "store chat usernames and aggregate behavior only when useful for attribution; do not persist sensitive personal claims without explicit approval"
  }
}
```

Suggested `observations.jsonl` record:

```json
{
  "observation_id": "obs_20260523_0001",
  "vod_id": "2776101332",
  "timestamp_start": 120.0,
  "timestamp_end": 180.0,
  "type": "inside_joke|personality_trait|voiceprint|common_chatter|guest_identity|clip_quality_lesson",
  "claim": "The abrasive donation alert is a recurring inside joke that may need explanation for new viewers.",
  "confidence": 0.86,
  "source": "stage3_final_selected.intelligence_report",
  "evidence": ["Streamer explains the alert to a new chatter", "Chat reacts with recognition"],
  "promote_to_profile": true,
  "created_at": "2026-05-23T00:00:00Z"
}
```

### Persistent intelligence lifecycle

```text
Before analysis:
  resolve streamer_id from VOD metadata / CLI override
  load data/streamer_intelligence/<streamer_id>/profile.json if present
  load voice profile refs for speaker recognition
  render compact profile context into Stage 1/3 prompts

During analysis:
  speaker attribution uses persistent voice profiles
  clip context includes known chatters, known guests, and relevant inside jokes
  Qwen uses profile-aware speaker context to infer attribution/framing risks from current evidence

After analysis:
  extract candidate observations from final selected clips + rejected clips + chat/speaker artifacts
  ask Qwen for a profile-update proposal with evidence refs
  deterministically merge high-confidence, evidence-backed observations
  append all accepted observations to observations.jsonl
  update profile.json summaries/counters/last_seen fields
```

### Profile context injected into prompts

Keep prompt context compact. Do not dump the full profile. Render only relevant, high-confidence facts:

```text
### STREAMER PROFILE CONTEXT (evidence-backed, advisory)
Streamer: Skitch
Known voice profile: streamer_skitch (confidence 0.91)
Recurring community bits:
- abrasive donation alert: inside joke; do not treat laughter alone as clip-worthy unless setup/payoff explains it.
Common chatters near this VOD: user_a, user_b, user_c
Known clip-quality lessons:
- strong clips often involve explaining a community bit to a newcomer.
- weak clips: transactional donation reactions without story/context.
```

Rules:
1. Profile context is advisory and must not override current VOD transcript/frame/chat evidence.
2. Every injected fact must have confidence `>=0.65` and at least one evidence ref.
3. Prefer current-VOD evidence over older profile facts when they conflict.
4. The profile should help Qwen recognize lore and attribution, not invent new lore.

---

## Dynamic Name Inference Rules

Implement deterministic candidate extraction first, then optionally ask Qwen to resolve ambiguity.

### High-confidence patterns

| Pattern | Rule | Confidence |
|---|---|---:|
| `I am <NAME>` / `I'm <NAME>` / `my name is <NAME>` | Current speaker is `<NAME>` | 0.95 |
| `<NAME> here` / `this is <NAME>` said by same label | Current speaker is `<NAME>` | 0.85 |
| `this is <NAME>` by host followed by another speaker responding | Responding speaker is `<NAME>` | 0.75 |

### Addressee patterns

| Pattern | Rule | Confidence |
|---|---|---:|
| `hey <NAME>` / `hi <NAME>` / `hello <NAME>` and another speaker responds within 2-8s | Responding speaker is likely `<NAME>` | 0.65-0.80 |
| `thanks <NAME>` after a previous utterance | Previous speaker may be `<NAME>` | 0.55-0.70 |
| `<NAME>, what do you think?` then next speaker responds | Next speaker is likely `<NAME>` | 0.70 |

### Twitch-specific safeguards

1. If `<NAME>` matches a chat username active in nearby chat and no voice speaker responds, mark as `addressee_type="chat"`, not a voiced speaker name.
2. If the streamer greets chatters rapidly (`hey bob, hey alice, hey everyone`) with no alternating voice turns, do **not** assign those names to diarized speakers.
3. Do not infer streamer identity from text alone if a voice profile exists and says otherwise.
4. If multiple names are mentioned in one utterance, lower confidence unless Qwen resolves it with explicit evidence.
5. Store all inferred names as candidates with evidence; only promote to `primary_name` above a confidence threshold, e.g. `>=0.70`.

---
