# Speaker Attribution Runbook

This runbook covers speaker attribution setup and operation for VOD Lens.

## Policy (non-negotiable)

- Speaker attribution is **prompt/report context only**.
- Do **not** add deterministic Stage-2 speaker-specific penalties or hard gates unless explicitly requested.

## Prerequisites

1. Optional SpeakerID dependencies:

```bash
cd /workspace/twitch-vod-lens
python3 -m pip install -r requirements-speakerid.txt
```

2. HuggingFace token and gated model access:

```bash
export HF_TOKEN=<your_token>
# or
export HUGGINGFACE_TOKEN=<your_token>
```

- Accept HuggingFace terms for pyannote gated models before running diarization.

## Voice enrollment (optional, recommended)

Use enrollment to create reference voice profiles for streamer recognition.

```bash
cd /workspace/twitch-vod-lens
PYTHONPATH=. python3 src/preprocessing/speaker_enroll.py --help
```

Common profile locations:
- Generic: `data/speaker_profiles/`
- Persistent per-streamer: `data/streamer_intelligence/<STREAMER_ID>/voice_profiles/`

## Generate `speaker_attribution_<VOD_ID>.json`

Current CLI contract:

```bash
cd /workspace/twitch-vod-lens
PYTHONPATH=. python3 -m src.preprocessing.run_speaker_attribution \
  --vod-id <VOD_ID> \
  --vod-media vods/phase4_<VOD_ID>/raw/<VOD_ID>.mp4 \
  --transcript vods/phase4_<VOD_ID>/transcript.json \
  --chat vods/phase4_<VOD_ID>/chat.json \
  --profiles-dir data/streamer_intelligence/<STREAMER_ID>/voice_profiles \
  --output vods/phase4_<VOD_ID>/speaker_attribution_<VOD_ID>.json \
  --hf-token "$HF_TOKEN" \
  --require-speaker-id
```

## Validate integration

Minimum checks:
- `vods/phase4_<VOD_ID>/speaker_attribution_<VOD_ID>.json` exists and parses.
- Synthesis output retains per-clip `speaker_attribution` payloads in final selected clips.
- Attribution fields can be surfaced in Stage-1/Stage-3 prompt/report context.

## Dynamic name inference (what it is / limitations)

- Name inference combines diarized segments + chat cues to propose speaker labels.
- It is heuristic, not guaranteed identity proof.
- Common failure mode: streamer greeting chat users can look like identity evidence when it is not.
- Use confidence/evidence references; treat uncertain mappings as tentative.

## Privacy and safety

- Voice profiles are biometric-ish artifacts; store and share carefully.
- Do not persist sensitive personal claims without explicit approval.
- Persistent profile facts must be evidence-backed and confidence-scored.

## Troubleshooting (quick)

| Symptom | Likely cause | Fix |
|---|---|---|
| Diarization auth/model error | Missing token or gated-model terms not accepted | Set `HF_TOKEN`/`HUGGINGFACE_TOKEN`; accept pyannote model terms |
| Import/module errors in speaker pipeline | Missing optional deps | `pip install -r requirements-speakerid.txt` |
| No streamer cluster recognized | No/weak profile match or poor enrollment sample | Re-enroll cleaner samples; verify profile dir path |
| Empty or low-confidence attribution | Weak audio segments / overlap / noisy transcript alignment | Re-run with better source audio and verify transcript/chat inputs |

