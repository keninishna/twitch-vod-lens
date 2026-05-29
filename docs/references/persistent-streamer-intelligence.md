# Persistent Streamer Intelligence Runbook

This runbook covers how VOD Lens loads and updates persistent streamer intelligence.

## What this feature does

Persistent intelligence stores reusable, evidence-backed streamer context under:

- `data/streamer_intelligence/<streamer_id>/profile.json`
- `data/streamer_intelligence/<streamer_id>/observations.jsonl`
- `data/streamer_intelligence/<streamer_id>/voice_profiles/` (optional)

It improves prompt grounding across runs (identity, recurring context, known voice/profile cues).

## Identity resolution safeguards

- `streamer_id` is resolved from VOD metadata by default.
- `--streamer-id` is an explicit override.
- If override conflicts with metadata-derived ID, a mismatch warning should be emitted and recorded.
- Do not silently merge data into the wrong streamer directory.

## Enable persistent intelligence in synthesis

```bash
cd /workspace/twitch-vod-lens
PYTHONPATH=. python3 src/synthesis/qwen_clip_analyzer_progressive.py \
  --vod-id <VOD_ID> \
  --enable-persistent-intelligence \
  --streamer-id <STREAMER_ID> \
  --profile-root data/streamer_intelligence \
  --update-streamer-profile \
  --profile-update-mode propose
```

Modes:
- `propose`: write proposal artifact only.
- `auto`: apply eligible observations automatically (with confidence/evidence safeguards).
- `off`: disable profile update flow.

## How profile updates are applied

1. Load existing profile context before Stage-1 analysis.
2. Generate candidate observations from final selected clips.
3. Score and partition observations:
   - accepted
   - queued/manual
   - rejected
4. Emit proposal artifact (always in propose/auto modes):
   - `profile_update_proposal_<VOD_ID>.json`
5. In `auto` mode, merge only accepted observations and append evidence-backed records.

## Validation workflow (WSL artifact-first)

Use the harness:

```bash
cd /workspace/twitch-vod-lens
bash scripts/validate_persistent_intelligence_wsl.sh <VOD_ID>
```

Minimum expected outcomes:
- `qwen_vision_progressive.json` includes `streamer_identity` metadata.
- Proposal file is written under resolved streamer profile directory.
- Override mismatch warnings are visible when applicable.
- Final selected clips retain attribution/audit metadata.

## Safety and data quality constraints

- Never promote observations without confidence and evidence references.
- Treat profile facts as revisable hypotheses, not immutable truths.
- Avoid storing sensitive personal claims without explicit approval.
- Keep stored context compact to reduce prompt contamination.

## Troubleshooting (quick)

| Symptom | Likely cause | Fix |
|---|---|---|
| Proposal not written | Update mode off or path mismatch | Enable `--update-streamer-profile` and use `propose`/`auto`; verify resolved streamer_id |
| Data written under wrong streamer | Manual override mismatch | Re-run with metadata-derived ID or correct override; inspect mismatch warning |
| No observations accepted in auto mode | Confidence/evidence thresholds not met | Review proposal contents; keep as queued/manual until more evidence exists |
| Profile context missing in prompts | Persistent mode disabled or bad profile-root | Enable `--enable-persistent-intelligence`; verify `--profile-root` and profile files |
