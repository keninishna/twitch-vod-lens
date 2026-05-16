# VOD Lens — Clip Intelligence Pipeline

> **Status:** Active Development
> **Last Updated:** May 15, 2026

## Current Pipeline Architecture

```
Twitch VOD
    │
    ▼
┌─────────────────────────────┐
│ Step 1: Preprocessing       │  Phase 1 — COMPLETE
│ yt-dlp + distil-whisper    │  Docker image vod-lens-worker:latest
│ large-v3 + YOLO             │  Runs on WSL2 RTX 5090
│ + PySceneDetect + Chat      │  distil-large-v3: 303.9x realtime ✅
│ → FusionResult              │  2,029 segments with word timestamps
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Step 2: Clip Selection      │  Phase 1 — COMPLETE
│ Score candidates from       │  YOLO detections + transcript
│ scenes, transcript, YOLO    │  + chat intensity fusion
│ → clip_manifest.json        │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Step 3: Qwen Vision         │  Phase 4 — FUNCTIONAL 🟢
│ Progressive chunking        │
│ w/ context carryover        │
│ 6 frames per clip (was 3)   │  Better temporal coverage
│ Uses requests (not curl)    │  Fixes ARG_MAX with base64 images
│ 06d frame format            │  Matches actual filenames ✅
│ + narrative evaluation      │
│ + platform intelligence     │  Per-platform scores + reasoning
│ + trim reasoning            │  trim_start_reason / trim_end_reason
│ + max_tokens=4096           │  No more JSON truncation
│ → qwen_vision_progressive   │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Step 3.5: Audio Check       │  Phase 5 — INTEGRATED INTO PIPELINE 🟢
│ Qwen2.5-Omni-7B offline     │  Automatic model swap during pipeline run
│ vllm:custom Docker image    │  Top N clips selected by clip_worthiness
│ → audio_analysis_output.txt │  Audio facts injected into synthesis context
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Step 3: Qwen Vision         │  Re-run with audio context
│ (Final Synthesis)           │  Synthesis prompts include audio facts
│ + platform_guide            │  Research-backed per-platform scoring
│ + dynamic trim              │  20-77s based on content, not hardcoded
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Post-Process: RMS Trim      │  NEW — Fallback trim narrowing
│ ffmpeg RMS energy analysis  │  Finds loudest peak, expands to boundaries
│ Dynamic length (15-60s)     │  Trusts Qwen's trim if Qwen narrowed
│ → narrow_boundaries         │  Only runs on full-window returns
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Step 4: Clip Extraction     │  MANUAL — works
│ ffmpeg from downloaded VOD  │
│ → MP4 clips (480p)          │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Step 5: Nextcloud Upload    │  MANUAL — works
│ SCP → docker cp →           │
│ occ files:scan →            │
│ OCS Share API               │
│ → share links                │
└─────────────────────────────┘
```

## What's Built and Working

### Phase 1 — Preprocessing (COMPLETE ✅)
- Docker image `vod-lens-worker:latest` on WSL2 (100.97.240.34)
- yt-dlp for VOD download (480p, sub-only via browser auth-token)
- **distil-whisper large-v3** (was large-v3) — 303.9x realtime, 2,029 segments (was 553)
- YOLOv11 nano object detection (2,927 frames, 5s interval)
- PySceneDetect scene boundaries
- Chat download via GQL VideoComments query
- FusionResult: unified timeline merging all signals
- Tested on VOD 2770929139 (asyajade, 4h lofi study)

### Phase 4 — Qwen Vision Synthesis (FUNCTIONAL 🟢)

**Infrastructure:**
- vLLM container `vllm-qwen` on WSL2, model `Ex0bit/Qwen3.6-35B-A3B-PRISM-NVFP4`
- API at `http://100.97.240.34:8000/v1/chat/completions`
- Uses Python `requests` library (not `curl` subprocess — fixes ARG_MAX with base64 images)

**Progressive Chunking Script:** `src/synthesis/qwen_clip_analyzer_progressive.py`

Key improvements:
- **6 frames per clip** (was 3) — evenly sampled across 120s window
- **`06d` frame format** — matches actual 6-digit frame filenames (`frame_000200.jpg`)
- **max_tokens=4096** for per-clip, **16384** for synthesis — no JSON truncation
- **`requests` library** instead of `curl` subprocess for API calls

**Narrative Evaluation & Clip Titles:**
Each clip analysis includes:
- `clip_point` — **Click-worthy title** following 1 of 5 proven patterns (reaction-based, question bait, short+punchy, etc.)
  - Prohibited: dry descriptions like "Streamer reacts to donation alert"
  - Examples: *"Chat absolutely roasts her for this simple mistake"*, *"The moment she realized chat knew more than she did"*
- `platform_recommendations` — **Explicit list** of which platforms to post to (e.g. `["tiktok", "twitter"]`), empty if none. Only platforms where score >= 6 and clip genuinely fits.
- `narrative_type` — `storytelling | chat_banter | transactional_reaction | organic_reaction | ambient | other`
- `has_narrative_payoff` — true/false
- `requires_context` — true/false
- `narrative_arc` — Chronological summary
- `suggested_trim_start/end` — Precise timestamps with reasoning
- `trim_start_reason` / `trim_end_reason` — WHY those seconds are the boundaries

**Donation Alert Context:**
- Chat messages from the streamer account are auto-bot responses being read aloud
- When a viewer's chat message appears verbatim in the transcript spoken by the streamer, it's a DONATION ALERT
- A laugh at a donation alert = transactional reaction (low clip value)
- Explaining an inside joke to a new viewer = story arc (high clip value)

### Phase 5 — Audio Analysis via vLLM Offline API (INTEGRATED 🟢)

**Model Swap Pipeline:**
```
Phase 1: Qwen 35B (vision) → Score clips by narrative potential
         ↓ Stop Qwen container, kill stale GPU processes, start Omni container
Phase 2: Qwen2.5-Omni-7B (offline Python) → Extract audio features from candidate clips
         ↓ Stop Omni container, kill stale GPU processes, start Qwen container
Phase 3: Qwen 35B (vision) → Final synthesis with audio-filtered scores
```

**Custom Docker Image:**
- Tag: `vllm:custom` (37.5 GB)
- Based on vLLM main branch (v0.21.0+) for `input_audio` support
- Audio deps (soundfile, librosa) installed at runtime

**Batch audio script:** `/vods/audio_batch.py` (runs inside Docker container)

### Post-Processing: RMS Audio Trim Narrowing

When Qwen returns the full clip window (didn't narrow), a fallback step runs:

1. Extracts raw PCM audio via ffmpeg (8kHz mono)
2. Computes RMS energy per second
3. Finds the loudest peak second
4. Expands outward until energy drops to dynamic threshold (25th percentile + 15% of peak)
5. Enforces 15-60s range
6. Updates `suggested_trim_start/end` with audio-backed timestamps

**If Qwen narrowed the clip at all (even to 77s), its timing is trusted** — RMS only runs on full-window returns.

### Platform Intelligence Scoring

Each clip scored independently for **5 platforms** based on research-backed criteria (`PLATFORM_SCORING_GUIDE`):

| Platform | Key Factors |
|----------|-------------|
| **TikTok** | Hook in 3s, fast pacing, vertical framing, text overlay, trend audio sync, loopability, 15-60s |
| **YouTube Shorts** | Loopability, info density, thumbnail potential, self-contained, replay value, 15-60s |
| **Twitter / X** | Context independence, punchline density, quote-tweet bait, engagement hook, 30-120s |
| **Twitch** | Streamer personality, community in-jokes, emote potential, emotional range, chat interaction |
| **Instagram Reels** | Visual aesthetic, shareability, text overlay, niche community fit, 15-60s |

**Output per clip:**
```json
{
  "platform_scores": {"tiktok": 6, "shorts": 6, "twitter": 7, "twitch": 8, "reels": 6},
  "platform_reasoning": {
    "tiktok": "Good storytelling hook, but pacing is moderate.",
    "twitch": "Showcases streamer personality. Engages chat naturally."
  }
}
```

### Clip Extraction & Sharing (WORKS)

**Extraction:** ffmpeg stream copy with re-encode to H.264 Main profile:
```bash
ffmpeg -ss START -to END -i input.mp4 -c:v libx264 -profile:v main -level 3.1 -preset fast -crf 23 -c:a aac -movflags +faststart output.mp4
```

**Upload to Nextcloud:**
1. SCP clip to fileserver (192.168.1.115)
2. `docker cp` into nextcloud-nextcloud-app-1 container
3. `php occ files:scan john`
4. OCS Share API: `curl -u 'john:NextcloudFan!2025' -X POST 'http://172.18.0.4/ocs/v2.php/apps/files_sharing/api/v1/shares' -d 'path=/VOD-Lens/clip.mp4' -d 'shareType=3' -d 'permissions=1'`
5. Extract `<url>` from XML response

## Infrastructure

| Machine | IP | Role |
|---------|-----|------|
| WSL2 (gaming PC) | 100.97.240.34 | GPU processing: distil-whisper, YOLO, Qwen vLLM, Qwen2.5-Omni-7B |
| Hermes Container | — | Orchestration, SSH coordination |
| Fileserver | 192.168.1.115 | Nextcloud file hosting + share links |
| SSH Key | ~/.ssh/id_ed25519 | Passwordless auth to fileserver |

**WSL2 SSH Access:**
- Username: `john`
- Password: `Sparky1234` (via Python PTY)
- SSH command: `ssh -o StrictHostKeyChecking=no john@100.97.240.34`

## Performance Benchmarks

| Model | 4h VOD Time | Segments | Realtime Speed | GPU Util |
|-------|:-----------:|:--------:|:--------------:|:--------:|
| large-v3 (tuned VAD) | ~8 min | 553 | ~45x | ~65% |
| **distil-large-v3** | **48s** | **2,029** | **303.9x** 🔥 | **~95%** |

## Known Issues & Working On

1. **Qwen mislabels clips** — sometimes labels are dry/factual instead of click-worthy. Prompts now instruct to generate engaging titles. Monitor next run.
2. **Clip extraction → upload → share link pipeline** — currently manual SCP + Docker steps. Needs automation.

## What's Next

### Short Term
1. ~~Upgrade Qwen pipeline from curl to `requests`~~ ✅ Done
2. ~~Add narrative evaluation to prompts~~ ✅ Done
3. ~~Fix clip extraction for browser compatibility (H.264 Main profile)~~ ✅ Done
4. ~~Download audio-capable models~~ ✅ Done (Qwen2.5-Omni-7B)
5. ~~Build vLLM from main branch for audio support~~ ✅ Done (vllm:custom)
6. ~~Test audio inference on real VOD clips~~ ✅ Done
7. ~~Integrate audio analysis into pipeline~~ ✅ Done
8. ~~Add platform intelligence scoring~~ ✅ Done
9. ~~Switch to distil-large-v3 for 10x speed~~ ✅ Done
10. ~~Fix dead air in clips~~ ✅ Done (Qwen prompted to penalize silence gaps)
11. **Automate clip extraction → upload → share link**

### Medium Term
12. End-to-end automation: one command from VOD ID to share links
13. Marketing intelligence scoring per platform (TikTok/Shorts/Twitter scores 0-100)
14. Weekly trend cache for trend-aware clip suggestion
15. Tag-based preference learning

## How to Run the Pipeline

All commands run **on WSL2** (`ssh john@100.97.240.34`, password: `Sparky1234`).

**Prerequisites:**
- Qwen 35B vLLM container running (`docker ps` should show `vllm-qwen` Up)
- VOD preprocessed (yt-dlp, WhisperX, YOLO, PySceneDetect, chat download complete)
- Fusion result + clip manifest exist in `~/twitch-vod-analyzer/vods/phase4_{VOD_ID}/`

**Full pipeline with audio integration (default):**
```bash
cd ~/twitch-vod-analyzer
python3 src/synthesis/qwen_clip_analyzer_progressive.py --vod-id <VOD_ID>
```

**Vision only (skip audio phase):**
```bash
python3 src/synthesis/qwen_clip_analyzer_progressive.py --vod-id <VOD_ID> --skip-audio
```

**Adjust audio scope:**
```bash
python3 src/synthesis/qwen_clip_analyzer_progressive.py --vod-id <VOD_ID> --top-clips 5
```

**What happens during the run:**
```
Phase 1     Qwen Vision (batches)   ─── 2-5 min (8 batches × 6 images × ~3s each)
Phase 1.5   Audio (if enabled)          ~6 min (swap → Omni → audio → restart)
Phase 2a    Provisional synthesis        ~15s
Phase 2c    Final synthesis              ~15s
Post        RMS trim narrowing           ~2s
────────────────────────────────
Total: ~8 min (skip-audio) or ~15 min (with audio)
```

**Expected output:** `phase4_{VOD_ID}/qwen_vision_progressive.json`

**Troubleshooting:**

| Problem | Likely Fix |
|---------|------------|
| "Sending 0 images" | Frame naming mismatch — ensure `frame_name()` uses `06d` format |
| "failed to parse JSON" | Response truncated — increase max_tokens (>2000 needed for platform scores) |
| Audio batch "No such file" | Docker mount path — `audio_batch.py` mounts at `/vods/audio_batch.py` |
| "nvidia-smi: not found" | Use `/usr/lib/wsl/lib/nvidia-smi` on WSL2 |
| Qwen API not responding | Cold start takes ~2.5 min — wait for `vllm-qwen` to initialize |
