# VOD Lens — Clip Intelligence Pipeline

> **Status:** Active Development
> **Last Updated:** May 15, 2026
> **GitHub:** https://github.com/keninishna/twitch-vod-lens

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
│ 6 frames per clip           │
│ + narrative evaluation      │  clip_point, narrative_type, payoff
│ + title style guide         │  5 proven clickbait patterns
│ + dead air detection        │  ⚠️ DEAD AIR DETECTED injected into transcript
│ + chat-transcript matching  │  ⚠️ CHAT-READ FLAGS when streamer reads chat
│ + platform intelligence     │  Per-platform scores + recommendations
│ + trim reasoning            │  trim_start_reason / trim_end_reason
│ + -5 penalty for dead air   │  Single gap >10s → max score 5
│ + batch_context w/ titles   │  title_given= prevents duplicate titles
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
│ Phase 2a: Provisional Synth │  PHASE 2 — FUNCTIONAL 🟢
│ Rank clips, request frames  │  No hard cap, frame requests enabled
│ → provisional_ranking       │
└─────────────────────────────┘
    │
    ▼ (if frame requests)
┌─────────────────────────────┐
│ Phase 2b: Frame Review      │  PHASE 2 — FUNCTIONAL 🟢
│ Extra frames for uncertain  │  Up to 3 clips, 5 extra frames each
│ clips → re-analysis         │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Phase 2c: Final Synthesis   │  PHASE 2 — FUNCTIONAL 🟢
│ Final ranked list           │  Deduplicated, click-worthy titles
│ → final_ranking             │  platform_recommendations per clip
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Post-Process: RMS Trim      │  SAFETY NET — narrows full-window clips
│ ffmpeg RMS energy analysis  │  Finds loudest peak, expands to boundaries
│ Dynamic length (15-60s)     │  Only runs when Qwen didn't narrow
│ → narrow_boundaries         │
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

### Click-Worthy Titles & Title Style Guide

Each clip gets a `clip_point` generated using one of **5 proven clickbait patterns**:

| Pattern | Formula | Examples |
|---------|---------|----------|
| **Reaction** | `[Streamer] [reaction] after [trigger]` | *"Streamer loses it after discovering her donation sound is broken"* |
| **Girl gets real** | `Girl [verb] [something] and [surprise]` | *"Girl gets real about why running on empty hits different"* |
| **The moment** | `The moment [streamer] realized [reveal]` | *"The moment she realized chat knew more than she did"* |
| **Question bait** | `What happens when [x]?` / `Can she [verb]?` | *"What happens when you ask chat to explain an inside joke?"* |
| **Short + punchy** | Emote-bait one-liner | *"She had ONE job"*, *"This chat needs to be stopped"* |

**Prohibited:** Dry descriptions like *"Streamer reacts to donation alert"*, *"Streamer talks about her day"*.

### Dead Air Detection & Penalty

Dead air is detected **programmatically in Python** (not left to Qwen's judgment):

1. `context_for_time()` computes gaps between consecutive transcript segments
2. Any gap > 5 seconds is flagged as dead air
3. The warning `⚠️ DEAD AIR DETECTED: Xs silence (Y% of window). Gaps: ...` is appended to the transcript
4. Qwen sees this warning and must apply:

   - **Single gap > 10 seconds → -5 penalty** (max score 5/10). A clip with 10+ seconds of dead silence is useless for content.
   - **Total silence > 30% of window → clip_worthiness MUST be ≤ 5**
   - **Trim must exclude dead air**: if gaps exist inside suggested trim range, narrow around them or discard (score ≤ 3)
   - **Silence ≠ Ambient**: deliberate atmosphere (lofi music, rain) is not dead air

**Verified:** Clip at 652s (50s gap) dropped from score 8 → 4 after this rule.

**Safety net:** RMS post-processing catches clips where Qwen ignores the rule.

### Chat Attribution (Transcription Matching)

When a viewer's chat message appears (or closely matches) text the streamer says aloud, the pipeline:

1. **Detects the match** in `context_for_time()` by substring comparison
2. **Injects `⚠️ CHAT-READ FLAGS`** into the transcript: *"@Buchaanan at 752s: '...meeting F1 McLaren owner...' — streamer reads this aloud."*
3. **Qwen must attribute correctly**: titles say *"reads a chat message about..."* not *"streamer reveals she..."*
4. The prompt also includes a broader rule: *"When a viewer chat message appears in the transcript, the story is the chatter's, not the streamer's."*

**Verified:** Title for F1 McLaren clip changed from *"Streamer reveals she met..."* ❌ to *"Streamer reads a chat story about..."* ✅

### Deduplication (No Duplicate Titles)

Three-layer protection against duplicate clip titles:

1. **`clip_id=` anchor** in `build_analysis_log_entry()` — each clip is identified by its start time
2. **`title_given=` in batch_context** — every analysis batch tells Qwen what titles it already assigned to previous clips
3. **DEDUP RULE** in all synthesis prompts: *"NEVER assign the same clip_point/title to two different clips"*

**Verified:** All recent runs produce 8/8 unique titles.

### Platform Intelligence Scoring & Recommendations

Each clip gets per-platform scores (1-10) AND explicit recommendations:

```json
{
  "platform_scores": {"tiktok": 7, "shorts": 7, "twitter": 8, "twitch": 9, "reels": 7},
  "platform_reasoning": {"tiktok": "Good hook, relatable confusion, fits 15-60s", "twitch": "High community relevance"},
  "platform_recommendations": ["tiktok", "twitter", "twitch"]
}
```

| Platform | Key Factors |
|----------|-------------|
| **TikTok** | Hook in 3s, fast pacing, vertical framing, text overlay, trend audio sync, loopability, 15-60s |
| **YouTube Shorts** | Loopability, info density, thumbnail potential, self-contained, replay value, 15-60s |
| **Twitter / X** | Context independence, punchline density, quote-tweet bait, engagement hook, 30-120s |
| **Twitch** | Streamer personality, community in-jokes, emote potential, emotional range, chat interaction |
| **Instagram Reels** | Visual aesthetic, shareability, text overlay, niche community fit, 15-60s |

### Post-Processing: RMS Audio Trim Narrowing

When Qwen returns the full clip window (didn't narrow), a fallback step runs:

1. Extracts raw PCM audio via ffmpeg (8kHz mono)
2. Computes RMS energy per second
3. Finds the loudest peak second
4. Expands outward until energy drops to dynamic threshold (25th percentile + 15% of peak)
5. Enforces 15-60s range
6. Updates `suggested_trim_start/end` with audio-backed timestamps

**If Qwen narrowed the clip at all (even to 77s), its timing is trusted** — RMS only runs on full-window returns.

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
| Hermes Container | — | Orchestration, SSH coordination, git source of truth |
| GitHub | github.com/keninishna/twitch-vod-lens | Repo hosting, history, collaboration |
| Fileserver | 192.168.1.115 | Nextcloud file hosting + share links |
| SSH Key | ~/.ssh/id_ed25519 | Passwordless auth to fileserver |

**WSL2 SSH Access:**
- Username: `john`
- Password: `Sparky1234` (via Python PTY)
- SSH command: `ssh -o StrictHostKeyChecking=no john@100.97.240.34`

## Workflow: Making Changes

```
Hermes (this container) ──git push──→ GitHub
  ↑ Make edits, commit, push

WSL2 (john) ──git pull──→ GitHub
  ↑ Pull latest, run pipeline
```

**To sync WSL2 with latest code:**
```bash
cd ~/twitch-vod-analyzer && git pull
```

**To push updates from Hermes:**
```bash
cd /workspace/twitch-vod-lens
git add -A
git commit -m "description of changes"
git push
```

## Performance Benchmarks

| Model | 4h VOD Time | Segments | Realtime Speed | GPU Util |
|-------|:-----------:|:--------:|:--------------:|:--------:|
| large-v3 (tuned VAD) | ~8 min | 553 | ~45x | ~65% |
| **distil-large-v3** | **48s** | **2,029** | **303.9x** 🔥 | **~95%** |

## Known Issues & Working On

1. **Qwen sometimes ignores score caps** — the -5 penalty and ≤5 hard cap for dead air work most of the time, but Qwen occasionally pushes scores 1-2 points above the cap. RMS post-processor is the safety net.
2. **Clip extraction → upload → share link pipeline** — currently manual SCP + Docker steps. Needs automation.
3. **Qwen 3.6 hallucinates attribution** — occasionally still generates *"streamer reveals she..."* instead of *"reads a chat about..."* despite chat-transcript matching. The injected CHAT-READ FLAGS help but aren't 100%.

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
10. ~~Fix dead air in clips~~ ✅ Done (-5 penalty, hard caps, trim rules)
11. ~~Add click-worthy title style guide~~ ✅ Done (5 patterns, example list)
12. ~~Add chat attribution (transcript matching)~~ ✅ Done (CHAT-READ FLAGS)
13. ~~Add deduplication protection~~ ✅ Done (clip_id, title_given, DEDUP RULE)
14. ~~Set up GitHub repo~~ ✅ Done (keninishna/twitch-vod-lens)
15. **Automate clip extraction → upload → share link**

### Medium Term
16. End-to-end automation: one command from VOD ID to share links
17. Marketing intelligence scoring per platform (TikTok/Shorts/Twitter scores 0-100)
18. Weekly trend cache for trend-aware clip suggestion
19. Tag-based preference learning

## How to Run the Pipeline

All commands run **on WSL2** (`ssh john@100.97.240.34`, password: `Sparky1234`).

**Prerequisites:**
- Qwen 35B vLLM container running (`docker ps` should show `vllm-qwen` Up)
- VOD preprocessed (yt-dlp, WhisperX, YOLO, PySceneDetect, chat download complete)
- Fusion result + clip manifest exist in `~/twitch-vod-analyzer/vods/phase4_{VOD_ID}/`
- Code up to date: `cd ~/twitch-vod-analyzer && git pull`

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
Phase 2b    Frame review (if needed)     ~3s per clip
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
| Stale code on WSL2 | Run `cd ~/twitch-vod-analyzer && git pull` |
