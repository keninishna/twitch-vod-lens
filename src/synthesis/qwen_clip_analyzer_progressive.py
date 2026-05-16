#!/usr/bin/env python3
"""
Qwen MoE Vision Analyzer - Progressive Chunking with Context Carryover.


Strategy:
  Clip candidates are processed in chronological batches (1 clip, 3 frames each).
  Each batch prompt includes a BATCH_CONTEXT block summarizing all previous
  analyses: what clips were found, their scores, standout moments, and cross-
  batch patterns.  Qwen sees the running state and can compare later clips
  against the full VOD context, not just the current frame in isolation.

  After all batches, an iterative synthesis phase runs:
    Phase 2a — text-only provisional: Qwen produces a ranked selection (any
               number of clips, no hard cap) and may request additional frames
               for clips it's uncertain about.
    Phase 2b — vision re-analysis: for each requested clip, additional frames
               are sampled and sent to Qwen for review.
    Phase 2c — text-only final synthesis: Qwen produces the final ranked list
               incorporating the re-analysed clips.

Usage (on WSL2):
  python qwen_clip_analyzer_progressive.py [--vod-id VOD_ID]
"""

import json
import os
import sys
import base64
import subprocess
import time
import argparse
from pathlib import Path

# ── Configuration (tweak per VOD / model) ────────────────────────────

QWEN_API_URL = "http://100.97.240.34:8000/v1/chat/completions"
QWEN_MODEL  = "Ex0bit/Qwen3.6-35B-A3B-PRISM-NVFP4"

# How many clips to analyse per API call (2 clips x 3 frames ~ 6 images).
CLIPS_PER_BATCH = 1
# How many frames to sample per clip window (start, mid, end).
FRAMES_PER_CLIP = 6
# Gap (seconds) between sampled frames so they aren't all identical.
FRAME_SPREAD   = 5

# Max rounds of additional frame requests during synthesis.
MAX_FRAME_REQUEST_ROUNDS = 20


# Audio analysis config
ENABLE_AUDIO       = True   # Run audio analysis between Phase 1 and Phase 2
AUDIO_CLIPS_TO_PROCESS = 5  # Top N clips to analyze via Omni-7B audio
AUDIO_BATCH_SCRIPT = os.environ.get("AUDIO_BATCH_SCRIPT",
    "/home/john/twitch-vod-analyzer/vods/audio_batch.py")
# VOD_MP4_PATH defined below after VOD_ID

# Paths – override with env vars for different VODs.
VOD_ID             = os.environ.get("VOD_ID", "2770929139")
VOD_DIR            = Path(os.environ.get("VOD_DIR",
                         f"/home/john/twitch-vod-analyzer/vods/phase4_{VOD_ID}"))
FRAMES_DIR         = VOD_DIR / "frames"
FUSION_PATH        = VOD_DIR / f"fusion_result_{VOD_ID}.json"
CLIP_MANIFEST_PATH = VOD_DIR / "clip_manifest.json"
OUTPUT_PATH        = VOD_DIR / "qwen_vision_progressive.json"
FRAME_INTERVAL_S   = 5       # frames were sampled at this interval

VOD_MP4_PATH       = os.environ.get("VOD_MP4_PATH",
    f"/home/john/twitch-vod-analyzer/vods/phase4_{VOD_ID}/raw/{VOD_ID}.mp4")

# ── Helpers ──────────────────────────────────────────────────────────

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def load_json(path):
    with open(path) as f:
        return json.load(f)

def frame_name(seconds):
    idx = round(seconds / FRAME_INTERVAL_S)
    return f"frame_{idx:06d}.jpg"

def encode_image(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{b64}"

def safe_json_parse(raw):
    """Try to parse JSON, stripping markdown fences and handling partials."""
    if not raw:
        return None
    content = raw.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None

def qwen_call(payload, timeout=90):
    """POST payload to Qwen vLLM endpoint. Returns parsed JSON content."""
    import requests
    try:
        resp = requests.post(QWEN_API_URL, json=payload, timeout=timeout)
        data = resp.json()
        raw = data["choices"][0]["message"]["content"]
        parsed = safe_json_parse(raw)
        if parsed is not None:
            return parsed
        return {"error": "failed to parse JSON response", "raw": raw[:200], "clip_worthy": 0}
    except Exception as e:
        return {"error": str(e), "clip_worthy": 0}

def sample_clip_frames(clip, count=FRAMES_PER_CLIP, frame_spread=FRAME_SPREAD):
    """
    Return (frame_paths, timestamps) for a clip window.
    Samples evenly across the clip window for better temporal coverage.
    """
    start = clip["start"]
    end   = clip["end"]
    dur   = end - start
    
    if dur <= 15:
        # Very short clip – just sample every second
        points = list(range(start + 1, end, max(1, dur // count)))
    else:
        # Spread count frames evenly across the window
        step = max(dur // (count + 1), 2)
        points = [start + (i + 1) * step for i in range(count)]
    
    paths = []
    for t in points:
        fn = frame_name(t)
        fp = FRAMES_DIR / fn
        if fp.exists():
            paths.append((str(fp), t))
        if len(paths) >= count:
            break
    return paths

def sample_extra_frames(clip, count=8):
    """
    Sample additional frames from a clip for deeper analysis.
    More granular than the initial sampling.
    """
    start = clip["start"]
    end   = clip["end"]
    dur   = end - start
    step  = max(1, dur // (count + 1))
    points = [start + (i + 1) * step for i in range(count)]
    paths = []
    for t in points:
        fn = frame_name(t)
        fp = FRAMES_DIR / fn
        if fp.exists():
            paths.append((str(fp), t))
    return paths


# ── Audio Analysis Phase ──────────────────────────────────────────

def run_audio_phase(clips, all_results, fusion, manifest):
    """
    Phase 1.5: Audio analysis via Qwen2.5-Omni-7B.
    
    1. Select top candidate clips by clip_worthiness
    2. Stop the Qwen vLLM container
    3. Run batch audio inference via vllm:custom Docker image
    4. Restart the Qwen container
    5. Attach audio results to clip analyses for synthesis
    """
    if not ENABLE_AUDIO:
        log("Audio analysis disabled (ENABLE_AUDIO=False). Skipping.")
        return all_results
    
    if not all_results:
        log("No clip results to analyze audio on. Skipping.")
        return all_results
    
    # Step 1: Select top clips
    scored = [(r, r.get("analysis", {}).get("clip_worthiness", 0)) for r in all_results]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_n = scored[:min(AUDIO_CLIPS_TO_PROCESS, len(scored))]
    
    log(f"\n{'='*60}")
    log(f"PHASE 1.5: AUDIO ANALYSIS")
    log(f"Selected {len(top_n)}/{len(all_results)} top clips for audio analysis")
    for r, score in top_n:
        log(f"  Clip at {r['start']:>5}s — clip_worthiness={score}/10 — {r.get('title','')[:50]}")
    
    # Step 2: Stop Qwen container
    log("\n  Stopping Qwen vLLM container to free GPU...")
    subprocess.run(["docker", "stop", "vllm-qwen"], capture_output=True, timeout=30)
    subprocess.run(["docker", "rm", "vllm-qwen"], capture_output=True, timeout=30)
    
    # Step 3: Kill stale GPU processes
    try:
        gpu_procs = subprocess.run(
            ["/usr/lib/wsl/lib/nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        pids = [p.strip() for p in gpu_procs.stdout.split('\n') if p.strip().isdigit()]
        if pids:
            log(f"  Killing {len(pids)} stale GPU processes: {pids}")
            subprocess.run(["kill", "-9"] + pids, capture_output=True, timeout=10)
    except Exception as e:
        log(f"  WARN: GPU cleanup: {e}")
    
    # Also kill leftover Python/vllm processes
    try:
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=10
        )
        vllm_pids = []
        for line in result.stdout.split('\n'):
            if 'python' in line and 'vllm' in line:
                parts = line.split()
                if parts:
                    vllm_pids.append(parts[1])
        if vllm_pids:
            log(f"  Killing {len(vllm_pids)} leftover vLLM Python processes")
            subprocess.run(["kill", "-9"] + vllm_pids, capture_output=True, timeout=10)
    except Exception as e:
        log(f"  WARN: process cleanup: {e}")
    
    # Wait for GPU to drain
    log("  Waiting for GPU memory to drain...")
    time.sleep(10)
    
    # Build audio batch input — paths relative to Docker mount
    output_dir = str(VOD_DIR)
    docker_vod_path = f"/vods/phase4_{VOD_ID}/raw/{VOD_ID}.mp4"
    batch_input = {
        "vod_id": VOD_ID,
        "vod_mp4": docker_vod_path,
        "output_dir": f"/vods/phase4_{VOD_ID}/",
        "clips": []
    }
    
    # Get transcript and chat from fusion for context
    transcript_segments = fusion.get("transcript", {}).get("segments", [])
    chat_messages = fusion.get("chat", {}).get("messages", [])
    
    def context_for_time(seconds, window=120):
        lo, hi = seconds - window, seconds + window
        txt = " ".join(s.get("text", "") for s in transcript_segments
                       if lo <= s.get("start", 0) <= hi)
        chats = [m for m in chat_messages if lo <= m.get("timestamp", 0) <= hi]
        # Check for donation alerts: viewer msg appearing in transcript
        donation_count = 0
        for m in chats:
            user = m.get("user", "")
            msg = m.get("message", "")
            if msg and msg in txt:
                donation_count += 1
        return txt[:2000], len(chats), donation_count
    
    for r, _ in top_n:
        transcript_excerpt, chat_count, donation_count = context_for_time(r["start"])
        batch_input["clips"].append({
            "start": r["start"],
            "end": r["end"],
            "title": r.get("title", f"clip_{r['start']}s"),
            "transcript_excerpt": transcript_excerpt,
            "chat_count": chat_count,
            "donation_alerts_detected": donation_count,
        })
    
    # Write batch input file
    batch_input_path = os.path.join(output_dir, "audio_batch_input.json")
    with open(batch_input_path, "w") as f:
        json.dump(batch_input, f, indent=2)
    log(f"  Audio batch input written to {batch_input_path}")
    
    # Step 5: Run audio inference via vllm:custom Docker
    log("\n  Starting Qwen2.5-Omni-7B audio inference...")
    log("  (This will take several minutes per clip)")
    
    audio_cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "--entrypoint", "python3",
        "-v", f"{os.path.expanduser('~')}/.cache/huggingface:/root/.cache/huggingface",
        "-v", f"{VOD_DIR.parent}:/vods",
        "vllm:custom",
        f"/vods/audio_batch.py",
        f"/vods/phase4_{VOD_ID}/audio_batch_input.json",
    ]
    
    t0 = time.time()
    log(f"  Running: {' '.join(audio_cmd[:4])} ... {' '.join(audio_cmd[-3:])}")
    audio_result = subprocess.run(audio_cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0
    log(f"  Audio phase complete in {elapsed:.0f}s")
    
    if audio_result.returncode != 0:
        log(f"  WARN: Audio batch returned code {audio_result.returncode}")
        log(f"  stderr: {audio_result.stderr[:500]}")
    else:
        log(f"  stdout (last 300 chars): {audio_result.stdout[-300:]}")
    
    # Step 6: Read audio batch output
    audio_output_path = os.path.join(output_dir, "audio_batch_output.json")
    audio_results = {}
    if os.path.exists(audio_output_path):
        with open(audio_output_path) as f:
            audio_data = json.load(f)
        for result in audio_data.get("results", []):
            start = result.get("start")
            audio_results[start] = result
        log(f"  Loaded {len(audio_results)} audio results")
    
    # Step 7: Attach audio results to clip analyses
    for r in all_results:
        start = r["start"]
        if start in audio_results:
            r["analysis"]["audio_analysis"] = audio_results[start]["analysis"]
            r["analysis"]["audio_extraction_time"] = audio_results[start].get("extraction_time_seconds")
            r["analysis"]["audio_inference_time"] = audio_results[start].get("inference_time_seconds")
    
    # Step 8: Restart Qwen container
    log("\n  Restarting Qwen 35B vLLM container...")
    
    # Inspect the original container to get its full run command
    try:
        inspect = subprocess.run(
            ["docker", "inspect", "vllm-qwen", "--format", "{{json .Config.Cmd}}"],
            capture_output=True, text=True, timeout=10
        )
        # If previous container is gone, need full docker run
        # Use the config from the model swap script's knowledge
    except Exception:
        pass
    
    # Kill any remaining GPU processes
    try:
        gpu_procs = subprocess.run(
            ["/usr/lib/wsl/lib/nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        pids = [p.strip() for p in gpu_procs.stdout.split('\n') if p.strip().isdigit()]
        if pids:
            subprocess.run(["kill", "-9"] + pids, capture_output=True, timeout=10)
    except Exception:
        pass
    
    time.sleep(5)
    
    # Start Qwen with the standard config
    qwen_run_cmd = (
        "docker run -d --name vllm-qwen --gpus all --restart unless-stopped --network host "
        "-v /home/john/.cache/huggingface:/root/.cache/huggingface "
        "-v /home/john/.cache/vllm:/root/.cache/vllm "
        "-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        "-e VLLM_USAGE_SOURCE=production-docker-image "
        "vllm/vllm-openai:latest "
        "--model Ex0bit/Qwen3.6-35B-A3B-PRISM-NVFP4 "
        "--trust-remote-code --host 0.0.0.0 --port 8000 "
        "--attention-backend flashinfer "
        "--kv-cache-dtype fp8_e4m3 "
        "--gpu-memory-utilization 0.93 "
        "--max-model-len 220000 "
        "--max-num-seqs 1 "
        "--max-num-batched-tokens 4096 "
        "--enable-chunked-prefill "
        "--enable-prefix-caching "
        "--enable-auto-tool-choice "
        "--tool-call-parser qwen3_xml "
        "--moe-backend flashinfer_cutlass "
        '--speculative-config \'{"method":"mtp","num_speculative_tokens":3}\''
    )
    
    log("  Starting Qwen 35B...")
    subprocess.run(qwen_run_cmd, shell=True, capture_output=True, timeout=30)
    
    # Step 9: Wait for Qwen API to be ready
    log("  Waiting for Qwen API to be ready (up to ~3 min cold start)...")
    import urllib.request
    import urllib.error
    
    api_ready = False
    api_wait_start = time.time()
    while time.time() - api_wait_start < 300:
        try:
            req = urllib.request.Request(
                "http://100.97.240.34:8000/v1/models"
            )
            resp = urllib.request.urlopen(req, timeout=5)
            api_ready = True
            break
        except Exception:
            elapsed = time.time() - api_wait_start
            if int(elapsed) % 30 < 5:
                log(f"    Waiting... ({elapsed:.0f}s)")
            time.sleep(5)
    
    if api_ready:
        log(f"  ✅ Qwen API ready after {time.time() - api_wait_start:.0f}s")
    else:
        log(f"  ❌ Qwen API did not become ready within timeout")
    
    total_audio_time = time.time() - t0
    log(f"\n  Audio phase total: {total_audio_time:.0f}s")
    log(f"  Audio results for {len(audio_results)} clips injected into synthesis context")
    
    return all_results


# ── Prompts ──────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """You are a Twitch clip analyst. I'll show you frames from a {clip_title} segment ({start}s - {end}s).

Transcript context: {transcript}
Chat messages:
{chat_messages}
YOLO detections: {yolo_objects}

PREVIOUS BATCH CONTEXT (what's been analysed so far in this VOD):
{batch_context}

IMPORTANT CONTEXT FOR UNDERSTANDING THE STREAM:
- Chat messages FROM the streamer account (username "asyajade") are auto-bot responses like "has redeemed their daily pickle" - the streamer reads these aloud as donation/sub alerts.
- When you see a viewer chat message, then the SAME or closely similar text appears in the TRANSCRIPT spoken by the streamer, the streamer is READING that chat message aloud (not speaking from personal experience). The story is the CHATTER'S, not the streamer's. Attribute correctly: e.g. "Streamer reads a chat message about..." not "Streamer says she..."
- This applies to BOTH timed donation/sub alerts AND regular chat messages the streamer chooses to read. The key signal is: chat message + same/similar transcript = streamer reading chat.
- The streamer may react emotionally to these alerts (laughing, commenting), then explain the inside joke to new viewers.
- A clip where the streamer explains an inside joke to a new viewer IS a story arc with setup and payoff. HIGH clip value.
- A clip where the streamer just reacts to an alert without explanation is a transactional reaction. LOW clip value.

Analyse these specific frames and return valid JSON only:
{{{{
  "clip_start": {start},
  "clip_end": {end},
  "person_visible": true/false,
  "face_visible": true/false,
  "primary_expression": "neutral|happy|sad|surprised|focused|talking|smiling|laughing|other",
  "visible_objects": ["list", "of", "objects"],
  "scene_description": "brief 1-line scene",
  "streamer_activity": "what they appear to be doing",
  "emotional_energy": 1-10,
  "visual_interest": 1-10,
  "clip_worthiness": 1-10,
  "clip_point": "CLICK-WORTHY TITLE (1 sentence max). Use one of these proven patterns:

PATTERN 1 — '[Streamer] [reaction] after [trigger]':
✓ 'Streamer loses it after discovering her donation sound is broken'
✓ 'Chat absolutely roasts her for this simple mistake'

PATTERN 2 — 'Girl [verb] [something] and [surprise]':
✓ 'Girl gets real about why running on empty hits different'
✓ 'Girl thought she muted her mic... she did not'

PATTERN 3 — 'The moment [streamer] realized [reveal]':
✓ 'The moment she realized chat knew more than she did'
✓ 'The split second her energy completely flipped'

PATTERN 4 — Question bait:
✓ 'What happens when you ask chat to explain an inside joke?'
✓ 'Can she keep a straight face? (spoiler: no)'

PATTERN 5 — Short + punchy (emote-bait):
✓ 'She had ONE job'
✓ 'This chat needs to be stopped'
✓ 'Energy check gone wrong'

DON'Ts — Dry descriptions (score these low):
✗ 'Streamer reacts to donation alert'
✗ 'Streamer talks about her day'
✗ 'Streamer explains why she's tired'

GOOD titles are specific, emotional, and make you want to click. BAD titles are generic and could describe any stream.
NO DUPLICATE TITLES: Check the batch_context for already-used titles (title_given=...). Your clip_point MUST be different from any title_given in previous clips. If the content overlaps with a previous clip, find a unique angle.",
  "narrative_type": "storytelling|chat_banter|transactional_reaction|organic_reaction|ambient|other",
  "has_narrative_payoff": true/false,
  "requires_context": true/false,
  "suggested_trim_start": "CRITICAL: Narrow to EXACT second the interesting moment starts. The input is ~2 min. You MUST usually narrow to 15-45s (up to 60s only when the narrative truly needs it). Return clip_start ONLY if the moment truly starts at the very beginning.",
  "suggested_trim_end": "CRITICAL: Narrow to EXACT second the payoff ends. You MUST usually keep total length 15-45s (up to 60s only when clearly justified). Returning the full input window should be rare and only when every second is essential.",
  "trim_start_reason": "WHY this exact second is where the interesting moment begins — reference the transcript timestamp that triggers it (e.g. 'donation alert at 885s' or 'story starts at 890s')",
  "trim_end_reason": "WHY this exact second is where the moment ends — reference what finishes (e.g. 'laughing ends by 915s' or 'punchline lands at 905s')",
  "narrative_arc": "Chronological summary of what happens in this clip window: what triggers each moment (donation alert? chat message? story?), what the streamer does, and what the actual interesting moment is.",
  "comparative_note": "How this compares to previously analysed clips in this VOD",
  "platform_scores": {{"tiktok": 1-10, "shorts": 1-10, "twitter": 1-10, "twitch": 1-10, "reels": 1-10}},
  "platform_reasoning": {{"tiktok": "why this score", "shorts": "why", "twitter": "why", "twitch": "why", "reels": "why"}},
  "platform_recommendations": ["tiktok", "twitter"],  /* EXPLICIT LIST of platforms worth posting to. Empty [] if none. Only include platforms where score >= 6 AND the clip genuinely fits. */
  "reason": "why this would (or wouldn't) make a good clip"
}}}}

NARRATIVE CATEGORIES (use these to classify):
- storytelling: Story with beginning/middle/end or joke with setup+punchline. HIGHEST clip value.
- chat_banter: Organic back-and-forth with chat with a clear punchline. HIGH clip value.
- organic_reaction: Genuine reaction to unexpected/funny moment. GOOD clip value.
- transactional_reaction: Response to donation/sub/raid/host alert. LOW clip value (the emotion is about a transaction, not the streamer's content).
- ambient: Calm background, no particular moment. LOW clip value.
- other: Doesn't fit above.

KEY RULES:
- A donation alert laugh may have high emotional energy but LOW clip value because the humor is the donors
- DEAD AIR RULE: Check the transcript timestamps and the ⚠️ DEAD AIR DETECTED warning above. If the ⚠️ DEAD AIR DETECTED warning shows a single silence gap > 10 seconds, that's a DEAD AIR CLIP — clip_worthiness MUST get a -5 PENALTY (max score 5/10). Example: if it would score 8, score becomes 3. If it would score 7, score becomes 2. A clip with 10+ seconds of dead silence is useless for content.
- TRIM MUST EXCLUDE DEAD AIR: If dead air gaps exist INSIDE your suggested_trim_start→suggested_trim_end range, your trim is wrong. Either (a) narrow the trim to cut around the gaps so no dead air remains in the clip, or (b) if the interesting content spans both sides of a dead air gap with no clean narrow, discard the clip (score ≤ 3). A clean clip has continuous speech/vocalization from trim_start to trim_end.
- SILENCE ≠ AMBIENT: Ambient (low-fi music, rain sounds) is a deliberate atmosphere choice. Dead air is silence with nothing happening — the streamer stopped talking, there's no music, no content. Differentiate these.

Use the PLATFORM SCORING GUIDE below to evaluate platform-specific clip value.

{platform_guide}

Previous batch context: {batch_context}"""

PROVISIONAL_SYNTHESIS_PROMPT = """You have just analysed {total_clips} clip candidates from a Twitch VOD titled "{vod_title}" by {streamer}.

Here is the complete analysis log from every batch:

{complete_log}

Audio analysis context:
{audio_context}

Now produce a provisional ranked synthesis. You may recommend ANY number of clips (0 to {total_clips}) — there is no fixed limit. Only include clips that are genuinely worth clipping.

IMPORTANT: If you need additional visual information to make a confident decision about a specific clip (e.g. you want to see more frames to confirm an expression, verify scene context, or check visual quality), specify those clip start times in "need_more_frames" and explain why in "frame_requests". Additional frames will be sampled and shown to you in a follow-up.

Return valid JSON only:
{{{{
  "vod_id": "{vod_id}",
  "selected_clips": [
    {{{{
      "rank": 1,
      "start": ...,
      "end": ...,
      "score": 1-10,
      "why": "why this clip is worth clipping",
      "platform_scores": {{"tiktok": 1-10, "shorts": 1-10, "twitter": 1-10, "twitch": 1-10, "reels": 1-10}},
      "platform_reasoning": {{"tiktok": "why", "shorts": "why", "twitter": "why", "twitch": "why", "reels": "why"}},
      "platform_recommendations": ["tiktok", "twitter"],
      "strengths": ["list"],
      "weaknesses": ["list"],
      "narrative_quality": "Describe the narrative: is there a clear setup+payoff or is this a transactional reaction?",
      "narrative_type": "storytelling|chat_banter|transactional_reaction|organic_reaction|ambient|other",
      "suggested_trim_start": "CRITICAL: Exact second the moment starts. MUST reference transcript timestamps below.",
      "suggested_trim_end": "CRITICAL: Exact second the moment ends.",
      "trim_start_reason": "Cite the exact trigger at this second (e.g. 'donation alert read at 885s', 'chat message appears at 120s', 'streamer starts story at 890s')",
      "trim_end_reason": "Cite what resolves/ends at this second (e.g. 'story payoff lands at 905s', 'laughing dies down by 915s', 'donation reaction ends at 770s')"
    }}}}
  ],
  "need_more_frames": [/* array of clip start times needing more frames, or empty [] */],
  "frame_requests": [
    {{{{
      "clip_start": 123,
      "reason": "why I need more frames (e.g. 'uncertain about expression', 'want to verify energy level')",
      "preferred_timestamps": [124, 125, 126]
    }}}}
  ],
  "overall_vod_assessment": "summary paragraph",
  "total_clips_evaluated": {total_clips}
}}}}

IMPORTANT RULES:
- "selected_clips" can be empty (no good clips), have 1 clip, or have 10+ clips. No fixed cap.
- "need_more_frames" should be an empty array [] if you are confident about all clips.
- "frame_requests" should be an empty array [] if no additional frames needed.
- If you do request frames, keep it to at most 3 clips that you're most uncertain about.
- NARRATIVE QUALITY matters more than emotional energy. A transactional reaction (donation/sub alert) is LOW value. A story or chat banter is HIGH value.
- DEAD AIR RULE: Check the ⚠️ DEAD AIR DETECTED warnings in the analysis log. If a clip has a single silence gap > 10 seconds, apply a -5 penalty to its score (max final score 5/10). If total silence > 30% of window, score ≤ 5. Discard clips with too much dead air.
- DEDUP RULE: Each clip has a unique clip_id (e.g. 'Clip at 998s'). NEVER assign the same clip_point/title to two different clips. If two clips have similar content, differentiate their titles. When in doubt, reference the clip_id as an anchor.
- TITLE RULE: Each clip's clip_point MUST be a click-worthy title following one of the proven patterns (reaction pattern, question bait, punchy one-liner, etc.). The title should make someone WANT to click, not just describe what happens.
- PLATFORM RULE: For each clip, provide platform_recommendations — an explicit list of which platforms to actually post to. Only include platforms where the clip genuinely fits (score >= 6). Can recommend multiple platforms. Empty list if none.
- TRIM RULE: Narrow suggested_trim_start/end as much as you can, but provide trim_start_reason and trim_end_reason explaining WHY those seconds are the boundaries (reference transcript timestamps).
- STRONG PREFERENCE: Do NOT return the full candidate window when it's 120s unless absolutely necessary. Most good clips should be narrowed to 15-45s (up to 60s when justified).
- RMS FALLBACK POLICY: Audio RMS fallback is a last resort for unresolved full-window 120s outputs. Your trim should stand on its own whenever possible."""

FRAME_REVIEW_PROMPT = """You previously analysed a clip at {start}s - {end}s ("{clip_title}") from the VOD "{vod_title}".

Your original analysis was:
{original_analysis}

You requested additional frames because: {request_reason}

Here are additional frames from this segment for a closer look:

Analyse these extra frames and update your assessment. Return valid JSON only:
{{{{
  "clip_start": {start},
  "clip_end": {end},
  "revised_clip_worthiness": 1-10,
  "revised_emotional_energy": 1-10,
  "revised_visual_interest": 1-10,
  "revised_expression": "updated expression if different",
  "revised_scene_description": "updated scene description if different",
  "revised_reason": "updated reason based on additional frames",
  "revised_platform_scores": {{"tiktok": 1-10, "shorts": 1-10, "twitter": 1-10, "twitch": 1-10, "reels": 1-10}},
  "confidence_change": "increased|decreased|unchanged",
  "final_verdict": "include|discard|need_more",
  "notes": "any additional observations from these frames"
}}}}

Previous analysis: {original_analysis}"""

FINAL_SYNTHESIS_PROMPT = """Here is the COMPLETE analysis log for the VOD "{vod_title}" by {streamer}, including any re-analyses from additional frame reviews:

{complete_log}

Audio analysis context:
{audio_context}

Produce the FINAL ranked synthesis. Recommend any number of clips (0 to {total_clips}) — no fixed cap. Only include clips genuinely worth clipping.

Return valid JSON only:
{{{{
  "vod_id": "{vod_id}",
  "final_selected_clips": [
    {{{{
      "rank": 1,
      "start": ...,
      "end": ...,
      "score": 1-10,
      "why": "final verdict",
      "platform_scores": {{"tiktok": 1-10, "shorts": 1-10, "twitter": 1-10, "twitch": 1-10, "reels": 1-10}},
      "platform_reasoning": {{"tiktok": "why", "shorts": "why", "twitter": "why", "twitch": "why", "reels": "why"}},
      "platform_recommendations": ["tiktok", "twitter"],
      "strengths": ["list"],
      "weaknesses": ["list"],
      "narrative_quality": "Narrative assessment: complete story/bit, transactional reaction, or other?",
      "narrative_type": "storytelling|chat_banter|transactional_reaction|organic_reaction|ambient|other",
      "suggested_trim_start": "CRITICAL: Exact second the moment starts. MUST reference transcript timestamps.",
      "suggested_trim_end": "CRITICAL: Exact second the moment ends.",
      "trim_start_reason": "Cite the exact trigger at this second.",
      "trim_end_reason": "Cite what resolves/ends at this second.",
      "clip_point": "CLICK-WORTHY TITLE (1 sentence max). Use a proven pattern: reaction-based ('Streamer [reaction] after [trigger]'), question bait ('What happens when...?'), or short + punchy ('She had ONE job'). NO dry descriptions.",
    }}}}
  ],
  "overall_vod_assessment": "final summary paragraph",
  "total_clips_evaluated": {total_clips},
  "clips_requesting_extra_frames": {frames_requested_count}
}}}}

IMPORTANT RULES:
- "final_selected_clips" can be empty, 1, or many. No fixed limit on the number of clips.
- NARRATIVE QUALITY matters more than emotional energy. Prioritize clips with stories, chat banter, or organic moments over transactional reactions.
- DEDUP RULE: Each clip has a unique clip_id (e.g. 'Clip at 998s'). NEVER assign the same clip_point/title to two different clips. Each clip MUST have a unique title. Differentiate similar clips by focusing on what makes each moment distinct.
- DEAD AIR RULE: Check for ⚠️ DEAD AIR DETECTED in the analysis log. If a single silence gap > 10 seconds exists, that clip must have a -5 penalty applied (max final score 5/10). If total silence > 30% of window, score ≤ 5. Discard clips with unacceptable dead air. Ambient atmosphere is NOT dead air — differentiate.
- For each selected clip, provide suggested_trim_start and suggested_trim_end to capture only the relevant moment.
- STRONG PREFERENCE: Avoid full-window outputs, especially 120s full windows. Most selected clips should be narrowed to 15-45s (up to 60s only when clearly justified by the narrative arc).

{platform_guide}"""


# ── Platform Intelligence ──────────────────────────────────────────

PLATFORM_SCORING_GUIDE = """
PLATFORM SCORING GUIDE — Rate each clip on every platform (1-10).

TIKTOK (score 1-10):
- Hook: Does the first 3 seconds grab attention? (alert sound, sudden movement, visual change, punchline start)
- Pacing: Fast cuts or rapid dialogue? Slow/ambient clips score low (3 or less).
- Vertical framing: Is the streamer's face centered and visible in 9:16 crop?
- Text overlay potential: Can key dialogue be captioned on screen?
- Trend audio sync: Does the moment sync with a potential audio trend?
- Loopability: Does the end lead back to the start? (highly desirable)
- Ideal length: 15-60 seconds. Clips under 15s or over 60s penalized.

YOUTUBE SHORTS (score 1-10):
- Loopability: Seamless start-to-end loop potential (critical for Shorts)
- Info density: Something interesting happens every 5-10 seconds
- Thumbnail potential: Is there a single frame that works as a thumbnail?
- Self-contained: Works without Twitch community knowledge
- Replay value: Would someone watch this more than once?
- Ideal length: 15-60 seconds

TWITTER / X (score 1-10):
- Context independence: Makes sense to someone who doesn't know the streamer
- Punchline density: Clear punchline or surprising moment
- Quote-tweet bait: Invites commentary or sharing
- Engagement hook: Makes you want to reply or react
- Length tolerance: 30-120 seconds accepted (longer than TikTok)
- Horizontal format: Twitter viewers tolerate horizontal video better

TWITCH CLIPS (score 1-10):
- Streamer personality: Does this showcase who the streamer is?
- Community in-joke: Will regular viewers recognize and share?
- Emote potential: Does this generate emote-spam moments?
- Emotional range: Rage, joy, surprise, laughter — genuine emotions score higher
- Chat interaction: Does the streamer engage with chat?
- Context dependence: OK if it requires community knowledge (expected for Twitch)
- Length: 30-120 seconds OK, horizontal format preferred

INSTAGRAM REELS (score 1-10):
- Visual aesthetic: Well-lit, composed, visually interesting?
- Shareability: Would someone send this to a friend via DM?
- Text overlay compatibility: Can key moments be highlighted with text?
- Niche community fit: Does this appeal to a specific Instagram niche?
- Polished production: Higher bar for visual quality
- Ideal length: 15-60 seconds, vertical format required
"""




# ── Core Pipeline ────────────────────────────────────────────────────

def build_analysis_log_entry(r):
    """Format a single clip result for inclusion in synthesis context."""
    a = r.get("analysis", {})
    label = f"Clip at {r['start']}s (clip_id={r['start']})"
    if a.get("revised_clip_worthiness") is not None:
        entry = (
            f"{label}: score={a.get('revised_clip_worthiness','?')}/10 (revised from {a.get('clip_worthiness','?')}), "
            f"energy={a.get('revised_emotional_energy', a.get('emotional_energy','?'))}/10, "
            f"interest={a.get('revised_visual_interest', a.get('visual_interest','?'))}/10, "
            f"expr={a.get('revised_expression', a.get('primary_expression','?'))}, "
            f"narrative={a.get('narrative_type','?')}, payoff={a.get('has_narrative_payoff','?')}, "
            f"trim={a.get('suggested_trim_start','?')}-{a.get('suggested_trim_end','?')}s, reason={a.get('trim_start_reason','')[:40]}->{a.get('trim_end_reason','')[:40]}, "
            f"arc={a.get('narrative_arc','')[:60]}, "
            f"platform_scores={a.get('platform_scores','?')}, "
            f"point={a.get('clip_point','')[:80]}, "
            f"reason={a.get('revised_reason', a.get('reason',''))[:120]}"
        )
    else:
        entry = (
            f"{label}: score={a.get('clip_worthiness','?')}/10, "
            f"energy={a.get('emotional_energy','?')}/10, "
            f"interest={a.get('visual_interest','?')}/10, "
            f"expr={a.get('primary_expression','?')}, "
            f"narrative={a.get('narrative_type','?')}, payoff={a.get('has_narrative_payoff','?')}, "
            f"trim={a.get('suggested_trim_start','?')}-{a.get('suggested_trim_end','?')}s, reason={a.get('trim_start_reason','')[:40]}->{a.get('trim_end_reason','')[:40]}, "
            f"arc={a.get('narrative_arc','')[:60]}, "
            f"platform_scores={a.get('platform_scores','?')}, "
            f"point={a.get('clip_point','')[:80]}, "
            f"reason={a.get('reason','')[:120]}"
        )
    return entry

def run():
    log(f"Loading data for VOD {VOD_ID} ...")
    fusion = load_json(FUSION_PATH)
    manifest = load_json(CLIP_MANIFEST_PATH)
    clips = manifest.get("clips", [])
    if not clips:
        log("ERROR: No clips in manifest.")
        sys.exit(1)

    clips.sort(key=lambda c: c["start"])

    # Build enrichment lookup from fusion data
    transcript_segments = fusion.get("transcript", {}).get("segments", [])
    chat_messages = fusion.get("chat", {}).get("messages", [])

    def context_for_time(seconds, window=120):
        """Get transcript + full chat messages around a timestamp."""
        lo, hi = seconds - window, seconds + window
        segs = [s for s in transcript_segments
                if lo <= s.get("start", 0) <= hi]
        txt_lines = []
        for s in segs:
            st = s.get("start", 0)
            et = s.get("end", 0)
            text = s.get("text", "")
            txt_lines.append(f"[{st:.0f}s-{et:.0f}s] {text}")
        txt = "\n".join(txt_lines)[:2000]
        # Compute dead air: gaps between transcript segments
        sorted_segs = sorted(segs, key=lambda s: s.get("start", 0))
        dead_air_gaps = []
        for i in range(1, len(sorted_segs)):
            gap = sorted_segs[i]["start"] - sorted_segs[i-1]["end"]
            if gap > 5:
                dead_air_gaps.append((sorted_segs[i-1]["end"], sorted_segs[i]["start"], gap))
        total_dead = sum(g for _, _, g in dead_air_gaps)
        window_duration = hi - lo
        if total_dead > 0:
            pct = total_dead / window_duration * 100
            details = "; ".join(f"{g:.0f}s gap at {s:.0f}s-{e:.0f}s" for s, e, g in dead_air_gaps)
            txt += f"\n\n⚠️ DEAD AIR DETECTED: {total_dead:.0f}s silence ({pct:.0f}% of {window_duration}s window). Gaps: {details}"
        # Check for chat messages that the streamer reads aloud
        full_txt = " ".join(s.get("text", "") for s in segs).lower()
        chat_reading_flags = []
        chats = [m for m in chat_messages if lo <= m.get("timestamp", 0) <= hi]
        for m in chats:
            user = m.get("user", "?")
            msg = m.get("message", "")
            ts = m.get("timestamp", 0)
            if msg and len(msg) > 20 and msg.lower()[:40] in full_txt:
                chat_reading_flags.append((ts, user, msg))
        if chat_reading_flags:
            txt += "\n\n⚠️ CHAT-READ FLAGS (streamer reading chat aloud — do NOT attribute to streamer):"
            for ts, user, msg in chat_reading_flags:
                txt += f"\n  @{user} at {ts:.0f}s: '{msg[:80]}...' — streamer reads this aloud."
            txt += "\nIMPORTANT: When a chat message appears in the transcript, it is the CHATTER's story, not the streamer's. Titles MUST say 'reads a chat message about...'."
        chat_lines = []
        for m in chats:
            user = m.get("user", "?")
            msg = m.get("message", "")
            ts = m.get("timestamp", 0)
            chat_lines.append(f"[{ts:.0f}s] @{user}: {msg}")
        chat_text = "\n".join(chat_lines)
        return txt, chat_text

    # ── Batch processing with context carryover ──
    all_results = []
    batch_context = "No clips analysed yet — this is the first batch."

    total = len(clips)
    batches = [clips[i:i+CLIPS_PER_BATCH] for i in range(0, total, CLIPS_PER_BATCH)]
    log(f"Found {total} clips in {len(batches)} batch(es) of {CLIPS_PER_BATCH}")

    for batch_idx, batch in enumerate(batches):
        log(f"\n{'='*60}")
        log(f"BATCH {batch_idx + 1}/{len(batches)} — clips {batch[0]['start']}s - {batch[-1]['end']}s")

        # Build the multimodal payload
        messages = []
        user_content = []

        for clip in batch:
            title = clip.get("title", f"clip at {clip['start']}s")
            transcript, chat_act = context_for_time(clip["start"])
            yolo_objs = clip.get("objects_detected", [])
            frame_samples = sample_clip_frames(clip)

            prompt = ANALYSIS_PROMPT.format(
                clip_title=title,
                start=clip["start"], end=clip["end"],
                transcript=transcript,
                chat_messages=chat_act,
                yolo_objects=", ".join(yolo_objs) if yolo_objs else "none",
                batch_context=batch_context,
                platform_guide=PLATFORM_SCORING_GUIDE,
            )

            user_content.append({"type": "text", "text": prompt})

            for fp, ts in frame_samples:
                log(f"  Encoding frame at {ts}s ...")
                try:
                    uri = encode_image(fp)
                    user_content.append({"type": "image_url", "image_url": {"url": uri}})
                except Exception as e:
                    log(f"  WARN: failed to encode {fp}: {e}")

        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": QWEN_MODEL,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.15,
            "response_format": {"type": "json_object"},
        }

        log(f"  Sending {len(user_content)-len(batch)} images to Qwen ...")
        t0 = time.time()
        analysis = qwen_call(payload)
        elapsed = time.time() - t0
        log(f"  Response in {elapsed:.1f}s")

        for clip in batch:
            clip_result = {
                "start": clip["start"],
                "end":   clip["end"],
                "title": clip.get("title", ""),
                "analysis": analysis if len(batch) == 1 else {},  # one result per batch for now
                "batch": batch_idx + 1,
            }

            # When multiple clips per batch, Qwen might return an array
            # Fallback: treat the single JSON as covering the last clip
            if len(batch) > 1 and "clip_start" in analysis:
                clip_result["analysis"] = analysis

            all_results.append(clip_result)

        # Build running batch_context for next batch
        high_watermark = [r for r in all_results
                          if r.get("analysis", {}).get("clip_worthiness", 0) >= 7]
        batch_context = (
            f"Analysed {len(all_results)}/{total} clips so far (through batch {batch_idx+1}).\n"
            f"Top clips identified so far:\n"
        )
        for r in sorted(all_results, key=lambda x: x.get("analysis", {}).get("clip_worthiness", 0), reverse=True)[:5]:
            a = r.get("analysis", {})
            cp_prev = a.get("clip_point", "")[:60]
            batch_context += (
                f"  - {r['start']}s \"{r.get('title','')[:50]}\": "
                f"clip_worthiness={a.get('clip_worthiness','?')}/10, "
                f"expression={a.get('primary_expression','?')}, "
                f"energy={a.get('emotional_energy','?')}/10, "
                f"narrative={a.get('narrative_type','?')}, "
                f"trim={a.get('suggested_trim_start','?')}-{a.get('suggested_trim_end','?')}s, reason={a.get('trim_start_reason','')[:30]}->{a.get('trim_end_reason','')[:30]}, "
                f"arc={a.get('narrative_arc','')[:60]}, "
                f"platform_scores={a.get('platform_scores','?')}, "
                f"title_given=\"{cp_prev}\"\n"
            )
        batch_context += (
            f"\nCross-batch observations so far: "
            f"{len(high_watermark)} clips scored 7+."
        )

        # Rate limit between batches
        if batch_idx < len(batches) - 1:
            log("  Cooling down 2s before next batch ...")
            time.sleep(2)


    # ── Phase 1.5: Audio Analysis (model swap → Omni → audio → restart Qwen) ──
    all_results = run_audio_phase(clips, all_results, fusion, manifest)

    # Build audio context for synthesis prompts
    audio_context = ""
    audio_clips_found = []
    for r in all_results:
        aa = r.get("analysis", {}).get("audio_analysis", "")
        if aa:
            audio_clips_found.append(r["start"])
            audio_context += f"Clip at {r['start']}s: {aa}\n\n"

    if audio_clips_found:
        log(f"Audio context built for {len(audio_clips_found)} clips: {audio_clips_found}")
    else:
        audio_context = "No audio analysis available for this VOD."


    # ── Phase 2a: Provisional synthesis (text-only, no hard cap, frame requests) ──
    log(f"\n{'='*60}")
    log("PHASE 2a: PROVISIONAL SYNTHESIS (no hard cap, frame requests enabled) ...")

    complete_log = ""
    for r in sorted(all_results, key=lambda x: x["start"]):
        complete_log += build_analysis_log_entry(r) + "\n"

    provisional_payload = {
        "model": QWEN_MODEL,
        "messages": [{
            "role": "user",
            "content": PROVISIONAL_SYNTHESIS_PROMPT.format(
                total_clips=len(all_results),
                vod_title=manifest.get("vod_title", "Unknown"),
                streamer=manifest.get("streamer", "Unknown"),
                complete_log=complete_log,
                vod_id=VOD_ID,
                audio_context=audio_context,
                platform_guide=PLATFORM_SCORING_GUIDE,
            )
        }],
        "max_tokens": 16384,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    t0 = time.time()
    provisional = qwen_call(provisional_payload)
    elapsed = time.time() - t0
    log(f"Provisional synthesis complete in {elapsed:.1f}s")

    # Extract frame requests
    need_more = provisional.get("need_more_frames", []) or []
    frame_requests = provisional.get("frame_requests", []) or []
    log(f"  Clips needing more frames: {len(need_more)}")
    if need_more:
        log(f"  Requested clips: {need_more}")

    # ── Phase 2b: Iterative frame request loop ──
    frames_served = 0
    for round_num in range(MAX_FRAME_REQUEST_ROUNDS):
        if not need_more:
            log("No more frame requests — proceeding to final synthesis.")
            break

        log(f"\n{'='*60}")
        log(f"PHASE 2b, ROUND {round_num + 1}: SERVING FRAME REQUESTS ...")

        # Build a lookup: clip_start -> request details
        request_map = {}
        for req in frame_requests:
            cs = req.get("clip_start")
            if cs:
                request_map[cs] = req

        newly_analysed = []

        for clip_start in need_more:
            # Find the clip in manifest and its existing result
            clip_src = next((c for c in clips if c["start"] == clip_start), None)
            clip_res = next((r for r in all_results if r["start"] == clip_start), None)

            if not clip_src or not clip_res:
                log(f"  WARN: clip at {clip_start}s not found, skipping.")
                continue

            request_reason = request_map.get(clip_start, {}).get("reason", "additional visual context needed")

            # Sample extra frames
            extra_frames = sample_extra_frames(clip_src, count=5)
            if not extra_frames:
                log(f"  WARN: no extra frames found for clip at {clip_start}s, skipping.")
                continue

            log(f"  Analysing extra frames for clip at {clip_start}s ({len(extra_frames)} frames) ...")

            # Build vision payload
            messages = []
            user_content = []

            user_content.append({
                "type": "text",
                "text": FRAME_REVIEW_PROMPT.format(
                    start=clip_src["start"], end=clip_src["end"],
                    clip_title=clip_src.get("title", f"clip at {clip_src['start']}s"),
                    vod_title=manifest.get("vod_title", "Unknown"),
                    original_analysis=json.dumps(clip_res["analysis"], indent=2),
                    request_reason=request_reason,
                )
            })

            for fp, ts in extra_frames:
                try:
                    uri = encode_image(fp)
                    user_content.append({"type": "image_url", "image_url": {"url": uri}})
                except Exception as e:
                    log(f"    WARN: failed to encode extra frame {fp}: {e}")

            messages.append({"role": "user", "content": user_content})

            payload = {
                "model": QWEN_MODEL,
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0.15,
                "response_format": {"type": "json_object"},
            }

            t0 = time.time()
            review = qwen_call(payload)
            elapsed = time.time() - t0
            log(f"    Review response in {elapsed:.1f}s")

            # Merge review into existing analysis
            if "error" not in review:
                clip_res["analysis"].update(review)
                clip_res["analysis"]["extra_frames_served"] = True
                clip_res["analysis"]["extra_frames_round"] = round_num + 1
                frames_served += 1

                verdict = review.get("final_verdict", "")
                need_more_further = review.get("final_verdict") == "need_more"

                newly_analysed.append(clip_start)

                log(f"    Verdict: {verdict}")

        # Update need_more for next round: only clips that still requested more frames
        need_more = []
        for cs in newly_analysed:
            clip_res = next((r for r in all_results if r["start"] == cs), None)
            if clip_res and clip_res.get("analysis", {}).get("final_verdict") == "need_more":
                need_more.append(cs)

        # Also re-check from the original set in case new requests emerged
        # (Qwen's frame review response might include a new `need_more` field)
        # Limit to prevent runaway

    if need_more:
        log(f"  {len(need_more)} clips still requesting frames after {MAX_FRAME_REQUEST_ROUNDS} rounds — proceeding to final synthesis anyway.")

    # ── Phase 2c: Final synthesis (text-only) ──
    log(f"\n{'='*60}")
    log("PHASE 2c: FINAL SYNTHESIS ...")

    # Rebuild complete log with revised analyses
    final_log = ""
    for r in sorted(all_results, key=lambda x: x["start"]):
        final_log += build_analysis_log_entry(r) + "\n"

    final_payload = {
        "model": QWEN_MODEL,
        "messages": [{
            "role": "user",
            "content": FINAL_SYNTHESIS_PROMPT.format(
                vod_title=manifest.get("vod_title", "Unknown"),
                streamer=manifest.get("streamer", "Unknown"),
                complete_log=final_log,
                total_clips=len(all_results),
                vod_id=VOD_ID,
                frames_requested_count=frames_served,
                audio_context=audio_context,
                platform_guide=PLATFORM_SCORING_GUIDE,
            )
        }],
        "max_tokens": 16384,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    t0 = time.time()
    final_synthesis = qwen_call(final_payload)
    elapsed = time.time() - t0
    log(f"Final synthesis complete in {elapsed:.1f}s")

    # ── Save ──
    output = {
        "vod_id": VOD_ID,
        "pipeline": "progressive-chunking-v2",
        "batches_processed": len(batches),
        "clips_analyzed": len(all_results),
        "clips_with_extra_frames": frames_served,
        "clip_details": all_results,
        "provisional_ranking": provisional,
        "final_ranking": final_synthesis,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    log(f"\nResults saved to {OUTPUT_PATH}")
    selected = final_synthesis.get("final_selected_clips", [])
    log(f"Final selection: {len(selected)} clip(s)")
    for s in selected:
        log(f"  Rank {s.get('rank','?')}: {s.get('start','?')}s — score={s.get('score','?')}/10 — {s.get('why','')[:100]}")

    # ── Post-process: narrow clips using audio RMS energy peaks ──
    log(f"\n{'='*60}")
    log("POST-PROCESS: Narrowing clip trim boundaries using audio RMS energy...")
    for s in selected:
        raw_start = s.get("start", 0)
        raw_end = s.get("end", 0)
        trim_start = s.get("suggested_trim_start", raw_start)
        trim_end = s.get("suggested_trim_end", raw_end)

        # Convert to int if string
        try:
            trim_start = int(float(trim_start))
        except (ValueError, TypeError):
            trim_start = raw_start
        try:
            trim_end = int(float(trim_end))
        except (ValueError, TypeError):
            trim_end = raw_end

        width = trim_end - trim_start
        candidate_width = raw_end - raw_start
        is_full_window = (trim_start == raw_start and trim_end == raw_end)

        if not is_full_window:
            log(f"  Clip at {raw_start}s: Qwen narrowed to {width}s ({trim_start}-{trim_end}s) — trusting Qwen's reasoning")
            continue

        # RMS fallback policy: only run RMS when Qwen returned the full 120s candidate window.
        if candidate_width != 120:
            log(f"  Clip at {raw_start}s: Qwen returned full {candidate_width}s window (not 120s) — keeping Qwen trim, skipping RMS")
            continue

        log(f"  Clip at {raw_start}s: Qwen returned full 120s window — finding moment via audio RMS...")

        # Use ffmpeg to extract per-second RMS energy
        try:
            dur = raw_end - raw_start
            cmd = [
                "ffmpeg", "-nostats", "-ss", str(raw_start), "-t", str(dur),
                "-i", VOD_MP4_PATH, "-f", "f32le", "-ac", "1", "-ar", "8000", "-"
            ]
            raw_audio = subprocess.check_output(cmd, timeout=60, stderr=subprocess.DEVNULL)
            # Compute RMS per second (8000 samples/sec at 8kHz)
            import struct
            floats = struct.unpack("<" + str(len(raw_audio)//4) + "f", raw_audio)
            step = 8000
            energies = []
            for i in range(0, len(floats), step):
                chunk = floats[i:i+step]
                if chunk:
                    rms = (sum(x*x for x in chunk) / len(chunk)) ** 0.5
                    energies.append(rms)
                else:
                    energies.append(0.0)

            if len(energies) < 15:
                log(f"    Too short ({len(energies)}s) — keeping Qwen's suggestion")
                continue

            # Find the peak second (loudest moment)
            peak_idx = max(range(len(energies)), key=lambda i: energies[i])
            peak_rms = energies[peak_idx]

            # Compute dynamic threshold based on music floor
            sorted_energies = sorted(energies)
            music_floor = sorted_energies[len(energies) // 4]  # 25th percentile = music baseline

            # Expand outward until energy settles near music floor
            threshold = music_floor + (peak_rms - music_floor) * 0.15
            seg_start = peak_idx
            seg_end = peak_idx + 1
            while seg_start > 0 and energies[seg_start - 1] > threshold:
                seg_start -= 1
            while seg_end < len(energies) and energies[seg_end] > threshold:
                seg_end += 1

            seg_len = seg_end - seg_start

            # Build a broader envelope with a looser threshold so short peaks can
            # still expand into a natural moment window (instead of collapsing to
            # fixed-length clips).
            broad_threshold = music_floor + (peak_rms - music_floor) * 0.05
            broad_start = peak_idx
            broad_end = peak_idx + 1
            while broad_start > 0 and energies[broad_start - 1] > broad_threshold:
                broad_start -= 1
            while broad_end < len(energies) and energies[broad_end] > broad_threshold:
                broad_end += 1

            # Enforce sensible range: 15-60s, but keep duration dynamic.
            if seg_len < 15:
                # Prefer the broader natural envelope first.
                if (broad_end - broad_start) > seg_len:
                    seg_start, seg_end = broad_start, broad_end
                    seg_len = seg_end - seg_start

                # If still too short, pad around peak only as much as needed.
                if seg_len < 15:
                    target_len = 15
                    need = target_len - seg_len
                    left = need // 2
                    right = need - left
                    seg_start = max(0, seg_start - left)
                    seg_end = min(len(energies), seg_end + right)
                    seg_len = seg_end - seg_start

                    # If we hit an edge, extend on the other side.
                    if seg_len < target_len:
                        extra = target_len - seg_len
                        if seg_start == 0:
                            seg_end = min(len(energies), seg_end + extra)
                        elif seg_end == len(energies):
                            seg_start = max(0, seg_start - extra)
                        seg_len = seg_end - seg_start

            if seg_len > 60:
                center = (seg_start + seg_end) // 2
                seg_start = max(0, center - 30)
                seg_end = min(len(energies), center + 30)
                seg_len = seg_end - seg_start

            narrow_start = raw_start + seg_start
            narrow_end = raw_start + seg_end
            log(f"    Audio RMS: dynamic trim = {narrow_start}-{narrow_end}s ({seg_len}s) around peak at {raw_start + peak_idx}s (RMS: {peak_rms:.6f})")
            s["suggested_trim_start"] = narrow_start
            s["suggested_trim_end"] = narrow_end

        except Exception as e:
            log(f"    Audio RMS failed: {e} — falling back to middle 45s")
            center = (raw_start + raw_end) // 2
            s["suggested_trim_start"] = max(raw_start, center - 22)
            s["suggested_trim_end"] = min(raw_end, center + 23)

    # Re-save with narrowed boundaries
    output["final_ranking"]["final_selected_clips"] = selected
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    log(f"Updated results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vod-id", default=VOD_ID)
    parser.add_argument("--batch-size", type=int, default=CLIPS_PER_BATCH)
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio analysis phase")
    parser.add_argument("--top-clips", type=int, default=AUDIO_CLIPS_TO_PROCESS, help="Number of top clips for audio analysis")
    parser.add_argument("--vod-mp4", default=None, help="Path to VOD MP4 for audio extraction")

    args, _ = parser.parse_known_args()
    VOD_ID = args.vod_id
    if args.skip_audio:
        ENABLE_AUDIO = False
    AUDIO_CLIPS_TO_PROCESS = args.top_clips
    if args.vod_mp4:
        VOD_MP4_PATH = args.vod_mp4

    CLIPS_PER_BATCH = args.batch_size
    run()
