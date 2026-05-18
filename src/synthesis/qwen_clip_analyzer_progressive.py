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

from src.synthesis.audio_normalization import normalize_audio_result
from src.synthesis.clip_context import build_clip_context, render_prompt_context
from src.synthesis.scoring import normalize_clip_analysis
from src.synthesis.stage1_discovery import (
    build_discovery_batch_context,
    map_analysis_to_discovery,
)
from src.synthesis.stitching import stitch_discoveries
from src.synthesis.title_dedup import finalize_stage3_candidates

# ── Configuration (tweak per VOD / model) ────────────────────────────

QWEN_API_URL = "http://100.97.240.34:8082/v1/chat/completions"
QWEN_MODEL  = "Qwen3.6-27B-Q5_K_S.gguf"

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

def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


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


def _to_int_or_none(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def duration_penalty_seconds(trim_start, trim_end):
    """
    Research-guided duration penalty for short-form clip viability.

    Returns: (penalty_points, duration_seconds)

    Target policy:
      - Optimal: 25-60s      -> 0 penalty
      - Acceptable: 20-24s   -> -1
      - Acceptable: 61-75s   -> -1
      - Risky: 15-19s        -> -2
      - Risky: 76-90s        -> -2
      - Poor fit: <15 or >90 -> -3
    """
    ts = _to_int_or_none(trim_start)
    te = _to_int_or_none(trim_end)
    if ts is None or te is None or te <= ts:
        return 3, None

    dur = te - ts
    if 25 <= dur <= 60:
        return 0, dur
    if 20 <= dur <= 24 or 61 <= dur <= 75:
        return 1, dur
    if 15 <= dur <= 19 or 76 <= dur <= 90:
        return 2, dur
    return 3, dur

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
        context = build_clip_context(
            seconds=seconds,
            transcript_segments=transcript_segments,
            chat_messages=chat_messages,
            window=window,
        )
        txt, _chat_text = render_prompt_context(context, transcript_char_limit=2000)
        chat_count = len(context.get("chat_messages", []))
        donation_count = len(context.get("chat_read_flags", []))
        return txt, chat_count, donation_count
    
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
            raw_audio = audio_results[start]
            r["analysis"]["audio_analysis"] = raw_audio.get("analysis")
            r["analysis"]["audio_extraction_time"] = raw_audio.get("extraction_time_seconds")
            r["analysis"]["audio_inference_time"] = raw_audio.get("inference_time_seconds")
            r["analysis"]["audio_structured"] = normalize_audio_result(raw_audio)
    log("  Starting Bee server...")
    # Kill any existing Bee/llama-server process
    subprocess.run("pkill -f llama-server 2>/dev/null; sleep 2", shell=True, capture_output=True, timeout=10)
    # Start Bee
    bee_cmd = (
        "nohup /home/john/beellama.cpp/build/bin/llama-server "
        "-m /home/john/models/bee-qwen36-27b/Qwen3.6-27B-Q5_K_S.gguf "
        "--mmproj /home/john/models/bee-qwen36-27b/mmproj-BF16.gguf "
        "--spec-draft-model /home/john/models/bee-qwen36-27b/dflash-draft-3.6-q4_k_m.gguf "
        "--spec-type dflash --spec-dflash-cross-ctx 1024 "
        "--port 8082 -np 1 --kv-unified -ngl all --spec-draft-ngl all "
        "-b 2048 -ub 256 --ctx-size 200000 "
        "--cache-type-k turbo4 --cache-type-v turbo3_tcq "
        "--flash-attn on --cache-ram 0 --jinja --no-mmap --mlock "
        "--reasoning on "
        '--chat-template-kwargs \'{"preserve_thinking":true}\' '
        "--temp 0.6 --top-k 20 --min-p 0.0 "
        "> /tmp/bee_server.log 2>&1 &"
    )
    subprocess.run(bee_cmd, shell=True, capture_output=True, timeout=10)

    # Step 9: Wait for Bee API to be ready
    log("  Waiting for Bee API to be ready (up to ~3 min cold start)...")
    import urllib.request
    import urllib.error
    
    api_ready = False
    api_wait_start = time.time()
    while time.time() - api_wait_start < 300:
        try:
            req = urllib.request.Request(
                "http://100.97.240.34:8082/v1/models"
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
        log(f"  ✅ Bee API ready after {time.time() - api_wait_start:.0f}s")
    else:
        log(f"  ❌ Qwen API did not become ready within timeout")
    
    total_audio_time = time.time() - t0
    log(f"\n  Audio phase total: {total_audio_time:.0f}s")
    log(f"  Audio results for {len(audio_results)} clips injected into synthesis context")
    
    return all_results


# ── Prompts ──────────────────────────────────────────────────────────

PHASE1_TITLE_RESEARCH_SUMMARY = """
CLICK-WORTHY TITLE RESEARCH SUMMARY (apply this when writing Stage 1 titles):
- YouTube Help (Thumbnail & title tips): strong titles are accurate, concise, and front-load key words; mismatch/clickbait hurts watch behavior and discoverability.
- Nielsen Norman Group microcontent guidance: titles must work out of context, be specific/useful, and stay succinct.
- Nature Human Behaviour (Negativity drives online news consumption, Upworthy RCTs): negative wording can increase CTR (+2.3% per extra negative word on average), but this is a weak nudge — do not force negativity when unsupported by evidence.
- Scientific Reports (When curiosity gaps backfire): best CTR comes from a balance of specificity + curiosity. Too vague OR too fully explained both underperform.

Operational rule for this pipeline:
- Build titles from trigger + payoff evidence in this clip.
- Keep a curiosity gap, but preserve factual accuracy.
- Avoid metadata phrasing ("chat message from...") and dry summaries.
"""

PHASE1_TITLE_EXAMPLES = """
TITLE EXAMPLES (use as style guidance)
GOOD:
- "She explains the inside joke right after chat calls it out"
- "What happens when chat asks about the mystery donation sound?"
- "The moment she realizes chat already solved it"
- "Chat drops one message and the whole story changes"

BAD:
- "Streamer reacts to donation alert"
- "Streamer reads a chat message about donation alert"
- "Funny stream moment"
- "Chat message from user about event"
"""

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

PHASE 1 TITLE RESEARCH BRIEF:
{phase1_title_research_summary}

{phase1_title_examples}

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
  "narrative_type": "storytelling|chat_banter|transactional_reaction|organic_reaction|ambient|other",
  "has_narrative_payoff": true/false,
  "requires_context": true/false,
  "trigger": "What event starts this moment",
  "payoff": "What resolution/punchline/payoff occurs",
  "suggested_trim_start": "CRITICAL: Narrow to EXACT second the interesting moment starts. Return clip_start ONLY if the moment truly starts at the very beginning.",
  "suggested_trim_end": "CRITICAL: Narrow to EXACT second the payoff ends. Returning the full input window should be extremely rare and only when every second is essential.",
  "trim_duration_seconds": "INTEGER = suggested_trim_end - suggested_trim_start",
  "duration_penalty_applied": "INTEGER 0..3 based on DURATION POLICY below (0 optimal, 3 worst)",
  "trim_start_reason": "WHY this exact second is where the interesting moment begins — reference the transcript timestamp that triggers it (e.g. 'donation alert at 885s' or 'story starts at 890s')",
  "trim_end_reason": "WHY this exact second is where the moment ends — reference what finishes (e.g. 'laughing ends by 915s' or 'punchline lands at 905s')",
  "narrative_arc": "Chronological summary of what happens in this clip window: what triggers each moment (donation alert? chat message? story?), what the streamer does, and what the actual interesting moment is.",
  "clip_point": "CLICK-WORTHY TITLE (max 12 words). Must follow PHASE 1 TITLE RESEARCH BRIEF and be evidence-grounded in trigger+payoff.",
  "title_why": "1 sentence: why this title balances specificity + curiosity and remains accurate to the clip evidence.",
  "comparative_note": "How this compares to previously analysed clips in this VOD",
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
- DURATION POLICY (research-guided):
  - No minimum trim length requirement.
  - Prefer the shortest trim that preserves clear setup + payoff and standalone clarity.
  - Keep clips tight to avoid filler; very long trims are usually weaker for retention.
- DURATION PENALTY: Compute trim_duration_seconds = suggested_trim_end - suggested_trim_start and apply penalty to clip_worthiness:
  - <=60s: penalty 0
  - 61-75s: penalty -1
  - 76-90s: penalty -2
  - >90s: penalty -3
- Final clip_worthiness MUST reflect this penalty. Set duration_penalty_applied explicitly.

SCORING RUBRIC (be strict and narrative-first):
- 9-10: Exceptional, highly clip-worthy. Clear setup + payoff, emotionally resonant, specific moment, minimal filler.
- 7-8: Strong clip candidate. Clear story/joke/banter arc with payoff and good standalone clarity.
- 5-6: Borderline. Some interesting signal, but weak payoff OR too context-dependent OR pacing issues.
- 3-4: Weak. Mostly transactional reaction, generic chatter, or limited narrative progression.
- 1-2: Poor. Ambient/dead air/no clear moment.

HARD CAPS:
- If narrative_type == transactional_reaction and there is NO explanation/inside-joke arc for new viewers, clip_worthiness MUST be ≤ 4.
- If has_narrative_payoff == false, clip_worthiness MUST be ≤ 5.
- If requires_context == true and the moment is not understandable as a standalone clip, clip_worthiness MUST be ≤ 5.

HIGH-SCORE GATE:
- To assign clip_worthiness >= 7, the moment MUST have all of:
  (1) clear trigger,
  (2) clear payoff,
  (3) non-transactional narrative value.

TITLE RULE (Stage 1):
- Provide clip_point for every clip.
- clip_point must be <=12 words, evidence-grounded in trigger+payoff, and avoid dry metadata phrasing.
- For chat-read clips, keep attribution to chat while preserving hook quality.

IMPORTANT: Stage 1 is discovery-only. Do not perform final title optimization or final platform recommendation decisions here.

Focus on narrative discovery quality and trim precision in this stage.

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
      "trim_duration_seconds": "INTEGER = suggested_trim_end - suggested_trim_start",
      "duration_penalty_applied": "INTEGER 0..3 based on DURATION POLICY below (0 optimal, 3 worst)",
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
- TITLE RULE: Each clip's clip_point MUST be click-worthy and curiosity-inducing. Do NOT use dry descriptions. For chat-read clips, preserve attribution but use hooky phrasing (e.g. 'What happens when chat drops a message about ...?'). Avoid bland forms like 'Streamer reads a chat message about ...'.
- PLATFORM RULE: For each clip, provide platform_recommendations — an explicit list of which platforms to actually post to. Only include platforms where the clip genuinely fits (score >= 6). Can recommend multiple platforms. Empty list if none.
- TRIM RULE: Narrow suggested_trim_start/end as much as you can, but provide trim_start_reason and trim_end_reason explaining WHY those seconds are the boundaries (reference transcript timestamps).
- DURATION POLICY (research-guided): no minimum trim length requirement. Prefer the shortest trim that preserves setup + payoff and standalone clarity.
- DURATION PENALTY: apply to the clip score and report duration_penalty_applied + trim_duration_seconds:
  - <=60s: 0
  - 61-75s: -1
  - 76-90s: -2
  - >90s: -3
- STRONG PREFERENCE: Do NOT return the full candidate window when it's 120s unless absolutely necessary. Default to the shortest trim that keeps the full narrative payoff; allow 60-90s only when there is clear multi-beat narrative continuity and no filler.
- RMS FALLBACK POLICY: Audio RMS fallback is a last resort for unresolved full-window 120s outputs. Your trim should stand on its own whenever possible.
- SELECTION GATE (strict): Prefer including clips with score >= 7. 5-6 is borderline and should usually be excluded unless it has clear narrative payoff and strong platform fit. <=4 should be excluded.
- TRANSACTIONAL GATE: A transactional_reaction clip should be excluded by default unless it includes a genuine explanation/story arc (e.g., inside joke explained to new viewers).
- STANDALONE GATE: Exclude clips that require too much outside context to be understood.
- FULL-WINDOW GATE: If a selected clip still uses the full 120s candidate window without a compelling justification, down-rank or exclude it.
- OUTPUT QUALITY OVER QUANTITY: It's better to output fewer high-quality clips than many mediocre clips."""

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

DURATION POLICY reminder for revised scoring:
- No minimum trim length requirement.
- Prefer the shortest trim that preserves setup + payoff and standalone clarity.
- Duration penalty by final trim length: <=60s (0), 61-75s (-1), 76-90s (-2), >90s (-3).
- If revised trim is long enough to trigger a penalty, revised_clip_worthiness should reflect it.

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
      "trim_duration_seconds": "INTEGER = suggested_trim_end - suggested_trim_start",
      "duration_penalty_applied": "INTEGER 0..3 based on DURATION POLICY below (0 optimal, 3 worst)",
      "trim_start_reason": "Cite the exact trigger at this second.",
      "trim_end_reason": "Cite what resolves/ends at this second.",
      "clip_point": "CLICK-WORTHY TITLE (1 sentence max). Use a proven pattern: reaction-based ('Streamer [reaction] after [trigger]'), question bait ('What happens when...?'), or short + punchy ('She had ONE job'). NO dry descriptions. For chat-read clips, keep attribution but make it hooky (e.g. 'What happens when chat drops a message about ...?').",
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
- TITLE RULE: clip_point must be click-worthy (reaction, question-bait, or punchy one-liner). Keep factual attribution in analysis fields, but title must maximize curiosity. For chat-read clips, keep attribution while still hooky (e.g. 'What happens when chat drops a message about ...?'). Avoid dry forms like 'Streamer reads a chat message about ...'.
- DEAD AIR RULE: Check for ⚠️ DEAD AIR DETECTED in the analysis log. If a single silence gap > 10 seconds exists, that clip must have a -5 penalty applied (max final score 5/10). If total silence > 30% of window, score ≤ 5. Discard clips with unacceptable dead air. Ambient atmosphere is NOT dead air — differentiate.
- For each selected clip, provide suggested_trim_start and suggested_trim_end to capture only the relevant moment.
- DURATION POLICY (research-guided): no minimum trim length requirement. Prefer the shortest trim that preserves setup + payoff and standalone clarity.
- DURATION PENALTY: apply to final score and report duration_penalty_applied + trim_duration_seconds:
  - <=60s: 0
  - 61-75s: -1
  - 76-90s: -2
  - >90s: -3
- STRONG PREFERENCE: Avoid full-window outputs, especially 120s full windows. Default to the shortest trim that keeps the full narrative payoff. Use 60-90s only when a complete multi-beat narrative requires it and there is no filler.
- FINAL SELECTION GATE (strict): Include primarily clips with score >= 7 and clear setup→payoff narrative value.
- BORDERLINE RULE: Score 5-6 clips should usually be excluded unless they are uniquely strong for a specific platform and still have a clear payoff.
- TRANSACTIONAL RULE: transactional_reaction clips are excluded by default unless they contain a clear explanation/inside-joke arc that creates standalone narrative value.
- STANDALONE RULE: Exclude clips that are confusing without prior stream context.
- FULL-WINDOW RULE: If trim remains the full 120s candidate with no compelling reason, exclude or heavily down-rank.
- QUALITY > QUANTITY: Prefer a short list of high-confidence clips over a long list of mediocre ones.

{platform_guide}"""


# ── Platform Intelligence ──────────────────────────────────────────

PLATFORM_SCORING_GUIDE = """
PLATFORM SCORING GUIDE — Rate each clip on every platform (1-10).

RESEARCH-BACKED LENGTH + RETENTION PRINCIPLES:
- Universal: strongest hooks happen in the first 2-3 seconds.
- Universal: completion/retention usually beats raw duration; do not add dead time.
- Universal trim policy for this pipeline:
  - No minimum trim length requirement.
  - Keep only the seconds needed for setup + payoff.
  - <=60s is generally strongest for clipped moments.
  - 60-90s only when a multi-beat narrative clearly needs it.
  - >90s is rare and usually weak for retention unless exceptional.

TIKTOK (score 1-10):
- Hook in first 2s is mandatory.
- Strong pattern from large-post analysis: videos >60s can outperform shorter ones on reach/watch time IF pacing stays strong.
- Because this pipeline outputs clipped moments (not full essays), prioritize:
  - concise punchy bits (often <=45s)
  - up to ~75s when a clear story arc needs the extra beat
  - Avoid >90s unless unusually compelling.
- Loopability, captionability, and fast payoff still matter more than absolute length.

YOUTUBE SHORTS (score 1-10):
- Shorts creation supports up to 3 minutes, but clip discovery still rewards concise, rewatchable moments.
- Prioritize concise, self-contained cuts (often <=50s).
- Use 50-90s only when there is clear narrative progression with no filler.
- Score down clips that require niche context or have slow ramps.

TWITTER / X (score 1-10):
- Context independence + quote-tweet bait matter most.
- Keep trims punchy and shareable (often <=60s).
- 60-90s acceptable if payoff stays strong.
- Penalize meandering clips even if emotional.

TWITCH CLIPS (score 1-10):
- Platform-native clips are short highlights (typically <=60s).
- Favor personality, chat interplay, and emote-spam potential.
- Keep clips concise by default; exceed 60s only when story continuity clearly requires it.

INSTAGRAM REELS (score 1-10):
- Reels can be longer, but discovery tends to favor shorter cuts.
- Strong practical target: <=90s, with best general distribution often under 90.
- Preferred range in this pipeline:
  - concise punchy shareability (often <=45s)
  - up to ~75s for narrative bits
  - 75-90s only if every beat contributes.
- Prioritize visual clarity, subtitle readability, and first-3-second hook.
"""




# ── Core Pipeline ────────────────────────────────────────────────────

def build_analysis_log_entry(r):
    """Format a single clip result for inclusion in synthesis context."""
    a = r.get("analysis", {})
    label = f"Clip at {r['start']}s (clip_id={r['start']})"
    title_hint = a.get("clip_point") or ""

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
            f"point={title_hint[:80]}, "
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
            f"point={title_hint[:80]}, "
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
        context = build_clip_context(
            seconds=seconds,
            transcript_segments=transcript_segments,
            chat_messages=chat_messages,
            window=window,
        )
        return render_prompt_context(context, transcript_char_limit=2000)

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
                phase1_title_research_summary=PHASE1_TITLE_RESEARCH_SUMMARY,
                phase1_title_examples=PHASE1_TITLE_EXAMPLES,
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
            "max_tokens": 16384,
            "temperature": 0.15,
        }

        log(f"  Sending {len(user_content)-len(batch)} images to Qwen ...")
        t0 = time.time()
        analysis = qwen_call(payload)
        elapsed = time.time() - t0
        log(f"  Response in {elapsed:.1f}s")

        for clip in batch:
            analysis_for_clip = analysis if len(batch) == 1 else {}

            # When multiple clips per batch, Qwen might return an array
            # Fallback: treat the single JSON as covering the last clip
            if len(batch) > 1 and "clip_start" in analysis:
                analysis_for_clip = analysis

            discovery = map_analysis_to_discovery(clip, analysis_for_clip)
            clip_result = {
                "start": clip["start"],
                "end": clip["end"],
                "title": clip.get("title", ""),
                "analysis": analysis_for_clip,
                "discovery": discovery,
                "batch": batch_idx + 1,
            }
            all_results.append(clip_result)

        # Build discovery-only running context for next batch
        batch_context = build_discovery_batch_context(
            all_results,
            total=total,
            batch_idx=batch_idx + 1,
        )

        # Rate limit between batches
        if batch_idx < len(batches) - 1:
            log("  Cooling down 2s before next batch ...")
            time.sleep(2)


    # ── Stage 1.5: Deterministic cross-window stitching ──
    discoveries = [r.get("discovery") for r in all_results if isinstance(r.get("discovery"), dict)]
    stitch_debug_decisions = []
    stitched_candidates = stitch_discoveries(
        discoveries,
        max_gap_seconds=20,
        max_bridge_gap_seconds=45,
        debug_decisions=stitch_debug_decisions,
    )
    merged_pair_count = sum(1 for d in stitch_debug_decisions if d.get("merged") is True)
    evaluated_pair_count = len([d for d in stitch_debug_decisions if "left_window" in d and "right_window" in d])
    log(
        f"Stage 1.5 stitching: {len(discoveries)} discovery candidate(s) -> "
        f"{len(stitched_candidates)} stitched arc(s); "
        f"pair evaluations={evaluated_pair_count}, merged pairs={merged_pair_count}"
    )

    # Stage 2 scoring depends on audio_structured context, so it runs after
    # the audio phase injection below.
    scored_candidates = []
    eligible_source_ids = set()
    analysis_by_stitched_id = {}

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

    # ── Stage 2: Deterministic scoring + hard gate (audio-aware) ──
    results_by_candidate_id = {}
    for r in all_results:
        d = r.get("discovery") or {}
        cid = d.get("candidate_id")
        if cid:
            results_by_candidate_id[cid] = r

    scored_candidates = []
    eligible_source_ids = set()
    analysis_by_stitched_id = {}
    rejected_clips = []

    for stitched in stitched_candidates:
        source_ids = stitched.get("source_candidate_ids") or []
        source_results = [results_by_candidate_id.get(cid) for cid in source_ids]
        source_results = [r for r in source_results if r is not None]
        if not source_results:
            continue

        representative = max(
            source_results,
            key=lambda r: _as_float(r.get("analysis", {}).get("clip_worthiness"), 0.0),
        )
        rep_analysis = dict(representative.get("analysis", {}))

        context_window = max(15, _as_int(stitched.get("end"), 0) - _as_int(stitched.get("start"), 0))
        stitched_context = build_clip_context(
            seconds=_as_int(stitched.get("start"), 0),
            transcript_segments=transcript_segments,
            chat_messages=chat_messages,
            window=context_window,
        )

        scored = normalize_clip_analysis(
            candidate=stitched,
            analysis=rep_analysis,
            context=stitched_context,
            audio=rep_analysis.get("audio_structured"),
        )
        scored_candidates.append(scored)

        stitched_id = str(stitched.get("stitched_id"))
        if rep_analysis.get("suggested_trim_start") is None:
            rep_analysis["suggested_trim_start"] = stitched.get("start")
        if rep_analysis.get("suggested_trim_end") is None:
            rep_analysis["suggested_trim_end"] = stitched.get("end")
        analysis_by_stitched_id[stitched_id] = rep_analysis

        if scored.get("eligible_for_final") and _as_float(scored.get("final_score"), 0.0) >= 3.0:
            for cid in source_ids:
                eligible_source_ids.add(cid)
        else:
            reason_codes = list(scored.get("rejection_reasons") or [])
            if not reason_codes:
                reason_codes = ["below_score_threshold"]
            rejected_clips.append({
                "stage": "stage2",
                "clip_id": stitched_id,
                "start": stitched.get("start"),
                "end": stitched.get("end"),
                "final_score": scored.get("final_score"),
                "eligible_for_final": scored.get("eligible_for_final"),
                "trim_source": scored.get("trim_source"),
                "reason_codes": reason_codes,
            })

    log(
        f"Stage 2 scoring (audio-aware): {len(scored_candidates)} stitched candidate(s), "
        f"{len(eligible_source_ids)} source clip(s) passed score>=3 hard gate"
    )


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
                "max_tokens": 16384,
                "temperature": 0.15,
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

    # Rebuild complete log with revised analyses, but only for clips that
    # passed deterministic Stage 2 hard gate (final_score >= 3).
    gated_results = [
        r for r in sorted(all_results, key=lambda x: x["start"])
        if ((r.get("discovery") or {}).get("candidate_id") in eligible_source_ids)
    ]

    final_log = ""
    for r in gated_results:
        final_log += build_analysis_log_entry(r) + "\n"

    if not gated_results:
        log("No clips passed Stage 2 hard gate (score >= 3). Skipping final synthesis call.")
        final_synthesis = {
            "final_selected_clips": [],
            "gating_summary": {
                "stage2_scored_candidates": len(scored_candidates),
                "eligible_source_clips": 0,
                "hard_gate": "final_score>=3",
            },
        }
    else:
        final_payload = {
            "model": QWEN_MODEL,
            "messages": [{
                "role": "user",
                "content": FINAL_SYNTHESIS_PROMPT.format(
                    vod_title=manifest.get("vod_title", "Unknown"),
                    streamer=manifest.get("streamer", "Unknown"),
                    complete_log=final_log,
                    total_clips=len(gated_results),
                    vod_id=VOD_ID,
                    frames_requested_count=frames_served,
                    audio_context=audio_context,
                    platform_guide=PLATFORM_SCORING_GUIDE,
                )
            }],
            "max_tokens": 16384,
            "temperature": 0.2,
        }

        t0 = time.time()
        final_synthesis = qwen_call(final_payload)
        elapsed = time.time() - t0
        log(f"Final synthesis complete in {elapsed:.1f}s")

    # Stage 3 deterministic verification + title/dedup pass.
    stage3_final_selected = finalize_stage3_candidates(
        scored_candidates=scored_candidates,
        stitched_candidates=stitched_candidates,
        analysis_by_candidate=analysis_by_stitched_id,
        min_score=3.0,
        fallback_top_n_when_empty=3,
    )
    passed_hard_gate = sum(
        1 for s in scored_candidates
        if s.get("eligible_for_final") and _as_float(s.get("final_score"), 0.0) >= 3.0
    )
    fallback_mode = passed_hard_gate == 0 and bool(stage3_final_selected)
    if fallback_mode:
        log(
            f"Stage 3 fallback activated: no clips scored >=3, selected top {len(stage3_final_selected)} clip(s) by final_score"
        )
    log(f"Stage 3 title/dedup pass selected {len(stage3_final_selected)} clip(s)")

    # Keep model ranking as reference, but enforce Stage 3 deterministic list as
    # canonical final_selected_clips for downstream extraction.
    if isinstance(final_synthesis, dict):
        final_synthesis["model_final_selected_clips"] = final_synthesis.get("final_selected_clips", [])
        final_synthesis["final_selected_clips"] = stage3_final_selected

    # ── Save ──
    output = {
        "vod_id": VOD_ID,
        "pipeline": "progressive-chunking-v2",
        "batches_processed": len(batches),
        "clips_analyzed": len(all_results),
        "clips_with_extra_frames": frames_served,
        "clip_details": all_results,
        "stage1_5_stitched": stitched_candidates,
        "stage1_5_stitch_debug": stitch_debug_decisions,
        "stage2_scored": scored_candidates,
        "stage3_final_selected": stage3_final_selected,
        "rejected_clips": rejected_clips,
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
            s["trim_source"] = "qwen"
            log(f"  Clip at {raw_start}s: Qwen narrowed to {width}s ({trim_start}-{trim_end}s) — trusting Qwen's reasoning")
            continue

        # RMS fallback policy: only run RMS when Qwen returned the full 120s candidate window.
        if candidate_width != 120:
            s["trim_source"] = "qwen"
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
                s["trim_source"] = "qwen"
                continue

            peak_idx = max(range(len(energies)), key=lambda i: energies[i])
            peak_rms = energies[peak_idx]
            sorted_energies = sorted(energies)
            music_floor = sorted_energies[len(energies) // 4]

            threshold = music_floor + (peak_rms - music_floor) * 0.15
            seg_start = peak_idx
            seg_end = peak_idx + 1
            while seg_start > 0 and energies[seg_start - 1] > threshold:
                seg_start -= 1
            while seg_end < len(energies) and energies[seg_end] > threshold:
                seg_end += 1

            seg_len = seg_end - seg_start

            broad_threshold = music_floor + (peak_rms - music_floor) * 0.05
            broad_start = peak_idx
            broad_end = peak_idx + 1
            while broad_start > 0 and energies[broad_start - 1] > broad_threshold:
                broad_start -= 1
            while broad_end < len(energies) and energies[broad_end] > broad_threshold:
                broad_end += 1

            if seg_len < 15:
                if (broad_end - broad_start) > seg_len:
                    seg_start, seg_end = broad_start, broad_end
                    seg_len = seg_end - seg_start

                if seg_len < 15:
                    target_len = 15
                    need = target_len - seg_len
                    left = need // 2
                    right = need - left
                    seg_start = max(0, seg_start - left)
                    seg_end = min(len(energies), seg_end + right)
                    seg_len = seg_end - seg_start

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
            s["trim_source"] = "rms_fallback"

        except Exception as e:
            log(f"    Audio RMS failed: {e} — falling back to middle 45s")
            center = (raw_start + raw_end) // 2
            s["suggested_trim_start"] = max(raw_start, center - 22)
            s["suggested_trim_end"] = min(raw_end, center + 23)
            s["trim_source"] = "rms_fallback"

    # Recompute deterministic score/eligibility after any trim changes from RMS.
    stitched_by_id = {str(c.get("stitched_id")): c for c in stitched_candidates}
    rescored_selected = []
    rms_rejected_clips = []
    for s in selected:
        clip_id = str(s.get("clip_id"))
        stitched = stitched_by_id.get(clip_id)
        if not stitched:
            continue

        rep_analysis = dict(analysis_by_stitched_id.get(clip_id, {}))
        rep_analysis["suggested_trim_start"] = s.get("suggested_trim_start", stitched.get("start"))
        rep_analysis["suggested_trim_end"] = s.get("suggested_trim_end", stitched.get("end"))
        rep_analysis["trim_source"] = s.get("trim_source", "qwen")

        context_window = max(15, _as_int(stitched.get("end"), 0) - _as_int(stitched.get("start"), 0))
        stitched_context = build_clip_context(
            seconds=_as_int(stitched.get("start"), 0),
            transcript_segments=transcript_segments,
            chat_messages=chat_messages,
            window=context_window,
        )

        rescored = normalize_clip_analysis(
            candidate=stitched,
            analysis=rep_analysis,
            context=stitched_context,
            audio=rep_analysis.get("audio_structured"),
        )

        s["score"] = rescored.get("final_score")
        s["normalized_score"] = rescored.get("final_score")
        s["raw_score"] = rescored.get("raw_score")
        s["trim_source"] = rescored.get("trim_source", s.get("trim_source"))
        s["rank"] = len(rescored_selected) + 1

        if fallback_mode:
            rescored_selected.append(s)
        elif rescored.get("eligible_for_final") and _as_float(rescored.get("final_score"), 0.0) >= 3.0:
            rescored_selected.append(s)
        else:
            reason_codes = list(rescored.get("rejection_reasons") or [])
            if "rms_rescore_below_threshold" not in reason_codes:
                reason_codes.append("rms_rescore_below_threshold")

            rms_rejected_clips.append({
                "stage": "post_rms_rescore",
                "clip_id": clip_id,
                "start": stitched.get("start"),
                "end": stitched.get("end"),
                "final_score": rescored.get("final_score"),
                "eligible_for_final": rescored.get("eligible_for_final"),
                "trim_source": rescored.get("trim_source"),
                "reason_codes": reason_codes,
            })
            log(
                f"  Dropping clip {clip_id} after RMS rescoring: "
                f"final_score={rescored.get('final_score')} eligible={rescored.get('eligible_for_final')}"
            )

    selected = rescored_selected

    # Re-save with narrowed boundaries and rescored eligibility.
    if rms_rejected_clips:
        output.setdefault("rejected_clips", []).extend(rms_rejected_clips)
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
