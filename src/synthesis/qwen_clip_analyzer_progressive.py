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

from src.intelligence.profile_context import render_streamer_profile_context
from src.intelligence.profile_update import (
    apply_profile_update_auto,
    build_profile_update_proposal,
    partition_observations_for_merge,
)
from src.intelligence.streamer_store import (
    append_observations,
    load_streamer_profile,
    resolve_streamer_id_context,
    save_streamer_profile,
)
from src.synthesis.audio_normalization import normalize_audio_result
from src.synthesis.bee_server import ensure_bee_api_ready
from src.synthesis.clip_context import build_clip_context, render_prompt_context
from src.synthesis.scoring import normalize_clip_analysis
from src.synthesis.stage1_discovery import (
    build_discovery_batch_context,
    map_analysis_to_discovery,
)
from src.synthesis.stitching import stitch_discoveries
from src.synthesis.title_dedup import finalize_stage3_candidates
from src.synthesis.fastpass_triage import (
    build_triage_chunks,
    compute_vision_budget,
    normalize_triage_candidate,
    select_gemma_frames_for_window,
    select_vision_shortlist,
    summarize_gemma_signals_for_triage,
)
from src.synthesis.gemma_enrichment import run_gemma_enrichment

# ── Configuration (tweak per VOD / model) ────────────────────────────

DEFAULT_BEE_URL = "http://localhost:8082"
BEE_URL = os.environ.get("BEE_URL", DEFAULT_BEE_URL)
START_BEE = False
DEFAULT_BEE_START_COMMAND = "bash /home/john/twitch-vod-analyzer/scripts/start_bee_prebuilt_wsl.sh"
BEE_START_COMMAND = os.environ.get("BEE_START_COMMAND", DEFAULT_BEE_START_COMMAND)


def bee_models_url() -> str:
    return f"{BEE_URL.rstrip('/')}/v1/models"


def qwen_api_url() -> str:
    return f"{BEE_URL.rstrip('/')}/v1/chat/completions"


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


FAST_PASS = os.environ.get("FAST_PASS", "0").lower() in {"1", "true", "yes", "on"}
FAST_PASS_MODE = os.environ.get("FAST_PASS_MODE", "gemma-enriched")
FAST_PASS_DRY_RUN = os.environ.get("FAST_PASS_DRY_RUN", "0").lower() in {"1", "true", "yes", "on"}
GEMMA_SMOKE_TEST_ONLY = os.environ.get("GEMMA_SMOKE_TEST_ONLY", "0").lower() in {"1", "true", "yes", "on"}
GEMMA_URL = os.environ.get("GEMMA_URL", "http://localhost:8084/v1")
GEMMA_MODEL = os.environ.get("GEMMA_MODEL", "gemma-4-12B-it")
GEMMA_WINDOW_SECONDS = int(os.environ.get("GEMMA_WINDOW_SECONDS", 30))
GEMMA_WINDOW_STRIDE_SECONDS = int(os.environ.get("GEMMA_WINDOW_STRIDE_SECONDS", 30))
GEMMA_MAX_WINDOWS = int(os.environ.get("GEMMA_MAX_WINDOWS", 0))
GEMMA_FRAMES_PER_WINDOW = int(os.environ.get("GEMMA_FRAMES_PER_WINDOW", 2))
GEMMA_AUDIO_MAX_SECONDS = int(os.environ.get("GEMMA_AUDIO_MAX_SECONDS", 30))
GEMMA_RESPONSE_TIMEOUT_SECONDS = int(os.environ.get("GEMMA_RESPONSE_TIMEOUT_SECONDS", 180))
GEMMA_CONCURRENT_WORKERS = int(os.environ.get("GEMMA_CONCURRENT_WORKERS", 3))
FAST_PASS_CHUNK_SECONDS = int(os.environ.get("FAST_PASS_CHUNK_SECONDS", 600))
FAST_PASS_OVERLAP_SECONDS = int(os.environ.get("FAST_PASS_OVERLAP_SECONDS", 60))
FAST_PASS_MAX_TRIAGE_CANDIDATES = int(os.environ.get("FAST_PASS_MAX_TRIAGE_CANDIDATES", 60))
FAST_PASS_VISION_RATIO = float(os.environ.get("FAST_PASS_VISION_RATIO", 0.20))
FAST_PASS_MIN_VISION_CANDIDATES = int(os.environ.get("FAST_PASS_MIN_VISION_CANDIDATES", 25))
FAST_PASS_MAX_VISION_CANDIDATES = int(os.environ.get("FAST_PASS_MAX_VISION_CANDIDATES", 50))
FAST_PASS_VISION_FRAMES = int(os.environ.get("FAST_PASS_VISION_FRAMES", 3))
FAST_PASS_SENTINEL_RATIO = float(os.environ.get("FAST_PASS_SENTINEL_RATIO", 0.05))
ENABLE_PERSISTENT_INTELLIGENCE = os.environ.get("ENABLE_PERSISTENT_INTELLIGENCE", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
STREAMER_ID_OVERRIDE = os.environ.get("STREAMER_ID")
STREAMER_PROFILE_ROOT = Path(os.environ.get("STREAMER_PROFILE_ROOT", "data/streamer_intelligence"))
UPDATE_STREAMER_PROFILE = os.environ.get("UPDATE_STREAMER_PROFILE", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
PROFILE_UPDATE_MODE = os.environ.get("PROFILE_UPDATE_MODE", "propose")

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
    # Handle doubled braces from Qwen copying template literally ({{...}} → {...})
    if content.startswith("{{"):
        inner = content[1:]  # strip one leading {
        # Try parsing after stripping leading {, optionally also trailing }
        for attempt in (inner, inner[:-1]):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue
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


def _format_hms(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    s = int(abs(seconds))
    h, s = s // 3600, s % 3600
    m, s = s // 60, s % 60
    sign = "-" if seconds < 0 else ""
    return f"{sign}{h}:{m:02d}:{s:02d}"


def _format_twitch_ts(seconds: float) -> str:
    """Convert seconds to Twitch URL timestamp format (e.g. 1h44m31s)."""
    s = int(abs(seconds))
    if s == 0:
        return "0s"
    h, s = s // 3600, s % 3600
    m, s = s // 60, s % 60
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    if s or not (h or m):
        parts.append(f"{s}s")
    return "".join(parts)


def _add_hms_and_links(clips: list[dict], vod_id: str) -> None:
    """Add ``_hms`` variants and ``vod_url`` fields to each clip record (in-place).

    Adds ``start_hms``, ``end_hms``, ``suggested_trim_start_hms``,
    ``suggested_trim_end_hms``, and ``vod_url`` (linking to the VOD at the
    clip window start) if the corresponding seconds field exists.
    """
    for clip in clips:
        for field in ("start", "end", "suggested_trim_start", "suggested_trim_end"):
            val = clip.get(field)
            if val is not None:
                try:
                    clip[f"{field}_hms"] = _format_hms(int(float(val)))
                except (ValueError, TypeError):
                    pass
        # Add VOD link at window start
        raw_start = clip.get("start")
        if raw_start is not None:
            try:
                ts = _format_twitch_ts(int(float(raw_start)))
                clip["vod_url"] = f"https://www.twitch.tv/videos/{vod_id}?t={ts}"
            except (ValueError, TypeError):
                pass


VOD_ID: str = "0001"


def qwen_call(payload, timeout=180):
    """POST payload to Qwen vLLM endpoint. Returns parsed JSON content."""
    import requests
    try:
        resp = requests.post(qwen_api_url(), json=payload, timeout=timeout)
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

def sample_clip_frames(clip, count=FRAMES_PER_CLIP, frame_spread=FRAME_SPREAD, *, fast_pass=False, suggested_trim_start=None, suggested_trim_end=None):
    """
    Return (frame_paths, timestamps) for a clip window.
    Samples evenly across the clip window for better temporal coverage.
    In fast-pass mode, prefer start/mid/end around the suggested trim and use the nearest existing frames.
    """
    start = _as_int(clip.get("start"), 0)
    end = _as_int(clip.get("end"), start + 1)
    if end <= start:
        end = start + 1
    count = max(1, _as_int(count, FRAMES_PER_CLIP))
    frame_spread = max(1, _as_int(frame_spread, FRAME_SPREAD))

    if fast_pass:
        trim_start = _as_int(
            suggested_trim_start if suggested_trim_start is not None else clip.get("suggested_trim_start"),
            start,
        )
        trim_end = _as_int(
            suggested_trim_end if suggested_trim_end is not None else clip.get("suggested_trim_end"),
            end,
        )
        if trim_end <= trim_start:
            trim_start, trim_end = start, end
        frame_paths = select_gemma_frames_for_window(
            {
                "start": trim_start,
                "end": trim_end,
            },
            str(FRAMES_DIR),
            frames_per_window=count,
        )
        sampled = []
        for fp in frame_paths:
            stem = Path(fp).stem
            timestamp = start
            if stem.startswith("frame_"):
                try:
                    timestamp = int(stem.split("_", 1)[1]) * FRAME_INTERVAL_S
                except (ValueError, IndexError):
                    timestamp = start
            sampled.append((fp, timestamp))
        return sampled

    dur = end - start
    if dur <= 15:
        points = list(range(start + 1, end, max(1, dur // count)))
    else:
        step = max(dur // (count + 1), max(2, frame_spread))
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


def _build_fast_pass_evidence_block(candidate: dict | None) -> str:
    candidate = candidate if isinstance(candidate, dict) else {}
    lines = [
        f"trigger: {candidate.get('trigger', 'unknown')}",
        f"payoff: {candidate.get('payoff', 'unknown')}",
        f"evidence_lines: {candidate.get('evidence_lines', [])}",
        f"risk_flags: {candidate.get('risk_flags', [])}",
    ]
    refs = candidate.get("gemma_annotation_refs") or []
    if refs:
        lines.append(f"gemma_annotation_refs: {refs}")
    gemma_block = candidate.get("gemma_evidence_block")
    if gemma_block:
        lines.append(f"gemma_evidence_block: {gemma_block}")
    return "\n".join(lines)


def _fast_pass_text_triage_prompt(*, chunk: dict, gemma_summary: dict, mode: str) -> str:
    transcript_lines = chunk.get("transcript_lines") or []
    chat_messages = chunk.get("chat_messages") or []
    gemma_evidence = gemma_summary.get("evidence_lines") or []
    return (
        "You are generating fast-pass text triage candidates for Twitch clip routing. "
        "Use transcript/chat chunks plus Gemma evidence, but verify/correct Gemma evidence rather than trusting it blindly. "
        "Return valid JSON only with a top-level object containing a 'candidates' array.\n\n"
        f"FAST_PASS_MODE={mode}\n"
        f"CHUNK_START={chunk.get('chunk_start')}\nCHUNK_END={chunk.get('chunk_end')}\n\n"
        f"TRANSCRIPT_LINES={json.dumps(transcript_lines, ensure_ascii=False)}\n\n"
        f"CHAT_MESSAGES={json.dumps(chat_messages, ensure_ascii=False)}\n\n"
        f"GEMMA_EVIDENCE={json.dumps(gemma_evidence, ensure_ascii=False)}\n"
        f"GEMMA_SUMMARY={json.dumps(gemma_summary, ensure_ascii=False)}\n\n"
        "Candidate schema: {candidate_id,start,end,suggested_trim_start,suggested_trim_end,narrative_type,trigger,payoff,"
        "evidence_lines,risk_flags,triage_score,triage_confidence,vision_need,selection_reasons,gemma_annotation_refs}. "
        "Prefer evidence-grounded candidates and keep the list compact."
    )


def _normalize_fast_pass_triage_candidates(raw_candidates: list[dict], fallback_start: int, fallback_end: int, limit: int) -> list[dict]:
    normalized = []
    for candidate in raw_candidates or []:
        if not isinstance(candidate, dict):
            continue
        normalized.append(normalize_triage_candidate(candidate, fallback_start, fallback_end))
    normalized = sorted(normalized, key=lambda c: (-float(c.get("triage_score", 0.0)), -float(c.get("triage_confidence", 0.0)), int(c.get("start", 0)), str(c.get("candidate_id", ""))))
    return normalized[: max(0, limit)]


def _run_fast_pass_text_triage(*, triage_chunks: list[dict], gemma_artifact: dict, mode: str) -> tuple[list[dict], dict]:
    gemma_windows = gemma_artifact.get("windows", []) if isinstance(gemma_artifact, dict) else []
    qwen_text_calls = 0
    candidates: list[dict] = []
    for chunk in triage_chunks:
        chunk_start = _as_int(chunk.get("chunk_start"), 0)
        chunk_end = _as_int(chunk.get("chunk_end"), chunk_start + 1)
        matching_windows = [w for w in gemma_windows if isinstance(w, dict) and _as_int(w.get("start"), -1) < chunk_end and _as_int(w.get("end"), -1) > chunk_start]
        gemma_summary = summarize_gemma_signals_for_triage(matching_windows)
        gemma_summary["annotation_refs"] = [w.get("window_id") for w in matching_windows if w.get("window_id")]
        prompt = _fast_pass_text_triage_prompt(chunk=chunk, gemma_summary=gemma_summary, mode=mode)
        payload = {"model": QWEN_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 4096, "temperature": 0.1}
        qwen_text_calls += 1
        response = qwen_call(payload)
        raw_candidates = response.get("candidates", []) if isinstance(response, dict) else []
        candidates.extend(_normalize_fast_pass_triage_candidates(raw_candidates, chunk_start, chunk_end, FAST_PASS_MAX_TRIAGE_CANDIDATES))
    candidates = sorted(candidates, key=lambda c: (-float(c.get("triage_score", 0.0)), -float(c.get("triage_confidence", 0.0)), int(c.get("start", 0)), str(c.get("candidate_id", ""))))
    candidates = candidates[:FAST_PASS_MAX_TRIAGE_CANDIDATES]
    stats = {"qwen_text_calls": qwen_text_calls, "mode": mode, "triage_candidate_count": len(candidates)}
    return candidates, stats

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

def run_audio_phase(clips, all_results, fusion, manifest, speaker_attribution=None):
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
            speaker_attribution=speaker_attribution,
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
    log(f"  Starting Bee server via: {BEE_START_COMMAND}")
    # Kill any existing Bee/llama-server process
    subprocess.run("pkill -f llama-server 2>/dev/null; sleep 2", shell=True, capture_output=True, timeout=10)
    bee_start = subprocess.run(BEE_START_COMMAND, shell=True, capture_output=True, text=True, timeout=30)
    if bee_start.returncode != 0:
        log(f"  WARN: Bee start command returned {bee_start.returncode}")
        if bee_start.stderr:
            log(f"  stderr: {bee_start.stderr[:500]}")

    # Step 9: Wait for Bee API to be ready
    log("  Waiting for Bee API to be ready (up to ~3 min cold start)...")
    bee_startup = ensure_bee_api_ready(
        base_url=BEE_URL,
        start_bee=False,
        timeout=300,
        check_interval=5,
        logger=lambda message: log(f"    {message}"),
    )
    if bee_startup.ready:
        log(f"  ✅ Bee API ready at {bee_models_url()}")
    else:
        log(f"  ❌ Qwen API did not become ready within timeout ({bee_models_url()})")
    
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

IMPORTANT CONTEXT FOR UNDERSTANDING THE STREAM:
- Chat messages FROM the streamer account (username "asyajade") are auto-bot responses like "has redeemed their daily pickle" - the streamer reads these aloud as donation/sub alerts.
- When you see a viewer chat message, then the SAME or closely similar text appears in the TRANSCRIPT spoken by the streamer, the streamer is READING that chat message aloud (not speaking from personal experience). The story is the CHATTER'S, not the streamer's. Attribute correctly: e.g. "Streamer reads a chat message about..." not "Streamer says she..."
- This applies to BOTH timed donation/sub alerts AND regular chat messages the streamer chooses to read. The key signal is: chat message + same/similar transcript = streamer reading chat.
- The streamer may react emotionally to these alerts (laughing, commenting), then explain the inside joke to new viewers.
- A clip where the streamer explains an inside joke to a new viewer IS a story arc with setup and payoff. HIGH clip value.
- A clip where the streamer just reacts to an alert without explanation is a transactional reaction. LOW clip value.

{streamer_profile_context}

PHASE 1 TITLE RESEARCH BRIEF:
{phase1_title_research_summary}

SPEAKER-FRAMING INFERENCE RULES:
- Use speaker attribution evidence in the transcript context (if present) to infer who is actually speaking.
- If a draft title frames this as a streamer reaction but the streamer is not the primary speaker, reframe to the real speaker/situation.
- If streamer absence weakens standalone narrative value, lower clip_worthiness accordingly and explain attribution risk.
- Guest-led clips may still be strong if attribution is accurate and payoff is clear.
- Do NOT assume deterministic speaker penalties/hard gates exist in Python. Handle this by inference/reframing in your analysis.

{phase1_title_examples}

CLIP-SPECIFIC CONTEXT:
Transcript context: {transcript}
Chat messages:
{chat_messages}
YOLO detections: {yolo_objects}
{fast_pass_evidence_context}

PREVIOUS BATCH CONTEXT (what's been analysed so far in this VOD):
{batch_context}

Analyse these specific frames and return valid JSON only:
{{
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
  "failure_modes": [
    {{
      "id": "A1|B2|C3|D1|E2",
      "category": "A|B|C|D|E",
      "reason": "short reason this failure applies",
      "suggested_penalty": "-0.5 to -5.0"
    }}
  ],
  "clip_point": "CLICK-WORTHY TITLE (max 12 words). Must follow PHASE 1 TITLE RESEARCH BRIEF and be evidence-grounded in trigger+payoff. Avoid duplicate words or repeated phrase structures (e.g. 'next to the X next to the Y').",
  "title_why": "1 sentence: why this title balances specificity + curiosity and remains accurate to the clip evidence.",
  "speaker_framing_assessment": {{
    "primary_speaker_identity": "streamer|guest|chat|unknown|mixed",
    "is_framed_as_streamer_reaction": true/false,
    "streamer_actually_speaking": true/false,
    "attribution_risk": "none|low|medium|high",
    "recommended_title_framing": "How title should be framed",
    "evidence": ["speaker evidence lines"]
  }},
  "comparative_note": "How this compares to previously analysed clips in this VOD",
  "reason": "why this would (or wouldn't) make a good clip"
}}

NARRATIVE CATEGORIES (use these to classify):
- storytelling: Story with beginning/middle/end or joke with setup+punchline. HIGHEST clip value.
- chat_banter: Organic back-and-forth with chat with a clear punchline. HIGH clip value.
- organic_reaction: Genuine reaction to unexpected/funny moment. GOOD clip value.
- transactional_reaction: Response to donation/sub/raid/host alert. LOW clip value (the emotion is about a transaction, not the streamer's content).
- ambient: Calm background, no particular moment. LOW clip value.
- other: Doesn't fit above.

KEY RULES:
- A donation alert laugh may have high emotional energy but LOW clip value because the humor is the donors
- DEAD AIR RULE: Check the transcript timestamps and the ⚠️ DEAD AIR DETECTED warning above. If the warning shows a single silence gap > 20 seconds, apply a -3 penalty and cap clip_worthiness at 6/10. If total silence > 30% of the window, cap score at 5/10.
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
- clip_point must be <=12 words, evidence-grounded in trigger+payoff, avoid dry metadata phrasing, and avoid duplicate words or repeated phrase structures.
- For chat-read clips, keep attribution to chat while preserving hook quality.

CLIP CRITICISM RULE (Stage 1):
- Populate failure_modes with any applicable structural/context/pacing/transactional/technical failures.
- Each failure should include suggested_penalty and concise reason.
- Use failure_modes conservatively; Stage 2 applies deterministic penalty cap.

IMPORTANT: Stage 1 is discovery-only. Do not perform final title optimization or final platform recommendation decisions here.

Focus on narrative discovery quality and trim precision in this stage."""

PROVISIONAL_SYNTHESIS_PROMPT = """You have just analysed {total_clips} clip candidates from a Twitch VOD titled "{vod_title}" by {streamer}.

Here is the complete analysis log from every batch:

{complete_log}

Audio analysis context:
{audio_context}

Now produce a provisional ranked synthesis. You may recommend ANY number of clips (0 to {total_clips}) — there is no fixed limit. Only include clips that are genuinely worth clipping.

IMPORTANT: If you need additional visual information to make a confident decision about a specific clip (e.g. you want to see more frames to confirm an expression, verify scene context, or check visual quality), specify those clip start times in "need_more_frames" and explain why in "frame_requests". Additional frames will be sampled and shown to you in a follow-up.

Return valid JSON only:
{{
  "vod_id": "{vod_id}",
  "selected_clips": [
    {{
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
    }}
  ],
  "need_more_frames": [/* array of clip start times needing more frames, or empty [] */],
  "frame_requests": [
    {{
      "clip_start": 123,
      "reason": "why I need more frames (e.g. 'uncertain about expression', 'want to verify energy level')",
      "preferred_timestamps": [124, 125, 126]
    }}
  ],
  "overall_vod_assessment": "summary paragraph",
  "total_clips_evaluated": {total_clips}
}}

IMPORTANT RULES:
- "selected_clips" can be empty (no good clips), have 1 clip, or have 10+ clips. No fixed cap.
- "need_more_frames" should be an empty array [] if you are confident about all clips.
- "frame_requests" should be an empty array [] if no additional frames needed.
- If you do request frames, keep it to at most 3 clips that you're most uncertain about.
- NARRATIVE QUALITY matters more than emotional energy. A transactional reaction (donation/sub alert) is LOW value. A story or chat banter is HIGH value.
- DEAD AIR RULE: Check the ⚠️ DEAD AIR DETECTED warnings in the analysis log. If a clip has a single silence gap > 20 seconds, apply a -3 penalty and cap score at 6/10. If total silence > 30% of window, score ≤ 5. Discard clips with too much dead air.
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
{{
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
}}

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
{{
  "vod_id": "{vod_id}",
  "final_selected_clips": [
    {{
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
      "clip_point": "CLICK-WORTHY TITLE (1 sentence max). Use a proven pattern: reaction-based ('Streamer [reaction] after [trigger]'), question bait ('What happens when...?'), or short + punchy ('She had ONE job'). NO dry descriptions. For chat-read clips, keep attribution but make it hooky (e.g. 'What happens when chat drops a message about ...?'). Avoid duplicate words or repeated phrase structures.",
    }}
  ],
  "overall_vod_assessment": "final summary paragraph",
  "total_clips_evaluated": {total_clips},
  "clips_requesting_extra_frames": {frames_requested_count}
}}

IMPORTANT RULES:
- "final_selected_clips" can be empty, 1, or many. No fixed limit on the number of clips.
- NARRATIVE QUALITY matters more than emotional energy. Prioritize clips with stories, chat banter, or organic moments over transactional reactions.
- DEDUP RULE: Each clip has a unique clip_id (e.g. 'Clip at 998s'). NEVER assign the same clip_point/title to two different clips. Each clip MUST have a unique title. Differentiate similar clips by focusing on what makes each moment distinct.
- TITLE RULE: clip_point must be click-worthy (reaction, question-bait, or punchy one-liner). Keep factual attribution in analysis fields, but title must maximize curiosity. For chat-read clips, keep attribution while still hooky (e.g. 'What happens when chat drops a message about ...?'). Avoid dry forms like 'Streamer reads a chat message about ...' and avoid duplicate words or repeated phrase structures.
- SPEAKER-FRAMING RULE: infer whether streamer is actually speaking from speaker-attribution context in the analysis log. If primary voice is guest/non-streamer, do not frame title as streamer reaction unless streamer speech is the payoff. Guest-led clips can still be selected when accurately attributed.
- SPEAKER POLICY: do not assume deterministic speaker-specific penalties/gates in Python; handle this through title/report inference and attribution-risk reasoning.
- DEAD AIR RULE: Check for ⚠️ DEAD AIR DETECTED in the analysis log. If a single silence gap > 20 seconds exists, that clip must take a -3 penalty with a cap at 6/10. If total silence > 30% of window, score ≤ 5. Discard clips with unacceptable dead air. Ambient atmosphere is NOT dead air — differentiate.
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



def _build_fast_pass_candidate(clip: dict, gemma_summary: dict | None = None) -> dict:
    gemma_summary = gemma_summary or {}
    evidence_lines = [f"[{clip['start']}s] clip window {clip['start']}-{clip['end']}s"]
    if gemma_summary.get("evidence_lines"):
        evidence_lines.extend(gemma_summary["evidence_lines"][:3])
    if gemma_summary.get("has_audio_alert"):
        evidence_lines.append("gemma_audio_alert")
    if gemma_summary.get("has_visual_reaction"):
        evidence_lines.append("gemma_visual_reaction")
    if gemma_summary.get("streamer_led_likelihood", 0.0) >= 0.6:
        evidence_lines.append("streamer_led_likelihood_high")
    if gemma_summary.get("transactional_alert_likelihood", 0.0) >= 0.6:
        evidence_lines.append("transactional_alert_likelihood_high")
    triage_score = min(
        10.0,
        1.0
        + (2.5 if gemma_summary.get("has_audio_alert") else 0.0)
        + (2.0 if gemma_summary.get("has_visual_reaction") else 0.0)
        + 2.0 * float(gemma_summary.get("streamer_led_likelihood", 0.0) or 0.0)
        + 1.5 * float(gemma_summary.get("transactional_alert_likelihood", 0.0) or 0.0),
    )
    triage_confidence = min(1.0, 0.35 + 0.1 * len(gemma_summary.get("evidence_lines", []) or []))
    candidate = normalize_triage_candidate(
        {
            "candidate_id": f"triage_{clip['start']}",
            "start": clip["start"],
            "end": clip["end"],
            "suggested_trim_start": clip.get("suggested_trim_start", clip["start"]),
            "suggested_trim_end": clip.get("suggested_trim_end", clip["end"]),
            "narrative_type": clip.get("narrative_type", "other"),
            "trigger": clip.get("trigger", "What starts the moment"),
            "payoff": clip.get("payoff", "What resolves or lands"),
            "evidence_lines": evidence_lines,
            "risk_flags": list(clip.get("risk_flags", [])) + [flag for flag in gemma_summary.get("risk_counts", {}).keys()],
            "triage_score": triage_score,
            "triage_confidence": triage_confidence,
            "vision_need": "critical" if gemma_summary.get("has_audio_alert") or gemma_summary.get("has_visual_reaction") else "verify_expression",
            "selection_reasons": ["text_top_rank"],
        },
        fallback_start=clip["start"],
        fallback_end=clip["end"],
    )
    candidate["gemma_annotation_refs"] = list(gemma_summary.get("annotation_refs", []) or [])
    candidate["selection_reasons"] = ["text_top_rank"] + (["gemma_audio_alert_or_laughter"] if gemma_summary.get("has_audio_alert") else []) + (["gemma_visual_reaction"] if gemma_summary.get("has_visual_reaction") else [])
    return candidate


def _build_fast_pass_artifacts(*, fusion: dict, manifest: dict, clips: list[dict], phase4_dir: Path, speaker_attribution: dict | None, dry_run: bool) -> dict:
    started_at = time.time()
    triage_chunks = build_triage_chunks(
        fusion.get("transcript", {}).get("segments", []),
        fusion.get("chat", {}).get("messages", []),
        vod_start=min((int(c.get("start", 0)) for c in clips), default=0),
        vod_end=max((int(c.get("end", 0)) for c in clips), default=1),
        chunk_seconds=FAST_PASS_CHUNK_SECONDS,
        overlap_seconds=FAST_PASS_OVERLAP_SECONDS,
    )
    for chunk in triage_chunks:
        chunk["signal_summary"] = chunk.get("signal_summary") or {}

    gemma_result = None
    if FAST_PASS_MODE == "text-only":
        gemma_artifact = {
            "backend": "disabled",
            "windows": [],
            "stats": {
                "total_windows": 0,
                "successful_windows": 0,
                "failed_windows": 0,
                "wall_clock_seconds": 0.0,
            },
        }
    else:
        gemma_result = run_gemma_enrichment(
            base_url=GEMMA_URL,
            model=GEMMA_MODEL,
            phase4_dir=str(phase4_dir),
            fusion={**fusion, "triage_chunks": triage_chunks},
            manifest=manifest,
            frames_dir=str(phase4_dir / "frames"),
            raw_vod_path=VOD_MP4_PATH,
            window_seconds=GEMMA_WINDOW_SECONDS,
            stride_seconds=GEMMA_WINDOW_STRIDE_SECONDS,
            frames_per_window=GEMMA_FRAMES_PER_WINDOW,
            max_windows=GEMMA_MAX_WINDOWS,
            timeout=GEMMA_RESPONSE_TIMEOUT_SECONDS,
            concurrent_workers=GEMMA_CONCURRENT_WORKERS,
            audio_max_seconds=GEMMA_AUDIO_MAX_SECONDS,
        )
        gemma_artifact = gemma_result["artifact"]

    bee_ready = ensure_bee_api_ready(
        base_url=BEE_URL,
        start_bee=START_BEE,
        start_command=BEE_START_COMMAND,
        timeout=300,
        check_interval=5,
        logger=lambda message: log(f"  {message}"),
    )
    if not bee_ready.ready:
        raise RuntimeError(bee_ready.message)
    if bee_ready.started:
        log(f"Bee API managed startup succeeded at {bee_models_url()}")
    else:
        log(f"Bee API already reachable at {bee_models_url()}")

    triage_candidates, triage_stats = _run_fast_pass_text_triage(
        triage_chunks=triage_chunks,
        gemma_artifact=gemma_artifact,
        mode=FAST_PASS_MODE,
    )

    if not triage_candidates and FAST_PASS_MODE != "text-only":
        gemma_windows = gemma_artifact.get("windows", []) if isinstance(gemma_artifact, dict) else []
        for clip in clips:
            matching = [
                w
                for w in gemma_windows
                if isinstance(w, dict)
                and _as_int(w.get("start"), -1) <= _as_int(clip.get("end"), 0)
                and _as_int(w.get("end"), -1) >= _as_int(clip.get("start"), 0)
            ]
            gemma_summary = summarize_gemma_signals_for_triage(matching)
            gemma_summary["annotation_refs"] = [w.get("window_id") for w in matching if w.get("window_id")]
            triage_candidates.append(_build_fast_pass_candidate(clip, gemma_summary))
        triage_stats["fallback_used"] = True

    triage_candidates = sorted(
        triage_candidates,
        key=lambda c: (-float(c.get("triage_score", 0.0)), -float(c.get("triage_confidence", 0.0)), int(c.get("start", 0)), str(c.get("candidate_id", ""))),
    )[:FAST_PASS_MAX_TRIAGE_CANDIDATES]
    text_triage_path = phase4_dir / "text_triage_candidates.json"
    text_triage_payload = {"mode": FAST_PASS_MODE, "candidates": triage_candidates}
    text_triage_path.write_text(json.dumps(text_triage_payload, indent=2))

    vision_budget = compute_vision_budget(len(triage_candidates), FAST_PASS_VISION_RATIO, FAST_PASS_MIN_VISION_CANDIDATES, FAST_PASS_MAX_VISION_CANDIDATES)
    shortlist = select_vision_shortlist(triage_candidates, manifest.get("clips", []), vision_budget=vision_budget, sentinel_ratio=FAST_PASS_SENTINEL_RATIO)
    shortlist_path = phase4_dir / "vision_shortlist.json"
    shortlist_payload = {"mode": FAST_PASS_MODE, "shortlist": shortlist}
    shortlist_path.write_text(json.dumps(shortlist_payload, indent=2))

    selection_reason_counts = {}
    for candidate in triage_candidates:
        for reason in candidate.get("selection_reasons") or []:
            selection_reason_counts[reason] = selection_reason_counts.get(reason, 0) + 1

    gemma_stats = gemma_artifact.get("stats", {}) if isinstance(gemma_artifact, dict) else {}
    gemma_windows_count = len(gemma_artifact.get("windows", []) if isinstance(gemma_artifact, dict) else [])

    return {
        "enabled": True,
        "mode": FAST_PASS_MODE,
        "dry_run": dry_run,
        "qwen_text_calls": triage_stats.get("qwen_text_calls", 0),
        "qwen_vision_calls": 0,
        "qwen_images_sent": 0,
        "escalated_frame_count": 0,
        "selection_reason_counts": selection_reason_counts,
        "gemma": {
            "artifact_path": gemma_result["artifact_path"] if gemma_result else "",
            "stats": gemma_stats,
            "backend": gemma_artifact.get("backend", "llama_cpp") if isinstance(gemma_artifact, dict) else "llama_cpp",
            "window_count": gemma_windows_count,
            "failure_count": int(gemma_stats.get("failed_windows", 0)),
        },
        "text_triage_path": str(text_triage_path),
        "vision_shortlist_path": str(shortlist_path),
        "artifact_paths": {
            "gemma_annotations": str(gemma_result["artifact_path"]) if gemma_result else "",
            "text_triage": str(text_triage_path),
            "vision_shortlist": str(shortlist_path),
        },
        "triage_candidates": triage_candidates,
        "vision_shortlist": shortlist,
        "gemma_artifact": gemma_artifact,
        "summary": {
            **(gemma_result.get("summary", {}) if gemma_result else {}),
            "triage_candidate_count": len(triage_candidates),
            "vision_shortlist_count": len(shortlist),
        },
        "wall_clock_seconds": round(time.time() - started_at, 3),
    }


def run():
    log(f"Loading data for VOD {VOD_ID} ...")

    # Standard path: Bee is required before Stage 1. Fast-pass can defer Bee
    # startup until after Gemma enrichment to maximize Gemma GPU headroom.
    if not FAST_PASS:
        preflight = ensure_bee_api_ready(
            base_url=BEE_URL,
            start_bee=START_BEE,
            start_command=BEE_START_COMMAND,
            timeout=300,
            check_interval=5,
            logger=lambda message: log(f"  {message}"),
        )
        if not preflight.ready:
            log("ERROR: Bee API preflight failed; aborting before Stage 1.")
            print(preflight.message)
            if not START_BEE:
                print("Hint: re-run with --start-bee and optionally --bee-start-command '<command>'.")
            sys.exit(2)
        if preflight.started:
            log(f"Bee API managed startup succeeded at {bee_models_url()}")
        else:
            log(f"Bee API already reachable at {bee_models_url()}")

    fusion = load_json(FUSION_PATH)
    manifest = load_json(CLIP_MANIFEST_PATH)

    speaker_attribution = None
    speaker_path = VOD_DIR / f"speaker_attribution_{VOD_ID}.json"
    if speaker_path.exists():
        try:
            speaker_attribution = load_json(speaker_path)
            seg_count = len((speaker_attribution or {}).get("segments", []) or [])
            log(f"Loaded speaker attribution: {speaker_path} (segments={seg_count})")
        except Exception as e:
            log(f"WARN: failed to load speaker attribution ({speaker_path}): {e}")
            speaker_attribution = None

    clips = manifest.get("clips", [])
    if not clips:
        log("ERROR: No clips in manifest.")
        sys.exit(1)

    clips.sort(key=lambda c: c["start"])
    original_clip_count = len(clips)

    fast_pass_state = None
    fast_pass_run_started_at = None
    if FAST_PASS or FAST_PASS_DRY_RUN or GEMMA_SMOKE_TEST_ONLY:
        fast_pass_run_started_at = time.time()
        fast_pass_state = _build_fast_pass_artifacts(
            fusion=fusion,
            manifest=manifest,
            clips=clips,
            phase4_dir=VOD_DIR,
            speaker_attribution=speaker_attribution,
            dry_run=FAST_PASS_DRY_RUN or GEMMA_SMOKE_TEST_ONLY,
        )
        if FAST_PASS_DRY_RUN or GEMMA_SMOKE_TEST_ONLY:
            log(f"Fast-pass dry-run complete: {fast_pass_state['gemma']['artifact_path']}")
            log(f"  text triage: {fast_pass_state['text_triage_path']}")
            log(f"  vision shortlist: {fast_pass_state['vision_shortlist_path']}")
            output = {
                "vod_id": VOD_ID,
                "pipeline": "progressive-chunking-v2",
                "fast_pass": {
                    **fast_pass_state,
                    "gemma_backend": fast_pass_state['gemma']['backend'],
                },
            }
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_PATH, "w") as f:
                json.dump(output, f, indent=2)
            log(f"Fast-pass dry-run results saved to {OUTPUT_PATH}")
            sys.exit(0)
        clips = list(fast_pass_state['vision_shortlist'])
        log(f"Fast-pass enabled: reduced clips from {original_clip_count} to {len(clips)}")

    streamer_profile_context = (
        "STREAMER PROFILE CONTEXT (evidence-backed, advisory): unavailable for this run."
    )
    vod_meta = fusion.get("vod_meta") if isinstance(fusion, dict) else {}
    streamer_identity = resolve_streamer_id_context(vod_meta or {}, STREAMER_ID_OVERRIDE)
    streamer_id_for_run = streamer_identity["streamer_id"]
    streamer_profile = None

    log(
        "Resolved streamer_id for run: "
        f"{streamer_id_for_run} (source={streamer_identity['source']}, "
        f"metadata={streamer_identity['metadata_streamer_id']}, "
        f"override={streamer_identity['override_streamer_id']})"
    )
    if streamer_identity.get("warning"):
        log(f"WARN: {streamer_identity['warning']}")

    if ENABLE_PERSISTENT_INTELLIGENCE:
        try:
            streamer_profile = load_streamer_profile(
                streamer_id=streamer_id_for_run,
                root=STREAMER_PROFILE_ROOT,
            )
            streamer_profile_context = render_streamer_profile_context(streamer_profile, max_chars=2000)
            log(
                f"Loaded persistent streamer profile context for '{streamer_id_for_run}' "
                f"from {STREAMER_PROFILE_ROOT}"
            )
        except Exception as e:
            log(f"WARN: failed to load persistent streamer profile context: {e}")

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
            speaker_attribution=speaker_attribution,
        )
        return render_prompt_context(context, transcript_char_limit=2000)

    # ── Batch processing with context carryover ──
    all_results = []
    batch_context = "No clips analysed yet — this is the first batch."

    total = len(clips)
    batches = [clips[i:i+CLIPS_PER_BATCH] for i in range(0, total, CLIPS_PER_BATCH)]
    log(f"Found {total} clips in {len(batches)} batch(es) of {CLIPS_PER_BATCH}")

    if fast_pass_state:
        fast_pass_state["qwen_vision_calls"] = 0
        fast_pass_state["qwen_images_sent"] = 0
        fast_pass_state["escalated_frame_count"] = 0

    for batch_idx, batch in enumerate(batches):
        log(f"\n{'='*60}")
        log(f"BATCH {batch_idx + 1}/{len(batches)} — clips {batch[0]['start']}s - {batch[-1]['end']}s")

        # Build the multimodal payload
        messages = []
        user_content = []
        batch_image_count = 0

        for clip in batch:
            title = clip.get("title", f"clip at {clip['start']}s")
            transcript, chat_act = context_for_time(clip["start"])
            yolo_objs = clip.get("objects_detected", [])
            frame_samples = sample_clip_frames(
                clip,
                count=FAST_PASS_VISION_FRAMES if FAST_PASS else FRAMES_PER_CLIP,
                fast_pass=FAST_PASS,
                suggested_trim_start=clip.get("suggested_trim_start"),
                suggested_trim_end=clip.get("suggested_trim_end"),
            )
            fast_pass_evidence_context = ""
            if FAST_PASS:
                evidence_block = _build_fast_pass_evidence_block(clip)
                fast_pass_evidence_context = (
                    "FAST-PASS TRIAGE EVIDENCE (advisory only — verify/correct this evidence rather than trusting it blindly):\n"
                    f"{evidence_block}"
                )

            prompt = ANALYSIS_PROMPT.format(
                clip_title=title,
                start=clip["start"], end=clip["end"],
                transcript=transcript,
                chat_messages=chat_act,
                yolo_objects=", ".join(yolo_objs) if yolo_objs else "none",
                fast_pass_evidence_context=fast_pass_evidence_context,
                batch_context=batch_context,
                streamer_profile_context=streamer_profile_context,
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
                    batch_image_count += 1
                except Exception as e:
                    log(f"  WARN: failed to encode {fp}: {e}")

        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": QWEN_MODEL,
            "messages": messages,
            "max_tokens": 16384,
            "temperature": 0.15,
        }

        log(f"  Sending {batch_image_count} images to Qwen ...")
        if fast_pass_state:
            fast_pass_state["qwen_vision_calls"] += 1
            fast_pass_state["qwen_images_sent"] += batch_image_count
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
        max_cluster_size=1,  # no merging — keep clips independent
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
    all_results = run_audio_phase(
        clips,
        all_results,
        fusion,
        manifest,
        speaker_attribution=speaker_attribution,
    )

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
            speaker_attribution=speaker_attribution,
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

        rep_analysis["speaker_attribution"] = {
            "primary_speaker_identity": stitched_context.get("primary_speaker_identity") or "unknown",
            "primary_speaker_name": stitched_context.get("primary_speaker_name"),
            "streamer_speaking_ratio": stitched_context.get("streamer_speaking_ratio", 0.0),
            "streamer_speaking_confidence": stitched_context.get("streamer_speaking_confidence", 0.0),
            "off_streamer_voice_detected": stitched_context.get("off_streamer_voice_detected", False),
            "evidence": list(stitched_context.get("speaker_name_evidence") or []),
        }

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

            if fast_pass_state:
                fast_pass_state["qwen_vision_calls"] += 1
                fast_pass_state["qwen_images_sent"] += len(extra_frames)
                fast_pass_state["escalated_frame_count"] += len(extra_frames)

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
        model_clips = final_synthesis.get("final_selected_clips", [])
        final_synthesis["model_final_selected_clips"] = model_clips
        final_synthesis["final_selected_clips"] = stage3_final_selected

        # Merge Qwen's richer final-synthesis fields into the deterministic payloads.
        # Match by `start` position (int/float), which is the most stable key across both paths.
        qwen_by_start = {}
        for mc in model_clips:
            ms = mc.get("start")
            if ms is not None:
                qwen_by_start[ms] = mc
            # Also index by suggested_trim_start since Qwen reports start as trim start
            ts = mc.get("suggested_trim_start")
            if ts is not None:
                qwen_by_start[ts] = mc

        for clip in stage3_final_selected:
            cstart = clip.get("start")
            cend = clip.get("end")
            # Try exact match first, then trim_start match, then range match
            qc = qwen_by_start.get(cstart)
            if not qc and cend:
                # Fallback: find any Qwen clip whose start falls within [cstart, cend]
                for mc in model_clips:
                    ms = mc.get("start")
                    if ms is not None and cstart <= ms <= cend:
                        qc = mc
                        break
            if not qc:
                continue

            # Qwen's platform data is more nuanced — use it when present.
            if qc.get("platform_scores"):
                clip["platform_scores"] = qc["platform_scores"]
            if qc.get("platform_recommendations"):
                clip["platform_recommendations"] = qc["platform_recommendations"]
            if qc.get("platform_reasoning"):
                clip["platform_reasoning"] = qc["platform_reasoning"]

            # Enrich the intelligence_report with Qwen's synthesis judgment.
            ir = clip.setdefault("intelligence_report", {})
            if qc.get("why"):
                ir["why_selected"] = qc["why"]
            for field in ("strengths", "weaknesses", "narrative_quality"):
                if qc.get(field):
                    ir[field] = qc[field]

    if fast_pass_state and fast_pass_run_started_at is not None:
        fast_pass_state["wall_clock_seconds"] = round(time.time() - fast_pass_run_started_at, 3)

    # ── Save ──
    output = {
        "vod_id": VOD_ID,
        "pipeline": "progressive-chunking-v2",
        "streamer_identity": streamer_identity,
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
        "fast_pass": fast_pass_state if fast_pass_state else {"enabled": False},
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    log(f"\nResults saved to {OUTPUT_PATH}")
    selected = final_synthesis.get("final_selected_clips", [])
    log(f"Final selection: {len(selected)} clip(s)")
    for s in selected:
        start = s.get('start', '?')
        hms = s.get('start_hms', '')
        pos = f"{start}s ({hms})" if hms else f"{start}s"
        log(f"  Rank {s.get('rank','?')}: {pos} — score={s.get('score','?')}/10 — {(s.get('intelligence_report') or {}).get('why_selected','')[:100]}")

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
            speaker_attribution=speaker_attribution,
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

    # Add HH:MM:SS format timestamps to selected clips for the report.
    _add_hms_and_links(selected, VOD_ID)

    # Re-save with narrowed boundaries and rescored eligibility.
    if rms_rejected_clips:
        output.setdefault("rejected_clips", []).extend(rms_rejected_clips)
    output["final_ranking"]["final_selected_clips"] = selected
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    log(f"Updated results saved to {OUTPUT_PATH}")

    if ENABLE_PERSISTENT_INTELLIGENCE and UPDATE_STREAMER_PROFILE and PROFILE_UPDATE_MODE != "off":
        try:
            if streamer_profile is None:
                streamer_profile = load_streamer_profile(
                    streamer_id=streamer_id_for_run,
                    root=STREAMER_PROFILE_ROOT,
                )

            proposal = build_profile_update_proposal(
                vod_id=VOD_ID,
                streamer_id=streamer_id_for_run,
                final_selected_clips=selected,
                mode=PROFILE_UPDATE_MODE,
                streamer_id_source=streamer_identity.get("source", "fallback"),
                metadata_streamer_id=streamer_identity.get("metadata_streamer_id"),
                override_streamer_id=streamer_identity.get("override_streamer_id"),
                mismatch_warning=streamer_identity.get("warning"),
            )
            proposal_path = OUTPUT_PATH.parent / f"profile_update_proposal_{VOD_ID}.json"
            with proposal_path.open("w", encoding="utf-8") as f:
                json.dump(proposal.model_dump(mode="json"), f, indent=2)

            output["profile_update"] = {
                "enabled": True,
                "mode": PROFILE_UPDATE_MODE,
                "streamer_id": streamer_id_for_run,
                "streamer_id_source": streamer_identity.get("source", "fallback"),
                "metadata_streamer_id": streamer_identity.get("metadata_streamer_id"),
                "override_streamer_id": streamer_identity.get("override_streamer_id"),
                "mismatch_warning": streamer_identity.get("warning"),
                "proposal_path": str(proposal_path),
                "candidate_observations": len(proposal.candidate_observations),
            }

            if PROFILE_UPDATE_MODE == "auto":
                updated_profile, accepted, queued, rejected = apply_profile_update_auto(
                    streamer_profile,
                    proposal,
                )
                if accepted:
                    append_observations(streamer_id_for_run, accepted, STREAMER_PROFILE_ROOT)
                save_streamer_profile(updated_profile, STREAMER_PROFILE_ROOT)
                output["profile_update"].update(
                    {
                        "accepted": len(accepted),
                        "queued": len(queued),
                        "rejected": len(rejected),
                    }
                )
                log(
                    "Persistent profile auto-update: "
                    f"accepted={len(accepted)} queued={len(queued)} rejected={len(rejected)}"
                )
            else:
                accepted, queued, rejected = partition_observations_for_merge(
                    proposal.candidate_observations
                )
                output["profile_update"].update(
                    {
                        "accepted_if_auto": len(accepted),
                        "queued_if_manual": len(queued),
                        "rejected_if_auto": len(rejected),
                    }
                )
                log(
                    "Persistent profile proposal generated "
                    f"(accepted_if_auto={len(accepted)}, queued_if_manual={len(queued)}, rejected_if_auto={len(rejected)}): "
                    f"{proposal_path}"
                )

            with open(OUTPUT_PATH, "w") as f:
                json.dump(output, f, indent=2)
            log(f"Updated results saved with profile-update metadata: {OUTPUT_PATH}")
        except Exception as e:
            log(f"WARN: persistent profile update flow failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vod-id", default=VOD_ID)
    parser.add_argument(
        "--bee-url",
        default=BEE_URL,
        help=f"Bee API base URL (default: env BEE_URL or {DEFAULT_BEE_URL})",
    )
    parser.add_argument(
        "--start-bee",
        action="store_true",
        help="Attempt managed Bee startup when API is not reachable",
    )
    parser.add_argument(
        "--bee-start-command",
        default=BEE_START_COMMAND,
        help="Command used with --start-bee (default: env BEE_START_COMMAND)",
    )
    parser.add_argument("--batch-size", type=int, default=CLIPS_PER_BATCH)
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio analysis phase")
    parser.add_argument("--top-clips", type=int, default=AUDIO_CLIPS_TO_PROCESS, help="Number of top clips for audio analysis")
    parser.add_argument("--vod-mp4", default=None, help="Path to VOD MP4 for audio extraction")
    parser.add_argument("--streamer-id", default=None, help="Override streamer ID for persistent intelligence")
    parser.add_argument(
        "--profile-root",
        default=None,
        help="Root dir for persistent streamer profiles (default: data/streamer_intelligence)",
    )
    parser.add_argument(
        "--enable-persistent-intelligence",
        action="store_true",
        help="Enable loading persistent streamer profile context for prompts",
    )
    parser.add_argument(
        "--update-streamer-profile",
        action="store_true",
        help="Enable writing profile_update_proposal and profile merge flow",
    )
    parser.add_argument(
        "--profile-update-mode",
        choices=["propose", "auto", "off"],
        default=PROFILE_UPDATE_MODE,
        help="Persistent profile update mode (propose, auto, off)",
    )
    parser.add_argument("--fast-pass", action="store_true", help="Enable Gemma-enriched fast-pass Stage 1 routing")
    parser.add_argument("--fast-pass-mode", choices=["gemma-enriched", "text-only"], default=FAST_PASS_MODE)
    parser.add_argument("--fast-pass-dry-run", action="store_true", help="Run Gemma/text shortlist generation and exit before Stage 1 vision")
    parser.add_argument("--gemma-smoke-test-only", action="store_true", help="Alias for a Gemma fast-pass dry run")
    parser.add_argument("--gemma-url", default=GEMMA_URL)
    parser.add_argument("--gemma-model", default=GEMMA_MODEL)
    parser.add_argument("--gemma-window-seconds", type=int, default=GEMMA_WINDOW_SECONDS)
    parser.add_argument("--gemma-window-stride-seconds", type=int, default=GEMMA_WINDOW_STRIDE_SECONDS)
    parser.add_argument("--gemma-max-windows", type=int, default=GEMMA_MAX_WINDOWS)
    parser.add_argument("--gemma-frames-per-window", type=int, default=GEMMA_FRAMES_PER_WINDOW)
    parser.add_argument("--gemma-audio-max-seconds", type=int, default=GEMMA_AUDIO_MAX_SECONDS)
    parser.add_argument("--gemma-response-timeout-seconds", type=int, default=GEMMA_RESPONSE_TIMEOUT_SECONDS)
    parser.add_argument("--fast-pass-chunk-seconds", type=int, default=FAST_PASS_CHUNK_SECONDS)
    parser.add_argument("--fast-pass-overlap-seconds", type=int, default=FAST_PASS_OVERLAP_SECONDS)
    parser.add_argument("--fast-pass-max-triage-candidates", type=int, default=FAST_PASS_MAX_TRIAGE_CANDIDATES)
    parser.add_argument("--fast-pass-vision-ratio", type=float, default=FAST_PASS_VISION_RATIO)
    parser.add_argument("--fast-pass-min-vision-candidates", type=int, default=FAST_PASS_MIN_VISION_CANDIDATES)
    parser.add_argument("--fast-pass-max-vision-candidates", type=int, default=FAST_PASS_MAX_VISION_CANDIDATES)
    parser.add_argument("--fast-pass-vision-frames", type=int, default=FAST_PASS_VISION_FRAMES)
    parser.add_argument("--fast-pass-sentinel-ratio", type=float, default=FAST_PASS_SENTINEL_RATIO)

    args, _ = parser.parse_known_args()
    VOD_ID = args.vod_id
    BEE_URL = args.bee_url
    START_BEE = args.start_bee
    BEE_START_COMMAND = args.bee_start_command
    # Recompute VOD-dependent paths now that VOD_ID is known.
    VOD_DIR = Path(os.environ.get("VOD_DIR", f"/home/john/twitch-vod-analyzer/vods/phase4_{VOD_ID}"))
    FRAMES_DIR = VOD_DIR / "frames"
    FUSION_PATH = VOD_DIR / f"fusion_result_{VOD_ID}.json"
    CLIP_MANIFEST_PATH = VOD_DIR / "clip_manifest.json"
    OUTPUT_PATH = VOD_DIR / "qwen_vision_progressive.json"
    VOD_MP4_PATH = os.environ.get("VOD_MP4_PATH", f"/home/john/twitch-vod-analyzer/vods/phase4_{VOD_ID}/raw/{VOD_ID}.mp4")
    if args.skip_audio:
        ENABLE_AUDIO = False
    AUDIO_CLIPS_TO_PROCESS = args.top_clips
    if args.vod_mp4:
        VOD_MP4_PATH = args.vod_mp4

    CLIPS_PER_BATCH = args.batch_size

    if args.streamer_id:
        STREAMER_ID_OVERRIDE = args.streamer_id
    if args.profile_root:
        STREAMER_PROFILE_ROOT = Path(args.profile_root)
    if args.enable_persistent_intelligence:
        ENABLE_PERSISTENT_INTELLIGENCE = True

    if args.update_streamer_profile:
        UPDATE_STREAMER_PROFILE = True
        ENABLE_PERSISTENT_INTELLIGENCE = True
    PROFILE_UPDATE_MODE = args.profile_update_mode

    FAST_PASS = args.fast_pass
    FAST_PASS_MODE = args.fast_pass_mode
    FAST_PASS_DRY_RUN = args.fast_pass_dry_run
    GEMMA_SMOKE_TEST_ONLY = args.gemma_smoke_test_only
    GEMMA_URL = args.gemma_url
    GEMMA_MODEL = args.gemma_model
    GEMMA_WINDOW_SECONDS = args.gemma_window_seconds
    GEMMA_WINDOW_STRIDE_SECONDS = args.gemma_window_stride_seconds
    GEMMA_MAX_WINDOWS = args.gemma_max_windows
    GEMMA_FRAMES_PER_WINDOW = args.gemma_frames_per_window
    GEMMA_AUDIO_MAX_SECONDS = args.gemma_audio_max_seconds
    GEMMA_RESPONSE_TIMEOUT_SECONDS = args.gemma_response_timeout_seconds
    FAST_PASS_CHUNK_SECONDS = args.fast_pass_chunk_seconds
    FAST_PASS_OVERLAP_SECONDS = args.fast_pass_overlap_seconds
    FAST_PASS_MAX_TRIAGE_CANDIDATES = args.fast_pass_max_triage_candidates
    FAST_PASS_VISION_RATIO = args.fast_pass_vision_ratio
    FAST_PASS_MIN_VISION_CANDIDATES = args.fast_pass_min_vision_candidates
    FAST_PASS_MAX_VISION_CANDIDATES = args.fast_pass_max_vision_candidates
    FAST_PASS_VISION_FRAMES = args.fast_pass_vision_frames
    FAST_PASS_SENTINEL_RATIO = args.fast_pass_sentinel_ratio

    run()
