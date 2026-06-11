from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

from src.synthesis.fastpass_triage import (
    build_gemma_annotation_windows,
    build_gemma_audio_extract_command,
    merge_gemma_annotations_into_chunk,
    normalize_gemma_annotation,
    summarize_gemma_signals_for_triage,
)


def ensure_gemma_api_ready(
    base_url: str = "http://localhost:8084",
    gemma_bin: str = "",
    model_path: str = "",
    mmproj_path: str = "",
    draft_model_path: str = "",
    timeout: int = 600,
    check_interval: int = 5,
    logger=print,
) -> bool:
    """Start Gemma server and wait for it to be reachable.

    Starts ``llama-server`` with the Gemma QAT model and, when the
    draft model is present, enables Gemma MTP speculative decoding.

    Returns ``True`` if Gemma becomes reachable within *timeout*.
    """
    import shutil

    # Set LD_LIBRARY_PATH so the build_compat llama-server can find CUDA
    cuda_lib = "/home/john/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
    if cuda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib}:{existing}" if existing else cuda_lib

    # Auto-discover paths on WSL if not explicitly provided
    if not gemma_bin:
        candidates = [
            "/home/john/llama.cpp/build/bin/llama-server",       # latest build (MTP + multimodal fixed)
            "/home/john/llama.cpp/build_compat/bin/llama-server", # old fallback (no MTP draft)
        ]
        gemma_bin = next((p for p in candidates if os.path.isfile(p)), "")
    if not model_path:
        model_path = "/home/john/models/gemma-4-12b/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
    if not mmproj_path:
        mmproj_path = "/home/john/models/gemma-4-12b/mmproj-F16.gguf"
    if not draft_model_path:
        candidate_draft = "/home/john/models/gemma-4-12b/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf"
        if os.path.isfile(candidate_draft):
            draft_model_path = candidate_draft

    is_new_build = "build/bin/llama-server" in str(gemma_bin)

    cmd = [
        gemma_bin,
        "--host", "0.0.0.0",
        "--port", "8084",
        "--model", model_path,
        "--mmproj", mmproj_path,
        "-c", "32768",
        "-ngl", "999",
        "--no-mmap",
        "--flash-attn", "on",
        "--reasoning", "on",
        "--no-host",
    ]
    if draft_model_path:
        cmd.extend([
            "--model-draft", draft_model_path,
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", "4",
            "-np", "1",
            "--kv-unified",
            "-b", "2048",
            "-ub", "512",
            "--jinja",
        ])
    logger(
        f"Using {'new build' if is_new_build else 'build_compat'} at {gemma_bin}"
        + (f" with MTP draft {draft_model_path}" if draft_model_path else " without MTP draft")
    )

    # Check if already running
    endpoint = f"{base_url.rstrip('/')}/v1/models"
    try:
        req = requests.get(endpoint, timeout=5)
        if req.status_code < 400:
            logger("Gemma API is already reachable — no restart needed.")
            return True
    except requests.ConnectionError:
        pass

    logger(f"Starting Gemma server: {' '.join(cmd)}")
    log_path = "/home/john/gemma_mtp_server.log"
    log_fd = open(log_path, "w")
    gemma_env = os.environ.copy()
    if cuda_lib not in gemma_env.get("LD_LIBRARY_PATH", ""):
        cuda_lib = "/home/john/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
        existing = gemma_env.get("LD_LIBRARY_PATH", "")
        gemma_env["LD_LIBRARY_PATH"] = f"{cuda_lib}:{existing}" if existing else cuda_lib
    process = subprocess.Popen(
        cmd,
        stdout=log_fd,
        stderr=subprocess.STDOUT,
        env=gemma_env,
    )

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = requests.get(endpoint, timeout=10)
            if req.status_code < 400:
                logger(f"Gemma API ready at {endpoint} after {int(time.time() - deadline + timeout)}s")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(check_interval)

    logger(f"Gemma API failed to start within {timeout}s at {endpoint}.")
    process.kill()
    return False


def shutdown_gemma(
    base_url: str = "http://localhost:8084",
    pid_file: str | None = None,
    logger=print,
) -> None:
    """Shut down the Gemma server to free VRAM for Bee.

    Tries a graceful shutdown via the API endpoint first, then
    falls back to ``pkill`` scoped to the Gemma binary and port.
    """
    # Try graceful API shutdown
    try:
        req = requests.post(f"{base_url.rstrip('/')}/shutdown", timeout=5)
        if req.status_code < 400:
            logger("Gemma shutdown via API succeeded.")
            return
    except requests.ConnectionError:
        pass

    # Fallback: kill by binary name + port signature
    import signal
    try:
        # Kill process listening on port 8084
        result = subprocess.run(
            ["fuser", "-k", "-n", "tcp", "8084"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        logger("Gemma killed via fuser -k on port 8084.")
        time.sleep(2)
        return
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        pass

    # Last resort: pkill (broad)
    try:
        subprocess.run(
            ["pkill", "-f", "gemma-4-12B-it-qat-UD-Q4_K_XL"],
            capture_output=True,
            timeout=10,
        )
        logger("Gemma killed via pkill.")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger("WARN: could not kill Gemma process.")


def build_gemma_enrichment_prompt(window: dict) -> str:
    start = int(window.get("start", 0))
    end = int(window.get("end", start + 1))
    return (
        "You are producing factual annotations only. "
        "Do not make final clip decisions or final platform recommendations. "
        f"Analyze the window from {start}s to {end}s. "
        "Report what you observe using the labeled sections below. "
        "Leave a section blank if nothing relevant is observed.\n\n"
        "AUDIO_EVENTS:\n"
        "Timestamp each distinct sound: what type (streamer_speech, non_streamer_speech, donation_alert, tts_alert, game_audio, music, laugh, silence), "
        "who is speaking (streamer, chat_tts, game_character, unknown), and your confidence (0.0-1.0). "
        "Example: '1238s: donation_alert likely TTS, speaker=unknown, confidence=0.8'\n\n"
        "VISUAL_EVENTS:\n"
        "Timestamp each visual change: what type (streamer_visible, face_visible, laughing, surprised, focused, gameplay_event, scene_change, visual_payoff), "
        "and your confidence. "
        "Example: '1241s: streamer laughing at screen, confidence=0.9'\n\n"
        "SPEAKER:\n"
        "Primary speaker identity: streamer, non_streamer, mixed, or unknown. "
        f"How likely is the streamer leading this window (0.0-1.0)? How likely is this a transactional/TTS/donation alert (0.0-1.0)?\n\n"
        "EMOTION:\n"
        "Streamer affect: amused, surprised, confused, flat, performative, focused, or unknown. "
        "Is the reaction organic or triggered by an alert/donation?\n\n"
        "RISK_FLAGS:\n"
        "Any concerns: possible_alert_reaction, game_audio_dominant, visual_context_required, speaker_uncertain"
    )


def _load_bytes(path: str | None) -> bytes | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return p.read_bytes()


def build_gemma_chat_payload(*, model: str, prompt: str, image_paths: list[str], audio_path: str | None, max_tokens: int = 1200) -> dict:
    """Build payload for Gemma 4 multimodal call.

    Modality order matters for optimal results (per Unsloth QAT guide):
      IMAGES first → TEXT second → AUDIO last
    """
    content: list[dict[str, Any]] = []
    # 1) Images first
    for image_path in image_paths or []:
        blob = _load_bytes(image_path)
        if blob is None:
            continue
        mime = "image/jpeg" if Path(image_path).suffix.lower() not in {".png", ".webp"} else "image/png"
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(blob).decode()}"}})
    # 2) Text second
    content.append({"type": "text", "text": prompt})
    # 3) Audio last
    if audio_path:
        blob = _load_bytes(audio_path)
        if blob is not None:
            fmt = Path(audio_path).suffix.lower().lstrip(".") or "wav"
            content.append({"type": "input_audio", "input_audio": {"data": base64.b64encode(blob).decode(), "format": fmt}})
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }


def _parse_json_response(text: str | None) -> tuple[bool, Any | None, str | None]:
    if not text:
        return False, None, "empty response"
    try:
        return True, json.loads(text), None
    except Exception as exc:
        return False, None, str(exc)


def parse_gemma_raw_output(text: str | None) -> dict:
    """Parse Gemma's raw natural-language observations into structured annotation dict."""
    result = {
        "audio_events": [],
        "visual_events": [],
        "speaker_nuance": {
            "primary_speaker": "unknown",
            "streamer_led_likelihood": 0.0,
            "non_streamer_voice_present": False,
            "non_streamer_voice_type": "unknown",
        },
        "emotion_nuance": {
            "streamer_affect": "unknown",
            "organic_reaction_likelihood": 0.0,
            "transactional_alert_likelihood": 0.0,
            "evidence": "",
        },
        "risk_flags": [],
        "clip_relevance_notes": [],
    }
    if not text:
        return result

    import re

    # Extract AUDIO_EVENTS section
    audio_match = re.search(r"AUDIO_EVENTS:\s*(.*?)(?=\n\nVISUAL_EVENTS:|\n\nSPEAKER:|$)", text, re.DOTALL)
    if audio_match:
        raw = audio_match.group(1).strip()
        for line in raw.split("\n"):
            line = line.strip()
            if not line or line.startswith("Example") or line.startswith("None"):
                continue
            ts_match = re.search(r"(\d+[\.\d]*)s", line)
            result["audio_events"].append({
                "timestamp": float(ts_match.group(1)) if ts_match else 0.0,
                "raw": line,
            })

    # Extract VISUAL_EVENTS section
    visual_match = re.search(r"VISUAL_EVENTS:\s*(.*?)(?=\n\nSPEAKER:|\n\nEMOTION:|$)", text, re.DOTALL)
    if visual_match:
        raw = visual_match.group(1).strip()
        for line in raw.split("\n"):
            line = line.strip()
            if not line or line.startswith("Example") or line.startswith("None"):
                continue
            ts_match = re.search(r"(\d+[\.\d]*)s", line)
            result["visual_events"].append({
                "timestamp": float(ts_match.group(1)) if ts_match else 0.0,
                "raw": line,
            })

    # Extract SPEAKER section
    speaker_match = re.search(r"SPEAKER:\s*(.*?)(?=\n\nEMOTION:|\n\nRISK_FLAGS:|$)", text, re.DOTALL)
    if speaker_match:
        raw = speaker_match.group(1).strip()
        # Extract streamer_led_likelihood
        sll = re.search(r"streamer.*?led.*?([\d\.]+)", raw, re.IGNORECASE)
        if sll:
            result["speaker_nuance"]["streamer_led_likelihood"] = float(sll.group(1))
        # Extract transactional_alert_likelihood
        tal = re.search(r"transactional.*?([\d\.]+)", raw, re.IGNORECASE)
        if tal:
            result["emotion_nuance"]["transactional_alert_likelihood"] = float(tal.group(1))
        if re.search(r"non_streamer|chat_tts|game_character", raw, re.IGNORECASE):
            result["speaker_nuance"]["non_streamer_voice_present"] = True
        for kw in ["streamer", "non_streamer", "mixed", "unknown"]:
            if re.search(kw, raw, re.IGNORECASE):
                result["speaker_nuance"]["primary_speaker"] = kw

    # Extract EMOTION section
    emotion_match = re.search(r"EMOTION:\s*(.*?)(?=\n\nRISK_FLAGS:|$)", text, re.DOTALL)
    if emotion_match:
        raw = emotion_match.group(1).strip()
        result["emotion_nuance"]["evidence"] = raw[:200]
        for aff in ["amused", "surprised", "confused", "flat", "performative", "focused"]:
            if re.search(aff, raw, re.IGNORECASE):
                result["emotion_nuance"]["streamer_affect"] = aff
                break
        if re.search(r"organic|genuine|natural", raw, re.IGNORECASE):
            result["emotion_nuance"]["organic_reaction_likelihood"] = min(
                result["emotion_nuance"]["organic_reaction_likelihood"] + 0.3, 1.0
            )
        if re.search(r"alert|donation|tts|transactional", raw, re.IGNORECASE):
            result["emotion_nuance"]["transactional_alert_likelihood"] = max(
                result["emotion_nuance"]["transactional_alert_likelihood"], 0.3
            )

    # Extract RISK_FLAGS section
    risk_match = re.search(r"RISK_FLAGS:\s*(.*)", text, re.DOTALL)
    if risk_match:
        raw = risk_match.group(1).strip()
        for flag in ["possible_alert_reaction", "game_audio_dominant",
                     "visual_context_required", "speaker_uncertain"]:
            if re.search(flag.replace("_", " "), raw, re.IGNORECASE):
                result["risk_flags"].append(flag)

    return result


def call_gemma_llamacpp(*, base_url: str, payload: dict, timeout: int) -> dict:
    url = f"{base_url.rstrip('/')}/chat/completions"
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    message = data.get("choices", [{}])[0].get("message", {})
    raw = message.get("content")
    parse_ok, parsed, error = _parse_json_response(raw)
    return {
        "http_status": response.status_code,
        "raw_content": raw,
        "parse_ok": parse_ok,
        "parsed": parsed,
        "error": error,
        "backend": "llama_cpp",
    }


def _extract_window_audio(vod_mp4: str, window: dict, output_wav: str, *, max_seconds: int) -> None:
    cmd = build_gemma_audio_extract_command(vod_mp4, window, output_wav, max_seconds=max_seconds)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_gemma_enrichment(*, base_url: str, model: str, phase4_dir: str, fusion: dict, manifest: dict, frames_dir: str, raw_vod_path: str, window_seconds: int, stride_seconds: int, frames_per_window: int, max_windows: int, timeout: int, audio_max_seconds: int = 30, concurrent_workers: int = 1) -> dict:
    phase4_path = Path(phase4_dir)
    phase4_path.mkdir(parents=True, exist_ok=True)
    windows = build_gemma_annotation_windows(
        fusion.get("triage_chunks", []) or fusion.get("chunks", []) or [],
        manifest.get("clips", []) if isinstance(manifest, dict) else [],
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        max_windows=max_windows,
    )
    artifact_path = phase4_path / "gemma_multimodal_annotations.json"
    artifacts: list[dict] = []
    errors: list[dict] = []
    artifacts_lock = __import__("threading").Lock()
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        def _process_one_window(window: dict) -> None:
            entry = dict(window)
            try:
                image_paths: list[str] = []
                selected: list[str] = []
                for p in (Path(p) for p in []):
                    selected.append(str(p))
                if frames_dir:
                    from src.synthesis.fastpass_triage import select_gemma_frames_for_window
                    image_paths = select_gemma_frames_for_window(window, frames_dir, frames_per_window=frames_per_window)
                audio_path = td_path / f"{window['window_id']}.wav"
                _extract_window_audio(raw_vod_path, window, str(audio_path), max_seconds=audio_max_seconds)
                prompt = build_gemma_enrichment_prompt(window)
                payload = build_gemma_chat_payload(model=model, prompt=prompt, image_paths=image_paths, audio_path=str(audio_path), max_tokens=1200)
                raw_response = call_gemma_llamacpp(base_url=base_url, payload=payload, timeout=timeout)
                parsed_gemma = parse_gemma_raw_output(raw_response.get("raw_content"))
                normalized = normalize_gemma_annotation(parsed_gemma, window)
                normalized["parse_ok"] = True
                normalized["error"] = None
                normalized["source_refs"] = {
                    "transcript_segment_ids": [],
                    "chat_message_ids": [],
                    "frame_paths": image_paths,
                    "audio_path": str(audio_path),
                }
                with artifacts_lock:
                    artifacts.append(normalized)
            except Exception as exc:
                with artifacts_lock:
                    errors.append({"window_id": window.get("window_id"), "error": str(exc)})
                    artifacts.append(normalize_gemma_annotation({"error": str(exc), "parse_ok": False}, window))

        if concurrent_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            futures = []
            with ThreadPoolExecutor(max_workers=concurrent_workers) as pool:
                for window in windows:
                    futures.append(pool.submit(_process_one_window, window))
                for f in as_completed(futures):
                    f.result()  # surface any unhandled exception
        else:
            for window in windows:
                _process_one_window(window)
    out = {
        "vod_id": str(manifest.get("vod_id") or fusion.get("vod_id") or ""),
        "model": model,
        "backend": "llama_cpp",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "windows": artifacts,
        "stats": {
            "total_windows": len(windows),
            "successful_windows": sum(1 for a in artifacts if a.get("parse_ok")),
            "failed_windows": sum(1 for a in artifacts if not a.get("parse_ok")),
            "wall_clock_seconds": round(time.time() - t0, 3),
        },
        "errors": errors,
    }
    artifact_path.write_text(json.dumps(out, indent=2))
    return {"artifact_path": str(artifact_path), "artifact": out, "summary": summarize_gemma_signals_for_triage(artifacts), "merged_preview": merge_gemma_annotations_into_chunk({"chunk_start": 0}, artifacts)}
