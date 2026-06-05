from __future__ import annotations

import base64
import json
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


def build_gemma_enrichment_prompt(window: dict) -> str:
    start = int(window.get("start", 0))
    end = int(window.get("end", start + 1))
    return (
        "You are producing factual annotations only. "
        "Return JSON only. Do not make final clip decisions or final platform recommendations. "
        f"Analyze the window from {start}s to {end}s and report timestamped audio/visual evidence. "
        "Focus on donation/TTS alerts vs streamer speech, game audio vs human reaction, "
        "streamer-led vs non-streamer-led moments, laughter/surprise/confusion/focused affect, "
        "and visual evidence of reaction or payoff. "
        "Use concise evidence strings and keep all claims fallible."
    )


def _load_bytes(path: str | None) -> bytes | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return p.read_bytes()


def build_gemma_chat_payload(*, model: str, prompt: str, image_paths: list[str], audio_path: str | None, max_tokens: int = 1200) -> dict:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_path in image_paths or []:
        blob = _load_bytes(image_path)
        if blob is None:
            continue
        mime = "image/jpeg" if Path(image_path).suffix.lower() not in {".png", ".webp"} else "image/png"
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(blob).decode()}"}})
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
        "response_format": {"type": "json_object"},
    }


def _parse_json_response(text: str | None) -> tuple[bool, Any | None, str | None]:
    if not text:
        return False, None, "empty response"
    try:
        return True, json.loads(text), None
    except Exception as exc:
        return False, None, str(exc)


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


def _extract_window_audio(vod_mp4: str, window: dict, output_wav: str) -> None:
    cmd = build_gemma_audio_extract_command(vod_mp4, window, output_wav)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_gemma_enrichment(*, base_url: str, model: str, phase4_dir: str, fusion: dict, manifest: dict, frames_dir: str, raw_vod_path: str, window_seconds: int, stride_seconds: int, frames_per_window: int, max_windows: int, timeout: int, concurrent_workers: int = 1) -> dict:
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
                _extract_window_audio(raw_vod_path, window, str(audio_path))
                prompt = build_gemma_enrichment_prompt(window)
                payload = build_gemma_chat_payload(model=model, prompt=prompt, image_paths=image_paths, audio_path=str(audio_path), max_tokens=1200)
                result = call_gemma_llamacpp(base_url=base_url, payload=payload, timeout=timeout)
                normalized = normalize_gemma_annotation(result if result.get("parsed") is None else (result.get("parsed") if isinstance(result.get("parsed"), dict) else {}), window)
                normalized["parse_ok"] = bool(result.get("parse_ok"))
                normalized["error"] = result.get("error")
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
