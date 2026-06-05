#!/usr/bin/env python3
"""Smoke test Gemma 4 12B llama.cpp OpenAI-compatible backend.

This script exercises four request shapes and records parseability and fallback
behavior. It is discovery-only and does not change pipeline state.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import requests


def _load_bytes(path: str | None) -> bytes | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return p.read_bytes()


def _data_url(path: str | None, mime: str) -> str | None:
    blob = _load_bytes(path)
    if blob is None:
        return None
    return f"data:{mime};base64,{base64.b64encode(blob).decode()}"


def _json_ok(text: str | None) -> tuple[bool, Any | None]:
    if not text:
        return False, None
    try:
        return True, json.loads(text)
    except Exception:
        return False, None


def build_payload(model: str, prompt: str, *, image_path: str | None = None, audio_path: str | None = None, use_response_format: bool = True) -> dict:
    messages = [{"role": "user", "content": []}]
    content = [{"type": "text", "text": prompt}]
    if image_path:
        url = _data_url(image_path, "image/jpeg") or image_path
        content.append({"type": "image_url", "image_url": {"url": url}})
    if audio_path:
        audio_bytes = _load_bytes(audio_path)
        if audio_bytes is not None:
            fmt = Path(audio_path).suffix.lower().lstrip('.') or 'wav'
            content.append({"type": "input_audio", "input_audio": {"data": base64.b64encode(audio_bytes).decode(), "format": fmt}})
    messages[0]["content"] = content
    payload = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 256}
    if use_response_format:
        payload["response_format"] = {"type": "json_object"}
    return payload


def call_chat(base_url: str, payload: dict, timeout: int) -> dict:
    url = f"{base_url.rstrip('/')}/chat/completions"
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"].get("content")
    ok, parsed = _json_ok(content)
    return {"http_status": resp.status_code, "content": content, "parse_ok": ok, "parsed": parsed}


def run_one(name: str, *, base_url: str, model: str, prompt: str, image_path: str | None, audio_path: str | None, timeout: int) -> dict:
    result = {"test": name, "requested_modalities": [m for m in ("text", "image" if image_path else None, "audio" if audio_path else None) if m], "response_format_used": True}
    try:
        payload = build_payload(model, prompt, image_path=image_path, audio_path=audio_path, use_response_format=True)
        try:
            result["request"] = payload
            result["response"] = call_chat(base_url, payload, timeout)
        except Exception as e:
            if "response_format" in str(e).lower():
                result["response_format_used"] = False
                payload = build_payload(model, prompt, image_path=image_path, audio_path=audio_path, use_response_format=False)
                result["request_fallback"] = payload
                result["response"] = call_chat(base_url, payload, timeout)
            else:
                raise
    except Exception as e:
        result["error"] = str(e)
        result["response"] = {"http_status": None, "content": None, "parse_ok": False, "parsed": None}
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--image")
    ap.add_argument("--audio")
    ap.add_argument("--output", required=True)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    tests = [
        ("text_only", "Return JSON with keys ok and mode.", None, None),
        ("image_text", "Describe the image in JSON with keys ok and mode.", args.image, None),
        ("audio_text", "Describe the audio in JSON with keys ok and mode.", None, args.audio),
        ("audio_image_text", "Describe the audio and image in JSON with keys ok and mode.", args.image, args.audio),
    ]
    results = [run_one(name, base_url=args.base_url, model=args.model, prompt=prompt, image_path=image_path, audio_path=audio_path, timeout=args.timeout) for name, prompt, image_path, audio_path in tests]
    out = {"base_url": args.base_url, "model": args.model, "results": results}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps({"output": str(out_path), "tests": len(results), "parseable": sum(1 for r in results if r["response"].get("parse_ok"))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
