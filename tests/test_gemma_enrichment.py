from pathlib import Path

import json

from src.synthesis.gemma_enrichment import (
    build_gemma_chat_payload,
    build_gemma_enrichment_prompt,
    call_gemma_llamacpp,
)


def test_prompt_is_json_only_and_mentions_evidence_rules():
    prompt = build_gemma_enrichment_prompt({"start": 10, "end": 40})
    assert "JSON only" in prompt
    assert "factual annotations only" in prompt
    assert "Do not make final clip decisions" in prompt


def test_payload_includes_image_url_and_input_audio(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image-bytes")
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audio-bytes")
    payload = build_gemma_chat_payload(model="gemma", prompt="hi", image_paths=[str(image)], audio_path=str(audio))
    content = payload["messages"][0]["content"]
    types = [item["type"] for item in content]
    assert "text" in types
    assert "image_url" in types
    assert "input_audio" in types


def test_call_parser_handles_json_content(monkeypatch):
    class Resp:
        status_code = 200
        def raise_for_status(self):
            return None
        def json(self):
            return {"choices": [{"message": {"content": json.dumps({"ok": True})}}]}
    monkeypatch.setattr("src.synthesis.gemma_enrichment.requests.post", lambda *a, **k: Resp())
    out = call_gemma_llamacpp(base_url="http://example/v1", payload={"model": "m", "messages": []}, timeout=1)
    assert out["parse_ok"] is True
    assert out["parsed"]["ok"] is True
