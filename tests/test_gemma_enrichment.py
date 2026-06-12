import json
from pathlib import Path

import requests

from src.synthesis.gemma_enrichment import (
    build_gemma_chat_payload,
    build_gemma_enrichment_prompt,
    call_gemma_llamacpp,
    ensure_gemma_api_ready,
)


def test_prompt_uses_labeled_sections_and_factual_annotations_only():
    prompt = build_gemma_enrichment_prompt({"start": 10, "end": 40})
    assert "factual annotations only" in prompt
    assert "Do not make final clip decisions" in prompt
    for label in ["AUDIO_EVENTS", "VISUAL_EVENTS", "SPEAKER", "EMOTION", "RISK_FLAGS"]:
        assert label in prompt


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


def test_call_parser_tolerates_trailing_dot_float_tokens(monkeypatch):
    raw = """
SPEAKER:
Primary speaker identity: unknown. streamer led likelihood 0.0. transactional alert likelihood 0.1.

EMOTION:
Streamer affect: amused.
""".strip()

    class Resp:
        status_code = 200
        def raise_for_status(self):
            return None
        def json(self):
            return {"choices": [{"message": {"content": raw}}]}

    monkeypatch.setattr("src.synthesis.gemma_enrichment.requests.post", lambda *a, **k: Resp())
    out = call_gemma_llamacpp(base_url="http://example/v1", payload={"model": "m", "messages": []}, timeout=1)
    assert out["parse_ok"] is True
    assert out["parsed"]["speaker_nuance"]["streamer_led_likelihood"] == 0.0
    assert out["parsed"]["emotion_nuance"]["transactional_alert_likelihood"] == 0.1


def test_ensure_gemma_api_ready_uses_mtp_draft_flags_when_draft_model_present(monkeypatch, tmp_path):
    log_path = tmp_path / "gemma.log"
    captured = {}

    class FakeProc:
        def kill(self):
            captured["killed"] = True

    calls = {"count": 0}

    def fake_get(url, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.ConnectionError("not ready")

        class Resp:
            status_code = 200

        return Resp()

    def fake_popen(cmd, stdout=None, stderr=None, env=None):
        captured["cmd"] = cmd
        captured["env"] = env
        return FakeProc()

    monkeypatch.setattr("src.synthesis.gemma_enrichment.requests.get", fake_get)
    monkeypatch.setattr("src.synthesis.gemma_enrichment.subprocess.Popen", fake_popen)
    monkeypatch.setattr("src.synthesis.gemma_enrichment.time.sleep", lambda *_: None)
    monkeypatch.setattr("src.synthesis.gemma_enrichment.open", lambda *a, **k: log_path.open("w"), raising=False)

    ready = ensure_gemma_api_ready(
        base_url="http://localhost:8084",
        gemma_bin="/fake/llama-server",
        model_path="/fake/main.gguf",
        mmproj_path="/fake/mmproj.gguf",
        draft_model_path="/fake/draft.gguf",
        parallel_slots=30,
        timeout=5,
        check_interval=0,
        logger=lambda *_: None,
    )

    assert ready is True
    cmd = captured["cmd"]
    assert "--model-draft" in cmd
    assert "/fake/draft.gguf" in cmd
    assert "--spec-type" in cmd
    assert "draft-mtp" in cmd
    assert "--spec-draft-n-max" in cmd
    assert "4" in cmd
    assert "-np" in cmd
    np_index = cmd.index("-np")
    assert cmd[np_index + 1] == "30"
    assert "--kv-unified" in cmd
    assert "--jinja" in cmd


def test_ensure_gemma_api_ready_does_not_duplicate_cuda_lib_in_ld_library_path(monkeypatch, tmp_path):
    log_path = tmp_path / "gemma.log"
    captured = {}
    cuda_lib = "/home/john/.local/lib/python3.12/site-packages/nvidia/cu13/lib"

    class FakeProc:
        def kill(self):
            captured["killed"] = True

    calls = {"count": 0}

    def fake_get(url, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.ConnectionError("not ready")

        class Resp:
            status_code = 200

        return Resp()

    def fake_popen(cmd, stdout=None, stderr=None, env=None):
        captured["env"] = env
        return FakeProc()

    monkeypatch.setattr("src.synthesis.gemma_enrichment.requests.get", fake_get)
    monkeypatch.setattr("src.synthesis.gemma_enrichment.subprocess.Popen", fake_popen)
    monkeypatch.setattr("src.synthesis.gemma_enrichment.time.sleep", lambda *_: None)
    monkeypatch.setattr("src.synthesis.gemma_enrichment.open", lambda *a, **k: log_path.open("w"), raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")

    ready = ensure_gemma_api_ready(
        base_url="http://localhost:8084",
        gemma_bin="/fake/llama-server",
        model_path="/fake/main.gguf",
        mmproj_path="/fake/mmproj.gguf",
        draft_model_path="/fake/draft.gguf",
        timeout=5,
        check_interval=0,
        logger=lambda *_: None,
    )

    assert ready is True
    ld_library_path = captured["env"]["LD_LIBRARY_PATH"]
    assert ld_library_path.count(cuda_lib) == 1
