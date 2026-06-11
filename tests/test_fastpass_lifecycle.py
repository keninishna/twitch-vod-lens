import importlib
from types import SimpleNamespace

import pytest

import src.synthesis.qwen_clip_analyzer_progressive as progressive


@pytest.fixture()
def fastpass_module(monkeypatch, tmp_path):
    module = importlib.reload(progressive)

    monkeypatch.setattr(module, "VOD_ID", "12345")
    monkeypatch.setattr(module, "VOD_DIR", tmp_path)
    monkeypatch.setattr(module, "FUSION_PATH", tmp_path / "fusion.json")
    monkeypatch.setattr(module, "CLIP_MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(module, "OUTPUT_PATH", tmp_path / "output.json")
    monkeypatch.setattr(module, "FRAMES_DIR", tmp_path / "frames")
    monkeypatch.setattr(module, "FAST_PASS", True)
    monkeypatch.setattr(module, "FAST_PASS_DRY_RUN", False)
    monkeypatch.setattr(module, "GEMMA_SMOKE_TEST_ONLY", False)
    monkeypatch.setattr(module, "FAST_PASS_MODE", "gemma-enriched")
    (tmp_path / "frames").mkdir()
    return module


def _stub_fastpass_inputs(monkeypatch, module):
    monkeypatch.setattr(module, "load_json", lambda path: {"clips": [{"start": 10, "end": 20}]})
    monkeypatch.setattr(module, "resolve_streamer_id_context", lambda vod_meta, override: {
        "streamer_id": "streamer",
        "source": "metadata",
        "metadata_streamer_id": "streamer",
        "override_streamer_id": override,
    })
    monkeypatch.setattr(module, "select_gemma_frames_for_window", lambda window, frames_dir, frames_per_window=2: [])
    monkeypatch.setattr(module, "qwen_call", lambda payload, timeout=180: {"candidates": []})
    monkeypatch.setattr(module, "run_gemma_enrichment", lambda **kwargs: {"artifact": {"artifact_path": "gemma.json", "backend": "gemma"}, "artifact_path": "gemma.json", "summary": {}})
    monkeypatch.setattr(module, "build_triage_chunks", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "compute_vision_budget", lambda *args, **kwargs: 0)
    monkeypatch.setattr(module, "select_vision_shortlist", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "_run_fast_pass_text_triage", lambda **kwargs: ([], {"qwen_text_calls": 0}))
    monkeypatch.setattr(module, "run_audio_phase", lambda clips, all_results, fusion, manifest, speaker_attribution=None: all_results)
    monkeypatch.setattr(module, "sys", SimpleNamespace(exit=lambda code=0: (_ for _ in ()).throw(SystemExit(code))))


def test_fastpass_gemma_enriched_shutdowns_before_bee_start(monkeypatch, fastpass_module):
    module = fastpass_module
    _stub_fastpass_inputs(monkeypatch, module)

    calls = []

    monkeypatch.setattr(module, "ensure_gemma_api_ready", lambda **kwargs: calls.append("ensure_gemma_api_ready") or True)
    monkeypatch.setattr(module, "shutdown_gemma", lambda **kwargs: calls.append("shutdown_gemma") or None)
    monkeypatch.setattr(module, "ensure_bee_api_ready", lambda **kwargs: calls.append("ensure_bee_api_ready") or SimpleNamespace(ready=True, started=False, message="ok"))

    with pytest.raises(SystemExit):
        module.run()

    assert calls.index("shutdown_gemma") < calls.index("ensure_bee_api_ready")


def test_fastpass_text_only_never_starts_gemma(monkeypatch, fastpass_module):
    module = fastpass_module
    module.FAST_PASS_MODE = "text-only"
    _stub_fastpass_inputs(monkeypatch, module)

    calls = []

    monkeypatch.setattr(module, "ensure_gemma_api_ready", lambda **kwargs: (_ for _ in ()).throw(AssertionError("Gemma should not start in text-only mode")))
    monkeypatch.setattr(module, "shutdown_gemma", lambda **kwargs: calls.append("shutdown_gemma") or None)
    monkeypatch.setattr(module, "ensure_bee_api_ready", lambda **kwargs: calls.append("ensure_bee_api_ready") or SimpleNamespace(ready=True, started=False, message="ok"))

    module.run()

    assert "shutdown_gemma" not in calls


def test_fastpass_gemma_smoke_test_shuts_down_before_exit(monkeypatch, fastpass_module):
    module = fastpass_module
    module.GEMMA_SMOKE_TEST_ONLY = True
    _stub_fastpass_inputs(monkeypatch, module)

    calls = []

    monkeypatch.setattr(module, "ensure_gemma_api_ready", lambda **kwargs: calls.append("ensure_gemma_api_ready") or True)
    monkeypatch.setattr(module, "shutdown_gemma", lambda **kwargs: calls.append("shutdown_gemma") or None)
    monkeypatch.setattr(module, "ensure_bee_api_ready", lambda **kwargs: calls.append("ensure_bee_api_ready") or SimpleNamespace(ready=True, started=False, message="ok"))

    with pytest.raises(SystemExit) as exc:
        module.run()

    assert exc.value.code == 0
    assert "shutdown_gemma" in calls
    assert "ensure_bee_api_ready" not in calls
