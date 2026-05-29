from __future__ import annotations

from urllib.error import URLError

from src.synthesis import bee_server


def test_wait_for_bee_api_success(monkeypatch):
    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout):
        assert req.full_url.endswith("/v1/models")
        assert timeout == 10
        return _Resp()

    monkeypatch.setattr(bee_server, "urlopen", fake_urlopen)

    assert bee_server.wait_for_bee_api("http://bee.local", timeout=1, check_interval=0, logger=lambda *_: None) is True


def test_wait_for_bee_api_timeout_failure(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        raise URLError("connection refused")

    # Make timeout deterministic and fast: one failed poll, then timeout.
    now_values = iter([0.0, 0.0, 1.0])
    monkeypatch.setattr(bee_server, "time", type("_T", (), {
        "time": staticmethod(lambda: next(now_values)),
        "sleep": staticmethod(lambda _seconds: None),
    })())
    monkeypatch.setattr(bee_server, "urlopen", fake_urlopen)

    ok = bee_server.wait_for_bee_api("http://bee.local", timeout=1, check_interval=0, logger=lambda *_: None)

    assert ok is False
    assert calls["n"] >= 1


def test_ensure_bee_api_ready_already_ready(monkeypatch):
    monkeypatch.setattr(bee_server, "wait_for_bee_api", lambda **kwargs: True)

    result = bee_server.ensure_bee_api_ready("http://bee.local", logger=lambda *_: None)

    assert result.ready is True
    assert result.started is False


def test_ensure_bee_api_ready_not_ready_no_start(monkeypatch):
    monkeypatch.setattr(bee_server, "wait_for_bee_api", lambda **kwargs: False)

    result = bee_server.ensure_bee_api_ready("http://bee.local", start_bee=False, logger=lambda *_: None)

    assert result.ready is False
    assert result.started is False
    assert "http://bee.local/v1/models" in result.message


def test_ensure_bee_api_ready_start_requested_no_command(monkeypatch):
    monkeypatch.setattr(bee_server, "wait_for_bee_api", lambda **kwargs: False)

    result = bee_server.ensure_bee_api_ready(
        "http://bee.local",
        start_bee=True,
        start_command=None,
        logger=lambda *_: None,
    )

    assert result.ready is False
    assert result.started is False
    assert "Provide a Bee start command" in result.message


def test_ensure_bee_api_ready_start_and_recheck_success(monkeypatch):
    outcomes = iter([False, True])
    calls = []

    def fake_wait_for_bee_api(**kwargs):
        calls.append(kwargs)
        return next(outcomes)

    started = {"command": None}

    def fake_start_bee_server(start_command, logger):
        started["command"] = start_command
        return object()

    monkeypatch.setattr(bee_server, "wait_for_bee_api", fake_wait_for_bee_api)
    monkeypatch.setattr(bee_server, "start_bee_server", fake_start_bee_server)

    result = bee_server.ensure_bee_api_ready(
        "http://bee.local",
        start_bee=True,
        start_command="bee serve --port 7777",
        logger=lambda *_: None,
    )

    assert len(calls) == 2
    assert started["command"] == "bee serve --port 7777"
    assert result.ready is True
    assert result.started is True
