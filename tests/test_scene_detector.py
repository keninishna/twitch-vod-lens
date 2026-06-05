import sys
import types
from pathlib import Path

import pytest

from src.preprocessing.scene_detector import detect_scenes


class _FakeTimecode:
    def __init__(self, seconds):
        self._seconds = float(seconds)

    def get_seconds(self):
        return self._seconds


class _FakeVideo:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


class _FakeSceneManager:
    instances = []

    def __init__(self):
        self.auto_downscale = True
        self.downscale = 1
        self.detectors = []
        self.video = None
        self.show_progress = None
        _FakeSceneManager.instances.append(self)

    def add_detector(self, detector):
        self.detectors.append(detector)

    def detect_scenes(self, video=None, show_progress=False, **kwargs):
        self.video = video
        self.show_progress = show_progress
        return 2

    def get_scene_list(self, start_in_scene=False):
        assert start_in_scene is True
        return [
            (_FakeTimecode(0.0), _FakeTimecode(12.5)),
            (_FakeTimecode(12.5), _FakeTimecode(30.0)),
        ]


class _FakeContentDetector:
    def __init__(self, threshold=27.0, **kwargs):
        self.kind = "content"
        self.threshold = threshold
        self.kwargs = kwargs


class _FakeAdaptiveDetector:
    def __init__(self, adaptive_threshold=3.0, **kwargs):
        self.kind = "adaptive"
        self.threshold = adaptive_threshold
        self.kwargs = kwargs


def _install_fake_scenedetect(monkeypatch):
    fake_video = _FakeVideo()
    _FakeSceneManager.instances.clear()

    fake_mod = types.ModuleType("scenedetect")
    fake_mod.SceneManager = _FakeSceneManager
    fake_mod.ContentDetector = _FakeContentDetector
    fake_mod.AdaptiveDetector = _FakeAdaptiveDetector
    fake_mod.open_video = lambda path: fake_video

    monkeypatch.setitem(sys.modules, "scenedetect", fake_mod)
    return fake_video


def test_detect_scenes_uses_python_api_and_converts_scene_list(tmp_path, monkeypatch):
    fake_video = _install_fake_scenedetect(monkeypatch)
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake-video")

    scenes = detect_scenes(video_path, threshold=12.0, downscale="480p", method="content")

    assert [(scene.start, scene.end, scene.duration) for scene in scenes] == [
        (0.0, 12.5, 12.5),
        (12.5, 30.0, 17.5),
    ]

    manager = _FakeSceneManager.instances[-1]
    assert manager.video is fake_video
    assert manager.show_progress is False
    assert manager.auto_downscale is True
    assert manager.detectors[0].kind == "content"
    assert manager.detectors[0].threshold == 12.0
    assert fake_video.released is True


def test_detect_scenes_supports_adaptive_method_and_disable_downscale(tmp_path, monkeypatch):
    _install_fake_scenedetect(monkeypatch)
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake-video")

    detect_scenes(video_path, threshold=9.5, downscale=None, method="adaptive")

    manager = _FakeSceneManager.instances[-1]
    assert manager.auto_downscale is False
    assert manager.downscale == 1
    assert manager.detectors[0].kind == "adaptive"
    assert manager.detectors[0].threshold == 9.5


def test_detect_scenes_raises_helpful_import_error_when_scenedetect_missing(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "scenedetect", raising=False)
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake-video")

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "scenedetect":
            raise ImportError("missing scenedetect")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ImportError, match="PySceneDetect is required"):
        detect_scenes(video_path)
