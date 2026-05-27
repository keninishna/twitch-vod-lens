from __future__ import annotations

from pathlib import Path

from src.intelligence.streamer_store import (
    append_observations,
    load_persistent_voice_profiles,
    load_recent_observations,
    load_streamer_profile,
    resolve_streamer_id_context,
    resolve_streamer_id,
    save_streamer_profile,
)
from src.intelligence.types import StreamerObservation, StreamerProfile


def test_resolve_streamer_id_prefers_override_and_sanitizes() -> None:
    vod_meta = {"streamer": "Skitch TV"}

    assert resolve_streamer_id(vod_meta, override="  Skitch.Main ") == "skitch_main"
    assert resolve_streamer_id(vod_meta) == "skitch_tv"


def test_resolve_streamer_id_context_defaults_to_metadata() -> None:
    ctx = resolve_streamer_id_context({"channel_login": "LostGirls27"}, override=None)

    assert ctx["streamer_id"] == "lostgirls27"
    assert ctx["metadata_streamer_id"] == "lostgirls27"
    assert ctx["override_streamer_id"] is None
    assert ctx["source"] == "metadata"
    assert ctx["override_mismatch"] is False
    assert ctx["warning"] is None


def test_resolve_streamer_id_context_reports_override_mismatch() -> None:
    ctx = resolve_streamer_id_context({"streamer": "LostGirls27"}, override="asyajade")

    assert ctx["streamer_id"] == "asyajade"
    assert ctx["metadata_streamer_id"] == "lostgirls27"
    assert ctx["override_streamer_id"] == "asyajade"
    assert ctx["source"] == "override"
    assert ctx["override_mismatch"] is True
    assert "override='asyajade'" in (ctx["warning"] or "")


def test_resolve_streamer_id_context_matching_override_does_not_warn() -> None:
    ctx = resolve_streamer_id_context({"streamer": "LostGirls27"}, override="LostGirls27")

    assert ctx["streamer_id"] == "lostgirls27"
    assert ctx["source"] == "override"
    assert ctx["override_mismatch"] is False
    assert ctx["warning"] is None


def test_resolve_streamer_id_context_missing_metadata_without_override_is_unknown() -> None:
    ctx = resolve_streamer_id_context({}, override=None)

    assert ctx["streamer_id"] == "unknown_streamer"
    assert ctx["source"] == "fallback"
    assert ctx["override_mismatch"] is False
    assert ctx["warning"] is None


def test_load_streamer_profile_creates_default_profile_when_missing(tmp_path: Path) -> None:
    profile = load_streamer_profile("Skitch", tmp_path)

    assert profile.streamer_id == "skitch"
    assert profile.profile_version == 1
    assert (tmp_path / "skitch" / "profile.json").exists()


def test_save_streamer_profile_persists_to_profile_json(tmp_path: Path) -> None:
    profile = StreamerProfile(streamer_id="skitch", display_name="Skitch")

    out = save_streamer_profile(profile, tmp_path)
    reloaded = load_streamer_profile("skitch", tmp_path)

    assert out == tmp_path / "skitch" / "profile.json"
    assert reloaded.display_name == "Skitch"


def test_append_and_load_recent_observations_with_limit(tmp_path: Path) -> None:
    observations = [
        StreamerObservation(
            vod_id="2776101332",
            timestamp_start=10.0,
            timestamp_end=20.0,
            type="inside_joke",
            claim="Recurring donation alert bit",
            evidence=["chat:101"],
            source="chat",
            confidence=0.8,
            evidence_refs=["vod:2776101332@10-20"],
        ),
        StreamerObservation(
            vod_id="2776101332",
            timestamp_start=21.0,
            timestamp_end=30.0,
            type="content_pattern",
            claim="Short setup+payoff clips perform best",
            evidence=["clip:21-30"],
            source="llm_summary",
            confidence=0.85,
            evidence_refs=["vod:2776101332@21-30"],
        ),
    ]

    obs_path = append_observations("skitch", observations, tmp_path)
    recent = load_recent_observations("skitch", tmp_path, limit=1)

    assert obs_path == tmp_path / "skitch" / "observations.jsonl"
    assert len(recent) == 1
    assert recent[0].claim == "Short setup+payoff clips perform best"


def test_load_recent_observations_ignores_invalid_lines(tmp_path: Path) -> None:
    streamer_dir = tmp_path / "skitch"
    streamer_dir.mkdir(parents=True)
    (streamer_dir / "observations.jsonl").write_text(
        "not-json\n"
        '{"vod_id":"2776101332","timestamp_start":1,"timestamp_end":2,"type":"other","claim":"ok","evidence":["x"],"source":"manual","confidence":0.7,"evidence_refs":["vod:1-2"]}\n',
        encoding="utf-8",
    )

    loaded = load_recent_observations("skitch", tmp_path, limit=10)

    assert len(loaded) == 1
    assert loaded[0].claim == "ok"


def test_load_persistent_voice_profiles_reads_paths_from_profile(tmp_path: Path) -> None:
    voice_dir = tmp_path / "voice_profiles"
    voice_dir.mkdir(parents=True)
    p1 = voice_dir / "streamer_skitch.json"
    p2 = voice_dir / "guest_alpha.json"
    p1.write_text('{"profile_id":"streamer_skitch","embedding":[1.0,0.0]}', encoding="utf-8")
    p2.write_text('{"profile_id":"guest_alpha","embedding":[0.0,1.0]}', encoding="utf-8")

    profile = StreamerProfile.model_validate(
        {
            "streamer_id": "skitch",
            "voice_profiles": [
                {
                    "profile_id": "streamer_skitch",
                    "path": str(p1),
                    "role": "streamer",
                    "confidence": 0.95,
                    "evidence_refs": ["vod:1@1-2"],
                },
                {
                    "profile_id": "guest_alpha",
                    "path": str(p2),
                    "role": "guest",
                    "confidence": 0.90,
                    "evidence_refs": ["vod:1@3-4"],
                },
            ],
        }
    )
    save_streamer_profile(profile, tmp_path)

    loaded_profiles = load_persistent_voice_profiles("skitch", tmp_path)
    assert {p.get("profile_id") for p in loaded_profiles} == {"streamer_skitch", "guest_alpha"}
