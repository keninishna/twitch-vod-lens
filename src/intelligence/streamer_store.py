"""Persistent streamer intelligence storage helpers (SpeakerID Phase 04 Task 15)."""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.intelligence.types import StreamerObservation, StreamerProfile


def _sanitize_streamer_id(value: str) -> str:
    cleaned = value.strip().lower()
    cleaned = re.sub(r"[^a-z0-9_-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned or "unknown_streamer"


def _resolve_streamer_id_from_metadata(vod_meta: dict[str, Any] | None) -> str:
    meta = vod_meta if isinstance(vod_meta, dict) else {}
    keys = (
        "streamer_id",
        "streamer",
        "channel",
        "channel_name",
        "channel_login",
        "user_login",
        "broadcaster_login",
        "display_name",
        "uploader_id",
        "uploader",
        "owner",
    )

    for key in keys:
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return _sanitize_streamer_id(value)

    return "unknown_streamer"


def resolve_streamer_id(vod_meta: dict, override: str | None = None) -> str:
    """Resolve stable streamer_id.

    Contract:
    - metadata-derived streamer_id is the default
    - override wins only when explicitly provided
    """

    if override and override.strip():
        return _sanitize_streamer_id(override)

    return _resolve_streamer_id_from_metadata(vod_meta)


def resolve_streamer_id_context(
    vod_meta: dict[str, Any] | None,
    override: str | None = None,
) -> dict[str, Any]:
    """Resolve streamer_id with provenance and mismatch diagnostics.

    Returns:
      {
        streamer_id,
        metadata_streamer_id,
        override_streamer_id,
        source,                 # metadata | override | fallback
        override_mismatch,      # True only when metadata is known and differs
        warning,                # warning text or None
      }
    """

    metadata_streamer_id = _resolve_streamer_id_from_metadata(vod_meta)
    override_streamer_id = (
        _sanitize_streamer_id(override)
        if isinstance(override, str) and override.strip()
        else None
    )

    if override_streamer_id:
        streamer_id = override_streamer_id
        source = "override"
    elif metadata_streamer_id != "unknown_streamer":
        streamer_id = metadata_streamer_id
        source = "metadata"
    else:
        streamer_id = "unknown_streamer"
        source = "fallback"

    override_mismatch = bool(
        override_streamer_id
        and metadata_streamer_id != "unknown_streamer"
        and override_streamer_id != metadata_streamer_id
    )

    warning = None
    if override_mismatch:
        warning = (
            "streamer_id override mismatch: "
            f"override='{override_streamer_id}' metadata='{metadata_streamer_id}'"
        )

    return {
        "streamer_id": streamer_id,
        "metadata_streamer_id": metadata_streamer_id,
        "override_streamer_id": override_streamer_id,
        "source": source,
        "override_mismatch": override_mismatch,
        "warning": warning,
    }


def _streamer_dir(root: Path, streamer_id: str) -> Path:
    return root / _sanitize_streamer_id(streamer_id)


def _profile_path(root: Path, streamer_id: str) -> Path:
    return _streamer_dir(root, streamer_id) / "profile.json"


def _observations_path(root: Path, streamer_id: str) -> Path:
    return _streamer_dir(root, streamer_id) / "observations.jsonl"


def _lock_path(root: Path, streamer_id: str) -> Path:
    return _streamer_dir(root, streamer_id) / ".profile.lock"


@contextmanager
def _profile_lock(root: Path, streamer_id: str, timeout_seconds: float = 5.0) -> Iterator[None]:
    """Simple lockfile guard for profile/observation writes."""

    lock_file = _lock_path(root, streamer_id)
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    while True:
        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            os.close(fd)
            break
        except FileExistsError:
            if time.time() - start >= timeout_seconds:
                raise TimeoutError(f"Timed out waiting for lock: {lock_file}")
            time.sleep(0.05)

    try:
        yield
    finally:
        try:
            lock_file.unlink(missing_ok=True)
        except Exception:
            pass


def save_streamer_profile(profile: StreamerProfile, root: Path) -> Path:
    """Atomically save profile.json under data/streamer_intelligence/<streamer_id>/."""

    streamer_id = _sanitize_streamer_id(profile.streamer_id)
    target_dir = _streamer_dir(root, streamer_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = _profile_path(root, streamer_id)

    payload = profile.model_copy(update={"streamer_id": streamer_id})

    with _profile_lock(root, streamer_id):
        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            payload.model_dump_json(indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(out_path)

    return out_path


def load_streamer_profile(streamer_id: str, root: Path) -> StreamerProfile:
    """Load profile or create a default profile.json when missing."""

    sid = _sanitize_streamer_id(streamer_id)
    path = _profile_path(root, sid)

    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        profile = StreamerProfile.model_validate(data)
        if profile.streamer_id != sid:
            profile = profile.model_copy(update={"streamer_id": sid})
            save_streamer_profile(profile, root)
        return profile

    profile = StreamerProfile(streamer_id=sid, profile_version=1)
    save_streamer_profile(profile, root)
    return profile


def append_observations(
    streamer_id: str,
    observations: list[StreamerObservation],
    root: Path,
) -> Path:
    """Append immutable observations to observations.jsonl."""

    sid = _sanitize_streamer_id(streamer_id)
    obs_path = _observations_path(root, sid)
    obs_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure profile exists (contract: missing profiles are auto-created).
    load_streamer_profile(sid, root)

    with _profile_lock(root, sid):
        with obs_path.open("a", encoding="utf-8") as handle:
            for observation in observations:
                validated = StreamerObservation.model_validate(observation)
                handle.write(validated.model_dump_json())
                handle.write("\n")

    return obs_path


def load_recent_observations(
    streamer_id: str,
    root: Path,
    limit: int = 200,
) -> list[StreamerObservation]:
    """Load up to `limit` most recent observations from JSONL."""

    sid = _sanitize_streamer_id(streamer_id)
    obs_path = _observations_path(root, sid)
    if not obs_path.exists():
        return []

    loaded: list[StreamerObservation] = []
    for raw_line in obs_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            loaded.append(StreamerObservation.model_validate_json(line))
        except Exception:
            continue

    if limit <= 0:
        return []

    return loaded[-limit:]


def load_persistent_voice_profiles(streamer_id: str, root: Path) -> list[dict[str, Any]]:
    """Load voice-profile JSON payloads referenced by StreamerProfile.voice_profiles.

    This lets speaker attribution reuse persistent voice thumbprints across VODs
    without requiring manual --profiles-dir wiring for every run.
    """

    from src.preprocessing.speaker_profiles import load_profiles_from_paths

    profile = load_streamer_profile(streamer_id, root)
    paths = [
        ref.path
        for ref in profile.voice_profiles
        if isinstance(ref.path, str) and ref.path.strip()
    ]
    if not paths:
        return []

    return load_profiles_from_paths(paths)
