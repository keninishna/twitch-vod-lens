#!/usr/bin/env bash
set -euo pipefail

# Validate persistent-intelligence end-to-end behavior on WSL artifacts.
#
# Usage:
#   bash scripts/validate_persistent_intelligence_wsl.sh <VOD_ID> [STREAMER_ID]
#
# Environment overrides:
#   PROFILE_ROOT=data/streamer_intelligence
#   MODE=propose|auto|off         (default: propose)
#   SKIP_AUDIO=1|0                (default: 1)
#   REPO_ROOT=/home/john/twitch-vod-analyzer

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <VOD_ID> [STREAMER_ID]" >&2
  exit 2
fi

VOD_ID="$1"
STREAMER_ID="${2:-}"
PROFILE_ROOT="${PROFILE_ROOT:-data/streamer_intelligence}"
MODE="${MODE:-propose}"
SKIP_AUDIO="${SKIP_AUDIO:-1}"
REPO_ROOT="${REPO_ROOT:-$PWD}"

if [[ "$MODE" != "propose" && "$MODE" != "auto" && "$MODE" != "off" ]]; then
  echo "ERROR: MODE must be one of propose|auto|off (got '$MODE')" >&2
  exit 2
fi

cd "$REPO_ROOT"
RUN_START_EPOCH="$(date +%s)"
PHASE4_DIR="vods/phase4_${VOD_ID}"
OUTPUT_JSON="${PHASE4_DIR}/qwen_vision_progressive.json"

echo "==> [1/3] Phase4 validation"
VALIDATE_CMD=(
  python3 src/preprocessing/validate_phase4_inputs.py
  --vod-id "$VOD_ID"
  --enable-persistent-intelligence
  --profile-root "$PROFILE_ROOT"
)
if [[ -n "$STREAMER_ID" ]]; then
  VALIDATE_CMD+=(--streamer-id "$STREAMER_ID")
fi
PYTHONPATH=. "${VALIDATE_CMD[@]}"

echo "==> [2/3] Synthesis run (persistent intelligence enabled)"
SYNTH_CMD=(
  python3 src/synthesis/qwen_clip_analyzer_progressive.py
  --vod-id "$VOD_ID"
  --enable-persistent-intelligence
  --profile-root "$PROFILE_ROOT"
  --profile-update-mode "$MODE"
)
if [[ "$MODE" != "off" ]]; then
  SYNTH_CMD+=(--update-streamer-profile)
fi
if [[ "$SKIP_AUDIO" == "1" ]]; then
  SYNTH_CMD+=(--skip-audio)
fi
if [[ -n "$STREAMER_ID" ]]; then
  SYNTH_CMD+=(--streamer-id "$STREAMER_ID")
fi
PYTHONPATH=. "${SYNTH_CMD[@]}"

echo "==> [3/3] Artifact checks"
python3 - "$VOD_ID" "$PROFILE_ROOT" "$MODE" "$RUN_START_EPOCH" "$OUTPUT_JSON" <<'PY'
import json
import os
import sys
from pathlib import Path

vod_id = sys.argv[1]
profile_root = Path(sys.argv[2])
mode = sys.argv[3]
run_start = int(sys.argv[4])
output_path = Path(sys.argv[5])


def fail(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}")
    raise SystemExit(code)

if not output_path.exists():
    fail(f"missing output JSON: {output_path}")

if int(output_path.stat().st_mtime) < run_start:
    fail(f"stale output JSON (mtime older than run start): {output_path}")

try:
    data = json.loads(output_path.read_text(encoding="utf-8"))
except Exception as exc:
    fail(f"failed to parse output JSON: {exc}")

identity = data.get("streamer_identity")
if not isinstance(identity, dict):
    fail("missing 'streamer_identity' block in output JSON")

resolved_id = identity.get("streamer_id")
source = identity.get("source")
if not isinstance(resolved_id, str) or not resolved_id.strip():
    fail("streamer_identity.streamer_id missing/empty")
if source not in {"metadata", "override", "fallback"}:
    fail(f"streamer_identity.source invalid: {source!r}")

print(
    "INFO: streamer_identity "
    f"streamer_id={resolved_id} source={source} "
    f"metadata={identity.get('metadata_streamer_id')} "
    f"override={identity.get('override_streamer_id')}"
)
if identity.get("warning"):
    print(f"WARN: {identity['warning']}")

if mode != "off":
    profile_update = data.get("profile_update")
    if not isinstance(profile_update, dict):
        fail("missing 'profile_update' block while update mode is enabled")

    proposal_path_raw = profile_update.get("proposal_path")
    if not isinstance(proposal_path_raw, str) or not proposal_path_raw.strip():
        fail("profile_update.proposal_path missing")

    proposal_path = Path(proposal_path_raw)
    if not proposal_path.exists():
        fail(f"missing profile update proposal file: {proposal_path}")

    if int(proposal_path.stat().st_mtime) < run_start:
        fail(f"stale profile update proposal (mtime older than run start): {proposal_path}")

    print(
        "INFO: profile_update "
        f"mode={profile_update.get('mode')} "
        f"candidate_observations={profile_update.get('candidate_observations')} "
        f"accepted={profile_update.get('accepted', 0)} "
        f"queued={profile_update.get('queued', profile_update.get('queued_if_manual', 0))} "
        f"rejected={profile_update.get('rejected', profile_update.get('rejected_if_auto', 0))}"
    )

    # Auto mode: if accepted > 0, ensure persistent artifacts exist and are non-empty.
    if mode == "auto":
        accepted = int(profile_update.get("accepted", 0) or 0)
        streamer_dir = profile_root / resolved_id
        observations = streamer_dir / "observations.jsonl"
        profile_json = streamer_dir / "profile.json"

        if accepted > 0:
            if not observations.exists() or observations.stat().st_size == 0:
                fail(
                    "accepted > 0 but observations.jsonl missing/empty: "
                    f"{observations}"
                )
            if not profile_json.exists() or profile_json.stat().st_size == 0:
                fail(
                    "accepted > 0 but profile.json missing/empty: "
                    f"{profile_json}"
                )
            print(
                "INFO: auto-mode persistence verified "
                f"(accepted={accepted}, streamer_dir={streamer_dir})"
            )
        else:
            print("INFO: auto-mode accepted=0 (valid outcome under confidence policy)")

print("OK: persistent-intelligence validation passed")
PY
