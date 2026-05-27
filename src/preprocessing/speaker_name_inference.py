"""Heuristic + LLM-assisted speaker name inference from diarized transcript."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from src.preprocessing.types import SpeakerNameCandidate

_SELF_INTRO_PATTERNS = [
    re.compile(r"\b(?:i am|i'm|my name is)\s+([A-Za-z][A-Za-z0-9_]{1,31})\b", re.IGNORECASE),
    re.compile(r"\bthis is\s+([A-Za-z][A-Za-z0-9_]{1,31})\b", re.IGNORECASE),
    re.compile(r"\b([A-Za-z][A-Za-z0-9_]{1,31})\s+here\b", re.IGNORECASE),
]

_NAME_MENTION_PATTERNS = [
    re.compile(r"\b(?:hey|hi|hello|thanks|thank you)\s+([A-Za-z][A-Za-z0-9_]{1,31})\b", re.IGNORECASE),
    re.compile(r"\b([A-Za-z][A-Za-z0-9_]{1,31})\s*,\s*what do you think\b", re.IGNORECASE),
]


def _norm_name(name: str) -> str:
    return name.strip().strip(".,!?\"'")


def extract_name_mentions(text: str) -> list[str]:
    """Extract probable names from direct-address patterns in text."""

    found: list[str] = []
    for pattern in _NAME_MENTION_PATTERNS + _SELF_INTRO_PATTERNS:
        for match in pattern.findall(text or ""):
            if isinstance(match, tuple):
                candidate = match[0]
            else:
                candidate = match
            name = _norm_name(str(candidate))
            if name:
                found.append(name)

    # stable unique
    out: list[str] = []
    seen: set[str] = set()
    for n in found:
        key = n.lower()
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out


def _coerce_candidate(obj: SpeakerNameCandidate | dict[str, Any]) -> SpeakerNameCandidate:
    if isinstance(obj, SpeakerNameCandidate):
        return obj
    return SpeakerNameCandidate.model_validate(obj)


def merge_name_candidates(
    heuristic_candidates: dict[str, list[SpeakerNameCandidate | dict[str, Any]]],
    qwen_candidates: Any,
) -> dict[str, list[SpeakerNameCandidate]]:
    """Merge candidate maps from heuristics and Qwen into deduped per-speaker lists."""

    merged: dict[str, dict[str, SpeakerNameCandidate]] = defaultdict(dict)

    def add(label: str, cand: SpeakerNameCandidate) -> None:
        key = cand.name.lower()
        current = merged[label].get(key)
        if current is None:
            merged[label][key] = cand
            return

        confidence = max(current.confidence, cand.confidence)
        evidence = list(dict.fromkeys([*current.evidence, *cand.evidence]))
        merged[label][key] = SpeakerNameCandidate(name=current.name, confidence=confidence, evidence=evidence)

    for label, candidates in (heuristic_candidates or {}).items():
        for c in candidates:
            add(label, _coerce_candidate(c))

    if isinstance(qwen_candidates, dict):
        # Either dict[label]=[...] or OpenAI-style payload wrapper.
        if "speaker_name_candidates" in qwen_candidates and isinstance(qwen_candidates["speaker_name_candidates"], list):
            for item in qwen_candidates["speaker_name_candidates"]:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("speaker_label", "")).strip()
                name = str(item.get("name", "")).strip()
                if not label or not name:
                    continue
                add(
                    label,
                    SpeakerNameCandidate(
                        name=name,
                        confidence=float(item.get("confidence", 0.0)),
                        evidence=list(item.get("evidence", [])) if isinstance(item.get("evidence"), list) else [],
                    ),
                )
        else:
            for label, candidates in qwen_candidates.items():
                if not isinstance(candidates, list):
                    continue
                for c in candidates:
                    add(str(label), _coerce_candidate(c))

    final: dict[str, list[SpeakerNameCandidate]] = {}
    for label, by_name in merged.items():
        final[label] = sorted(by_name.values(), key=lambda x: x.confidence, reverse=True)
    return final


def infer_names_heuristic(
    diarized_transcript: list[dict[str, Any]],
    chat_messages: list[dict[str, Any]] | None = None,
) -> dict[str, list[SpeakerNameCandidate]]:
    """Infer speaker-name candidates from transcript dialogue patterns."""

    chat_users = {
        str((m.get("user") or m.get("username") or "")).strip().lower()
        for m in (chat_messages or [])
        if isinstance(m, dict)
    }
    chat_users.discard("")

    candidates: dict[str, list[SpeakerNameCandidate]] = defaultdict(list)

    def add(label: str, name: str, confidence: float, evidence: str) -> None:
        candidates[label].append(
            SpeakerNameCandidate(name=name, confidence=max(0.0, min(1.0, confidence)), evidence=[evidence])
        )

    for idx, seg in enumerate(diarized_transcript):
        label = str(seg.get("speaker_label") or "UNKNOWN")
        text = str(seg.get("text") or "")
        start = float(seg.get("start") or 0.0)

        # High-confidence self-introduction patterns.
        for pat in _SELF_INTRO_PATTERNS:
            for m in pat.finditer(text):
                name = _norm_name(m.group(1))
                if not name:
                    continue
                conf = 0.95 if "i am" in m.group(0).lower() or "i'm" in m.group(0).lower() or "my name is" in m.group(0).lower() else 0.85
                add(label, name, conf, f"self-identification at {start:.1f}s: '{m.group(0)}'")

        mentions = []
        for pat in _NAME_MENTION_PATTERNS:
            mentions.extend(_norm_name(m.group(1)) for m in pat.finditer(text))
        mentions = [m for m in mentions if m]
        if not mentions:
            continue

        unique_mentions = list(dict.fromkeys(mentions))
        multi_name_penalty = 0.12 if len(unique_mentions) > 1 else 0.0

        next_seg = diarized_transcript[idx + 1] if idx + 1 < len(diarized_transcript) else None
        has_voice_response = False
        responder_label = None

        if isinstance(next_seg, dict):
            next_label = str(next_seg.get("speaker_label") or "UNKNOWN")
            next_start = float(next_seg.get("start") or start)
            if next_label != label and 0.0 <= (next_start - start) <= 8.0:
                has_voice_response = True
                responder_label = next_label

        for mentioned_name in unique_mentions:
            if has_voice_response and responder_label:
                conf = 0.72 - multi_name_penalty
                evidence = (
                    f"addressed by name '{mentioned_name}' at {start:.1f}s; "
                    f"{responder_label} responds within 8s"
                )
                if mentioned_name.lower() in chat_users:
                    evidence += " (name also appears in chat usernames)"
                add(responder_label, mentioned_name, conf, evidence)
            else:
                # Twitch safeguard: without voice turn response, treat as chat-addressed.
                if mentioned_name.lower() in chat_users:
                    continue
                # Generic no-response mention is too weak; keep out by default.
                continue

    return merge_name_candidates(candidates, {})


def build_qwen_name_resolution_prompt(
    diarized_transcript: list[dict[str, Any]],
    chat_messages: list[dict[str, Any]] | None = None,
) -> str:
    """Build compact prompt asking Qwen to resolve ambiguous speaker names in JSON."""

    transcript_lines = []
    for row in diarized_transcript[:200]:
        transcript_lines.append(
            f"[{float(row.get('start', 0.0)):.1f}s] {row.get('speaker_label','UNKNOWN')}: {row.get('text','')}"
        )

    chat_preview = []
    for msg in (chat_messages or [])[:100]:
        if not isinstance(msg, dict):
            continue
        ts = float(msg.get("timestamp", 0.0))
        user = msg.get("user") or msg.get("username") or "unknown"
        text = msg.get("message", "")
        chat_preview.append(f"[{ts:.1f}s] @{user}: {text}")

    return (
        "You are resolving speaker names from diarized stream transcript.\n"
        "Output ONLY valid JSON matching this schema:\n"
        "{\n"
        "  \"speaker_name_candidates\": [\n"
        "    {\n"
        "      \"speaker_label\": \"SPEAKER_01\",\n"
        "      \"name\": \"Skitch\",\n"
        "      \"confidence\": 0.72,\n"
        "      \"evidence\": [\"...\"],\n"
        "      \"reasoning_short\": \"Addressed by name and responded immediately\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Use transcript evidence first.\n"
        "- If a name looks chat-addressed with no voice response, do not assign.\n"
        "- Prefer conservative confidence.\n\n"
        "DIARIZED TRANSCRIPT:\n"
        + "\n".join(transcript_lines)
        + "\n\nCHAT PREVIEW:\n"
        + "\n".join(chat_preview)
    )
