"""Prompt-context renderer for persistent streamer intelligence (Phase 04 Task 16)."""

from __future__ import annotations

from src.intelligence.types import StreamerProfile


def _clip_text(value: str, max_len: int = 160) -> str:
    txt = " ".join(value.strip().split())
    if len(txt) <= max_len:
        return txt
    return txt[: max_len - 1].rstrip() + "…"


def render_streamer_profile_context(profile: StreamerProfile, max_chars: int = 2000) -> str:
    """Render compact evidence-backed profile context for LLM prompts.

    Rules:
    - Include only facts with confidence >= 0.65 and non-empty evidence refs.
    - Keep concise and advisory.
    - Explicitly state conflict rule: current VOD evidence overrides profile context.
    """

    lines: list[str] = [
        "STREAMER PROFILE CONTEXT (evidence-backed, advisory):",
        f"- streamer_id: {profile.streamer_id}",
        "- conflict rule: current VOD evidence overrides profile context when they differ.",
    ]

    voices = [
        v for v in profile.voice_profiles
        if v.confidence >= 0.65 and len(v.evidence_refs) > 0
    ]
    if voices:
        lines.append("- voice profiles:")
        for v in voices[:6]:
            display = v.display_name or v.profile_id
            lines.append(
                f"  - {display} ({v.role}), conf={v.confidence:.2f}, ref={v.evidence_refs[0]}"
            )

    traits = [
        t for t in profile.personality_traits
        if t.confidence >= 0.65 and len(t.evidence_refs) > 0
    ]
    if traits:
        lines.append("- personality/style traits:")
        for t in traits[:8]:
            core = t.description or t.trait
            lines.append(f"  - {_clip_text(core)}, conf={t.confidence:.2f}, ref={t.evidence_refs[0]}")

    jokes = [
        j for j in profile.inside_jokes
        if j.confidence >= 0.65 and len(j.evidence_refs) > 0
    ]
    if jokes:
        lines.append("- inside jokes/community bits:")
        for j in jokes[:8]:
            lines.append(
                f"  - {j.key}: {_clip_text(j.description)}, conf={j.confidence:.2f}, ref={j.evidence_refs[0]}"
            )

    patterns = [
        p for p in profile.content_patterns
        if p.confidence >= 0.65 and len(p.evidence_refs) > 0
    ]
    if patterns:
        lines.append("- clip quality lessons/patterns:")
        for p in patterns[:8]:
            lines.append(
                f"  - [{p.impact}] {_clip_text(p.pattern)} — {_clip_text(p.description)}, conf={p.confidence:.2f}, ref={p.evidence_refs[0]}"
            )

    chatters = [
        c for c in profile.community_chatters
        if c.confidence >= 0.65 and len(c.evidence_refs) > 0
    ]
    if chatters:
        # Prioritize recent/high-activity chatters.
        ordered = sorted(
            chatters,
            key=lambda c: (c.message_count, c.last_seen_vod_id or ""),
            reverse=True,
        )
        lines.append("- recurring community chatters:")
        for c in ordered[:8]:
            label = c.username
            if c.role:
                label = f"{label} ({c.role})"
            lines.append(
                f"  - {label}, msgs={c.message_count}, conf={c.confidence:.2f}, ref={c.evidence_refs[0]}"
            )

    rendered = "\n".join(lines).strip()
    if len(rendered) <= max_chars:
        return rendered

    # Soft truncate while preserving header/conflict lines.
    hard = rendered[: max_chars - 1].rstrip()
    return hard + "…"
