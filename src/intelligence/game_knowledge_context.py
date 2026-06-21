from __future__ import annotations

from src.intelligence.game_knowledge_types import GameCategorySegment, GameKnowledgeProfile


def render_game_knowledge_context(profile: GameKnowledgeProfile | None, segment: GameCategorySegment | None, max_chars: int = 1800) -> str:
    lines = ["GAME CONTEXT (evidence-backed, advisory)"]
    if segment:
        lines.append(f"Active Twitch category/game: {segment.game_name}")
        if segment.game_id:
            lines.append(f"Game ID: {segment.game_id}")
    if not profile:
        lines.append("Profile unavailable.")
    else:
        lines.append(f"Profile: {profile.game_name}")
        if profile.summary:
            lines.append(profile.summary)
    lines.append("Caution: actual transcript/chat/visual evidence wins.")
    text = "\n".join(lines)
    if len(text) > max_chars:
        if max_chars <= 3:
            return text[:max_chars]
        return text[: max_chars - 3] + "..."
    return text
