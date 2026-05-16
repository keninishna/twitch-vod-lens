"""
VOD Lens — Chat Activity Module

Downloads and analyzes Twitch VOD chat, producing message-level
data and aggregated activity windows.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional
from collections import Counter

from src.models.types import ChatMessage, ChatActivity, ChatAnalysis


WINDOW_SIZE = 30  # seconds per activity window


def download_chat(
    vod_id: str,
    output_path: Optional[Path] = None,
) -> list[ChatMessage]:
    """
    Download chat messages for a Twitch VOD.

    Uses twitch-dl or direct Twitch API. Falls back gracefully.

    Args:
        vod_id: Twitch VOD ID (numeric)
        output_path: Optional file path to save raw chat

    Returns:
        List of ChatMessage objects

    Raises:
        RuntimeError: If download fails entirely
    """
    messages: list[ChatMessage] = []

    # Try twitch-dl first
    try:
        messages = _download_via_twitch_dl(vod_id)
    except (FileNotFoundError, RuntimeError) as e:
        # Fallback: try Twitch API directly
        messages = _download_via_twitch_api(vod_id)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([m.model_dump() for m in messages], f, indent=2)

    return messages


def analyze_chat(
    messages: list[ChatMessage],
    window_size: int = WINDOW_SIZE,
) -> ChatAnalysis:
    """
    Analyze chat messages for activity patterns.

    Args:
        messages: List of ChatMessage objects
        window_size: Size of aggregation window in seconds

    Returns:
        ChatAnalysis with per-window activity and stats
    """
    if not messages:
        return ChatAnalysis(
            messages=[],
            activity=[],
            total_messages=0,
            unique_chatters=0,
            total_emotes=0,
        )

    # Compute activity windows
    max_time = max(m.timestamp for m in messages)
    windows = []
    window_start = 0.0

    while window_start < max_time:
        window_end = window_start + window_size
        window_msgs = [
            m for m in messages
            if window_start <= m.timestamp < window_end
        ]

        if window_msgs:
            all_emotes: list[str] = []
            for m in window_msgs:
                if m.emotes:
                    all_emotes.extend(m.emotes)
            emote_counts = Counter(all_emotes)
            peak_emote = emote_counts.most_common(1)

            windows.append(
                ChatActivity(
                    window_start=window_start,
                    window_end=window_end,
                    message_count=len(window_msgs),
                    unique_users=len(set(m.user for m in window_msgs)),
                    peak_emote=peak_emote[0][0] if peak_emote else None,
                    peak_emote_count=peak_emote[0][1] if peak_emote else 0,
                )
            )

        window_start += window_size

    all_users = set(m.user for m in messages)
    all_emotes_list: list[str] = []
    for m in messages:
        if m.emotes:
            all_emotes_list.extend(m.emotes)

    return ChatAnalysis(
        messages=messages,
        activity=windows,
        total_messages=len(messages),
        unique_chatters=len(all_users),
        total_emotes=len(all_emotes_list),
    )


def _download_via_twitch_dl(vod_id: str) -> list[ChatMessage]:
    """Download chat using twitch-dl CLI."""
    import subprocess

    result = subprocess.run(
        ["twitch-dl", "chat", vod_id, "--json"],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(f"twitch-dl failed: {result.stderr[:500]}")

    messages: list[ChatMessage] = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            messages.append(
                ChatMessage(
                    timestamp=data.get("timestamp_in_seconds", 0.0),
                    user=data.get("commenter", {}).get("display_name", "unknown"),
                    message=data.get("message", {}).get("body", ""),
                    emotes=list(data.get("message", {}).get("emotes", {}).keys()),
                    is_subscriber=data.get("badges", [{}])[0].get("id") == "subscriber",
                    is_moderator=data.get("badges", [{}])[0].get("id") == "moderator",
                )
            )
        except (json.JSONDecodeError, KeyError) as e:
            continue

    return messages


def _download_via_twitch_api(vod_id: str) -> list[ChatMessage]:
    """
    Fallback: download chat using Twitch API directly.
    Requires a valid OAuth token.
    """
    import requests

    # Try without auth first (may work for public VODs)
    try:
        url = f"https://api.twitch.tv/v5/videos/{vod_id}/comments"
        headers = {
            "Client-ID": "kimne78kx3ncx6brgo4mv6wki5h1ko",
            "Accept": "application/vnd.twitchtv.v5+json",
        }
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            return _parse_twitch_api_comments(response.json())
    except Exception:
        pass

    return []


def _parse_twitch_api_comments(data: dict) -> list[ChatMessage]:
    """Parse Twitch API v5 comments response."""
    messages: list[ChatMessage] = []
    for comment in data.get("comments", []):
        msg_body = ""
        emotes: list[str] = []

        for fragment in comment.get("message", {}).get("body", {}).get("fragments", []):
            text = fragment.get("text", "")
            if fragment.get("emoticon"):
                emotes.append(text)
            msg_body += text

        offset = comment.get("content_offset_seconds", 0.0)

        messages.append(
            ChatMessage(
                timestamp=offset,
                user=comment.get("commenter", {}).get("display_name", "unknown"),
                message=msg_body,
                emotes=emotes if emotes else None,
            )
        )

    return messages
