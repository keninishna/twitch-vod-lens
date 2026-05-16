"""
VOD Lens — Preprocessing Pipeline Orchestrator

Runs the full preprocessing pipeline: download, transcribe, scene
detect, chat analyze, and fuse results into a unified output.
"""

from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from typing import Optional

from src.models.types import FusionResult
from src.preprocessing.downloader import download_vod, cleanup, extract_vod_id
from src.preprocessing.transcriber import transcribe
from src.preprocessing.scene_detector import detect_scenes
from src.preprocessing.chat_analyzer import download_chat, analyze_chat
from src.preprocessing.fusion import fuse

logger = logging.getLogger("vod-lens.pipeline")


def run_pipeline(
    url: str,
    workdir: Optional[Path] = None,
    model_size: str = "large-v3",
    language: str = "en",
    scene_threshold: float = 12.0,
    download_audio_only: bool = True,
    output_path: Optional[Path] = None,
    skip_download: bool = False,
    audio_path_hint: Optional[Path] = None,
) -> FusionResult:
    """
    Run the full VOD preprocessing pipeline end-to-end.

    Args:
        url: Twitch VOD URL
        workdir: Working directory for temp files (default: /tmp/vod_<id>)
        model_size: Whisper model size
        language: Language code
        scene_threshold: Scene detection threshold
        download_audio_only: Download audio only (smaller/faster)
        output_path: Path to save final JSON output
        skip_download: Skip download step (use existing files)
        audio_path_hint: Use existing audio file instead of downloading

    Returns:
        FusionResult with all processing outputs

    Raises:
        RuntimeError: If any pipeline step fails
    """
    start_time = time.time()
    vod_id = extract_vod_id(url)

    if workdir is None:
        workdir = Path(f"/tmp/vod_{vod_id}")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Pipeline started: {url} -> {workdir}")

    try:
        # Step 1: Download
        if skip_download and audio_path_hint:
            logger.info("Skipping download, using existing audio")
            audio_path = Path(audio_path_hint)
            vod_meta = None  # Will be filled from existing metadata
        else:
            logger.info("Step 1/5: Downloading VOD...")
            audio_path, vod_meta = download_vod(
                url=url,
                output_dir=workdir,
                audio_only=download_audio_only,
            )
            logger.info(f"  Downloaded: {audio_path} ({vod_meta.duration_seconds}s)")

        # If audio-only download, we need the video for scene detection
        if download_audio_only and not audio_path_hint:
            logger.info("Step 1b/5: Downloading video for scene detection...")
            video_path, vod_meta_video = download_vod(
                url=url,
                output_dir=workdir,
                audio_only=False,
                format_spec="bestvideo[height<=720]/bestvideo",
            )
            if video_path.suffix != '.mp4':
                # Re-download with explicit merge
                from src.preprocessing.downloader import _find_downloaded_file
                video_path = _find_downloaded_file(workdir, "vod_input")
        else:
            video_path = audio_path

        # Step 2: Transcribe
        logger.info("Step 2/5: Transcribing audio (Whisper)...")
        transcript = transcribe(
            audio_path=audio_path,
            model_size=model_size,
            language=language,
        )
        logger.info(f"  Transcribed: {len(transcript.segments)} segments")

        # Save transcript to file
        _save_json(workdir / "transcript.json", transcript.model_dump())

        # Step 3: Scene detection
        logger.info("Step 3/5: Detecting scenes...")
        scenes = detect_scenes(
            video_path=video_path,
            threshold=scene_threshold,
        )
        logger.info(f"  Detected: {len(scenes)} scenes")

        # Save scenes to file
        _save_json(workdir / "scenes.json", [s.model_dump() for s in scenes])

        # Step 4: Chat analysis
        logger.info("Step 4/5: Downloading and analyzing chat...")
        chat_messages = download_chat(vod_id)
        chat = analyze_chat(chat_messages)
        logger.info(f"  Chat: {chat.total_messages} messages, {chat.unique_chatters} chatters")

        # Save chat to file
        _save_json(workdir / "chat.json", chat.model_dump())

        # Step 5: Fusion
        logger.info("Step 5/5: Fusing results into unified timeline...")
        if vod_meta is None:
            vod_meta = vod_meta_video

        result = fuse(
            vod_meta=vod_meta,
            transcript=transcript,
            scenes=scenes,
            chat=chat,
        )
        logger.info(f"  Fusion complete: {len(result.timeline)} timeline entries")

        # Add total processing time
        result.processing_time_seconds = time.time() - start_time

        # Save final output
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _save_json(output_path, result.model_dump())
            logger.info(f"Final output saved: {output_path}")

        logger.info(f"Pipeline complete in {result.processing_time_seconds:.1f}s")
        return result

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


def run_pipeline_minimal(
    audio_path: Path,
    url: Optional[str] = None,
    workdir: Optional[Path] = None,
    model_size: str = "large-v3",
    language: str = "en",
) -> FusionResult:
    """
    Run a minimal pipeline that skips download and uses existing audio.
    Useful for testing with pre-downloaded files.

    Args:
        audio_path: Path to existing audio file (16kHz mono WAV)
        url: Optional VOD URL for metadata
        workdir: Working directory

    Returns:
        FusionResult
    """
    vod_id = extract_vod_id(url) if url else "test"
    if workdir is None:
        workdir = Path(f"/tmp/vod_{vod_id}")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    vod_meta = None

    # Transcribe
    transcript = transcribe(
        audio_path=audio_path,
        model_size=model_size,
        language=language,
    )
    _save_json(workdir / "transcript.json", transcript.model_dump())

    # Scene detection on the same file won't work without video
    scenes: list = []

    # Chat is empty for test
    chat_messages = download_chat(vod_id) if url else []
    chat = analyze_chat(chat_messages)
    _save_json(workdir / "chat.json", chat.model_dump())

    if url:
        from src.preprocessing.downloader import download_vod
        _, vod_meta = download_vod(url, workdir, audio_only=False)

    if vod_meta is None:
        from src.models.types import VodMeta
        vod_meta = VodMeta(
            id=vod_id,
            title=f"Test_{vod_id}",
            duration_seconds=int(transcript.duration_seconds),
            url=url or "",
            streamer="test",
        )

    result = fuse(
        vod_meta=vod_meta,
        transcript=transcript,
        scenes=scenes,
        chat=chat,
    )

    result.processing_time_seconds = 0.0

    return result


def _save_json(path: Path, data: dict | list) -> None:
    """Save data as pretty-printed JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
