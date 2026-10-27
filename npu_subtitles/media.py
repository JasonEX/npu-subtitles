"""FFmpeg media access and process launch utilities."""

from __future__ import annotations
from pathlib import Path
import json
import math
import os
import shutil
import subprocess


def hidden_options() -> dict:
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def executable(value: str | None, name: str) -> str:
    candidates = [value] if value else [shutil.which(name)]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
        if candidate and shutil.which(candidate):
            return str(shutil.which(candidate))
    raise RuntimeError(f"{name} not found; supply --{name} PATH")


def probe_media(ffprobe: str, video: Path, audio_track: int) -> tuple[float, int]:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)],
        capture_output=True,
        timeout=30,
        check=True,
        **hidden_options(),
    )
    metadata = json.loads(result.stdout)
    audio = [stream for stream in metadata["streams"] if stream.get("codec_type") == "audio"]
    if not 0 <= audio_track < len(audio):
        raise ValueError(
            f"--audio-track {audio_track} is invalid; file has {len(audio)} audio tracks (zero-based)"
        )
    duration = float(metadata.get("format", {}).get("duration", 0))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("A local media file with a finite duration is required")
    return duration, int(audio[audio_track]["index"])
