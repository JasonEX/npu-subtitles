"""Subtitle values and SRT serialization."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import re
import time


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


def timestamp(seconds: float) -> str:
    millis = max(0, round(seconds * 1000))
    hours, millis = divmod(millis, 3600000)
    minutes, millis = divmod(millis, 60000)
    seconds, millis = divmod(millis, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def merge_cues(cues: list[Cue]) -> list[Cue]:
    """Remove exact repeated overlap text without suppressing later dialogue."""
    result: list[Cue] = []
    key = lambda text: re.sub(r"[^\w]+", "", text.casefold())
    for cue in sorted(cues, key=lambda item: (item.start, item.end)):
        if result and key(result[-1].text) == key(cue.text) and cue.start <= result[-1].end + 0.25:
            old = result[-1]
            result[-1] = Cue(old.start, max(old.end, cue.end), old.text)
        else:
            result.append(cue)
    return result


def render_srt(cues: list[Cue]) -> str:
    return "\n\n".join(
        f"{index}\n{timestamp(cue.start)} --> {timestamp(cue.end)}\n{cue.text}"
        for index, cue in enumerate(merge_cues(cues), 1)
    ) + ("\n" if cues else "")


def atomic_srt(path: Path, cues: list[Cue]) -> None:
    temporary = path.with_suffix(".srt.tmp")
    temporary.write_text(render_srt(cues), encoding="utf-8")
    # A reader may briefly hold the old file on Windows.
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05)
