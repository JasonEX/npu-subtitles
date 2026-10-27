"""ASS escaping and sidecar rendering without Windows player imports."""

from __future__ import annotations
from pathlib import Path
import math
import os
import re
import time

DEFAULT_FONT = "Microsoft YaHei"
# osd-overlay coordinate space; player/mpv/npu_subtitles.lua uses the same values.
OVERLAY_RES_X, OVERLAY_RES_Y = 1280, 720


def ass_timestamp(milliseconds: int) -> str:
    """Round SRT's millisecond time to ASS's nearest 10 ms without float drift."""
    centiseconds = (milliseconds + 5) // 10
    hours, centiseconds = divmod(centiseconds, 360000)
    minutes, centiseconds = divmod(centiseconds, 6000)
    seconds, centiseconds = divmod(centiseconds, 100)
    return f"{hours}:{minutes:02}:{seconds:02}.{centiseconds:02}"


def ass_text(text: str) -> str:
    """Keep model text visible rather than interpreting it as ASS markup.

    Full-width equivalents prevent literal braces/backslashes from introducing
    override blocks, drawing commands, or synthetic line breaks. Commas are
    safe because Text is the final event field. Only real newlines become \\N.
    """
    safe = text.replace("\r", "").replace("\t", " ").replace("\0", "")
    safe = safe.translate(str.maketrans({"\\": "＼", "{": "｛", "}": "｝"}))
    return safe.replace("\n", r"\N")


def overlay_event(text: str, font: str = DEFAULT_FONT, font_size: float = 24) -> str:
    """One bottom-centred osd-overlay event, or "" to clear; text cannot inject tags."""
    if not text:
        return ""
    return (
        f"{{\\an2\\pos({OVERLAY_RES_X // 2},{OVERLAY_RES_Y - 42})\\fn{font}\\fs{font_size:g}"
        "\\bord1.6\\shad0.5\\1c&HFFFFFF&\\3c&H000000&}" + ass_text(text)
    )


def srt_to_ass(srt: str, font: str = DEFAULT_FONT, font_size: float = 28) -> str:
    """Convert this project's generated SRT to a sized bilingual ASS sidecar."""
    if not font or any(character in font for character in ",\r\n\0"):
        raise ValueError("Subtitle font must be non-empty and contain no commas or line breaks")
    if not math.isfinite(font_size) or font_size <= 0:
        raise ValueError("Subtitle font size must be positive and finite")
    header = (
        "[Script Info]\n"
        "Title: Local generated subtitles\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {OVERLAY_RES_X}\nPlayResY: {OVERLAY_RES_Y}\n"
        "WrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: LocalBilingual,{font},{font_size:g},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
        "0,0,0,0,100,100,0,0,1,1.5,0.5,2,44,44,30,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    srt = srt.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not srt:
        return header
    timing = re.compile(r"^(\d+):(\d{2}):(\d{2}),(\d{3}) --> (\d+):(\d{2}):(\d{2}),(\d{3})$")
    events = []
    for block in re.split(r"\n\s*\n", srt):
        lines = block.split("\n")
        match = timing.fullmatch(lines[1]) if len(lines) >= 3 and lines[0].isdigit() else None
        if match is None:
            raise ValueError(
                "Invalid generated SRT cue; refusing to replace the player's subtitles"
            )
        values = [int(value) for value in match.groups()]
        if any(values[index] >= 60 for index in (1, 2, 5, 6)):
            raise ValueError("Invalid SRT minute or second value")
        start = ((values[0] * 60 + values[1]) * 60 + values[2]) * 1000 + values[3]
        end = ((values[4] * 60 + values[5]) * 60 + values[6]) * 1000 + values[7]
        if end <= start:
            raise ValueError("SRT cue must end after it starts")
        # Preserve tiny positive cues when both ends round to the same 10 ms.
        if (end + 5) // 10 <= (start + 5) // 10:
            end = ((start + 5) // 10 + 1) * 10
        events.append(
            f"Dialogue: 0,{ass_timestamp(start)},{ass_timestamp(end)},LocalBilingual,,0,0,0,,"
            + ass_text("\n".join(lines[2:]))
        )
    return header + "\n".join(events) + "\n"


def write_ass(source: Path, target: Path, font: str, font_size: float) -> None:
    content = srt_to_ass(source.read_text(encoding="utf-8-sig"), font, font_size)
    temporary = target.with_suffix(".ass.tmp")
    temporary.write_text(content, encoding="utf-8-sig")
    for attempt in range(10):
        try:
            os.replace(temporary, target)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05)
