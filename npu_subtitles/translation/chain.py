"""One deadline and one publication point across translation stages."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from typing import Any

from npu_subtitles.translation.marian import TranslationExpired, TranslationRejected


def split_for_model(text: str, model: Any) -> list[str]:
    """Split on punctuation or space without discarding source characters."""
    text = " ".join(text.split())
    limit = int(model.config["source_length"]) - 1
    parts = []
    while text:
        if len(model.source.encode(text)) <= limit:
            parts.append(text)
            break
        low, high = 1, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if len(model.source.encode(text[:mid])) <= limit:
                low = mid
            else:
                high = mid - 1
        cuts = list(re.finditer(r"[。！？.!?;,，；\s]", text[:low]))
        cut = cuts[-1].end() if cuts and cuts[-1].end() >= low // 2 else low
        piece = text[:cut].strip()
        if not piece or len(model.source.encode(piece)) > limit:
            raise TranslationRejected("Subtitle cannot be split within the model token limit")
        parts.append(piece)
        text = text[cut:].strip()
    return parts


def translate_stages(
    text: str,
    stages: Iterable[tuple[str, Any]],
    result: dict,
    *,
    deadline: float | None = None,
    cancelled: Callable[[], bool] | None = None,
    split: bool = False,
) -> str:
    """Publish only a complete chain; intermediate text remains diagnostic data."""
    started = time.monotonic()
    result.update(status="started", stages={}, stage_seconds={})
    for name, model in stages:
        stage_started = time.monotonic()
        try:
            parts = split_for_model(text, model) if split else [text]
            translated = []
            for part in parts:
                value = model.translate(part, deadline=deadline, cancelled=cancelled)
                if part.strip() and not value.strip():
                    raise TranslationRejected(f"Empty translation from {name}")
                translated.append(value)
            separator = "" if model.target_language == "zh-Hans" else " "
            text = separator.join(translated)
            if name != "en_zh":
                result["pivot_text"] = text
        except TranslationExpired:
            result.update(status=model.last_result.get("status", "expired"), failed_stage=name)
            raise
        except Exception as exc:
            result.update(status="failed", failed_stage=name, error=type(exc).__name__)
            raise
        finally:
            result["stages"][name] = dict(model.last_result)
            result["stage_seconds"][name] = time.monotonic() - stage_started
            result["elapsed_seconds"] = time.monotonic() - started
    result.update(status="completed", translation=text)
    return text
