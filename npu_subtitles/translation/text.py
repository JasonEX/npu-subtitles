"""Conservative, auditable English preparation for a small translation model.

These are string/grammar operations, not a language model or an ASR correction
layer. They never consult subtitles, infer missing speech, or repair homophones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from collections.abc import Iterable, Mapping


@dataclass
class PreparedTranslation:
    source: str
    model_input: str
    output_prefix: str = ""
    changes: list[dict[str, str]] = field(default_factory=list)


_ASTRONOMICAL = re.compile(
    r"\b(?:constellation|zodiac|horoscope|astrology|astronomy|star\s+sign)\b", re.IGNORECASE
)


def prepare_translation(
    text: str,
    *,
    person_names: Iterable[str] | Mapping[str, str] = (),
) -> PreparedTranslation:
    """Normalize whitespace and preserve explicitly configured leading names."""
    normalized = " ".join(text.split())
    prepared = PreparedTranslation(source=text, model_input=normalized)
    if normalized != text:
        prepared.changes.append(
            {
                "rule": "whitespace",
                "before": text,
                "after": normalized,
                "reason": "Normalize whitespace without repairing or completing speech.",
            }
        )
    names = (
        dict(person_names)
        if isinstance(person_names, Mapping)
        else {name: name for name in person_names}
    )
    # Explicitly configured names only. Protect a leading vocative by moving it
    # outside NMT, without creating an unreliable placeholder inside the model.
    for name in sorted(names, key=len, reverse=True):
        if not name.strip() or not names[name].strip():
            raise ValueError("Person names and their display forms must not be empty")
        match = re.fullmatch(
            r"\s*(" + re.escape(name) + r"),\s+(\S[\s\S]*)", prepared.model_input, re.IGNORECASE
        )
        if match and not _ASTRONOMICAL.search(match.group(2)):
            prepared.output_prefix = names[name] + "，"
            prepared.model_input = match.group(2)
            prepared.changes.append(
                {
                    "rule": "explicit_vocative_name",
                    "before": match.group(1),
                    "after": names[name],
                    "reason": "Caller declared this person name; preserve the leading address outside NMT.",
                }
            )
            break

    return prepared
