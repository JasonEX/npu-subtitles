"""Language-dependent text units, without content-specific correction rules.

Japanese CTC ownership uses characters (Latin runs remain words), not spaces or
BPE tokens. This keeps occurrence matching independent of tokenizer segmentation.
Characters sharing a CTC token share its evidence; they are not new observations.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

LANGUAGES = ("en", "ja")
JAPANESE_MAX_CHARACTERS = 42
JAPANESE_CHARACTERS_PER_SECOND = 7.0
_JAPANESE_UNITS = re.compile(r"[0-9]+(?:[.,][0-9]+)+|[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*|[^\s]")
_OPENING = set("「『（【〈《［([{")
_CLOSING = "\"'”’)]}」』）】〉》］"


def check_language(language: str) -> str:
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported subtitle source language: {language!r}")
    return language


def lexical_spans(text: str, language: str) -> list[tuple[int, int]]:
    """Return source character ranges, attaching punctuation to lexical units."""
    check_language(language)
    if language == "en":
        return [match.span() for match in re.finditer(r"\S+", text)]
    spans: list[tuple[int, int]] = []
    opening = None
    for match in _JAPANESE_UNITS.finditer(text):
        part = match.group()
        if part in _OPENING:
            if opening is None:
                opening = match.start()
        elif any(char.isalnum() for char in part):
            spans.append((opening if opening is not None else match.start(), match.end()))
            opening = None
        elif spans:
            # Dakuten/variation selectors and punctuation keep the preceding
            # acoustic unit. They never receive fabricated independent times.
            spans[-1] = (spans[-1][0], match.end())
    if opening is not None and spans:
        spans[-1] = (spans[-1][0], len(text))
    return spans


def join_units(units: Iterable[str], language: str) -> str:
    check_language(language)
    if language == "en":
        return " ".join(units)
    text = ""
    for unit in units:
        if (
            text
            and unit
            and text[-1].isascii()
            and text[-1].isalnum()
            and unit[0].isascii()
            and unit[0].isalnum()
        ):
            text += " "
        text += unit
    return text


def reading_units(text: str, language: str) -> int:
    check_language(language)
    if language == "en":
        return sum(any(char.isalnum() for char in word) for word in text.split())
    return sum(not char.isspace() and not unicodedata.combining(char) for char in text)


def japanese_sentence_end(text: str) -> bool:
    terminal = text.rstrip(_CLOSING)
    if terminal.endswith(("..", "…", "‥", "-", "–", "—")):
        return False
    return terminal.endswith(("。", "！", "？", "!", "?", ".", ";", "；"))


def japanese_clause_end(text: str) -> bool:
    return text.rstrip(_CLOSING).endswith(("、", "，", ",", "：", ":"))


class JapaneseBoundaries:
    """Dictionary morphology chooses cuts; it never rewrites recognized text.

    Long-unit (C) boundaries keep compounds together. Particles, auxiliaries and
    suffixes stay with their preceding unit. This is host string processing,
    without a neural model, user dictionary or content-dependent vocabulary.
    """

    def __init__(self):
        from sudachipy import Dictionary, SplitMode

        self.tokenizer = Dictionary(dict="core").create()
        self.mode = SplitMode.C

    def ends(self, text: str) -> set[int]:
        morphemes = list(self.tokenizer.tokenize(text, self.mode))
        ends = set()
        for index, morpheme in enumerate(morphemes):
            following = morphemes[index + 1] if index + 1 < len(morphemes) else None
            if following is None or following.part_of_speech()[0] not in {
                "助詞",
                "助動詞",
                "接尾辞",
            }:
                ends.add(morpheme.end())
        return ends
