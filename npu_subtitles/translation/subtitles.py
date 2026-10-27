"""Translate existing text subtitles independently of player sessions."""

from __future__ import annotations

import time

from npu_subtitles.devices import require_npu
from npu_subtitles.paths import ROOT
from npu_subtitles.translation.chain import translate_stages
from npu_subtitles.translation.languages import PIVOT_LANGUAGES, SUBTITLE_LANGUAGES
from npu_subtitles.translation.marian import MTranNPUTranslator


class SubtitleTranslator:
    def __init__(self, source: str):
        if source not in SUBTITLE_LANGUAGES:
            raise ValueError("Unsupported subtitle language")
        self.source_language = source
        self.pivot = self.chinese = None
        self.last_result = {}
        try:
            self.chinese = MTranNPUTranslator(
                ROOT / "models/mtran-en-zh-npu", ROOT / ".cache/live-npu/mtran"
            )
            if (self.chinese.source_language, self.chinese.target_language) != ("en", "zh-Hans"):
                raise ValueError("Expected the fixed English-to-Chinese model")
            if source in PIVOT_LANGUAGES:
                cache = ROOT / (
                    ".cache/live-npu/japanese-translation/ja-en"
                    if source == "ja"
                    else f".cache/live-npu/subtitle-translation/{source}-en"
                )
                self.pivot = MTranNPUTranslator(ROOT / f"models/mtran-{source}-en-npu", cache)
                if (self.pivot.source_language, self.pivot.target_language) != (source, "en"):
                    raise ValueError("Subtitle source does not match the fixed pivot model")
                require_npu(self.pivot.execution_devices, "Source-to-English translation")
            require_npu(self.chinese.execution_devices, "English translation")
        except BaseException:
            self.close()
            raise

    def translate(self, text, display, cancelled, *, deadline=None):
        if deadline is None:
            deadline = time.monotonic() + 12
        self.last_result = dict(source_language=self.source_language, pivot_text=text)
        stages = [(f"{self.source_language}_en", self.pivot)] if self.pivot else []
        stages.append(("en_zh", self.chinese))
        chinese = translate_stages(
            text, stages, self.last_result, deadline=deadline, cancelled=cancelled, split=True
        )
        if not chinese.strip():
            raise ValueError("Empty subtitle translation")
        return chinese + "\n" + (text if display == "ja" else self.last_result["pivot_text"])

    def close(self):
        for model in (self.chinese, self.pivot):
            if model is not None:
                model.close()
