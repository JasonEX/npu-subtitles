"""Japanese -> English -> Simplified Chinese, with one shared NPU deadline.

The pivot is diagnostic evidence, never a displayed subtitle or a replacement
for the original Japanese. Neither stage renews the deadline or falls back to
another execution device. No partial translation is published.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from npu_subtitles.translation.marian import MTranNPUTranslator
from npu_subtitles.translation.chain import translate_stages

from npu_subtitles.paths import ROOT


class JapaneseChineseNPUTranslator:
    def __init__(
        self,
        japanese_model: str | Path = ROOT / "models/mtran-ja-en-npu",
        chinese_model: str | Path = ROOT / "models/mtran-en-zh-npu",
        cache_dir: str | Path = ROOT / ".cache/live-npu/japanese-translation",
    ):
        self.japanese = self.chinese = None
        self.last_result = {}
        try:
            self.japanese = MTranNPUTranslator(japanese_model, Path(cache_dir) / "ja-en")
            if (self.japanese.source_language, self.japanese.target_language) != ("ja", "en"):
                raise ValueError("Japanese translation stage must be ja -> en")
            self.chinese = MTranNPUTranslator(chinese_model, Path(cache_dir) / "en-zh")
            if (self.chinese.source_language, self.chinese.target_language) != ("en", "zh-Hans"):
                raise ValueError("Chinese translation stage must be en -> zh-Hans")
        except BaseException:
            self.close()
            raise

    def details(self) -> dict:
        return dict(
            backend="mozilla-marian-japanese-pivot-npu",
            source_language="ja",
            target_language="zh-Hans",
            pivot_language="en",
            device="NPU",
            cpu_fallback=False,
            execution_devices={
                "ja_en": self.japanese.execution_devices,
                "en_zh": self.chinese.execution_devices,
            },
            stages={"ja_en": self.japanese.details(), "en_zh": self.chinese.details()},
            deadline="one monotonic deadline and cancellation callback across both stages",
            output="Chinese only; pivot retained in diagnostics; no source correction",
        )

    def translate(
        self,
        text: str,
        *,
        deadline: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> str:
        self.last_result = dict(
            source=text, source_language="ja", target_language="zh-Hans", pivot_language="en"
        )
        return translate_stages(
            text,
            [("ja_en", self.japanese), ("en_zh", self.chinese)],
            self.last_result,
            deadline=deadline,
            cancelled=cancelled,
        )

    def close(self) -> None:
        for stage in (self.chinese, self.japanese):
            if stage is not None:
                stage.close()
