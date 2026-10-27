"""NPU-only general Japanese translation and cross-stage cancellation probes."""

import json
import time

from npu_subtitles.paths import ROOT
from npu_subtitles.translation.japanese import JapaneseChineseNPUTranslator
from npu_subtitles.devices import require_npu
from npu_subtitles.translation.marian import TranslationCancelled, TranslationExpired


def main():
    translator = JapaneseChineseNPUTranslator()
    report = dict(details=translator.details(), examples=[], contracts={})
    require_npu(report["details"]["execution_devices"], "Japanese pivot")
    try:
        for source in [
            "明日の会議は午後三時からです。",
            "会議は午後三時ではなく、四時からです。",
            "私は魚を食べません。",
            "この電車は京都に止まりますか。",
            "このボタンを二回押してください。",
            "ここで待っていてください。",
            "彼はまだ来ていません。",
            "今日は雨ですが、明日は晴れるでしょう。",
            "私は彼に本を貸しました。",
            "荷物は駅に忘れてしまいました。",
            "私は行けます。",
            "私は行けません。",
        ]:
            chinese = translator.translate(source, deadline=time.monotonic() + 1.5)
            report["examples"].append(
                dict(source=source, chinese=chinese, details=translator.last_result)
            )
        try:
            translator.translate("これは期限切れです。", deadline=time.monotonic() - 1.0)
            raise AssertionError("Expired request was returned")
        except TranslationExpired:
            report["contracts"]["expired_before_first_stage"] = translator.last_result
        try:
            translator.translate(
                "ここで待っていてください。",
                deadline=time.monotonic() + 1.5,
                cancelled=lambda: "pivot_text" in translator.last_result,
            )
            raise AssertionError("Cancelled pivot was returned")
        except TranslationCancelled:
            if translator.last_result.get("failed_stage") != "en_zh":
                raise
            report["contracts"]["cancelled_between_stages"] = translator.last_result
        report["contracts"]["next_request_after_cancellation"] = translator.translate(
            "私は行けます。", deadline=time.monotonic() + 1.5
        )
        path = ROOT / "outputs/validation/japanese-npu/general-translation.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                dict(examples=len(report["examples"]), contracts=list(report["contracts"])),
                ensure_ascii=False,
            )
        )
    finally:
        translator.close()


if __name__ == "__main__":
    main()
