"""Language/ownership/deadline contracts; no models or reference transcripts."""

import unittest
from unittest.mock import Mock

from npu_subtitles.asr.transcript import (
    AcousticTranscriptLedger,
    ObservationWord,
    WindowObservation,
)
from npu_subtitles.captions.language import (
    JapaneseBoundaries,
    join_units,
    lexical_spans,
    japanese_sentence_end,
    reading_units,
)
from npu_subtitles.asr.granite import GraniteNPU
from npu_subtitles.captions.sentences import GraniteSentenceAssembler
from npu_subtitles.translation.japanese import JapaneseChineseNPUTranslator
from npu_subtitles.captions.commit import CueCommitter
from npu_subtitles.inference import InferenceSettings, inference_worker
from npu_subtitles.devices import require_npu
from npu_subtitles.app import parser
from npu_subtitles.translation.marian import (
    TranslationCancelled,
    TranslationExpired,
    TranslationRejected,
)


def units(text, language="ja"):
    return [text[a:b] for a, b in lexical_spans(text, language)]


def event(text, *, context=1, start=0.0, reason="input_end", language="ja"):
    words = [
        ObservationWord(
            word,
            start + i * 0.08,
            start + (i + 1) * 0.08,
            0.95,
            (i,),
            starts_word=True,
            ends_word=True,
        )
        for i, word in enumerate(units(text, language))
    ]
    end = start + len(words) * 0.08 + 1.0
    ledger = AcousticTranscriptLedger(context, start)
    snapshot = ledger.observe(
        WindowObservation(0, 1, start, end, tuple(words)), final_reason=reason
    )
    return dict(
        id=context,
        revision=1,
        source_language=language,
        start=start,
        end=end,
        encoded_until=end,
        received_until=end,
        is_final=True,
        closure_reason=reason,
        transcript=snapshot,
    )


class JapaneseTextTests(unittest.TestCase):
    def test_round_trip_scripts_quotes_numbers_and_repetitions(self):
        for text in [
            "今日は雨です。明日は？",
            "「はい、はい」と答えた。",
            "OpenVINO NPUで3.14倍、1,000円。",
            "か\u3099んは\u3099る。",
            "まだ…",
        ]:
            with self.subTest(text=text):
                self.assertEqual(join_units(units(text), "ja"), text)
        self.assertIn("3.14", units("3.14倍。"))
        self.assertFalse(japanese_sentence_end("3.14"))
        self.assertFalse(japanese_sentence_end("まだ…"))
        self.assertTrue(japanese_sentence_end("か？」"))

    def test_english_unit_contract_unchanged(self):
        text = 'Dr. Smith said, "Hello!"'
        self.assertEqual(units(text, "en"), text.split())
        self.assertEqual(join_units(units(text, "en"), "en"), text)
        self.assertFalse(CueCommitter.sentence_end("Dr."))

    def test_token_segmentation_does_not_change_character_ownership(self):
        class Tokenizer:
            def __init__(self, pieces):
                self.pieces = pieces

            def decode(self, ids, **kwargs):
                return "".join(self.pieces[i] for i in ids)

            def id_to_token(self, identity):
                return self.pieces[identity]

        outputs = []
        for pieces in [("日本", "語です。"), ("日", "本語", "です。")]:
            adapter = GraniteNPU.__new__(GraniteNPU)
            adapter.source_language, adapter.tokenizer = "ja", Tokenizer(pieces)
            adapter.segment_start = adapter.start_time = 0.0
            adapter.processed_samples = 16000
            adapter.alignment_details = {}
            tokens = [
                dict(id=i, start=i * 0.08, end=(i + 1) * 0.08, confidence=0.95)
                for i in range(len(pieces))
            ]
            words = adapter._observation_words(tokens)
            outputs.append([word.text for word in words])
            ledger = AcousticTranscriptLedger(1, 0.0)
            first = ledger.observe(WindowObservation(0, 1, 0.0, 1.0, words))
            self.assertFalse(first["sealed_words"])
            closed = ledger.observe(
                WindowObservation(0, 2, 0.0, 6.0, words), final_reason="input_end"
            )
            self.assertEqual(
                "".join(word["text"] for word in closed["sealed_words"]), "日本語です。"
            )
            self.assertEqual(len({word["occurrence_id"] for word in closed["sealed_words"]}), 5)
        self.assertEqual(outputs[0], outputs[1])

    def test_short_question_and_42_character_budget_without_text_loss(self):
        text = "本当？" + "これは長い文章で同じ文字も消さずに扱います" * 5 + "。"
        assembler = GraniteSentenceAssembler(CueCommitter.sentence_end, source_language="ja")
        source = event(text)
        proposals = assembler.feed(source, 0.0, 0.0)
        self.assertEqual(proposals[0]["text"], "本当？")
        self.assertEqual("".join(proposal["text"] for proposal in proposals), text)
        self.assertTrue(all(reading_units(proposal["text"], "ja") <= 42 for proposal in proposals))
        owned = [identity for proposal in proposals for identity in proposal["word_ids"]]
        self.assertEqual(len(owned), len(set(owned)))
        self.assertEqual(len(owned), len(source["transcript"]["sealed_words"]))

    def test_capacity_roll_does_not_force_japanese_sentence(self):
        assembler = GraniteSentenceAssembler(CueCommitter.sentence_end, source_language="ja")
        first = event("今日は", reason="capacity_limit")
        # A capacity roll does not supply a silence gap or sentence ending.
        first["end"] = first["encoded_until"] = first["received_until"] = 0.24
        first["transcript"]["sealed_until"] = 0.24
        self.assertEqual(assembler.feed(first, 0.0, 0.0), [])
        proposals = assembler.feed(event("雨です。", context=2, start=0.24), 1.0, 0.0)
        self.assertEqual("".join(p["text"] for p in proposals), "今日は雨です。")
        self.assertEqual(len(proposals), 1)
        self.assertEqual(len(proposals[0]["source_ranges"]), 2)

    def test_reading_budget_respects_dictionary_word_and_particle_boundaries(self):
        text = "これは" * 12 + "インターネットを利用します。"
        assembler = GraniteSentenceAssembler(CueCommitter.sentence_end, source_language="ja")
        proposals = assembler.feed(event(text), 0.0, 0.0)
        ends, offset = JapaneseBoundaries().ends(text), 0
        for proposal in proposals:
            offset += len(proposal["text"])
            self.assertIn(offset, ends)
        self.assertEqual("".join(p["text"] for p in proposals), text)

    def test_observed_long_next_word_releases_only_its_sealed_prefix(self):
        prefix = "これは" * 12
        source = event(prefix + "インターネットを利用します。")
        transcript = source["transcript"]
        words = transcript["sealed_words"]
        transcript["sealed_words"], transcript["mutable_words"] = words[:36], words[36:]
        transcript["sealed_until"] = words[35]["end"]
        source["is_final"], source["closure_reason"] = False, None
        assembler = GraniteSentenceAssembler(CueCommitter.sentence_end, source_language="ja")
        proposals = assembler.feed(source, 0.0, 0.0)
        self.assertEqual([p["text"] for p in proposals], [prefix])
        self.assertEqual(proposals[0]["word_ids"], [w["occurrence_id"] for w in words[:36]])
        self.assertEqual(
            proposals[0]["boundary_evidence"]["capacity_trigger"],
            "observed_continuation_at_capacity",
        )

    def test_prospective_cut_uses_same_pause_preference_as_final_cut(self):
        source = event("文章です" * 12)
        transcript = source["transcript"]
        words = transcript["sealed_words"]
        for word in words[16:]:
            word["start"] += 0.4
            word["end"] += 0.4
        source["end"] += 0.4
        source["encoded_until"] += 0.4
        source["received_until"] += 0.4
        transcript["sealed_words"], transcript["mutable_words"] = words[:16], words[16:]
        transcript["sealed_until"] = words[15]["end"]
        source["is_final"], source["closure_reason"] = False, None
        assembler = GraniteSentenceAssembler(CueCommitter.sentence_end, source_language="ja")
        proposals = assembler.feed(source, 0.0, 0.0)
        self.assertEqual([p["text"] for p in proposals], ["文章です" * 4])

    def test_language_is_explicit_and_default_english_is_preserved(self):
        self.assertEqual(parser().parse_args(["example.mp4"]).source_language, "en")
        self.assertEqual(
            parser()
            .parse_args(["example.mp4", "--asr", "granite", "--language", "ja"])
            .source_language,
            "ja",
        )
        assembler = GraniteSentenceAssembler(CueCommitter.sentence_end, source_language="ja")
        with self.assertRaisesRegex(RuntimeError, "source language"):
            assembler.feed(event("hello", language="en"), 0.0, 0.0)


class JapanesePivotTests(unittest.TestCase):
    def translator(self):
        translator = JapaneseChineseNPUTranslator.__new__(JapaneseChineseNPUTranslator)
        translator.japanese = Mock(last_result={"status": "completed"})
        translator.chinese = Mock(last_result={"status": "completed"})
        translator.japanese.translate.return_value = "A test."
        translator.chinese.translate.return_value = "测试。"
        return translator

    def test_shared_deadline_and_cancellation_no_pivot_display(self):
        translator, cancelled = self.translator(), lambda: False
        result = translator.translate("例です。", deadline=123.0, cancelled=cancelled)
        self.assertEqual(result, "测试。")
        translator.japanese.translate.assert_called_once_with(
            "例です。", deadline=123.0, cancelled=cancelled
        )
        translator.chinese.translate.assert_called_once_with(
            "A test.", deadline=123.0, cancelled=cancelled
        )
        self.assertEqual(translator.last_result["pivot_text"], "A test.")

    def test_first_stage_expiry_never_starts_second(self):
        translator = self.translator()
        translator.japanese.last_result = {"status": "expired"}
        translator.japanese.translate.side_effect = TranslationExpired("after_encoder")
        with self.assertRaises(TranslationExpired):
            translator.translate("例です。")
        translator.chinese.translate.assert_not_called()
        self.assertEqual(translator.last_result["failed_stage"], "ja_en")
        self.assertNotIn("translation", translator.last_result)

    def test_second_stage_cancel_never_returns_partial_chinese(self):
        translator = self.translator()
        translator.chinese.last_result = {"status": "cancelled"}
        translator.chinese.translate.side_effect = TranslationCancelled("before_token_2")
        with self.assertRaises(TranslationCancelled):
            translator.translate("例です。")
        self.assertEqual(translator.last_result["status"], "cancelled")
        self.assertNotIn("translation", translator.last_result)

    def test_nested_device_reports_reject_any_cpu_or_auto(self):
        require_npu(
            {"ja_en": {"encoder": ["NPU"], "decoder": ["NPU.0"]}, "en_zh": ["NPU"]}, "pivot"
        )
        for invalid in ["CPU", "GPU", "AUTO:NPU,CPU"]:
            with self.assertRaises(RuntimeError):
                require_npu({"ja_en": ["NPU"], "en_zh": ["NPU", invalid]}, "pivot")

    def test_rejected_sentence_keeps_worker_alive_without_partial_translation(self):
        connection = Mock()
        connection.recv.side_effect = [
            dict(kind="translate", generation=0, id=index, revision=1, text="例です。")
            for index in (1, 2)
        ] + [dict(kind="stop")]
        asr = Mock(execution_devices=["NPU"])
        asr.details.return_value = {}
        translator = Mock(last_result={"status": "failed"})
        translator.details.return_value = {
            "execution_devices": {"ja_en": ["NPU"], "en_zh": ["NPU"]}
        }
        translator.translate.side_effect = [TranslationRejected("output bound"), "示例。"]
        inference_worker(
            connection, InferenceSettings("", "", ""), factory=lambda **kwargs: (asr, translator)
        )
        messages = [call.args[0] for call in connection.send.call_args_list]
        self.assertEqual(
            [m["kind"] for m in messages],
            ["startup", "ready", "translation_failed", "translation", "stopped"],
        )
        self.assertNotIn("text", messages[2])
        self.assertEqual(messages[3]["text"], "示例。")
        translator.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
