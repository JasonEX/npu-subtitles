"""Behavioral checks for the refactored runtime boundaries."""

import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from npu_subtitles.captions.pipeline import CaptionPipeline
from npu_subtitles.translation.chain import split_for_model, translate_stages
from npu_subtitles.translation.marian import TranslationCancelled, TranslationRejected


class CaptionLifecycleTests(unittest.TestCase):
    def pipeline(self):
        pipeline = CaptionPipeline()
        pipeline.feed(
            dict(id=1, revision=1, text="Please wait here.", start=0.0, end=3.0, is_final=True),
            0,
            1.0,
            0.0,
        )
        candidate = pipeline.translation_candidate(1.0, 0.0)
        self.assertIsNotNone(candidate)
        pipeline.mark_submitted(candidate["key"])
        return pipeline, candidate

    def test_translation_preserves_expiry_and_seek_rejects_old_results(self):
        pipeline, candidate = self.pipeline()
        expiry = candidate["expires_at"]
        result = dict(
            id=candidate["id"],
            revision=candidate["revision"],
            generation=0,
            source=candidate["text"],
            text="请在这里等候。",
        )
        self.assertTrue(pipeline.translated(result, 0, 1.1, 0.0))
        frame = pipeline.frame(2.0, candidate["due_at"])
        self.assertEqual(frame["chinese"], result["text"])
        self.assertEqual(frame["deadline"], expiry)
        pipeline.reset(1)
        self.assertFalse(pipeline.translated(result, 0, 3.0, 0.0))
        self.assertIsNone(pipeline.frame(3.0, 0.0))

    def test_late_translation_does_not_renew_caption_lifetime(self):
        pipeline, candidate = self.pipeline()
        result = dict(
            id=candidate["id"],
            revision=candidate["revision"],
            generation=0,
            source=candidate["text"],
            text="请在这里等候。",
        )
        self.assertFalse(pipeline.translated(result, 0, 20.0, candidate["expires_at"] + 1))
        self.assertIsNone(pipeline.frame(20.0, candidate["expires_at"] + 1))


class TranslationChainTests(unittest.TestCase):
    def model(self, target, translate):
        return SimpleNamespace(
            target_language=target,
            config={"source_length": 9},
            source=SimpleNamespace(encode=list),
            last_result={"status": "completed"},
            translate=Mock(side_effect=translate),
        )

    def test_chunked_pivot_uses_one_deadline_and_keeps_all_text(self):
        pivot = self.model("en", lambda text, **kwargs: text)
        chinese = self.model("zh-Hans", lambda text, **kwargs: text)
        cancelled = lambda: False
        source = "one two three four five"
        result = {}
        translated = translate_stages(
            source,
            [("fr_en", pivot), ("en_zh", chinese)],
            result,
            deadline=123.0,
            cancelled=cancelled,
            split=True,
        )
        self.assertEqual(translated.replace(" ", ""), source.replace(" ", ""))
        for model in (pivot, chinese):
            self.assertGreater(model.translate.call_count, 1)
            for call in model.translate.call_args_list:
                self.assertEqual(call.kwargs, dict(deadline=123.0, cancelled=cancelled))
        self.assertEqual(result["status"], "completed")

    def test_cancelled_chunk_never_publishes_partial_translation(self):
        def translate(text, **kwargs):
            if text == "three":
                model.last_result = {"status": "cancelled"}
                raise TranslationCancelled("before_encoder")
            return text

        model = self.model("zh-Hans", translate)
        result = {}
        with self.assertRaises(TranslationCancelled):
            translate_stages("one two three", [("en_zh", model)], result, split=True)
        self.assertNotIn("translation", result)
        self.assertEqual(result["status"], "cancelled")

    def test_token_limit_never_truncates_unbroken_text(self):
        model = self.model("en", lambda text: text)
        source = "abcdefghijklmno"
        parts = split_for_model(source, model)
        self.assertEqual("".join(parts), source)
        self.assertTrue(all(len(part) <= 8 for part in parts))
        model.config["source_length"] = 1
        with self.assertRaises(TranslationRejected):
            split_for_model(source, model)

    def test_empty_intermediate_does_not_start_chinese(self):
        pivot = self.model("en", lambda text, **kwargs: "")
        chinese = self.model("zh-Hans", lambda text, **kwargs: text)
        with self.assertRaises(TranslationRejected):
            translate_stages("bonjour", [("fr_en", pivot), ("en_zh", chinese)], {})
        chinese.translate.assert_not_called()


class ImportBoundaryTests(unittest.TestCase):
    def test_routing_and_help_do_not_load_models_or_preparation_tools(self):
        code = """import sys
import mpv_plugin
from npu_subtitles.app import parser
parser().parse_args(['example.mkv'])
assert not any(name.startswith('tools.models') for name in sys.modules)
assert not {'numpy', 'openvino', 'openvino_genai', 'onnx', 'customtkinter'} & sys.modules.keys()
"""
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, timeout=10)


class TranslationPreparationTests(unittest.TestCase):
    def test_preserves_configured_name_and_normalizes_whitespace(self):
        from npu_subtitles.translation.text import prepare_translation

        result = prepare_translation("Alice,  please wait.\n", person_names=("Alice",))
        self.assertEqual(result.model_input, "please wait.")
        self.assertEqual(result.output_prefix, "Alice，")

    def test_leaves_unconfigured_names_and_grammar_unchanged(self):
        from npu_subtitles.translation.text import prepare_translation

        text = "Should you see flesh-possessing spirits, I will help."
        self.assertEqual(prepare_translation(text).model_input, text)
        self.assertEqual(prepare_translation("Alice, please wait.").output_prefix, "")


if __name__ == "__main__":
    unittest.main()
