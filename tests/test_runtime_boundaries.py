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


class FakeBackend:
    execution_devices = ["NPU"]

    def details(self):
        return {"execution_devices": {"en_zh": ["NPU"]}}

    def close(self):
        pass


def fake_backends(**settings):
    # Module-level so the spawned inference worker can import it.
    return FakeBackend(), FakeBackend()


class FakePluginSession:
    """The subset of mpv_plugin.Session used by the audio loops."""

    def __init__(self, positions, durations, reader, stop_when):
        self.positions, self.durations = iter(positions), iter(durations)
        self.reader, self.stop_when = reader, stop_when
        self.ipc = SimpleNamespace(events=[])
        self.overlays, self.statuses = [], []

    def current(self):
        return not self.stop_when(self)

    def status(self, value, **details):
        self.statuses.append(value)

    def message(self, name, value):
        self.overlays.append(value)

    def clear(self):
        self.message("npu-overlay", "")

    def get(self, name):
        return {"pause": False}[name]

    def get_optional(self, name):
        return next(self.positions if name == "time-pos" else self.durations)

    def audio_reader(self, track):
        return self.reader


class FakeReader:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.submitted, self.cancelled = [], []

    def submit(self, start_sample, end_sample, seen, generation, read_limit):
        from concurrent.futures import Future

        self.submitted.append(generation)
        future = Future()
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            future.set_exception(outcome)
        else:
            future.set_result({"pcm": outcome})
        return future

    def cancel_before(self, generation):
        self.cancelled.append(generation)


class FakeSenseVoice:
    execution_devices = ["NPU"]

    def __init__(self, *args, **kwargs):
        pass

    def generate(self, pcm, language, **kwargs):
        return SimpleNamespace(
            chunks=[dict(start_ts=0.5, end_ts=3.0, text="你好")],
            texts=["hello there"],
            detected_language="en",
            language_score=0.95,
        )


class PluginRobustnessTests(unittest.TestCase):
    def test_session_keeps_one_audio_reader_per_track(self):
        from pathlib import Path
        from unittest.mock import patch
        import mpv_plugin

        args = SimpleNamespace(ipc="pipe", pid=1, token="t", video=Path("v.mkv"), ffmpeg=None)
        with (
            patch("mpv_plugin.MpvIPC"),
            patch("mpv_plugin.executable", return_value="ffmpeg"),
        ):
            session = mpv_plugin.Session(args)
            first, second = session.audio_reader(0), session.audio_reader(1)
            self.assertIs(session.audio_reader(0), first)
            self.assertIsNot(first, second)
            self.assertEqual((first.audio_track, second.audio_track), (0, 1))
            session.close()
        self.assertTrue(first.closed and second.closed)
        self.assertEqual(session.readers, {})

    def test_chinese_asr_skips_undecodable_window_and_waits_for_duration(self):
        import contextlib
        import io
        from unittest.mock import patch
        import numpy as np
        import mpv_plugin

        speech = np.full(16000 * 8, 0.1, dtype="<f4").tobytes()
        reader = FakeReader(RuntimeError("ffmpeg: corrupt packet"), speech)
        shown = lambda session: any("你好" in text for text in session.overlays)  # noqa: E731
        session = FakePluginSession(
            positions=[0.0] + [0.5 * i for i in range(400)],
            durations=[None] + [60.0] * 400,
            reader=reader,
            stop_when=shown,
        )
        output = io.StringIO()
        with (
            patch("npu_subtitles.asr.sensevoice.SenseVoicePipeline", FakeSenseVoice),
            contextlib.redirect_stdout(output),
        ):
            mpv_plugin.chinese_loop(session, 0)
        self.assertTrue(shown(session))
        self.assertIn("chinese_audio_skipped", output.getvalue())
        self.assertEqual(reader.cancelled, [1])
        self.assertEqual(reader.submitted, [1, 1])

    def test_language_probe_retries_when_duration_is_unavailable(self):
        from unittest.mock import patch
        import numpy as np
        import mpv_plugin

        clock = iter(range(0, 10_000, 1))
        reader = FakeReader(np.full(16000 * 6, 0.1, dtype="<f4").tobytes())
        session = FakePluginSession(
            positions=[1.0, 5.0],
            durations=[None, 60.0],
            reader=reader,
            stop_when=lambda session: False,
        )
        with (
            patch("npu_subtitles.asr.sensevoice.SenseVoicePipeline", FakeSenseVoice),
            patch("mpv_plugin.time", SimpleNamespace(monotonic=lambda: next(clock), sleep=Mock())),
        ):
            self.assertEqual(mpv_plugin.detect_audio(session, 0), "en")
        self.assertEqual(reader.submitted, [0])


class AttachedErrorLoggingTests(unittest.TestCase):
    def test_error_is_logged_when_player_closes_at_the_same_time(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from npu_subtitles import app

        with tempfile.TemporaryDirectory() as directory:
            args = app.parser().parse_args(
                ["v.mkv", "--output-dir", directory, "--cache-dir", directory]
            )
            args.asr_model = Path(directory) / "asr"
            ipc = Mock()
            ipc.get_optional.side_effect = RuntimeError("IPC failed")
            attachment = Mock()
            attachment.current.return_value = True
            # poll() == 0 is how an attached session reports that it ended.
            attachment.connect.return_value = (Mock(pid=1, poll=Mock(return_value=0)), ipc, 1)
            with patch.object(
                app, "validate", return_value=(Path(directory) / "v.mkv", "ffmpeg", 10.0, 1)
            ):
                self.assertEqual(app.run(args, fake_backends, attachment=attachment), 0)
            (log_path,) = Path(directory).glob("*.jsonl")
            events = [
                json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
        closed = [event for event in events if event["event"] == "player_closed"]
        self.assertEqual(closed[0]["error"], "RuntimeError: IPC failed")


class ExecutableDiscoveryTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "Windows executable discovery")
    def test_mpv_executable_takes_priority_over_console_wrapper(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from npu_subtitles.media import executable

        with tempfile.TemporaryDirectory() as directory:
            exe = Path(directory) / "mpv.exe"
            wrapper = Path(directory) / "mpv.com"
            exe.touch()
            wrapper.touch()
            with patch(
                "npu_subtitles.media.shutil.which",
                side_effect=lambda name: str(exe if name == "mpv.exe" else wrapper),
            ):
                self.assertEqual(executable(None, "mpv"), str(exe.resolve()))
                self.assertEqual(executable(str(wrapper), "mpv"), str(wrapper.resolve()))


if __name__ == "__main__":
    unittest.main()
