"""Sample alignment across container packet seeks, using local FFmpeg only."""

from pathlib import Path
import contextlib
import io
import json
import shutil
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
from npu_subtitles.playback import AudioReader

FFMPEG = shutil.which("ffmpeg")


@unittest.skipUnless(FFMPEG, "FFmpeg is required for the container seek regression")
class AudioSliceTests(unittest.TestCase):
    def test_plugin_seek_cancels_running_ffmpeg_and_recognizes_new_position(self):
        import mpv_plugin

        # Throttle only the old read so the seek deterministically interrupts a
        # live FFmpeg process. The replacement read uses the real decoder too.
        reading = threading.Event()
        processes = []
        popen = subprocess.Popen

        def start_ffmpeg(command, **kwargs):
            command = list(command)
            first = not processes
            if first:
                command.insert(command.index("-i"), "-re")
            process = popen(command, **kwargs)
            processes.append(process)
            if first:
                communicate = process.communicate

                def communicating(*args, **options):
                    reading.set()
                    return communicate(*args, **options)

                process.communicate = communicating
            return process

        class SeekingSession:
            def __init__(self, reader):
                self.reader = reader
                self.ipc = SimpleNamespace(events=[])
                self.position = 0.0
                self.sought = False
                self.overlays = []
                self.deadline = time.monotonic() + 10

            def current(self):
                if time.monotonic() > self.deadline:
                    raise TimeoutError("Plugin did not recover after seek")
                return not any("新位置" in text for text in self.overlays)

            def get_optional(self, name):
                if name == "duration":
                    return 40.0
                if reading.is_set() and not self.sought:
                    # Check the condition that distinguishes this from a seek
                    # before decoding starts or after the old read completes.
                    assert processes[0].poll() is None
                    self.sought = True
                    self.position = 20.0
                    self.ipc.events.append({"event": "seek"})
                return self.position

            def get(self, name):
                assert name == "pause"
                return False

            def audio_reader(self, track):
                return self.reader

            def status(self, *args, **kwargs):
                pass

            def message(self, name, text):
                self.overlays.append(text)

            def clear(self):
                self.message("npu-overlay", "")

        model = SimpleNamespace(
            execution_devices=["NPU"],
            generate=Mock(return_value=SimpleNamespace(
                chunks=[dict(start_ts=0.0, end_ts=3.0, text="新位置")]
            )),
        )
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "seek.wav"
            wave = np.full(40 * 16000, 0.1, dtype=np.float32)
            wave[20 * 16000:] = 0.2
            sf.write(source, wave, 16000, subtype="FLOAT")
            reader = AudioReader(FFMPEG, source, 0)
            session = SeekingSession(reader)
            output = io.StringIO()
            try:
                with (
                    patch("npu_subtitles.playback.subprocess.Popen", side_effect=start_ffmpeg),
                    patch("npu_subtitles.asr.sensevoice.SenseVoicePipeline", return_value=model),
                    patch.object(reader, "cancel_before", wraps=reader.cancel_before) as cancel,
                    contextlib.redirect_stdout(output),
                ):
                    mpv_plugin.chinese_loop(session, 0)
                self.assertEqual([call.args[0] for call in cancel.call_args_list], [1, 2])
                self.assertTrue(session.sought)
                self.assertEqual(len(processes), 2)
                self.assertNotEqual(processes[0].returncode, 0)
                self.assertIsNotNone(processes[0].returncode)
                self.assertEqual(processes[1].returncode, 0)
                model.generate.assert_called_once()
                np.testing.assert_allclose(model.generate.call_args.args[0], wave[320000:448000])
                entries = [json.loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual([entry["event"] for entry in entries], ["chinese_recognized"])
                self.assertEqual(entries[0]["cues"][0]["start"], 20.0)
                self.assertTrue(any("新位置" in text for text in session.overlays))
            finally:
                reader.close()
            self.assertTrue(all(process.poll() is not None for process in processes))

    def test_flac_container_seeks_deliver_exact_samples_without_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            wave = (np.sin(np.arange(round(101.6 * 16000)) * 0.173) * 0.1).astype(np.float32)
            source, video = directory / "source.wav", directory / "seek.mkv"
            sf.write(source, wave, 16000, subtype="FLOAT")
            subprocess.run(
                [
                    FFMPEG,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=s=32x32:r=5",
                    "-i",
                    str(source),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-c:a",
                    "flac",
                    "-t",
                    "101.6",
                    str(video),
                ],
                check=True,
            )
            reader = AudioReader(FFMPEG, video, 0)
            try:
                for start in [0.0, 0.8, 99.2, 100.0, 100.8]:
                    begin, end = round(start * 16000), round((start + 0.8) * 16000)
                    result = reader.submit(begin, end, start, 0, start + 0.8).result()
                    decoded = np.frombuffer(result["pcm"], dtype="<f4")
                    self.assertEqual(len(decoded), end - begin)
                    np.testing.assert_allclose(decoded, wave[begin:end], rtol=0, atol=1.2e-7)
            finally:
                reader.close()


if __name__ == "__main__":
    unittest.main()
