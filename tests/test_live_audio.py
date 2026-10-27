"""Sample alignment across container packet seeks, using local FFmpeg only."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import soundfile as sf
from npu_subtitles.playback import AudioReader

FFMPEG = shutil.which("ffmpeg")


@unittest.skipUnless(FFMPEG, "FFmpeg is required for the container seek regression")
class AudioSliceTests(unittest.TestCase):
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
