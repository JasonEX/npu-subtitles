"""One asset root, independent of the current working directory, and audio constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000
# Default RMS energy below which mono float audio is treated as silence.
SPEECH_RMS_THRESHOLD = 0.008
