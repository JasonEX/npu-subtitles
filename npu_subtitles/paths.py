"""One asset root, independent of the current working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000
