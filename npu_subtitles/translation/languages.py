"""Fixed, bounded subtitle translation coverage; no automatic model downloads."""

from pathlib import Path
import json

# Chosen for common film/subtitle coverage and reuse of the same Marian graphs.
# This list does not extend speech recognition or imply a popularity ranking.
EXTRA_SUBTITLE_LANGUAGES = ("ko", "es", "fr", "de", "pt", "it", "ru")
PIVOT_LANGUAGES = ("ja", *EXTRA_SUBTITLE_LANGUAGES)
SUBTITLE_LANGUAGES = ("en", *PIVOT_LANGUAGES)
MARIAN_PAIRS = frozenset({"en-zh", *(f"{lang}-en" for lang in PIVOT_LANGUAGES)})


def available_subtitle_languages(root: Path) -> set[str]:
    """Existing routes stay compatible; new routes require prepared graphs.

    Check small file metadata, never hash/read all weights on a playback path.
    Actual graph devices are verified by the adapter when it loads.
    """
    available = {"en", "ja"}
    for lang in EXTRA_SUBTITLE_LANGUAGES:
        directory = root / f"models/mtran-{lang}-en-npu"
        required = (
            "encoder.xml",
            "encoder.bin",
            "decoder.xml",
            "decoder.bin",
            "config.json",
            f"vocab.{lang}en.spm",
            "npu-validation.json",
        )
        if all((directory / name).is_file() for name in required):
            try:
                if (
                    json.loads((directory / "npu-validation.json").read_text(encoding="utf-8")).get(
                        "status"
                    )
                    == "passed"
                ):
                    available.add(lang)
            except (OSError, ValueError, AttributeError):
                pass  # interrupted preparation is unavailable, not an English guess
    return available
