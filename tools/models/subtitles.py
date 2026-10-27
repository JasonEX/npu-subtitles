"""Prepare the fixed additional subtitle models; no model search or playback download."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from tools.models.mtran import _download_bytes
from npu_subtitles.translation.languages import EXTRA_SUBTITLE_LANGUAGES

from npu_subtitles.paths import ROOT

MANIFEST = ROOT / "third_party/mozilla-subtitle-models.json"


def verify(language: str, directory: Path):
    """One-time device readiness check for the already quality-reviewed weights.

    This is not another quality benchmark. It makes a fresh installation usable
    without downloading FLORES or running the development evaluation suite.
    """
    from npu_subtitles.translation.subtitles import SubtitleTranslator
    from npu_subtitles.process import InferenceLease

    samples = dict(
        ko="지금 몇 시입니까?",
        es="¿Qué hora es ahora?",
        fr="Quelle heure est-il maintenant ?",
        de="Wie spät ist es jetzt?",
        pt="Que horas são agora?",
        it="Che ore sono adesso?",
        ru="Который сейчас час?",
    )
    marker = directory / "npu-validation.json"
    marker.unlink(missing_ok=True)
    with InferenceLease():
        translator = SubtitleTranslator(language)
        try:
            output = translator.translate(samples[language], "en", lambda: False)
            proof = dict(
                status="passed",
                method="Fixed supported weights: local NPU readiness smoke, not a quality score",
                source_language=language,
                output=output,
                execution_devices=dict(
                    source_en=translator.pivot.execution_devices,
                    en_zh=translator.chinese.execution_devices,
                ),
            )
        finally:
            translator.close()
    files = [
        "encoder.xml",
        "encoder.bin",
        "decoder.xml",
        "decoder.bin",
        "config.json",
        "source.json",
        f"vocab.{language}en.spm",
    ]
    proof["source_sha256"] = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in files
    }
    marker.write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{language}->en->zh ready on NPU", flush=True)


def download(language: str, *, export=True, force_verify=False):
    import zstandard

    if language not in EXTRA_SUBTITLE_LANGUAGES:
        raise ValueError("Language is outside the fixed subtitle support scope")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entry = manifest["models"][language]
    directory = ROOT / f"models/mtran-{language}-en-npu"
    directory.mkdir(parents=True, exist_ok=True)
    changed = False
    for item in entry["files"]:
        path = directory / item["name"]
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]:
            continue
        changed = True
        print(f"Downloading fixed {language}->en: {item['name']}", flush=True)
        packed = _download_bytes(item["url"])
        if hashlib.sha256(packed).hexdigest() != item["compressed_sha256"]:
            raise ValueError(f"Compressed checksum mismatch: {item['name']}")
        raw = zstandard.ZstdDecompressor().decompress(packed, max_output_size=item["bytes"])
        if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"Source checksum mismatch: {item['name']}")
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(raw)
        temporary.replace(path)
    receipt = dict(
        source=manifest["source"],
        revision=manifest["revision"],
        direction=f"{language}->en",
        architecture=entry["architecture"],
        version=entry["version"],
        model_license=manifest["license"],
        metadata_url=entry["metadata_url"],
        files=entry["files"],
    )
    (directory / "source.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    shutil.copyfile(ROOT / "third_party/Mozilla-Translations-LICENSE", directory / "LICENSE")
    required = ("encoder.xml", "encoder.bin", "decoder.xml", "decoder.bin", "config.json")
    if changed or not all((directory / name).is_file() for name in required):
        # Rebuilt weights must pass validation again before automatic routing.
        (directory / "npu-validation.json").unlink(missing_ok=True)
        if export:
            from tools.models.marian_export import convert_model

            convert_model(directory, pair=f"{language}-en")
        elif changed:
            # Do not let a later preparation reuse graphs from previous weights.
            for name in required:
                (directory / name).unlink(missing_ok=True)
    if export and (force_verify or not (directory / "npu-validation.json").is_file()):
        verify(language, directory)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=EXTRA_SUBTITLE_LANGUAGES,
        default=list(EXTRA_SUBTITLE_LANGUAGES),
    )
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Repeat the bounded NPU readiness check for prepared models",
    )
    args = parser.parse_args()
    for lang in dict.fromkeys(args.languages):
        download(lang, export=not args.download_only, force_verify=args.verify)


if __name__ == "__main__":
    main()
