"""Prepare pinned Mozilla Japanese -> English -> Simplified Chinese NPU models.

The small SSRU models share the existing Marian graph adapter. Preparation only
constructs graphs; all neural execution in the player is explicitly on NPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from tools.models.mtran import BASE_URL, _download_bytes, download as download_english

from npu_subtitles.paths import ROOT

REVISION = "e7957fc407441a5e3e35bbcbf9d60d9b35764618"
FILES = (
    (
        "model.jaen.intgemm.alphas.bin",
        "53a52e41-5abb-4705-9ba8-8dbc07614bdc.zst",
        "c0c9f2e48979f022e16271049d96dbef9aa28f8fddc2d1db8fa0c399dec903f7",
        "a9bf800679bba570520e1161d7b4fbfcb957add32ca35812134add85689752ad",
    ),
    (
        "vocab.jaen.spm",
        "2566c4c6-4b9f-4d94-8898-38dbda231813.zst",
        "f41f32e36f879643a47fedbd0616fa12f416da62e23af8620ee4fc7705217acc",
        "5cb217758bae05877bb3f0c2f612e4e7c1e4cb03c10db11f4a47098d7ae62919",
    ),
)


def download(directory: Path) -> None:
    import zstandard

    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / "source.json"
    receipts = []
    for name, location, archive_hash, raw_hash in FILES:
        target = directory / name
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != raw_hash:
            print(f"Downloading pinned Japanese translation file: {name}", flush=True)
            packed = _download_bytes(BASE_URL + location)
            if hashlib.sha256(packed).hexdigest() != archive_hash:
                raise ValueError(f"Compressed checksum mismatch: {name}")
            raw = zstandard.ZstdDecompressor().decompress(packed, max_output_size=100_000_000)
            if hashlib.sha256(raw).hexdigest() != raw_hash:
                raise ValueError(f"Decompressed checksum mismatch: {name}")
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(raw)
            temporary.replace(target)
        receipts.append(
            dict(
                file=name,
                url=BASE_URL + location,
                compressed_sha256=archive_hash,
                sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                bytes=target.stat().st_size,
            )
        )
    shutil.copyfile(ROOT / "third_party/Mozilla-Translations-LICENSE", directory / "LICENSE")
    receipt_path.write_text(
        json.dumps(
            dict(
                source="Mozilla firefox-translations-models",
                revision=REVISION,
                matching_model_metadata=f"https://github.com/mozilla/firefox-translations-models/blob/{REVISION}/models/base/jaen/metadata.json",
                direction="ja->en",
                version="3.0",
                model_license="MPL-2.0",
                files=receipts,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--japanese-output", type=Path, default=ROOT / "models/mtran-ja-en-npu")
    parser.add_argument("--english-output", type=Path, default=ROOT / "models/mtran-en-zh-npu")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if args.japanese_output.resolve() == args.english_output.resolve():
        parser.error("Japanese and English model stages require separate output directories")
    download(args.japanese_output)
    download_english(args.english_output)
    if not args.download_only:
        from tools.models.marian_export import convert_model

        convert_model(args.japanese_output, pair="ja-en")
        # Reuse the established English stage without rebuilding its graphs.
        if not all(
            (args.english_output / name).is_file()
            for name in ("encoder.xml", "encoder.bin", "decoder.xml", "decoder.bin", "config.json")
        ):
            convert_model(args.english_output)


if __name__ == "__main__":
    main()
