"""Download pinned Mozilla en→zh-Hans Marian weights used by MTranServer v5.

Conversion uses CPU graph construction only; it never executes a neural model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from npu_subtitles.paths import ROOT

MTRAN_COMMIT = "ea3e8b1523c70115d16f2d48027daaf9caeb3e8e"
BASE_URL = (
    "https://firefox-settings-attachments.cdn.mozilla.net/main-workspace/translations-models-v2/"
)
FILES = (
    (
        "model.enzh.intgemm.alphas.bin",
        "321191b6-2c42-4bca-84ec-b7b074b67bd6.zst",
        "f0c96e546b0af7c1fe84647d450b1eba0b0591c167e1ead135b334e4b54f6e08",
        "4e5accc141373565ddc8fa1565bceaa8d0c3482a82cab8131c719ebcc6c2157c",
    ),
    (
        "srcvocab.enzh.spm",
        "83278466-fec4-4944-a99d-c2372fa9de88.zst",
        "ec495ad8f0c80ba4eae5ac3b9a8e2ccb4a3a7becb1992efcfea5f4653ffdece1",
        "bd9b65504acc6d9726dd281f7defc2adb7c2c22d0688fe2f84697de25197c8c5",
    ),
    (
        "trgvocab.enzh.spm",
        "84236654-1b66-42fc-9ba2-74c37f27e5eb.zst",
        "25c6c984a7556cb6478b9f1748ff68d8fd433c0e7d472bfdbd4d64a2dffa21f1",
        "aded6993c36e440284d11cec3f6b8aef9c0e43188a772d80be342a713adf223d",
    ),
)


def _download_bytes(url: str) -> bytes:
    # Bounded range reads also avoid stalled large CDN responses on some proxies.
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
        size = int(response.headers["Content-Length"])
    chunk = 4 * 1024 * 1024

    def part(start: int) -> bytes:
        end = min(size, start + chunk) - 1
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(end - start + 2)
            if len(data) != end - start + 1:
                raise ValueError("Model server returned an invalid byte range")
            return data

    with ThreadPoolExecutor(max_workers=4) as pool:
        return b"".join(pool.map(part, range(0, size, chunk)))


def download(directory: Path) -> None:
    import zstandard

    directory.mkdir(parents=True, exist_ok=True)
    receipts = []
    for name, location, archive_hash, raw_hash in FILES:
        target = directory / name
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != raw_hash:
            print(f"Downloading {name}", flush=True)
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
            {
                "file": name,
                "url": BASE_URL + location,
                "sha256": raw_hash,
                "compressed_sha256": archive_hash,
                "bytes": target.stat().st_size,
            }
        )
    license_file = ROOT / "third_party/Mozilla-Translations-LICENSE"
    shutil.copyfile(license_file, directory / "LICENSE")
    (directory / "source.json").write_text(
        json.dumps(
            {
                "source": "Mozilla firefox-translations-models-v2",
                "version": "3.0",
                "direction": "en→zh-Hans",
                "mtran_reference_commit": MTRAN_COMMIT,
                "model_license": "MPL-2.0",
                "license_source": "https://github.com/mozilla/firefox-translations-models/blob/e7957fc407441a5e3e35bbcbf9d60d9b35764618/LICENSE",
                "matching_model_metadata": "https://github.com/mozilla/firefox-translations-models/blob/e7957fc407441a5e3e35bbcbf9d60d9b35764618/models/base-memory/enzh/metadata.json",
                "files": receipts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models/mtran-en-zh-npu")
    parser.add_argument("--source-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=384)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    download(args.output)
    if not args.download_only:
        from tools.models.marian_export import convert_model

        convert_model(args.output, args.source_length, args.max_target_length)


if __name__ == "__main__":
    main()
