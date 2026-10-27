"""Download pinned, MIT-licensed Moonshine v2 incremental ONNX graphs.

Only model data is downloaded; remote Python code is never executed. The FP32
graphs are used because the optional ORT dynamic INT8 graphs are not NPU IRs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from npu_subtitles.paths import ROOT

MODELS = {
    "small": ("Masterx/moonshine-streaming-small-ONNX", "dafc41f7b33daa11c0b19866e90eb493e98650c1"),
    "medium": (
        "Masterx/moonshine-streaming-medium-ONNX",
        "3c59ebafc67091803df529732365a259b0cc7b8a",
    ),
}
FILES = [
    "frontend.onnx",
    "encoder.onnx",
    "adapter.onnx",
    "cross_kv.onnx",
    "decoder_kv.onnx",
    "streaming_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
    "generation_config.json",
    "README.md",
    "LICENSE",
]
CHUNK = 8 * 1024 * 1024


def file_hash(path: Path, algorithm: str = "sha256", *, git_blob: bool = False) -> str:
    digest = hashlib.new(algorithm)
    if git_blob:
        digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def verified_file(path: Path, metadata) -> bool:
    if not path.is_file() or path.stat().st_size != metadata.size:
        return False
    expected = getattr(metadata.lfs, "sha256", None)
    if expected:
        return file_hash(path) == expected
    # HF's blob_id is Git's SHA-1 including the blob header, not plain SHA-1.
    return file_hash(path, "sha1", git_blob=True) == metadata.blob_id


def download_ranges(repo: str, revision: str, files: list, output: Path, parts: Path) -> None:
    """Resume bounded ranges, then verify complete objects before publication."""
    import httpx2 as httpx

    parts.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    def fetch(metadata, index):
        start = index * CHUNK
        end = min(metadata.size, start + CHUNK) - 1
        length = end - start + 1
        part = parts / f"{metadata.rfilename}.{index:04d}.part"
        if part.is_file() and part.stat().st_size == length:
            return length
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{metadata.rfilename}?download=true&range={start}"
        for attempt in range(5):
            try:
                response = httpx.get(
                    url,
                    headers={"Range": f"bytes={start}-{end}"},
                    follow_redirects=True,
                    timeout=60,
                )
                response.raise_for_status()
                if metadata.size > CHUNK and (
                    response.status_code != 206
                    or response.headers.get("content-range")
                    != f"bytes {start}-{end}/{metadata.size}"
                ):
                    raise ValueError("Model server did not honor the requested byte range")
                if len(response.content) != length:
                    raise ValueError("Downloaded byte range has the wrong size")
                temporary = part.with_suffix(".tmp")
                temporary.write_bytes(response.content)
                temporary.replace(part)
                return length
            except Exception as exc:
                print(
                    f"Retry {metadata.rfilename} part {index}, attempt {attempt + 1}: {type(exc).__name__}",
                    flush=True,
                )
                if attempt == 4:
                    raise
                time.sleep(min(2**attempt, 5))

    total = sum(item.size for item in files)
    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [
            pool.submit(fetch, item, i)
            for item in files
            for i in range((item.size + CHUNK - 1) // CHUNK)
        ]
        for job in as_completed(jobs):
            done += job.result()
            print(f"Downloaded {done / 1e6:.1f}/{total / 1e6:.1f} MB", flush=True)
    for item in files:
        assembled = parts / f"{item.rfilename}.assembled"
        with assembled.open("wb") as stream:
            for index in range((item.size + CHUNK - 1) // CHUNK):
                stream.write((parts / f"{item.rfilename}.{index:04d}.part").read_bytes())
        if not verified_file(assembled, item):
            raise RuntimeError(f"Downloaded object failed hash verification: {item.rfilename}")
        assembled.replace(output / item.rfilename)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Moonshine v2 incremental ONNX (Small ~497 MB; Medium ~984 MB)"
    )
    parser.add_argument(
        "--size",
        "--model",
        choices=tuple(MODELS),
        default="small",
        help="Model size; default keeps the existing Small model",
    )
    parser.add_argument(
        "--output", type=Path, help="Destination; defaults to models/moonshine-streaming-<size>"
    )
    args = parser.parse_args()
    repo, revision = MODELS[args.size]
    output = (args.output or ROOT / f"models/moonshine-streaming-{args.size}").resolve()
    other = "medium" if args.size == "small" else "small"
    if output == (ROOT / f"models/moonshine-streaming-{other}").resolve():
        raise ValueError(f"Refusing to overwrite the default {other} model directory")
    receipt = output / "source.json"
    if receipt.is_file():
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        if previous.get("repo_id") != repo or previous.get("revision") != revision:
            raise ValueError("Destination contains a different model revision; use a new directory")
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    os.environ.setdefault("HF_XET_CACHE", str(ROOT / ".cache/huggingface/xet"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError("Model server returned a different revision")
    metadata = {item.rfilename: item for item in info.siblings}
    missing = []
    for filename in FILES:
        if filename not in metadata:
            raise RuntimeError(f"Pinned model revision is missing required file {filename}")
        if not verified_file(output / filename, metadata[filename]):
            missing.append(metadata[filename])
    if missing:
        download_ranges(
            repo, revision, missing, output, ROOT / ".cache/moonshine-ranges" / args.size / revision
        )
    checksums = {}
    for filename in FILES:
        if not verified_file(output / filename, metadata[filename]):
            raise RuntimeError(f"Hash mismatch for {filename}")
        checksums[filename] = file_hash(output / filename)
    source = {
        "repo_id": repo,
        "revision": revision,
        "license": "MIT",
        "variant": args.size,
        "base_model": f"moonshine-ai/moonshine-streaming-{args.size}",
        "conversion": "Community five-graph incremental ONNX; local OpenVINO NPU adaptation",
        "lfs_sha256_verified": True,
        "git_blob_hashes_verified": True,
        "sha256": checksums,
        "total_bytes": sum(metadata[name].size for name in FILES),
        "files": {
            name: {
                "size": metadata[name].size,
                "lfs_sha256": getattr(metadata[name].lfs, "sha256", None),
                "git_blob_id": metadata[name].blob_id,
            }
            for name in FILES
        },
    }
    (output / "source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    print(f"Downloaded and verified: {output}")


if __name__ == "__main__":
    main()
