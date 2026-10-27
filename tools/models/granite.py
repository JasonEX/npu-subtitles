"""Prepare pinned Granite TurboCTC weights and a fixed-window OpenVINO graph.

The isolated ``tools/granite-export`` uv project is needed only for conversion.
Playback loads the saved graph on NPU; this utility never compiles a CPU/GPU fallback.

From the repository root, with the dedicated conversion environment:
    $env:UV_PROJECT_ENVIRONMENT = "$PWD\\.venv-granite-export"
    .venv-granite-export/Scripts/python.exe -m tools.models.granite --export-only
Use --download-only from the ordinary runtime environment to fetch weights, or
--export-only to verify and convert an already downloaded source without network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from tools.models.moonshine import download_ranges, file_hash, verified_file

from npu_subtitles.paths import ROOT

REPO = "ibm-granite/granite-speech-5.0-470m-turboctc"
REVISION = "1dd23f46f6d4e6eae2531982ba2e33220086dd97"
FILES = (
    "README.md",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "model.sig",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
EXPORT_VERSION = 1


def download_source(output: Path) -> dict:
    from huggingface_hub import HfApi

    source = output / "source"
    receipt = source / "source.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        if (previous.get("repo_id"), previous.get("revision")) != (REPO, REVISION):
            raise ValueError("Destination contains a different model revision")
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise RuntimeError("Model server returned a different revision")
    metadata = {item.rfilename: item for item in info.siblings}
    missing = [metadata[name] for name in FILES if not verified_file(source / name, metadata[name])]
    if missing:
        download_ranges(REPO, REVISION, missing, source, ROOT / ".cache/granite-ranges" / REVISION)
    for name in FILES:
        if not verified_file(source / name, metadata[name]):
            raise RuntimeError(f"Model object failed hash verification: {name}")
    provenance = {
        "repo_id": REPO,
        "revision": REVISION,
        "license": "Apache-2.0",
        "lfs_sha256_verified": True,
        "git_blob_hashes_verified": True,
        "total_bytes": sum(metadata[name].size for name in FILES),
        "sha256": {name: file_hash(source / name) for name in FILES},
    }
    receipt.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def export_static(output: Path, provenance: dict, window_seconds: float) -> Path:
    import importlib.metadata

    print("Loading isolated conversion dependencies...", flush=True)
    import numpy as np
    import openvino as ov
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    torch.set_num_threads(4)
    source = output / "source"
    suffix = f"{window_seconds:g}".replace(".", "p")
    destination = output / f"npu-static-{suffix}s"
    destination.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "revision": REVISION,
                "source_hashes": provenance["sha256"],
                "export_version": EXPORT_VERSION,
                "window_seconds": window_seconds,
                "openvino": ov.__version__,
                "transformers": importlib.metadata.version("transformers"),
                "torch": torch.__version__,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    manifest_path = destination / "artifact.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            previous.get("fingerprint") == fingerprint
            and all(
                (destination / name).is_file() and file_hash(destination / name) == expected
                for name, expected in previous.get("sha256", {}).items()
            )
            and previous.get("sha256")
        ):
            print(f"Verified existing static graph: {destination}", flush=True)
            return destination
        raise RuntimeError(
            "Static output differs from the requested export; choose a fresh output directory"
        )

    print("Loading pinned processor and saving runtime coefficients...", flush=True)
    processor = AutoProcessor.from_pretrained(
        source, local_files_only=True, trust_remote_code=False
    )
    extractor = processor.feature_extractor
    np.savez(
        destination / "runtime_preprocessor.npz",
        mel_filters=extractor.mel_filters.mel_scale.fb.detach().numpy(),
        window=extractor.mel_filters.spectrogram.window.detach().numpy(),
    )
    preprocessing = {
        name: getattr(extractor, name)
        for name in (
            "sampling_rate",
            "n_fft",
            "win_length",
            "hop_length",
            "num_mel_bins",
            "delta_win_length",
            "logmel_floor_db",
            "frame_stacking",
            "delta_expansion",
        )
    }
    preprocessing.update(
        {
            "center": True,
            "pad_mode": "reflect",
            "power": 2.0,
            "mask_rule": "ceil(floor(num_samples / hop_length) / frame_stacking)",
            "output_mask_rule": "floor(input_valid_frames / 4)",
        }
    )
    (destination / "preprocessor.json").write_text(
        json.dumps(preprocessing, indent=2), encoding="utf-8"
    )
    print("Loading pinned model weights for graph conversion...", flush=True)
    model = AutoModelForCTC.from_pretrained(
        source, dtype=torch.float32, local_files_only=True, trust_remote_code=False
    ).eval()
    # These zeros are a conversion example, never a playback inference fallback.
    samples = round(window_seconds * 16000)
    features = processor([np.zeros(samples, dtype=np.float32)], sampling_rate=16000)
    example = {
        "input_features": features["input_features"],
        "attention_mask": features["attention_mask"],
    }
    print(
        "Export inputs:", {name: list(value.shape) for name, value in example.items()}, flush=True
    )

    class EncoderCTC(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, input_features, attention_mask):
            return self.wrapped(input_features=input_features, attention_mask=attention_mask).logits

    converted = ov.convert_model(
        EncoderCTC(model),
        example_input=example,
        input=[(name, list(value.shape)) for name, value in example.items()],
    )
    for port, name in zip(converted.inputs, example):
        port.get_tensor().set_names({name})
    converted.output(0).get_tensor().set_names({"logits"})
    ov.save_model(converted, destination / "model.xml", compress_to_fp16=True)
    manifest = {
        "architecture": "GraniteSpeech5ForCTC",
        "repo_id": REPO,
        "revision": REVISION,
        "export_version": EXPORT_VERSION,
        "fingerprint": fingerprint,
        "window_seconds": window_seconds,
        "sample_rate": 16000,
        "samples": samples,
        "inputs": {
            p.get_any_name(): {"shape": list(p.shape), "dtype": p.element_type.get_type_name()}
            for p in converted.inputs
        },
        "outputs": {
            p.get_any_name(): {"shape": list(p.shape), "dtype": p.element_type.get_type_name()}
            for p in converted.outputs
        },
        "blank_token_id": int(model.config.pad_token_id),
        "weights": "FP16 compressed, FP32 model arithmetic",
        "special_token_ids": processor.tokenizer.all_special_ids,
        "preprocessing": preprocessing,
        "source_subdir": "../source",
        "runtime_device_required": "NPU",
        "openvino": ov.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "torch": torch.__version__,
        "source_sha256": provenance["sha256"],
        "sha256": {
            name: file_hash(destination / name)
            for name in ("model.xml", "model.bin", "runtime_preprocessor.npz", "preprocessor.json")
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved static graph (not yet NPU-validated): {destination}", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and export pinned Granite TurboCTC for NPU playback",
        epilog="Conversion requires the separate tools/granite-export uv project; playback does not require PyTorch or Transformers.",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "models/granite-speech-5.0-470m-turboctc"
    )
    parser.add_argument("--window-seconds", type=float, default=12.8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download-only", action="store_true")
    mode.add_argument(
        "--export-only",
        action="store_true",
        help="Verify the local receipt and export without network access",
    )
    args = parser.parse_args()
    if not 1 <= args.window_seconds <= 30:
        parser.error("window-seconds must be between 1 and 30")
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    output = args.output.resolve()
    if args.export_only:
        source = output / "source"
        provenance = json.loads((source / "source.json").read_text(encoding="utf-8"))
        if (provenance.get("repo_id"), provenance.get("revision")) != (REPO, REVISION):
            raise ValueError("Local source receipt has a different model revision")
        print("Verifying pinned local source hashes...", flush=True)
        for name in FILES:
            expected = provenance.get("sha256", {}).get(name)
            if (
                not expected
                or not (source / name).is_file()
                or file_hash(source / name) != expected
            ):
                raise RuntimeError(f"Local source hash mismatch: {name}")
    else:
        print("Verifying/downloading pinned source objects...", flush=True)
        provenance = download_source(output)
    if not args.download_only:
        export_static(output, provenance, args.window_seconds)


if __name__ == "__main__":
    main()
