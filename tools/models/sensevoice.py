"""Download a pinned community SenseVoice FP16 IR, without executing remote code."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from npu_subtitles.paths import ROOT

REPO = "apinge/sensevoice-small-ov"
REVISION = "942b912ad020e94545a9e0dcb95cee645aac51e5"
FILES = [
    "openvino_model.xml",
    "openvino_model.bin",
    "openvino_detokenizer.xml",
    "openvino_detokenizer.bin",
    "am.mvn",
    "config.yaml",
]


def main() -> None:
    p = argparse.ArgumentParser(description="Download SenseVoice Small OpenVINO FP16 (~472 MB)")
    p.add_argument("--output", type=Path, default=ROOT / "models/sensevoice-small-ov")
    args = p.parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    os.environ.setdefault("HF_XET_CACHE", str(ROOT / ".cache/huggingface/xet"))
    from huggingface_hub import snapshot_download

    print(f"Downloading community conversion {REPO}@{REVISION} -> {args.output}")
    snapshot_download(REPO, revision=REVISION, local_dir=str(args.output), allow_patterns=FILES)
    source = {
        "repo_id": REPO,
        "revision": REVISION,
        "source": f"https://huggingface.co/{REPO}",
        "original_model": "FunAudioLLM/SenseVoiceSmall",
        "precision": "FP16",
        "conversion": "Community conversion by apinge; not an Intel-official model",
    }
    (args.output / "source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    print(f"Done: {args.output.resolve()}")


if __name__ == "__main__":
    main()
