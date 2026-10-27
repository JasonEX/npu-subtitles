"""Evaluate fixed public audio through Granite and Japanese pivot on NPU.

Recognition receives waveform only. Reference text is consulted after inference.
This is an offline component/ownership check, not a live playback latency test.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import unicodedata

from npu_subtitles.paths import ROOT
import soundfile as sf
from npu_subtitles.captions.language import join_units
from npu_subtitles.captions.sentences import GraniteSentenceAssembler
from npu_subtitles.captions.commit import CueCommitter
from npu_subtitles.inference import create_backends
from npu_subtitles.devices import require_npu


def normalized(text):
    return "".join(c.casefold() for c in unicodedata.normalize("NFKC", text) if c.isalnum())


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["preflight", "acceptance"], required=True)
    parser.add_argument(
        "--selection",
        type=Path,
        default=ROOT / "outputs/validation/japanese-npu/fleurs/selection.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/validation/japanese-npu")
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    asr, translator = create_backends(
        str(ROOT / "models/granite-speech-5.0-470m-turboctc"),
        str(ROOT / "models/mtran-en-zh-npu"),
        str(ROOT / ".cache/live-npu"),
        asr_backend="granite",
        source_language="ja",
    )
    report = dict(
        group=args.group,
        selection_sha256=hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        evaluation="fixed audio, offline incremental 0.8s chunks; not wall-clock live latency",
        asr=asr.details(),
        translation=translator.details(),
        rows=[],
    )
    require_npu(asr.execution_devices, "ASR")
    require_npu(translator.details()["execution_devices"], "translation")
    path = args.output / f"{args.group}-components.json"
    try:
        for row in selection["rows"]:
            if row["group"] != args.group:
                continue
            audio_path = Path(row["audio_path"])
            if hashlib.sha256(audio_path.read_bytes()).hexdigest() != row["audio_sha256"]:
                raise ValueError("Validation audio hash mismatch")
            audio, rate = sf.read(audio_path, dtype="float32")
            if rate != 16000 or audio.ndim != 1:
                raise ValueError("Expected original 16 kHz mono")
            asr.reset()
            assembler = GraniteSentenceAssembler(CueCommitter.sentence_end, source_language="ja")
            updates, proposals, finals = [], [], []
            started = time.monotonic()
            for offset in range(0, len(audio), 12800):
                chunk = audio[offset : offset + 12800]
                for update in asr.accept_audio(chunk, final=offset + len(chunk) == len(audio)):
                    updates.append(update)
                    proposals.extend(assembler.feed(update, time.monotonic(), 0.0))
                    if update["is_final"]:
                        finals.append(update["text"])
            proposals.extend(assembler.finalize(time.monotonic(), 0.0))
            asr_seconds = time.monotonic() - started
            captions = []
            for proposal in proposals:
                translated = translator.translate(proposal["text"], deadline=time.monotonic() + 1.5)
                captions.append(
                    dict(proposal, chinese=translated, translation=translator.last_result)
                )
            source = join_units(finals, "ja")
            segmented = join_units((p["text"] for p in proposals), "ja")
            owned = [identity for p in proposals for identity in p["word_ids"]]
            if source != segmented or len(owned) != len(set(owned)):
                raise AssertionError(
                    "Subtitle segmentation dropped or duplicated ASR text/ownership"
                )
            reference = row.get("raw_transcription", row["transcription"])
            ref, hyp = normalized(reference), normalized(source)
            result = dict(
                row_index=row["row_index"],
                seconds=row["seconds"],
                reference=reference,
                hypothesis=source,
                reference_characters=len(ref),
                character_errors=edit_distance(ref, hyp),
                asr_seconds=asr_seconds,
                captions=captions,
                updates=updates,
            )
            report["rows"].append(result)
            report["reference_characters"] = sum(r["reference_characters"] for r in report["rows"])
            report["character_errors"] = sum(r["character_errors"] for r in report["rows"])
            report["cer"] = report["character_errors"] / report["reference_characters"]
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                json.dumps(
                    dict(
                        row=row["row_index"],
                        seconds=row["seconds"],
                        asr_seconds=asr_seconds,
                        reference=reference,
                        hypothesis=source,
                        captions=len(captions),
                        errors=result["character_errors"],
                    ),
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        translator.close()
        asr.close()


if __name__ == "__main__":
    main()
