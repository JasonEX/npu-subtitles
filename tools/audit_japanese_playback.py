"""Audit live Japanese logs without changing the pipeline or reference text."""

import argparse
from collections import Counter
import json
from pathlib import Path

from npu_subtitles.paths import ROOT
from npu_subtitles.devices import require_npu
from tools.validate_japanese_npu import normalized, edit_distance
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--group", choices=["preflight", "acceptance"], default="acceptance")
    parser.add_argument("--lifecycle", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.log.read_text(encoding="utf-8").splitlines()]
    ready = next(row for row in rows if row["event"] == "models_ready")
    require_npu(ready["asr_execution_devices"], "ASR")
    require_npu(ready["translation_details"]["execution_devices"], "pivot")
    commits = {
        (row["generation"], row["sentence_id"]): row
        for row in rows
        if row["event"] == "caption_state" and row.get("reason") == "sentence_committed"
    }
    displays = [
        row for row in rows if row["event"] in {"bilingual_displayed", "japanese_displayed"}
    ]
    translated = [
        row for row in rows if row["event"] == "worker_result" and row.get("kind") == "translation"
    ]
    finals = [
        update
        for row in rows
        if row["event"] == "worker_result" and row.get("kind") == "asr"
        for update in row["updates"]
        if update["is_final"]
    ]
    violations, owned, shown = [], set(), set()
    for key, commit in commits.items():
        for identity in commit["word_ids"]:
            owner = (commit["generation"], identity)
            if owner in owned:
                violations.append("duplicate committed occurrence")
            owned.add(owner)
        if commit["source_end"] > commit["sealed_until_at_commit"] + 1e-5:
            violations.append("unsealed source committed")
    epoch = 0
    for row in rows:
        if row["event"] == "stream_reset":
            epoch = row["generation"]
        if row["event"] in {"bilingual_displayed", "japanese_displayed"}:
            key = (row["generation"], row["sentence_id"])
            if row["generation"] != epoch:
                violations.append("stale display epoch")
            if key in shown:
                violations.append("same committed card displayed twice")
            shown.add(key)
            if key not in commits or row["text"] != commits[key]["text"]:
                violations.append("display text changed")
            if row["position"] < row["due_at"] - 1e-5:
                violations.append("early display")
            if row["position"] >= row["expires_at"]:
                violations.append("expired display")
        if row["event"] == "audio_submitted" and row["audio_end"] > row["read_limit"] + 1e-6:
            violations.append("audio beyond read horizon")

    def distribution(values):
        return (
            dict(
                count=len(values),
                median=float(np.median(values)),
                p95=float(np.percentile(values, 95)),
                maximum=max(values),
            )
            if values
            else dict(count=0)
        )

    errors = [row for row in rows if row["event"] == "error"]
    finished = [row for row in rows if row["event"] == "finished"]
    report = dict(
        log=str(args.log.resolve()),
        finished=finished,
        errors=errors,
        contract_violations=violations,
        asr_execution_devices=ready["asr_execution_devices"],
        translation_execution_devices=ready["translation_details"]["execution_devices"],
        committed=len(commits),
        displayed=len(displays),
        bilingual=sum(bool(row["chinese"]) for row in displays),
        japanese_only=sum(not row["chinese"] for row in displays),
        commit_reasons=dict(Counter(row["boundary_reason"] for row in commits.values())),
        event_counts=dict(Counter(row["event"] for row in rows)),
        translation_seconds=distribution([row["details"]["elapsed_seconds"] for row in translated]),
        bilingual_delay_from_ctc_emission_seconds=distribution(
            [row["position"] - row["source_start"] for row in displays if row["chinese"]]
        ),
        worker_peak_working_set_bytes=max(
            (row.get("memory", {}).get("os_peak_working_set_bytes", 0) for row in rows), default=0
        ),
        captions=[
            dict(
                source=row["text"],
                chinese=row["chinese"],
                generation=row["generation"],
                position=row["position"],
            )
            for row in displays
        ],
    )
    if not args.lifecycle:
        selection = json.loads(
            (ROOT / "outputs/validation/japanese-npu/fleurs/selection.json").read_text(
                encoding="utf-8"
            )
        )
        reference = normalized(
            "".join(
                row.get("raw_transcription", row["transcription"])
                for row in selection["rows"]
                if row["group"] == args.group
            )
        )
        asr_text = normalized("".join(row["text"] for row in finals))
        displayed_text = normalized("".join(row["text"] for row in displays))
        report.update(
            reference_characters=len(reference),
            final_asr_errors=edit_distance(reference, asr_text),
            displayed_source_errors=edit_distance(reference, displayed_text),
            final_asr_cer=edit_distance(reference, asr_text) / len(reference),
            displayed_source_cer=edit_distance(reference, displayed_text) / len(reference),
            final_equals_displayed=asr_text == displayed_text,
        )
    path = args.log.with_suffix(".audit.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"captions", "event_counts", "finished", "log"}
            },
            ensure_ascii=False,
        )
    )
    if errors or violations or not finished:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
