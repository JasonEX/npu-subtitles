"""Mozilla/MTran Marian encoder and single-token SSRU decoder, strictly on NPU.

Graph equations follow MTranServer's Apache-2.0-licensed converter at commit
ea3e8b1523c70115d16f2d48027daaf9caeb3e8e (src/converter/graph.ts).
This adapter replaces its dynamic ONNX Loop with externally managed recurrent
state and bounded static shapes. SentencePiece, argmax and state copies are host
control operations; embeddings, attention, SSRU, FFN and projection are NPU ops.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

from npu_subtitles.translation.text import prepare_translation
from npu_subtitles.translation.languages import MARIAN_PAIRS
from npu_subtitles.paths import ROOT


class TranslationExpired(TimeoutError):
    """A live caption became obsolete; no partial translation is returned."""

    def __init__(self, stage: str):
        self.stage = stage
        super().__init__(f"Translation deadline expired at {stage}")


class TranslationCancelled(TranslationExpired):
    """A newer caption invalidated this request; no partial result is returned."""

    def __init__(self, stage: str):
        self.stage = stage
        TimeoutError.__init__(self, f"Translation cancelled at {stage}")


class TranslationRejected(RuntimeError):
    """This sentence exceeds model bounds; retain source text and keep playing."""


class MTranNPUTranslator:
    def __init__(
        self,
        model_dir: str | Path = ROOT / "models/mtran-en-zh-npu",
        cache_dir: str | Path | None = None,
        *,
        person_names: Iterable[str] | Mapping[str, str] = (),
    ):
        import openvino as ov
        import sentencepiece as spm

        self.person_names = (
            dict(person_names) if isinstance(person_names, Mapping) else tuple(person_names)
        )
        self._cancelled: Callable[[], bool] | None = None
        prepare_translation("", person_names=self.person_names)

        self.directory = Path(model_dir).resolve()
        self.config = json.loads((self.directory / "config.json").read_text(encoding="utf-8"))
        if self.config.get("format") != "mtran-npu-static-v1":
            raise ValueError("Unknown MTran NPU model format")
        self.source_language = self.config.get("source_language", "en")
        self.target_language = self.config.get("target_language", "zh-Hans")
        pair = f"{self.source_language}-{self.target_language}"
        if (self.source_language, self.target_language) != ("en", "zh-Hans") and not (
            self.target_language == "en" and pair in MARIAN_PAIRS
        ):
            raise ValueError("Unsupported Marian translation direction")
        if self.source_language != "en" and self.person_names:
            raise ValueError("Person-name preservation requires an English source")
        self.source = spm.SentencePieceProcessor(
            model_file=str(self.directory / self.config.get("source_vocab", "srcvocab.enzh.spm"))
        )
        self.target = spm.SentencePieceProcessor(
            model_file=str(self.directory / self.config.get("target_vocab", "trgvocab.enzh.spm"))
        )
        self.core = ov.Core()
        if "NPU" not in self.core.available_devices:
            raise RuntimeError("Intel NPU is unavailable; CPU fallback is disabled")
        cache = Path(cache_dir) if cache_dir else self.directory / "npu-cache"
        cache.mkdir(parents=True, exist_ok=True)
        properties = {"CACHE_DIR": str(cache.resolve()), "PERFORMANCE_HINT": "LATENCY"}
        self.compiled = {}
        self.execution_devices = {}
        for name in ("encoder", "decoder"):
            print(f"Compiling MTran {name} on NPU", flush=True)
            compiled = self.core.compile_model(
                str(self.directory / f"{name}.xml"), "NPU", properties
            )
            reported_devices = compiled.get_property("EXECUTION_DEVICES")
            devices = (
                [reported_devices] if isinstance(reported_devices, str) else list(reported_devices)
            )
            if not devices or any(not str(device).startswith("NPU") for device in devices):
                raise RuntimeError(f"MTran {name} executed on unexpected devices {devices}")
            self.compiled[name] = compiled
            self.execution_devices[name] = devices
        self.encoder = self.compiled["encoder"].create_infer_request()
        self.decoder = self.compiled["decoder"].create_infer_request()
        self.last_result: dict[str, Any] = {}

    def details(self) -> dict[str, Any]:
        return {
            "backend": "mtran-marian-static-openvino",
            "model": str(self.directory),
            "device": "NPU",
            "execution_devices": self.execution_devices,
            "cpu_fallback": False,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "source_length": self.config["source_length"],
            "max_target_length": self.config["max_target_length"],
            "weight_format": self.config["weight_format"],
            "decoding": "greedy",
            "person_names": self.person_names,
            "input_preparation": "whitespace plus explicitly configured leading names",
            "deadline": "monotonic; checked before/after encoder and every decoder token",
            "cancellation": "request callback checked at every deadline checkpoint",
        }

    def _check_deadline(self, deadline: float | None, stage: str) -> None:
        if self._cancelled is not None and self._cancelled():
            self.last_result.update(status="cancelled", cancelled_stage=stage)
            raise TranslationCancelled(stage)
        if deadline is not None and time.monotonic() >= deadline:
            self.last_result.update(status="expired", expired_stage=stage)
            raise TranslationExpired(stage)

    def translate(
        self,
        text: str,
        *,
        deadline: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> str:
        """Translate one committed source revision with cooperative cancellation.

        The instance owns one infer request and is intended for one worker.
        Cancellation is checked around NPU calls; it cannot interrupt a kernel
        already executing. No cancelled or expired partial Chinese is returned.
        """
        started = time.monotonic()
        self._cancelled = cancelled
        self.last_result = {
            "source": text,
            "normalized_source": text,
            "text_changes": [],
            "status": "started",
        }
        try:
            if cancelled is not None and not callable(cancelled):
                raise TypeError("Translation cancellation must be a callable or None")
            if deadline is not None and not np.isfinite(deadline):
                raise ValueError("Translation deadline must be a finite monotonic timestamp")
            self._check_deadline(deadline, "before_preparation")
            prepared = prepare_translation(
                text,
                person_names=self.person_names,
            )
            model_input = prepared.model_input
            prefix = prepared.output_prefix
            self.last_result.update(
                normalized_source=model_input, text_changes=prepared.changes, output_prefix=prefix
            )
            if not model_input:
                self._check_deadline(deadline, "before_return")
                self.last_result.update(status="completed", raw_translation="", translation="")
                return ""
            raw = self._translate_model(model_input, deadline=deadline)
            self._check_deadline(deadline, "before_return")
            result = prefix + raw
            self.last_result.update(status="completed", raw_translation=raw, translation=result)
            return result
        except TranslationExpired:
            raise
        except Exception as exc:
            self.last_result.update(status="failed", error=type(exc).__name__)
            raise
        finally:
            self._cancelled = None
            self.last_result["elapsed_seconds"] = time.monotonic() - started

    def _translate_model(self, text: str, *, deadline: float | None = None) -> str:
        if not text.strip():
            return ""
        ids = self.source.encode(text, out_type=int) + [self.source.eos_id()]
        length = int(self.config["source_length"])
        if len(ids) > length:
            raise TranslationRejected(
                f"MTran source contains {len(ids)} tokens, exceeding {length}; no input was truncated"
            )
        source_ids = np.full((length,), self.source.eos_id(), dtype=np.int32)
        source_ids[: len(ids)] = ids
        mask = np.full((1, 1, 1, length), -10000, dtype=np.float32)
        mask[..., : len(ids)] = 0
        self.last_result["source_tokens"] = len(ids)
        self._check_deadline(deadline, "before_encoder")
        self.encoder.infer({"input_ids": source_ids, "source_mask": mask})
        self._check_deadline(deadline, "after_encoder")
        state = {"source_mask": mask}
        for layer in range(1, self.config["decoder_depth"] + 1):
            for kind in ("key", "value"):
                name = f"{kind}_{layer}"
                state[name] = self.encoder.get_tensor(name).data.copy()
            state[f"cell_{layer}"] = np.zeros((1, 1, 1, self.config["dim"]), dtype=np.float32)
        generated = []
        maximum = min(3 * len(ids) + 1, self.config["max_target_length"])
        terminated = False
        for position in range(maximum):
            self._check_deadline(deadline, f"before_token_{position}")
            state["previous_token"] = np.asarray(
                [generated[-1] if generated else self.target.eos_id()], dtype=np.int32
            )
            state["position"] = np.asarray([position], dtype=np.int32)
            state["embedding_multiplier"] = np.asarray([1 if position else 0], dtype=np.float32)
            self.decoder.infer(state)
            self._check_deadline(deadline, f"after_token_{position}")
            logits = self.decoder.get_tensor("logits").data
            if not np.isfinite(logits).all():
                raise RuntimeError("MTran NPU produced non-finite logits")
            token = int(np.argmax(logits))
            if token == self.target.eos_id():
                terminated = True
                break
            generated.append(token)
            for layer in range(1, self.config["decoder_depth"] + 1):
                state[f"cell_{layer}"] = self.decoder.get_tensor(f"next_cell_{layer}").data.copy()
        result = self.target.decode(generated)
        self.last_result.update(
            source_tokens=len(ids),
            generated_tokens=len(generated),
            eos=terminated,
            truncated=not terminated,
        )
        if not terminated:
            raise TranslationRejected(
                "MTran NPU reached its output bound before EOS; no partial output was returned"
            )
        return result

    def close(self) -> None:
        self._cancelled = None
        self.encoder = self.decoder = None
        self.compiled.clear()
        self.core = None
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/mtran-en-zh-npu")
    parser.add_argument("--text", default="Don't open the door unless I tell you to.")
    args = parser.parse_args()
    translator = MTranNPUTranslator(args.model_dir)
    try:
        print(
            json.dumps(
                {
                    "details": translator.details(),
                    "source": args.text,
                    "translation": translator.translate(args.text),
                    "generation": translator.last_result,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        translator.close()


if __name__ == "__main__":
    main()
