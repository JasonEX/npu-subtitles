"""SenseVoice Small with explicit OpenVINO NPU execution and CTC subtitle times.

Uses the pinned community FP16 IR conversion in python -m tools.models.sensevoice.
The neural model runs on the requested device; fbank, CTC decoding and text
detokenization run on CPU. Timestamps approximate CTC emission frames (60 ms),
not forced-aligned word boundaries. See the README for accuracy limitations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Callable

import numpy as np

from npu_subtitles.paths import SAMPLE_RATE

MAX_SECONDS = 30
STATIC_FRAMES = 500
LANGUAGE_IDS = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}


def load_cmvn(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("<LearnRateCoef>"):
            values.append(np.array(line.split("[", 1)[1].split("]", 1)[0].split(), np.float32))
    if len(values) != 2 or any(value.size != 560 for value in values):
        raise ValueError(f"Expected two 560-dimensional CMVN vectors in {path}")
    return values[0], values[1]


def extract_features(audio: np.ndarray, cmvn: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """FunASR-compatible fbank -> LFR 7/6 -> CMVN, with deterministic dither=0."""
    import kaldi_native_fbank as knf

    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = SAMPLE_RATE
    opts.frame_opts.dither = 0
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.frame_shift_ms = 10.0
    opts.frame_opts.frame_length_ms = 25.0
    opts.frame_opts.snip_edges = True
    opts.mel_opts.num_bins = 80
    opts.mel_opts.debug_mel = False
    opts.energy_floor = 0
    if audio.size < 400:
        audio = np.pad(audio, (0, 400 - audio.size))
    fb = knf.OnlineFbank(opts)
    fb.accept_waveform(SAMPLE_RATE, (audio * 32768).tolist())
    fb.input_finished()
    mel = np.stack([fb.get_frame(index) for index in range(fb.num_frames_ready)])
    frame_count = int(np.ceil(len(mel) / 6))
    padded = np.pad(mel, ((3, 6), (0, 0)), mode="edge")
    lfr = np.stack([padded[index * 6 : index * 6 + 7].reshape(-1) for index in range(frame_count)])
    return ((lfr + cmvn[0]) * cmvn[1]).astype(np.float32)


@dataclass(frozen=True)
class TokenSpan:
    token: int
    start: float
    end: float


def ctc_spans(logits: np.ndarray, valid_frames: int, duration: float) -> list[TokenSpan]:
    """Collapse adjacent repeats before dropping blanks, preserving true repeats."""
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("Expected SenseVoice logits shaped [1, time, vocabulary]")
    labels = logits[0, : max(0, valid_frames)].argmax(axis=-1)
    spans: list[TokenSpan] = []
    previous = None
    # Four task frames precede the speech frames: language/emotion/event/ITN.
    for frame in range(4, len(labels)):
        token = int(labels[frame])
        start = max(0.0, (frame - 4) * 0.06 - 0.03)
        end = min(duration, (frame - 4 + 1) * 0.06 - 0.03)
        if token == previous and token != 0 and spans:
            old = spans[-1]
            spans[-1] = TokenSpan(token, old.start, end)
        elif token != 0 and start < duration and end > start:
            spans.append(TokenSpan(token, start, end))
        previous = token
    return spans


def subtitle_chunks(
    spans: list[TokenSpan], decode: Callable[[list[int]], str], duration: float
) -> list[dict]:
    """Group CTC spans at pauses, punctuation and word boundaries, without overlap."""
    chunks: list[dict] = []
    active: list[TokenSpan] = []
    active_text = ""

    def flush(next_start: float | None = None) -> None:
        nonlocal active, active_text
        text = " ".join(active_text.split())
        if active and any(char.isalnum() for char in text):
            start = active[0].start
            end = min(duration, active[-1].end + 0.25)
            if next_start is not None:
                end = min(end, next_start)
            if chunks:
                chunks[-1]["end_ts"] = min(chunks[-1]["end_ts"], start)
            if end > start:
                chunks.append({"start_ts": start, "end_ts": end, "text": text})
        active, active_text = [], ""

    for span in spans:
        combined = decode([item.token for item in active] + [span.token])
        suffix = combined[len(active_text) :] if combined.startswith(active_text) else ""
        at_word_boundary = suffix.startswith(" ")
        if active and (
            span.start - active[-1].end > 0.7
            or (at_word_boundary and (span.end - active[0].start > 6 or len(active_text) >= 80))
        ):
            flush(span.start)
            combined = decode([span.token])
        if not active and not any(char.isalnum() for char in combined):
            continue  # Do not show isolated punctuation during silence.
        active.append(span)
        active_text = combined
        ends_sentence = re.search(r"[.!?。！？][\"'”’]?$", active_text)
        abbreviation = re.search(r"\b(?:Dr|Mr|Mrs|Ms|Prof|St|[A-Z])\.$", active_text)
        if ends_sentence and not abbreviation:
            flush()
    flush()
    return chunks


def make_static_model(model, frames: int = STATIC_FRAMES):
    """Fix the allocation width while preserving the real-length attention mask."""
    import openvino as ov
    from openvino.passes import ConstantFolding, Manager

    model.reshape(
        {"speech": [1, frames, 560], "speech_lengths": [1], "language": [1], "textnorm": [1]}
    )
    ranges = [node for node in model.get_ordered_ops() if node.friendly_name == "/encoder/Range"]
    if len(ranges) != 1:
        raise ValueError(
            "Unsupported SenseVoice IR graph. Use python -m tools.models.sensevoice's pinned model."
        )
    ranges[0].output(0).replace(
        ov.opset13.constant(np.arange(frames + 4, dtype=np.float32)).output(0)
    )
    model.validate_nodes_and_infer_types()
    passes = Manager()
    passes.register_pass(ConstantFolding())
    passes.run_passes(model)
    if any(
        port.partial_shape.is_dynamic for node in model.get_ordered_ops() for port in node.outputs()
    ):
        raise RuntimeError(
            "SenseVoice graph still contains dynamic shapes; refusing NPU compilation."
        )
    return model


class SenseVoicePipeline:
    def __init__(self, model_path: str | Path, *, cache_dir: str | Path | None = None):
        import openvino as ov
        import openvino_tokenizers  # noqa: F401 - registers the detokenizer ops

        self.model_path = Path(model_path).resolve()
        self.device = "NPU"
        required = [
            "openvino_model.xml",
            "openvino_model.bin",
            "openvino_detokenizer.xml",
            "openvino_detokenizer.bin",
            "am.mvn",
        ]
        missing = [name for name in required if not (self.model_path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing SenseVoice files: {', '.join(missing)}. Run uv run --group prepare -m tools.models.sensevoice"
            )
        self.cmvn = load_cmvn(self.model_path / "am.mvn")
        core = ov.Core()
        if self.device not in core.available_devices:
            raise RuntimeError(
                f"Requested SenseVoice device {self.device} is unavailable. No fallback was attempted."
            )
        model = make_static_model(core.read_model(str(self.model_path / "openvino_model.xml")))
        cache = Path(cache_dir) if cache_dir else self.model_path / ".openvino-cache"
        cache.mkdir(parents=True, exist_ok=True)
        config = {"CACHE_DIR": str(cache)}
        self.model = core.compile_model(model, self.device, config)
        actual_devices = self.model.get_property("EXECUTION_DEVICES")
        self.execution_devices = (
            [actual_devices] if isinstance(actual_devices, str) else list(actual_devices)
        )
        if self.device not in self.execution_devices:
            raise RuntimeError(f"SenseVoice requested {self.device}, got {self.execution_devices}")
        self.detokenizer = core.compile_model(
            str(self.model_path / "openvino_detokenizer.xml"), "CPU"
        )

    def details(self) -> dict:
        return {
            "backend": "sensevoice",
            "model_path": str(self.model_path),
            "device": self.device,
            "execution_devices": self.execution_devices,
            "static_frames": STATIC_FRAMES,
            "max_audio_seconds": MAX_SECONDS,
            "timestamp_method": "approximate CTC emission frames (60 ms)",
            "preprocessing_device": "CPU",
            "detokenizer_device": "CPU",
        }

    def _decode(self, tokens: list[int]) -> str:
        if not tokens:
            return ""
        result = self.detokenizer([np.array([tokens], np.int64)])
        return re.sub(r"<\|[^|]*\|>", "", str(result[self.detokenizer.output(0)][0])).strip()

    def _language_token(self, token: int) -> str:
        # The existing detokenizer deliberately strips special tokens. Read its
        # embedded SentencePiece vocabulary, rather than hardcoding token IDs
        # or loading a second language-identification model.
        if not hasattr(self, "_language_vocabulary"):
            import xml.etree.ElementTree as ET
            import sentencepiece as spm

            path = self.model_path / "openvino_detokenizer.xml"
            tree = ET.parse(path)
            layer = tree.find(".//layer[@type='SentencepieceDetokenizer']")
            if layer is None:
                raise ValueError("SenseVoice detokenizer has no SentencePiece vocabulary")
            edge = tree.find(f".//edge[@to-layer='{layer.attrib['id']}'][@to-port='0']")
            data = tree.find(f".//layer[@id='{edge.attrib['from-layer']}']/data")
            offset, size = int(data.attrib["offset"]), int(data.attrib["size"])
            with path.with_suffix(".bin").open("rb") as stream:
                stream.seek(offset)
                self._language_vocabulary = spm.SentencePieceProcessor(
                    model_proto=stream.read(size)
                )
        piece = self._language_vocabulary.id_to_piece(token)
        match = re.fullmatch(r"<\|([a-z]{2,8})\|>", piece)
        return match.group(1) if match else ""

    def generate(
        self,
        audio,
        language: str = "en",
        return_timestamps: bool = True,
    ):
        language = language.removeprefix("<|").removesuffix("|>").lower()
        if language not in LANGUAGE_IDS:
            raise ValueError(
                f"Unsupported SenseVoice language {language!r}; choose {', '.join(LANGUAGE_IDS)}"
            )
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("SenseVoice expects finite mono audio sampled at 16 kHz")
        if audio.size > SAMPLE_RATE * MAX_SECONDS:
            raise ValueError("SenseVoice accepts at most 30 seconds per window")
        if audio.size == 0 or not np.any(audio):
            return SimpleNamespace(texts=[""], chunks=[])
        duration = audio.size / SAMPLE_RATE
        feat = extract_features(audio, self.cmvn)
        valid_frames = len(feat)
        if valid_frames > STATIC_FRAMES:
            raise ValueError("SenseVoice features exceed the static NPU window")
        padded = np.pad(feat, ((0, STATIC_FRAMES - valid_frames), (0, 0)))
        result = self.model(
            {
                "speech": padded[None],
                "speech_lengths": np.array([valid_frames], np.int32),
                "language": np.array([LANGUAGE_IDS[language]], np.int32),
                "textnorm": np.array([14], np.int32),
            }
        )
        logits = result[self.model.output("ctc_logits")]
        valid = min(int(result[self.model.output("encoder_out_lens")][0]), valid_frames + 4)
        spans = ctc_spans(logits, valid, duration)
        text = self._decode([span.token for span in spans])
        chunks = subtitle_chunks(spans, self._decode, duration) if return_timestamps else []
        detected, score = "", 0.0
        if language == "auto":
            first = logits[0, 0].astype(np.float64)
            token = int(first.argmax())
            detected = self._language_token(token)
            score = float(1 / np.exp(first - first[token]).sum())
        return SimpleNamespace(
            texts=[text], chunks=chunks, detected_language=detected, language_score=score
        )
