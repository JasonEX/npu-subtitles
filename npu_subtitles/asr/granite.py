"""Bounded-window Granite Speech 5 TurboCTC recognition on Intel NPU only.

The Conformer/CTC graph always executes on NPU. NumPy performs the official
log-mel/delta feature calculation, energy VAD and greedy CTC bookkeeping on the
host. This is incremental window re-evaluation, not a stateful streaming model:
only received audio is used and each hypothesis may revise its bounded context.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np

from npu_subtitles.asr.transcript import (
    AcousticTranscriptLedger,
    ObservationWord,
    WindowObservation,
)
from npu_subtitles.captions.language import check_language, join_units, lexical_spans

from npu_subtitles.paths import ROOT

DEFAULT_MODEL_DIR = ROOT / "models/granite-speech-5.0-470m-turboctc"
from npu_subtitles.paths import SAMPLE_RATE, SPEECH_RMS_THRESHOLD

VAD_FRAME_SAMPLES = 320
PRE_ROLL_SAMPLES = 7680
MODEL_REPO = "ibm-granite/granite-speech-5.0-470m-turboctc"
MODEL_REVISION = "1dd23f46f6d4e6eae2531982ba2e33220086dd97"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_manifest(model_dir: Path, static_folder: Path) -> dict[str, Any]:
    manifest = json.loads((static_folder / "artifact.json").read_text(encoding="utf-8"))
    source = json.loads((model_dir / "source" / "source.json").read_text(encoding="utf-8"))
    for record in (manifest, source):
        if (record.get("repo_id"), record.get("revision")) != (MODEL_REPO, MODEL_REVISION):
            raise ValueError(
                "Granite source and graph must identify the pinned supported model revision"
            )
    if (
        manifest.get("architecture") != "GraniteSpeech5ForCTC"
        or manifest.get("runtime_device_required") != "NPU"
        or manifest.get("source_sha256") != source.get("sha256")
    ):
        raise ValueError("Granite exported graph does not match its recorded source")
    required = {"model.xml", "model.bin", "runtime_preprocessor.npz", "preprocessor.json"}
    if set(manifest.get("sha256", {})) != required:
        raise ValueError("Granite export must hash every graph and preprocessing artifact")
    for name in sorted(required):
        if _sha256(static_folder / name) != manifest["sha256"][name]:
            raise ValueError(f"Granite artifact hash mismatch: {name}")
    for name in ("config.json", "tokenizer.json"):
        if _sha256(model_dir / "source" / name) != source["sha256"].get(name):
            raise ValueError(f"Granite source hash mismatch: {name}")
    return manifest


class GraniteFeatures:
    """NumPy equivalent of the pinned official Granite Speech 5 extractor.

    Filter/window coefficients are saved by the export tool from the official
    processor, avoiding a second implementation of mel-filter construction.
    The static buffer is waveform-padded before feature extraction, exactly as
    the processor's max_length path. Its valid-frame mask excludes that tail.
    """

    def __init__(self, folder: Path, input_frames: int):
        path = folder / "runtime_preprocessor.npz"
        with np.load(path, allow_pickle=False) as data:
            self.mel_filters = np.asarray(data["mel_filters"], np.float32)
            window = np.asarray(data["window"], np.float32)
        config_path = folder / "preprocessor.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        # Processor bundles may nest the feature extractor configuration.
        config = config.get("feature_extractor", config)
        self.n_fft = int(config.get("n_fft", 512))
        self.hop_length = int(config.get("hop_length", 160))
        self.win_length = int(config.get("win_length", 400))
        self.delta_window = int(config.get("delta_win_length", 3))
        self.logmel_floor = float(config.get("logmel_floor_db", 8.0))
        self.input_frames = int(input_frames)
        self.capacity_samples = self.input_frames * self.hop_length * 2
        if (
            int(config.get("sampling_rate", SAMPLE_RATE)) != SAMPLE_RATE
            or self.mel_filters.shape != (self.n_fft // 2 + 1, 80)
            or window.shape != (self.win_length,)
            or self.n_fft < self.win_length
            or self.delta_window < 3
            or self.delta_window % 2 != 1
        ):
            raise ValueError("Unsupported Granite feature-extractor configuration")
        pad = self.n_fft - self.win_length
        self.window = np.pad(window, (pad // 2, pad - pad // 2))

    def __call__(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        samples = len(audio)
        if not 0 < samples <= self.capacity_samples:
            raise ValueError("Granite audio does not fit the static feature buffer")
        waveform = np.zeros(self.capacity_samples, np.float32)
        waveform[:samples] = audio
        waveform = np.pad(waveform, (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        frames = np.lib.stride_tricks.sliding_window_view(waveform, self.n_fft)[:: self.hop_length]
        frames = frames[: self.input_frames * 2]
        spectrum = np.fft.rfft(frames * self.window, axis=-1)
        power = (spectrum.real**2 + spectrum.imag**2).astype(np.float32)
        logmel = np.log10(np.maximum(power @ self.mel_filters, np.float32(1e-10)))
        logmel = np.maximum(logmel, logmel.max() - self.logmel_floor) / np.float32(4) + np.float32(
            1
        )
        radius = self.delta_window // 2
        padded = np.pad(logmel, ((radius, radius), (0, 0)), mode="edge")
        deltas = np.zeros_like(logmel)
        for shift in range(1, radius + 1):
            deltas += shift * (
                padded[radius + shift : radius + shift + len(logmel)]
                - padded[radius - shift : radius - shift + len(logmel)]
            )
        deltas /= np.float32(2 * sum(shift * shift for shift in range(1, radius + 1)))
        features = np.concatenate((logmel, deltas), axis=-1).reshape(1, self.input_frames, 320)
        valid_frames = min(self.input_frames, (samples // self.hop_length + 1) // 2)
        mask = (np.arange(self.input_frames)[None, :] < valid_frames).astype(np.int64)
        return np.ascontiguousarray(features, dtype=np.float32), mask, valid_frames


class GraniteNPU:
    """Incremental PCM with independent physical views and logical text.

    Overlapping views share one frame grid. Their observations update a bounded
    acoustic ledger; only its sealed prefix is eligible for subtitle ownership.
    A physical window change never finalizes a sentence.
    """

    def __init__(
        self,
        model_dir: Path | str = DEFAULT_MODEL_DIR,
        cache_dir: Path | str | None = None,
        *,
        energy_threshold: float = SPEECH_RMS_THRESHOLD,
        silence_seconds: float = 1.0,
        max_segment_seconds: float = 12.0,
        partial_seconds: float = 0.8,
        pause_seconds: float = 0.20,
        partial_decode_budget: float = 0.55,
        max_partial_seconds: float = 3.0,
        source_language: str = "en",
    ):
        import openvino as ov
        from tokenizers import Tokenizer

        if not np.isfinite(energy_threshold) or not 0 <= energy_threshold <= 1:
            raise ValueError("energy_threshold must be finite and in [0, 1]")
        if not np.isfinite(silence_seconds) or silence_seconds <= 0:
            raise ValueError("silence_seconds must be finite and positive")
        if not np.isfinite(max_segment_seconds) or not 0.8 <= max_segment_seconds <= 12:
            raise ValueError("max_segment_seconds must be between 0.8 and 12 seconds")
        if not np.isfinite(partial_seconds) or partial_seconds <= 0:
            raise ValueError("partial_seconds must be finite and positive")
        if not np.isfinite(pause_seconds) or not 0 < pause_seconds <= silence_seconds:
            raise ValueError("pause_seconds must be positive and no greater than silence_seconds")
        if not 0 < partial_decode_budget <= 1 or not partial_seconds <= max_partial_seconds <= 12:
            raise ValueError("Invalid Granite partial inference budget")
        self.source_language = check_language(source_language)
        self.model_dir = Path(model_dir)
        self.static_folder = self.model_dir / "npu-static-12p8s"
        self.manifest = _verified_manifest(self.model_dir, self.static_folder)
        tokenizer_path = self.model_dir / "source" / "tokenizer.json"
        if not tokenizer_path.is_file():
            tokenizer_path = self.model_dir / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        config_path = self.model_dir / "source" / "config.json"
        if not config_path.is_file():
            config_path = self.model_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.blank_id = int(config["pad_token_id"])
        self.output_stride = 2 ** len(config["encoder_config"]["subsample_layers"])
        self.energy_threshold = energy_threshold
        self.release_threshold = energy_threshold * 0.65
        self.silence_seconds = silence_seconds
        self.max_segment_seconds = max_segment_seconds
        # Physical windows are deliberately shorter than the exported graph.
        # A fresh view of a turn is possible without ending its logical text.
        self.output_hop_samples = 1280
        self.capacity_samples = (
            int(min(max_segment_seconds, 8.0) * SAMPLE_RATE)
            // self.output_hop_samples
            * self.output_hop_samples
        )
        self.overlap_samples = min(4 * SAMPLE_RATE, self.capacity_samples // 2)
        self.overlap_samples = (
            self.overlap_samples // self.output_hop_samples * self.output_hop_samples
        )
        self.window_hop_samples = self.capacity_samples - self.overlap_samples
        self.logical_capacity_seconds = 48.0
        self.logical_capacity_words = 256
        self.partial_seconds = partial_seconds
        self.pause_seconds = pause_seconds
        self.partial_decode_budget = partial_decode_budget
        self.max_partial_seconds = max_partial_seconds
        self.last_decode_seconds = 0.0
        core = ov.Core()
        if "NPU" not in core.available_devices:
            raise RuntimeError("Granite requires Intel NPU; CPU fallback is disabled")
        self.model = core.compile_model(
            self.static_folder / "model.xml",
            "NPU",
            {
                "CACHE_DIR": str(Path(cache_dir or ROOT / ".cache/live-npu/granite").resolve()),
                "PERFORMANCE_HINT": "LATENCY",
            },
        )
        devices = self.model.get_property("EXECUTION_DEVICES")
        devices = [devices] if isinstance(devices, str) else list(devices)
        if not devices or any(str(value).split(".", 1)[0] != "NPU" for value in devices):
            raise RuntimeError(f"Granite graph must execute only on NPU, reported {devices!r}")
        self.execution_devices = {"encoder_ctc": devices}
        inputs = {port.get_any_name(): tuple(port.shape) for port in self.model.inputs}
        if set(inputs) != {"input_features", "attention_mask"}:
            raise ValueError(f"Unexpected Granite graph inputs: {inputs}")
        self.features = GraniteFeatures(self.static_folder, inputs["input_features"][1])
        if (
            inputs["input_features"] != (1, self.features.input_frames, 320)
            or inputs["attention_mask"] != (1, self.features.input_frames)
            or self.capacity_samples > self.features.capacity_samples
        ):
            raise ValueError("Granite graph cannot hold the bounded physical window")
        if (
            self.manifest.get("blank_token_id") != self.blank_id
            or len(self.model.outputs) != 1
            or tuple(self.model.output(0).shape)
            != (1, self.features.input_frames // self.output_stride, int(config["vocab_size"]))
        ):
            raise ValueError("Granite graph output does not match the source CTC configuration")
        for name, shape in inputs.items():
            if list(shape) != self.manifest["inputs"][name]["shape"]:
                raise ValueError(f"Granite compiled input shape differs from its artifact: {name}")
        self.frame_seconds = self.features.hop_length * 2 * self.output_stride / SAMPLE_RATE
        self.request = self.model.create_infer_request()
        # Some NPU kernels initialize lazily on the first request. Complete
        # that work before playback using synthetic silence, without reading
        # media or letting a cold-start cost alter the partial cadence.
        warmup_started = time.perf_counter()
        warmup_features, warmup_mask, warmup_frames = self.features(
            np.zeros(round(0.8 * SAMPLE_RATE), np.float32)
        )
        self.request.infer({"input_features": warmup_features, "attention_mask": warmup_mask})
        warmup_logits = self.request.get_output_tensor(0).data[
            0, : warmup_frames // self.output_stride
        ]
        if not np.isfinite(warmup_logits).all():
            raise RuntimeError("Granite NPU warmup returned nonfinite valid CTC logits")
        self.warmup_seconds = time.perf_counter() - warmup_started
        self._next_id = 0
        self.reset()

    def details(self) -> dict[str, Any]:
        return dict(
            backend="granite-speech5-turboctc-static-openvino",
            variant="470m-turboctc",
            adapter_version=3,
            source_language=self.source_language,
            source_unit_basis="characters_and_latin_runs"
            if self.source_language == "ja"
            else "space_delimited_words",
            model_dir=str(self.model_dir.resolve()),
            static_folder=str(self.static_folder.resolve()),
            artifact=self.manifest,
            execution_devices=self.execution_devices,
            device="NPU",
            cpu_fallback=False,
            sample_rate=SAMPLE_RATE,
            max_segment_seconds=self.max_segment_seconds,
            static_window_seconds=self.features.capacity_samples / SAMPLE_RATE,
            physical_window_seconds=self.capacity_samples / SAMPLE_RATE,
            overlap_seconds=self.overlap_samples / SAMPLE_RATE,
            window_hop_seconds=self.window_hop_samples / SAMPLE_RATE,
            logical_capacity_seconds=self.logical_capacity_seconds,
            logical_capacity_words=self.logical_capacity_words,
            physical_rollover_ends_logical_context=False,
            partial_seconds=self.partial_seconds,
            partial_decode_budget=self.partial_decode_budget,
            max_partial_seconds=self.max_partial_seconds,
            energy_threshold=self.energy_threshold,
            silence_seconds=self.silence_seconds,
            pause_seconds=self.pause_seconds,
            onset_seconds=0.04,
            vad_frame_seconds=VAD_FRAME_SAMPLES / SAMPLE_RATE,
            frame_seconds=self.frame_seconds,
            timing_kind="ctc_emission_spans",
            segmentation_profile="granite_ctc_v3",
            warmup_seconds=self.warmup_seconds,
            warmup_input="single stateless NPU request on 0.8 seconds of synthetic silence",
            is_final_scope="logical_asr_context",
            decoder="greedy_ctc",
            alignment="bounded acoustic transcript ledger; full mutable path replacement",
            alignment_algorithm_version=4,
            lexical_integrity="source BPE boundaries and complete ordered source-path blocks",
            transcript_revision_seconds=self.overlap_samples / SAMPLE_RATE,
            transcript_retention_seconds=self.overlap_samples / SAMPLE_RATE + 2.0,
            frame_grid="80ms grid anchored to the current playback epoch; physical hops preserve phase",
            streaming_kind="overlapping_window_re_evaluation_with_logical_transcript",
            padding="masked_static_waveform_tail; no unread media used",
        )

    def _clear_context(self) -> None:
        self.active = False
        self.window_audio = np.empty(0, np.float32)
        self.pre_roll = np.empty(0, np.float32)
        self.onset_samples = self.silent_samples = 0
        self.pause_detected = False
        self.paused_at = self.activity_start = self.last_speech_audio_time = None
        self.acoustic_pauses: list[dict[str, float]] = []
        self.last_text = ""
        self.word_first_seen: list[float] = []
        self.word_timestamps: list[dict[str, float]] = []
        self.word_confidence: list[float] = []
        self.word_token_ranges: list[list[int]] = []
        self.transcript_ledger: AcousticTranscriptLedger | None = None
        self.transcript_snapshot: dict[str, Any] | None = None
        self.raw_window_observation: dict[str, Any] | None = None
        self.window_observation_revision = 0
        self.alignment_details: dict[str, Any] = {"kind": "acoustic_transcript_ledger"}
        self.confirmed_prefix_words = 0
        self.revision = 0
        self.physical_window_index = 0
        self.last_decode_sample = self.processed_samples

    def reset(self, start_time: float = 0.0) -> None:
        if not np.isfinite(start_time) or start_time < 0:
            raise ValueError("start_time must be finite and nonnegative")
        self.start_time = float(start_time)
        self.received_samples = self.processed_samples = 0
        self.pending = np.empty(0, np.float32)
        self._clear_context()

    def _next_partial_samples(self) -> int:
        seconds = max(
            self.partial_seconds,
            min(self.max_partial_seconds, self.last_decode_seconds / self.partial_decode_budget),
        )
        return int(np.ceil(seconds * SAMPLE_RATE / VAD_FRAME_SAMPLES)) * VAD_FRAME_SAMPLES

    def _time(self, sample: int) -> float:
        return self.start_time + sample / SAMPLE_RATE

    def _tokens_from_logits(
        self, logits: np.ndarray, start_sample: int
    ) -> tuple[list[dict], list[dict]]:
        ids = np.argmax(logits, axis=-1)
        peaks = logits[np.arange(len(ids)), ids]
        confidence = 1 / np.exp(logits - peaks[:, None]).sum(axis=-1)
        groups: list[dict[str, Any]] = []
        for i, value in enumerate(ids.tolist()):
            begin = self._time(start_sample + i * self.output_hop_samples)
            end = self._time(start_sample + (i + 1) * self.output_hop_samples)
            if groups and groups[-1]["id"] == value:
                groups[-1]["end"] = end
                groups[-1]["confidence"] = max(groups[-1]["confidence"], float(confidence[i]))
                groups[-1]["frames"] += 1
            else:
                groups.append(
                    dict(
                        id=int(value),
                        start=begin,
                        end=end,
                        confidence=float(confidence[i]),
                        frames=1,
                    )
                )
        tokens = []
        for i, group in enumerate(groups):
            if group["id"] == self.blank_id:
                continue
            previous = groups[i - 1] if i else None
            following = groups[i + 1] if i + 1 < len(groups) else None
            tokens.append(
                dict(
                    group,
                    blank_before=(
                        previous["end"] - previous["start"]
                        if previous and previous["id"] == self.blank_id
                        else 0.0
                    ),
                    blank_after=(
                        following["end"] - following["start"]
                        if following and following["id"] == self.blank_id
                        else 0.0
                    ),
                )
            )
        # Full frame evidence is bounded by one <=8s view, not by movie length.
        frames = [
            dict(
                sample=start_sample + i * self.output_hop_samples,
                id=int(value),
                confidence=float(confidence[i]),
            )
            for i, value in enumerate(ids.tolist())
        ]
        return tokens, frames

    def _ctc_words(
        self, tokens: list[dict]
    ) -> tuple[str, list[dict], list[float], list[list[int]], list[str]]:
        ids: list[int] = []
        text = ""
        char_tokens: list[int] = []
        for index, item in enumerate(tokens):
            ids.append(item["id"])
            decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
            mapping = [index] * len(decoded)
            for tag, a, b, c, d in SequenceMatcher(
                None, text, decoded, autojunk=False
            ).get_opcodes():
                if tag == "equal":
                    mapping[c:d] = char_tokens[a:b]
            text, char_tokens = decoded, mapping
        words, spans, confidences, ranges = [], [], [], []
        for begin, end in lexical_spans(text, self.source_language):
            indices = [char_tokens[i] for i in range(begin, end) if text[i].isalnum()]
            if not indices:
                # Keep punctuation text with the preceding word; it has no
                # independent acoustic duration or lexical token ownership.
                if words:
                    words[-1] += text[begin:end]
                continue
            lo, hi = min(indices), max(indices)
            selected = [tokens[i] for i in set(indices)]
            words.append(text[begin:end])
            spans.append(
                dict(
                    start=max(self.segment_start, min(item["start"] for item in selected)),
                    end=min(
                        self._time(self.processed_samples), max(item["end"] for item in selected)
                    ),
                )
            )
            confidences.append(float(min(item["confidence"] for item in selected)))
            ranges.append([lo, hi + 1])
        if any(
            current["start"] < previous["start"] - 1e-7
            for previous, current in zip(spans, spans[1:])
        ):
            self.alignment_details["word_timestamps_withheld_nonmonotonic"] = True
            spans = []
        return join_units(words, self.source_language), spans, confidences, ranges, words

    def _observation_words(self, tokens: list[dict]) -> tuple[ObservationWord, ...]:
        """Keep source BPE word boundaries that decoded text alone discards."""
        text, spans, confidence, ranges, words = self._ctc_words(tokens)
        if len(spans) != len(words):
            raise RuntimeError("Granite observation did not provide ordered lexical word evidence")
        pieces = [self.tokenizer.id_to_token(item["id"]) or "" for item in tokens]
        result = []
        for index, (word, span, score, (lo, hi)) in enumerate(
            zip(words, spans, confidence, ranges)
        ):
            starts_word = (
                self.source_language == "ja"
                or pieces[lo].startswith("▁")
                or (lo > 0 and pieces[lo - 1] == "▁")
            )
            # Another decoded word or explicit punctuation closes this lexical
            # candidate. A trailing CTC blank alone does not prove a word end.
            ends_word = (
                self.source_language == "ja" or index + 1 < len(words) or word[-1:] in ".,;:!?"
            )
            result.append(
                ObservationWord(
                    word,
                    span["start"],
                    span["end"],
                    score,
                    tuple(item["id"] for item in tokens[lo:hi]),
                    lo,
                    hi,
                    tuple(pieces[lo:hi]),
                    starts_word,
                    ends_word,
                )
            )
        return tuple(result)

    def _decode(self, *, final_reason: str | None = None) -> bool:
        if len(self.window_audio) < self.output_hop_samples:
            if final_reason is None:
                return False
            tokens = []  # Less than one output frame cannot contain CTC evidence.
        else:
            features, mask, valid_frames = self.features(self.window_audio)
            self.request.infer({"input_features": features, "attention_mask": mask})
            logits = self.request.get_output_tensor(0).data
            count = min(logits.shape[1], valid_frames // self.output_stride)
            valid_logits = logits[0, :count]
            if not np.isfinite(valid_logits).all():
                raise RuntimeError("Granite NPU returned nonfinite CTC logits")
            tokens, _ = self._tokens_from_logits(valid_logits, self.window_start_sample)
        if self.transcript_ledger is None:
            raise RuntimeError("Granite active context has no transcript ledger")
        self.window_observation_revision += 1
        observation = WindowObservation(
            window_id=self.physical_window_index,
            revision=self.window_observation_revision,
            audio_start=self._time(self.window_start_sample),
            audio_end=self._time(self.processed_samples),
            words=self._observation_words(tokens),
            tokens=tuple(tokens),
        )
        self.raw_window_observation = observation.record()
        self.transcript_snapshot = self.transcript_ledger.observe(
            observation, final_reason=final_reason
        )
        self.alignment_details = self.transcript_ledger.selection
        words = self.transcript_snapshot["sealed_words"] + self.transcript_snapshot["mutable_words"]
        self.last_text = join_units((word["text"] for word in words), self.source_language)
        self.word_timestamps = [dict(start=word["start"], end=word["end"]) for word in words]
        self.word_confidence = [word["confidence"] for word in words]
        self.word_first_seen = [
            min(
                (source["audio_end"] for source in word["sources"]),
                default=self._time(self.processed_samples),
            )
            for word in words
        ]
        self.word_token_ranges = []  # Token indices are local to source observations, never a global ownership key.
        self.confirmed_prefix_words = len(self.transcript_snapshot["sealed_words"])
        self.revision = self.transcript_snapshot["revision"]
        return True

    def _event(
        self, *, final: bool = False, reason: str | None = None, activity_only: bool = False
    ) -> dict[str, Any]:
        now = self._time(self.processed_samples)
        advanced, accepted, decode_seconds = False, True, None
        if not activity_only:
            advanced = self.processed_samples > self.last_decode_sample
            started = time.perf_counter()
            try:
                accepted = self._decode(final_reason=reason if final else None)
            finally:
                decode_seconds = time.perf_counter() - started
                self.last_decode_seconds = decode_seconds
            self.last_decode_sample = self.processed_samples
        return dict(
            id=self.segment_id,
            revision=self.revision,
            text=self.last_text,
            source_language=self.source_language,
            start=self.segment_start,
            end=now,
            encoded_until=now,
            received_until=self._time(self.received_samples),
            evidence_until=now,
            observed_audio_time=now,
            event_kind="activity"
            if activity_only
            else "hypothesis"
            if accepted
            else "alignment_pending",
            is_final=final,
            is_final_scope="logical_asr_context",
            hypothesis_advanced=advanced and accepted,
            word_first_seen=self.word_first_seen.copy(),
            word_timestamps=self.word_timestamps.copy(),
            word_confidence=self.word_confidence.copy(),
            word_token_ranges=self.word_token_ranges.copy(),
            confirmed_prefix_words=self.confirmed_prefix_words,
            timing_kind="ctc_emission_spans",
            alignment_kind="ctc_emission_spans",
            segmentation_profile="granite_ctc_v3",
            transcript=self.transcript_snapshot,
            raw_window_observation=None if activity_only else self.raw_window_observation,
            word_time_basis="lexical_ctc_emissions_excluding_punctuation_not_forced_alignment",
            decode_seconds=decode_seconds,
            next_partial_seconds=self._next_partial_samples() / SAMPLE_RATE,
            activity_start=self.activity_start,
            last_speech_audio_time=self.last_speech_audio_time,
            pause_detected=self.pause_detected,
            paused_at=self.paused_at,
            acoustic_pauses=self.acoustic_pauses.copy(),
            physical_window_index=self.physical_window_index,
            physical_window_start=self._time(self.window_start_sample),
            physical_window_end=now,
            window_alignment=self.alignment_details,
            final_reason=reason,
            context_end_reason=reason,
            closure_reason=reason,
            continuation_expected=reason == "capacity_limit",
        )

    def _roll_window(self) -> None:
        new_start = self.window_start_sample + self.window_hop_samples
        self.window_audio = self.window_audio[self.window_hop_samples :].copy()
        self.window_start_sample = new_start
        self.physical_window_index += 1
        self.window_observation_revision = 0

    def _start_context(self, sample_start: int, frame: np.ndarray) -> None:
        available_start = sample_start - len(self.pre_roll)
        # Align every physical view to one epoch-wide 80ms grid. Rounding only
        # trims idle preroll; it does not insert fabricated samples.
        start = (
            (available_start + self.output_hop_samples - 1)
            // self.output_hop_samples
            * self.output_hop_samples
        )
        start = min(sample_start, start)
        combined = np.concatenate((self.pre_roll, frame))
        self.window_audio = combined[start - available_start :].copy()
        self.window_start_sample = self.logical_start_sample = start
        self.segment_start = self._time(start)
        self.segment_id = self._next_id
        self._next_id += 1
        self.transcript_ledger = AcousticTranscriptLedger(
            self.segment_id,
            self.segment_start,
            revision_seconds=self.overlap_samples / SAMPLE_RATE,
            retention_seconds=self.overlap_samples / SAMPLE_RATE + 2.0,
        )
        self.transcript_snapshot = self.transcript_ledger.snapshot()
        self.active = True
        self.last_decode_sample = start
        self.activity_start = self._time(sample_start + len(frame) - self.onset_samples)
        self.pre_roll = np.empty(0, np.float32)

    def _frame(
        self, frame: np.ndarray, sample_start: int, *, final: bool = False
    ) -> list[dict[str, Any]]:
        self.processed_samples = sample_start + len(frame)
        frame_start, frame_end = self._time(sample_start), self._time(self.processed_samples)
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) if len(frame) else 0.0
        changed = False
        if not self.active:
            self.onset_samples = (
                self.onset_samples + len(frame) if rms >= self.energy_threshold else 0
            )
            if self.onset_samples < round(0.04 * SAMPLE_RATE):
                self.pre_roll = np.concatenate((self.pre_roll, frame))[-PRE_ROLL_SAMPLES:]
                return []
            self._start_context(sample_start, frame)
            changed = True
        else:
            self.window_audio = np.concatenate((self.window_audio, frame))
        if rms >= self.release_threshold:
            if self.pause_detected:
                if self.paused_at is not None:
                    self.acoustic_pauses.append(dict(start=self.paused_at, end=frame_start))
                    self.acoustic_pauses = self.acoustic_pauses[-64:]
                self.activity_start, changed = frame_start, True
            self.pause_detected, self.paused_at, self.silent_samples = False, None, 0
            self.last_speech_audio_time = frame_end
        else:
            self.silent_samples += len(frame)
            if not self.pause_detected and self.silent_samples >= round(
                self.pause_seconds * SAMPLE_RATE
            ):
                self.pause_detected, self.paused_at, changed = (
                    True,
                    self.last_speech_audio_time,
                    True,
                )
        reason = "input_end" if final else None
        if reason is None and self.silent_samples >= round(self.silence_seconds * SAMPLE_RATE):
            reason = "long_silence"
        if reason is None and (
            self.processed_samples - self.logical_start_sample
            >= round(self.logical_capacity_seconds * SAMPLE_RATE)
            or len(self.word_timestamps) >= self.logical_capacity_words
        ):
            reason = "capacity_limit"
        if reason is not None:
            event = self._event(final=True, reason=reason)
            idle_tail = (
                self.window_audio[-PRE_ROLL_SAMPLES:].copy() if reason == "long_silence" else None
            )
            self._clear_context()
            if idle_tail is not None:
                # Retain already received silence so rounding the next onset
                # to the global feature grid never needs to trim speech.
                self.pre_roll = idle_tail
            return [event]
        if len(self.window_audio) >= self.capacity_samples:
            if len(self.window_audio) > self.capacity_samples:
                raise RuntimeError("Granite physical window exceeded its aligned PCM capacity")
            event = self._event()
            self._roll_window()
            return [event]
        if self.processed_samples - self.last_decode_sample >= self._next_partial_samples():
            return [self._event()]
        return [self._event(activity_only=True)] if changed else []

    def accept_audio(self, audio: np.ndarray, final: bool = False) -> list[dict[str, Any]]:
        values = np.asarray(audio, dtype=np.float32)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Granite audio must be finite mono float32 at 16000 Hz")
        self.received_samples += len(values)
        self.pending = np.concatenate((self.pending, values))
        events = []
        while len(self.pending) >= VAD_FRAME_SAMPLES:
            frame = self.pending[:VAD_FRAME_SAMPLES].copy()
            self.pending = self.pending[VAD_FRAME_SAMPLES:]
            events.extend(
                self._frame(frame, self.processed_samples, final=final and not len(self.pending))
            )
        if final:
            if len(self.pending):
                frame, self.pending = self.pending.copy(), np.empty(0, np.float32)
                events.extend(self._frame(frame, self.processed_samples, final=True))
            elif self.active:
                events.append(self._event(final=True, reason="input_end"))
                self._clear_context()
            self.pre_roll = np.empty(0, np.float32)
        return events

    def close(self) -> None:
        self.request = None
        self.model = None
        gc.collect()
