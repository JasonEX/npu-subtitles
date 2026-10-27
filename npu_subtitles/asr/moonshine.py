"""Causal Moonshine v2 Small/Medium streaming ASR on Intel NPU only.

The five neural graphs run on NPU. Host work is audio buffering, energy-based
endpoint detection, moving KV buffers and greedy token decoding. There is no
CPU inference backend. Padding is excluded from every attention layer; encoder
lookahead is supplied only after those audio samples have arrived.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np

from npu_subtitles.paths import ROOT

DEFAULT_MODEL_DIR = ROOT / "models/moonshine-streaming-small"
MEDIUM_MODEL_DIR = ROOT / "models/moonshine-streaming-medium"
from npu_subtitles.paths import SAMPLE_RATE, SPEECH_RMS_THRESHOLD

CHUNK_SAMPLES = 6400
NEW_FRAMES = 20
ENCODER_HOP_SAMPLES = CHUNK_SAMPLES // NEW_FRAMES
VAD_FRAME_SAMPLES = 320
SELF_CAPACITY = 96
MEMORY_CAPACITY = 640
BUILD_VERSION = 2
GRAPH_NAMES = ("frontend", "encoder", "adapter", "cross_kv", "decoder_kv")


@dataclass(frozen=True)
class ModelDimensions:
    encoder_dim: int
    decoder_dim: int
    encoder_layers: int
    decoder_layers: int
    heads: int
    head_dim: int
    left_context: int
    lookahead: int
    conv1_shape: tuple[int, ...]
    conv2_shape: tuple[int, ...]
    bos_id: int
    eos_id: int

    @property
    def encoder_frames(self) -> int:
        return self.left_context + NEW_FRAMES + self.lookahead

    def kv_shape(self, length: int) -> tuple[int, ...]:
        return (self.decoder_layers, 1, self.heads, length, self.head_dim)


def read_dimensions(model_dir: Path) -> ModelDimensions:
    """Read and validate the actual export, never infer shapes from its name."""
    config_path = model_dir / "streaming_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing {config_path}; run python -m tools.models.moonshine")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    windows = config["windows"]
    dimensions = ModelDimensions(
        encoder_dim=int(config["encoder_dim"]),
        decoder_dim=int(config["decoder_dim"]),
        encoder_layers=len(windows),
        decoder_layers=int(config["depth"]),
        heads=int(config["nheads"]),
        head_dim=int(config["head_dim"]),
        left_context=int(config["total_left_context"]),
        lookahead=int(config["total_lookahead"]),
        conv1_shape=tuple(config["frontend_state_shapes"]["conv1_buffer"]),
        conv2_shape=tuple(config["frontend_state_shapes"]["conv2_buffer"]),
        bos_id=int(config["bos_id"]),
        eos_id=int(config["eos_id"]),
    )
    if int(config["frame_len"]) != 80 or CHUNK_SAMPLES % (int(config["frame_len"]) * 4):
        raise ValueError("Unsupported Moonshine frontend frame stride")
    if dimensions.left_context != sum(
        int(window[0]) for window in windows
    ) or dimensions.lookahead != sum(int(window[1]) for window in windows):
        raise ValueError("Moonshine accumulated attention context does not match its layer windows")
    if (
        dimensions.conv1_shape != (1, int(config["d_model_frontend"]), 4)
        or dimensions.conv2_shape != (1, int(config["c1"]), 4)
        or dimensions.encoder_dim != int(config["c2"])
        or dimensions.decoder_dim != dimensions.heads * dimensions.head_dim
    ):
        raise ValueError("Inconsistent Moonshine frontend or decoder dimensions")
    if any(
        value <= 0
        for value in (
            dimensions.encoder_dim,
            dimensions.decoder_dim,
            dimensions.encoder_layers,
            dimensions.decoder_layers,
            dimensions.heads,
            dimensions.head_dim,
        )
    ):
        raise ValueError("Moonshine model dimensions must be positive")
    return dimensions


def conversion_manifest(model_dir: Path, dimensions: ModelDimensions) -> dict[str, Any]:
    source_path = model_dir / "source.json"
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Missing pinned model source record {source_path}; rerun python -m tools.models.moonshine"
        )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not source.get("revision") or not source.get("repo_id"):
        raise ValueError("Moonshine source.json must identify its repository and pinned revision")
    files = {}
    for name in GRAPH_NAMES:
        path = model_dir / f"{name}.onnx"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing Moonshine graph {name}; run python -m tools.models.moonshine"
            )
        stat = path.stat()
        files[path.name] = {
            "sha256": source.get("sha256", {}).get(path.name),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return {
        "build_version": BUILD_VERSION,
        "source_repo": source["repo_id"],
        "source_revision": source["revision"],
        "base_model": source.get("base_model"),
        "variant": source.get("variant")
        or {620: "small", 768: "medium"}.get(dimensions.encoder_dim, "custom"),
        "source_files": files,
        "streaming_config_sha256": hashlib.sha256(
            (model_dir / "streaming_config.json").read_bytes()
        ).hexdigest(),
        "dimensions": json.loads(json.dumps(asdict(dimensions))),
        "precision": "FP16 weights, explicit NPU execution",
        "chunk_samples": CHUNK_SAMPLES,
        "new_frames": NEW_FRAMES,
        "encoder_frames": dimensions.encoder_frames,
        "self_capacity": SELF_CAPACITY,
        "memory_capacity": MEMORY_CAPACITY,
        "padding": "all-layer key masks; no future file audio",
    }


def prepare_npu_models(model_dir: Path) -> Path:
    """Convert ONNX on host; this never executes a neural model on host."""
    import openvino as ov
    from openvino import opset13 as op

    model_dir = Path(model_dir)
    dimensions = read_dimensions(model_dir)
    manifest = conversion_manifest(model_dir, dimensions)
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]
    target = model_dir / f"npu-static-v{BUILD_VERSION}-{fingerprint}"
    manifest_path = target / "conversion.json"
    if manifest_path.is_file():
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old_manifest == manifest and all(
            (target / f"{name}.xml").is_file() and (target / f"{name}.bin").is_file()
            for name in GRAPH_NAMES
        ):
            return target
    target.mkdir(parents=True, exist_ok=True)
    core = ov.Core()

    def save(model: Any, name: str) -> None:
        model.validate_nodes_and_infer_types()
        if any(output.partial_shape.is_dynamic for output in model.outputs):
            raise RuntimeError(f"Moonshine {name} retains dynamic NPU outputs")
        ov.save_model(model, target / f"{name}.xml", compress_to_fp16=True)

    frontend = core.read_model(model_dir / "frontend.onnx")
    # Every chunk is an exact multiple of the 80-sample frontend frame. There
    # are no pending raw samples; frame_count is an accounting-only output.
    for parameter in list(frontend.get_parameters()):
        name = parameter.output(0).get_any_name()
        if name in {"sample_len", "sample_buffer", "frame_count"}:
            dtype = np.float32 if name == "sample_buffer" else np.int64
            parameter.output(0).replace(
                op.constant(np.zeros(parameter.output(0).shape, dtype=dtype)).output(0)
            )
            frontend.remove_parameter(parameter)
    for result in list(frontend.get_results()):
        if result.input_value(0).get_any_name() not in {
            "features",
            "conv1_buffer_out",
            "conv2_buffer_out",
        }:
            frontend.remove_result(result)
    frontend.reshape({"audio_chunk": [1, CHUNK_SAMPLES]})
    for node in frontend.get_ordered_ops():
        if node.get_type_name() == "Asinh":
            x = node.input_value(0)
            a = op.abs(x)
            # The sign/abs formulation avoids cancellation for negative x.
            replacement = op.multiply(
                op.sign(x),
                op.log(op.add(a, op.sqrt(op.add(op.multiply(a, a), op.constant(1, np.float32))))),
            )
            node.output(0).replace(replacement.output(0))
    save(frontend, "frontend")
    del frontend

    encoder = core.read_model(model_dir / "encoder.onnx")
    encoder.reshape({"features": [1, dimensions.encoder_frames, dimensions.encoder_dim]})
    key_mask = op.parameter([1, 1, 1, dimensions.encoder_frames], ov.Type.f32, name="key_mask")
    key_mask.output(0).set_names({"key_mask"})
    encoder.add_parameters([key_mask])
    count = 0
    for node in encoder.get_ordered_ops():
        if node.get_type_name() == "Softmax":
            node.input(0).replace_source_output(op.add(node.input_value(0), key_mask).output(0))
            count += 1
    if count != dimensions.encoder_layers:
        raise ValueError(f"Unexpected Moonshine encoder graph: {count} attention layers")
    save(encoder, "encoder")
    del encoder
    for name, shape in [
        ("adapter", {"encoded": [1, NEW_FRAMES, dimensions.encoder_dim]}),
        ("cross_kv", {"memory": [1, NEW_FRAMES, dimensions.decoder_dim]}),
    ]:
        model = core.read_model(model_dir / f"{name}.onnx")
        model.reshape(shape)
        save(model, name)
        del model

    decoder = core.read_model(model_dir / "decoder_kv.onnx")
    decoder.reshape(
        {
            "token": [1, 1],
            "k_self": dimensions.kv_shape(SELF_CAPACITY),
            "v_self": dimensions.kv_shape(SELF_CAPACITY),
            "out_k_cross": dimensions.kv_shape(MEMORY_CAPACITY),
            "out_v_cross": dimensions.kv_shape(MEMORY_CAPACITY),
        }
    )
    # The physical cache is padded. RoPE must use the real token position,
    # rather than Shape(k_self), and attention must ignore unused cache slots.
    position = op.parameter([1], ov.Type.i32, name="position")
    position.output(0).set_names({"position"})
    self_mask = op.parameter([1, 1, 1, SELF_CAPACITY + 1], ov.Type.f32, name="self_mask")
    self_mask.output(0).set_names({"self_mask"})
    cross_mask = op.parameter([1, 1, 1, MEMORY_CAPACITY], ov.Type.f32, name="cross_mask")
    cross_mask.output(0).set_names({"cross_mask"})
    decoder.add_parameters([position, self_mask, cross_mask])
    frequencies = next(
        n.get_data().reshape(-1)
        for n in decoder.get_ordered_ops()
        if n.get_type_name() == "Constant" and n.friendly_name == "expand"
    )
    angles = np.arange(SELF_CAPACITY, dtype=np.float32)[:, None] * frequencies[None, :]
    angles = np.concatenate([angles, angles], axis=-1)
    for node in decoder.get_ordered_ops():
        if node.get_type_name() in {"Cos", "Sin"}:
            table = np.cos(angles) if node.get_type_name() == "Cos" else np.sin(angles)
            value = op.unsqueeze(
                op.gather(
                    op.constant(table.astype(np.float32)), position, op.constant(0, np.int64)
                ),
                op.constant([0], np.int64),
            )
            node.output(0).replace(value.output(0))
    counts = {SELF_CAPACITY + 1: 0, MEMORY_CAPACITY: 0}
    for node in decoder.get_ordered_ops():
        if node.get_type_name() == "Softmax":
            length = int(node.input_value(0).shape[-1])
            if length not in counts:
                raise ValueError(f"Unexpected decoder attention length {length}")
            node.input(0).replace_source_output(
                op.add(
                    node.input_value(0), self_mask if length == SELF_CAPACITY + 1 else cross_mask
                ).output(0)
            )
            counts[length] += 1
    if list(counts.values()) != [dimensions.decoder_layers, dimensions.decoder_layers]:
        raise ValueError(f"Unexpected decoder attention counts {counts}")
    for result in list(decoder.get_results()):
        if result.input_value(0).get_any_name() in {"out_k_cross", "out_v_cross"}:
            decoder.remove_result(result)
    save(decoder, "decoder_kv")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


class MoonshineNPU:
    """Incremental 16 kHz mono input; events use absolute reset media times.

    Hypothesis events revise one bounded ASR context, independently of subtitle
    sentences. ``is_final`` closes that context, not a grammatical sentence.
    Activity events only report energy-VAD changes and must not confirm text
    stability. Times bound received/encoded audio; they are not word alignment.
    Use ``reset`` for seeks or an actual audio discontinuity, not player pause.
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
    ):
        import openvino as ov
        from tokenizers import Tokenizer

        if not CHUNK_SAMPLES / SAMPLE_RATE <= max_segment_seconds <= 12:
            raise ValueError(
                "max_segment_seconds must be between one frontend block (0.4s) and 12s"
            )
        if not np.isfinite(energy_threshold) or not 0 <= energy_threshold <= 1:
            raise ValueError("energy_threshold must be finite and in [0, 1]")
        if not np.isfinite(silence_seconds) or silence_seconds <= 0:
            raise ValueError("silence_seconds must be finite and positive")
        if not np.isfinite(partial_seconds) or partial_seconds <= 0:
            raise ValueError("partial_seconds must be finite and positive")
        if not np.isfinite(pause_seconds) or not 0 < pause_seconds <= silence_seconds:
            raise ValueError("pause_seconds must be positive and no greater than silence_seconds")
        if not np.isfinite(partial_decode_budget) or not 0 < partial_decode_budget <= 1:
            raise ValueError("partial_decode_budget must be in (0, 1]")
        if not np.isfinite(max_partial_seconds) or not partial_seconds <= max_partial_seconds <= 12:
            raise ValueError("max_partial_seconds must be between partial_seconds and 12 seconds")
        self.model_dir = Path(model_dir)
        self.dimensions = read_dimensions(self.model_dir)
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.energy_threshold = energy_threshold
        self.silence_seconds = silence_seconds
        self.max_segment_seconds = max_segment_seconds
        # Prefer an existing pause near capacity, without making ordinary
        # short pauses reset the context. For small custom capacities, limit
        # this search window to the final half of the context.
        self.capacity_pause_window_seconds = min(2.0, max_segment_seconds / 2)
        self.capacity_pause_from_seconds = max_segment_seconds - self.capacity_pause_window_seconds
        self.partial_seconds = partial_seconds
        self.pause_seconds = pause_seconds
        self.partial_frames = max(
            1, int(np.ceil(partial_seconds * SAMPLE_RATE / ENCODER_HOP_SAMPLES))
        )
        self.partial_decode_budget = partial_decode_budget
        self.max_partial_seconds = max_partial_seconds
        self.max_partial_frames = max(
            1, int(np.ceil(max_partial_seconds * SAMPLE_RATE / ENCODER_HOP_SAMPLES))
        )
        # Retain the last measured compute cost across context rollovers and
        # seeks. Only audio/text state resets; a new context does not grant
        # the decoder another burst of the entire shared NPU budget.
        self.last_decode_seconds = 0.0
        self.release_threshold = energy_threshold * 0.65
        folder = prepare_npu_models(self.model_dir)
        self.static_folder = folder
        self.manifest = json.loads((folder / "conversion.json").read_text(encoding="utf-8"))
        core = ov.Core()
        if "NPU" not in core.available_devices:
            raise RuntimeError("Moonshine requires Intel NPU; CPU fallback is disabled")
        config = {"CACHE_DIR": str(Path(cache_dir or ROOT / ".cache/live-npu/moonshine").resolve())}
        self.models = {}
        self.requests = {}
        self.execution_devices = {}
        for name in GRAPH_NAMES:
            model = core.compile_model(folder / f"{name}.xml", "NPU", config)
            devices = model.get_property("EXECUTION_DEVICES")
            devices = [devices] if isinstance(devices, str) else list(devices)
            if not devices or any(not str(d).startswith("NPU") for d in devices):
                raise RuntimeError(
                    f"Moonshine {name} did not execute exclusively on NPU: {devices}"
                )
            self.execution_devices[name] = devices
            self.models[name] = model
            self.requests[name] = model.create_infer_request()
        self._next_id = 0
        self.reset()

    def details(self) -> dict[str, Any]:
        return {
            "backend": "moonshine-streaming-static-openvino",
            "variant": self.manifest["variant"],
            "source_repo": self.manifest["source_repo"],
            "source_revision": self.manifest["source_revision"],
            "model_dir": str(self.model_dir.resolve()),
            "static_folder": str(self.static_folder.resolve()),
            "dimensions": asdict(self.dimensions),
            "execution_devices": self.execution_devices,
            "device": "NPU",
            "cpu_fallback": False,
            "sample_rate": SAMPLE_RATE,
            "chunk_samples": CHUNK_SAMPLES,
            "energy_threshold": self.energy_threshold,
            "silence_seconds": self.silence_seconds,
            "pre_roll_seconds": CHUNK_SAMPLES / SAMPLE_RATE,
            "vad_frame_seconds": VAD_FRAME_SAMPLES / SAMPLE_RATE,
            "onset_seconds": 0.04,
            "release_threshold": self.release_threshold,
            "pause_seconds": self.pause_seconds,
            "partial_seconds": self.partial_seconds,
            "short_pause_resets_context": True,
            "short_pause_reset_policy": "only_in_capacity_tail_window",
            "partial_decode_budget": self.partial_decode_budget,
            "max_partial_seconds": self.max_partial_seconds,
            "decode_cost_estimate": "last_decode_wall_seconds_retained_across_contexts_and_resets",
            "partial_frame_step_seconds": NEW_FRAMES * ENCODER_HOP_SAMPLES / SAMPLE_RATE,
            "next_partial_seconds": self._next_partial_frames() * ENCODER_HOP_SAMPLES / SAMPLE_RATE,
            "max_segment_seconds": self.max_segment_seconds,
            "capacity_pause_window_seconds": self.capacity_pause_window_seconds,
            "capacity_pause_from_seconds": self.capacity_pause_from_seconds,
            "is_final_scope": "asr_context",
            "alignment_kind": "region_bound",
            "capacity_rollover": "block_aligned_center_with_real_right_context_and_raw_audio_replay",
            "rollover_right_context_seconds": CHUNK_SAMPLES / SAMPLE_RATE,
            "rollover_required_lookahead_seconds": self.dimensions.lookahead
            * ENCODER_HOP_SAMPLES
            / SAMPLE_RATE,
            "rollover_center_alignment_seconds": CHUNK_SAMPLES / SAMPLE_RATE,
            "rollover_pause_alignment": "cut_must_not_precede_last_speech",
            "internal_end_padding": False,
            "eof_padding_allowed": True,
            "self_capacity": SELF_CAPACITY,
            "memory_capacity": MEMORY_CAPACITY,
        }

    def _run(self, name: str, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        request = self.requests[name]
        request.infer(values)
        return {
            output.get_any_name(): request.get_tensor(output).data.copy()
            for output in self.models[name].outputs
        }

    def _next_partial_frames(self) -> int:
        """Minimum encoded-audio progress before another full hypothesis.

        Encoder output advances in 20-frame blocks, so the actual next event
        may be rounded up by one block. Context finalization always bypasses
        this partial-only compute budget.
        """
        budget_frames = int(
            np.ceil(
                self.last_decode_seconds
                / self.partial_decode_budget
                * SAMPLE_RATE
                / ENCODER_HOP_SAMPLES
            )
        )
        return max(self.partial_frames, min(self.max_partial_frames, budget_frames))

    def _reset_context(self) -> None:
        self.active = False
        self.conv1 = np.zeros(self.dimensions.conv1_shape, np.float32)
        self.conv2 = np.zeros(self.dimensions.conv2_shape, np.float32)
        self.features = np.empty((0, self.dimensions.encoder_dim), np.float32)
        self.emitted = 0
        self.cross_k = np.zeros(self.dimensions.kv_shape(MEMORY_CAPACITY), np.float32)
        self.cross_v = np.zeros_like(self.cross_k)
        self.memory_len = 0
        self.segment_samples = 0
        self.segment_audio = np.empty(0, np.float32)
        self.last_frontend_audio = np.empty(0, np.float32)
        self.pending_rollover: dict[str, Any] | None = None
        self.silent_samples = 0
        self.onset_samples = 0
        self.activity_start: float | None = None
        self.last_speech_audio_time: float | None = None
        self.pause_detected = False
        self.paused_at: float | None = None
        self.last_decode_frames = 0
        self.last_text = ""
        self.revision = 0
        self.observed_words: list[str] = []
        self.word_first_seen: list[float] = []
        self.observation_history: list[tuple[list[str], list[float]]] = []
        # Up to 400 ms of raw already-arrived audio. This also retains the
        # first high-energy onset frame while waiting for onset confirmation.
        self.pre_roll = np.empty(0, np.float32)

    def reset(self, start_time: float = 0.0) -> None:
        if not np.isfinite(start_time) or start_time < 0:
            raise ValueError("start_time must be finite and nonnegative")
        self.start_time = float(start_time)
        self.processed_samples = 0
        self.received_samples = 0
        self.pending = np.empty(0, np.float32)
        self._reset_context()

    def _encode_available(self, *, flush: bool = False, center_frames: int | None = None) -> None:
        total = len(self.features)
        limit = total if center_frames is None else min(total, center_frames)
        while self.emitted < limit and (
            flush or total - self.emitted >= NEW_FRAMES + self.dimensions.lookahead
        ):
            new_count = min(NEW_FRAMES, limit - self.emitted)
            left = max(0, self.emitted - self.dimensions.left_context)
            right = min(total, self.emitted + NEW_FRAMES + self.dimensions.lookahead)
            offset = self.dimensions.left_context - (self.emitted - left)
            values = np.zeros(
                (1, self.dimensions.encoder_frames, self.dimensions.encoder_dim), np.float32
            )
            values[0, offset : offset + right - left] = self.features[left:right]
            mask = np.full((1, 1, 1, self.dimensions.encoder_frames), -10000, np.float32)
            mask[..., offset : offset + right - left] = 0
            encoded = self._run("encoder", {"features": values, "key_mask": mask})["encoded"]
            new_encoded = encoded[
                :, self.dimensions.left_context : self.dimensions.left_context + NEW_FRAMES
            ]
            memory = self._run(
                "adapter",
                {"encoded": new_encoded, "pos_offset": np.array([self.emitted], np.int64)},
            )["memory"]
            kv = self._run("cross_kv", {"memory": memory})
            end = self.memory_len + new_count
            if end > MEMORY_CAPACITY:
                raise RuntimeError(
                    "Moonshine streaming memory exceeded its configured segment capacity"
                )
            self.cross_k[..., self.memory_len : end, :] = kv["k_cross"][..., :new_count, :]
            self.cross_v[..., self.memory_len : end, :] = kv["v_cross"][..., :new_count, :]
            self.memory_len = end
            self.emitted += new_count

    def _decode(self) -> str:
        if not self.memory_len:
            return ""
        k_self = np.zeros(self.dimensions.kv_shape(SELF_CAPACITY), np.float32)
        v_self = np.zeros_like(k_self)
        cross_mask = np.full((1, 1, 1, MEMORY_CAPACITY), -10000, np.float32)
        cross_mask[..., : self.memory_len] = 0
        token = self.dimensions.bos_id
        tokens = []
        for position in range(SELF_CAPACITY):
            mask = np.full((1, 1, 1, SELF_CAPACITY + 1), -10000, np.float32)
            mask[..., :position] = 0
            mask[..., SELF_CAPACITY] = 0
            result = self._run(
                "decoder_kv",
                {
                    "token": np.array([[token]], np.int64),
                    "k_self": k_self,
                    "v_self": v_self,
                    "out_k_cross": self.cross_k,
                    "out_v_cross": self.cross_v,
                    "position": np.array([position], np.int32),
                    "self_mask": mask,
                    "cross_mask": cross_mask,
                },
            )
            if not np.isfinite(result["logits"]).all():
                raise RuntimeError("Moonshine NPU decoder produced nonfinite logits")
            k_self[..., position, :] = result["out_k_self"][..., SELF_CAPACITY, :]
            v_self[..., position, :] = result["out_v_self"][..., SELF_CAPACITY, :]
            token = int(np.argmax(result["logits"][0, -1]))
            if token == self.dimensions.eos_id:
                break
            tokens.append(token)
        else:
            raise RuntimeError(
                "Moonshine decoder_token_capacity_limit: 96 tokens without EOS; no truncated hypothesis is published or finalized, and the context is not silently reset"
            )
        return " ".join(self.tokenizer.decode(tokens, skip_special_tokens=True).split())

    def _word_observation_times(self, words: list[str], observed_at: float) -> list[float]:
        """Conservatively align observation ages across hypothesis revisions.

        These are not acoustic word timestamps. Corresponding replacement
        words retain their ages, while extra words at the actual end can be
        new speech. Historical hypotheses only recover a shared prefix: an
        old mistaken suffix must not age every later sentence that replaces
        it. Empty hypotheses do not erase the last nonempty observation.
        """
        if not self.observation_history:
            return [observed_at] * len(words)

        def key(word: str) -> str:
            return word.casefold().strip(".,;:!?\"'“”‘’()[]{}")

        result = [observed_at] * len(words)
        old_words, old_times = self.observation_history[-1]
        matcher = SequenceMatcher(
            None, [key(word) for word in old_words], [key(word) for word in words], autojunk=False
        )
        for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
            if tag in {"equal", "replace"}:
                paired = min(old_end - old_start, new_end - new_start)
                for offset in range(paired):
                    result[new_start + offset] = old_times[old_start + offset]
                # In a trailing replacement, new words beyond the old span
                # may be newly heard speech. An insertion inside an existing
                # clause remains as old as its neighboring corrected words.
                if tag == "replace" and not (old_end == len(old_words) and new_end == len(words)):
                    result[new_start + paired : new_end] = [old_times[old_end - 1]] * (
                        new_end - new_start - paired
                    )
            elif tag == "insert" and old_start < len(old_times):
                result[new_start:new_end] = [old_times[old_start]] * (new_end - new_start)
        # Recover deleted-and-restored prefix words without re-aligning old
        # mistakes to unrelated later words (for example an old "of" to a
        # newly spoken "out of the window"). This also preserves age after an
        # empty withdrawal and a subsequent restoration of the same words.
        for historical_words, historical_times in self.observation_history:
            for index, (old, new) in enumerate(zip(historical_words, words)):
                if key(old) != key(new):
                    break
                result[index] = min(result[index], historical_times[index])
        return result

    def _event(
        self, *, final: bool = False, final_reason: str | None = None, activity_only: bool = False
    ) -> dict[str, Any]:
        observed_at = self.start_time + self.processed_samples / SAMPLE_RATE
        advanced = False
        decode_seconds = None
        if activity_only:
            text = self.last_text
            first_seen = self.word_first_seen.copy() if text else []
        else:
            advanced = self.memory_len > self.last_decode_frames
            started = time.perf_counter()
            try:
                text = self._decode()
            finally:
                decode_seconds = time.perf_counter() - started
                self.last_decode_seconds = decode_seconds
            self.last_decode_frames = self.memory_len
            if text != self.last_text:
                self.revision += 1
            self.last_text = text
            words = text.split()
            first_seen = self._word_observation_times(words, observed_at)
            if words:
                if words != self.observed_words:
                    self.observation_history.append((words.copy(), first_seen.copy()))
                self.observed_words = words
                self.word_first_seen = first_seen
        context_end = self.segment_start + self.segment_samples / SAMPLE_RATE
        encoded_samples = min(self.segment_samples, self.emitted * (CHUNK_SAMPLES // NEW_FRAMES))
        # Progress is explicit: activity-only events and an unchanged final
        # with no newly encoded frames cannot count as a second observation.
        return {
            "id": self.segment_id,
            "text": text,
            "start": self.segment_start,
            "end": context_end,
            "event_kind": "activity" if activity_only else "hypothesis",
            "is_final": final,
            "revision": self.revision,
            "is_final_scope": "asr_context",
            "hypothesis_advanced": advanced,
            "decode_seconds": decode_seconds,
            "next_partial_seconds": self._next_partial_frames() * ENCODER_HOP_SAMPLES / SAMPLE_RATE,
            "received_until": self.start_time + self.received_samples / SAMPLE_RATE,
            "encoded_until": self.segment_start + encoded_samples / SAMPLE_RATE,
            # Observation time may include real right context beyond the
            # center end. It is still bounded by genuinely processed PCM.
            "evidence_until": max(context_end, observed_at),
            "alignment_kind": "region_bound",
            "observed_audio_time": observed_at,
            "word_first_seen": first_seen,
            "activity_start": self.activity_start,
            "last_speech_audio_time": self.last_speech_audio_time,
            "pause_detected": self.pause_detected,
            "paused_at": self.paused_at,
            "final_reason": final_reason,
            "context_end_reason": final_reason,
            "continuation_expected": final_reason in {"capacity_pause", "capacity_limit"},
            "word_time_basis": "first_asr_observation_not_forced_alignment",
        }

    def _feed_frontend(self, block: np.ndarray, actual_samples: int) -> None:
        result = self._run(
            "frontend",
            {
                "audio_chunk": block.reshape(1, -1),
                "conv1_buffer": self.conv1,
                "conv2_buffer": self.conv2,
            },
        )
        self.conv1 = result["conv1_buffer_out"]
        self.conv2 = result["conv2_buffer_out"]
        count = min(NEW_FRAMES, (actual_samples + ENCODER_HOP_SAMPLES - 1) // ENCODER_HOP_SAMPLES)
        self.features = np.concatenate([self.features, result["features"][0, :count]], axis=0)
        self.last_frontend_audio = block[:actual_samples].copy()

    def _append_segment_audio(self, values: np.ndarray) -> None:
        self.segment_audio = np.concatenate([self.segment_audio, values])
        self.segment_samples += len(values)
        while len(self.segment_audio) >= CHUNK_SAMPLES:
            block = self.segment_audio[:CHUNK_SAMPLES].copy()
            self.segment_audio = self.segment_audio[CHUNK_SAMPLES:]
            self._feed_frontend(block, CHUNK_SAMPLES)

    def _finish_context(self, reason: str) -> list[dict[str, Any]]:
        if reason != "input_end":
            raise RuntimeError(
                "Internal context endings require real right context, not EOF padding"
            )
        if len(self.segment_audio):
            count = len(self.segment_audio)
            self._feed_frontend(np.pad(self.segment_audio, (0, CHUNK_SAMPLES - count)), count)
            self.segment_audio = np.empty(0, np.float32)
        self._encode_available(flush=True)
        event = self._event(final=True, final_reason=reason)
        self._reset_context()
        return [event]

    def _begin_rollover(self, reason: str) -> list[dict[str, Any]]:
        """Close a block-aligned center after one real frontend block of lookahead.

        The trailing PCM belongs to the next context and is replayed there;
        it must never advance the global input cursor a second time.
        """
        requested_samples = self.segment_samples
        center_samples = requested_samples // CHUNK_SAMPLES * CHUNK_SAMPLES
        if center_samples == 0:
            return []
        trailing = self.segment_audio.copy()
        # A long silence usually already contains a complete right-context
        # block. Reuse it immediately if moving the center back stays silent.
        earlier = center_samples - CHUNK_SAMPLES
        if (
            reason == "long_silence"
            and earlier >= CHUNK_SAMPLES
            and self.last_speech_audio_time is not None
            and self.segment_start + earlier / SAMPLE_RATE >= self.last_speech_audio_time - 1e-7
        ):
            center_samples = earlier
            trailing = np.concatenate([self.last_frontend_audio, trailing])
        center_frames = center_samples // ENCODER_HOP_SAMPLES
        if self.emitted > center_frames:
            raise RuntimeError(
                "Moonshine rollover cannot move behind already emitted center frames"
            )
        center_end = self.segment_start + center_samples / SAMPLE_RATE
        center_sample = int(round((center_end - self.start_time) * SAMPLE_RATE))
        self.pending_rollover = {
            "reason": reason,
            "requested_end": self.segment_start + requested_samples / SAMPLE_RATE,
            "center_end": center_end,
            "center_sample": center_sample,
            "center_frames": center_frames,
            "audio": trailing,
        }
        self.segment_samples = center_samples
        self.segment_audio = np.empty(0, np.float32)
        if self.last_speech_audio_time is not None:
            self.last_speech_audio_time = min(self.last_speech_audio_time, center_end)
        if self.activity_start is not None and self.activity_start > center_end:
            self.activity_start = None
        if len(trailing) >= CHUNK_SAMPLES:
            return self._complete_rollover()
        return []

    def _advance_rollover(
        self, frame: np.ndarray, sample_start: int, *, input_end: bool
    ) -> list[dict[str, Any]]:
        rollover = self.pending_rollover
        if rollover is None:
            raise RuntimeError("Missing Moonshine pending rollover")
        if sample_start != rollover["center_sample"] + len(rollover["audio"]):
            raise RuntimeError("Moonshine rollover right-context PCM is not contiguous")
        rollover["audio"] = np.concatenate([rollover["audio"], frame])
        if len(rollover["audio"]) >= CHUNK_SAMPLES or input_end:
            return self._complete_rollover(input_end=input_end)
        return []

    def _complete_rollover(self, *, input_end: bool = False) -> list[dict[str, Any]]:
        rollover = self.pending_rollover
        if rollover is None:
            return []
        trailing = rollover["audio"]
        center_frames = rollover["center_frames"]
        if len(trailing) < CHUNK_SAMPLES and not input_end:
            raise RuntimeError("Moonshine internal rollover needs a real right-context block")
        # Long-silence completion may already have encoded this full block.
        # Only actual stream EOF may supply a shorter block or masked tail.
        if len(self.features) < center_frames + NEW_FRAMES and len(trailing):
            actual = min(CHUNK_SAMPLES, len(trailing))
            block = trailing[:actual]
            if actual < CHUNK_SAMPLES:
                block = np.pad(block, (0, CHUNK_SAMPLES - actual))
            self._feed_frontend(block, actual)
        self._encode_available(flush=input_end, center_frames=center_frames)
        if self.emitted != center_frames:
            raise RuntimeError("Moonshine rollover did not encode its complete center")
        event = self._event(final=True, final_reason=rollover["reason"])
        event.update(
            rollover_requested_end=rollover["requested_end"],
            rollover_center_end=rollover["center_end"],
            rollover_boundary_adjustment_seconds=rollover["requested_end"] - rollover["center_end"],
            right_context_until=rollover["center_end"]
            + min(len(trailing), CHUNK_SAMPLES) / SAMPLE_RATE,
            right_context_samples=min(len(trailing), CHUNK_SAMPLES),
            right_context_truncated_at_input_end=input_end and len(trailing) < CHUNK_SAMPLES,
            rollover_replayed_samples=len(trailing),
        )
        replay_start = rollover["center_sample"]
        self._reset_context()
        events = [event]
        for offset in range(0, len(trailing), VAD_FRAME_SAMPLES):
            frame = trailing[offset : offset + VAD_FRAME_SAMPLES]
            events.extend(
                self._frame(
                    frame,
                    sample_start=replay_start + offset,
                    force_final=input_end and offset + len(frame) == len(trailing),
                )
            )
        return events

    def _frame(
        self, frame: np.ndarray, *, force_final: bool = False, sample_start: int | None = None
    ) -> list[dict[str, Any]]:
        """Advance an energy-only VAD by at most 20 ms of received audio."""
        events = []
        actual_samples = len(frame)
        sample_start = self.processed_samples if sample_start is None else sample_start
        self.processed_samples = max(self.processed_samples, sample_start + actual_samples)
        if self.pending_rollover is not None:
            return self._advance_rollover(frame, sample_start, input_end=force_final)
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) if actual_samples else 0.0
        frame_start = self.start_time + sample_start / SAMPLE_RATE
        frame_end = frame_start + actual_samples / SAMPLE_RATE
        activity_changed = False
        if not self.active:
            self.onset_samples = (
                self.onset_samples + actual_samples if rms >= self.energy_threshold else 0
            )
            if self.onset_samples < int(0.04 * SAMPLE_RATE):
                self.pre_roll = np.concatenate([self.pre_roll, frame])[-CHUNK_SAMPLES:].copy()
                return events
            self.active = True
            self.segment_id = self._next_id
            self._next_id += 1
            self.segment_start = frame_start - len(self.pre_roll) / SAMPLE_RATE
            self.activity_start = frame_end - self.onset_samples / SAMPLE_RATE
            self._append_segment_audio(self.pre_roll)
            self.pre_roll = np.empty(0, np.float32)
            activity_changed = True
        self._append_segment_audio(frame)
        if rms >= self.release_threshold:
            if self.pause_detected:
                self.activity_start = frame_start
                activity_changed = True
            self.pause_detected = False
            self.paused_at = None
            self.silent_samples = 0
            self.last_speech_audio_time = frame_end
        else:
            self.silent_samples += actual_samples
            if not self.pause_detected and self.silent_samples >= int(
                self.pause_seconds * SAMPLE_RATE
            ):
                self.pause_detected = True
                self.paused_at = self.last_speech_audio_time
                activity_changed = True
        # Ordinary short pauses only describe activity. Near capacity, borrow
        # an existing pause for rollover before the hard limit cuts speech.
        reason = "input_end" if force_final else None
        if reason is None and self.silent_samples >= int(self.silence_seconds * SAMPLE_RATE):
            reason = "long_silence"
        if (
            reason is None
            and self.segment_samples >= int(self.capacity_pause_from_seconds * SAMPLE_RATE)
            and self.pause_detected
            and self.silent_samples >= int(self.pause_seconds * SAMPLE_RATE)
            and self.last_speech_audio_time is not None
            and self.segment_start
            + (self.segment_samples // CHUNK_SAMPLES * CHUNK_SAMPLES) / SAMPLE_RATE
            >= self.last_speech_audio_time - 1e-7
        ):
            # This is a capacity rollover borrowing a pause, not a semantic
            # sentence boundary. The hard limit remains the fallback.
            reason = "capacity_pause"
        if reason is None and self.segment_samples >= int(self.max_segment_seconds * SAMPLE_RATE):
            reason = "capacity_limit"
        if reason is not None:
            return (
                self._finish_context(reason)
                if reason == "input_end"
                else self._begin_rollover(reason)
            )
        self._encode_available()
        if self.memory_len - self.last_decode_frames >= self._next_partial_frames():
            events.append(self._event())
        elif activity_changed:
            events.append(self._event(activity_only=True))
        return events

    def accept_audio(self, audio: np.ndarray, final: bool = False) -> list[dict[str, Any]]:
        values = np.asarray(audio, dtype=np.float32)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Moonshine audio must be finite mono float32 at 16000 Hz")
        self.received_samples += len(values)
        self.pending = np.concatenate([self.pending, values])
        events = []
        while len(self.pending) >= VAD_FRAME_SAMPLES:
            frame = self.pending[:VAD_FRAME_SAMPLES].copy()
            self.pending = self.pending[VAD_FRAME_SAMPLES:]
            events.extend(self._frame(frame, force_final=final and len(self.pending) == 0))
        if final:
            if len(self.pending):
                frame = self.pending.copy()
                self.pending = np.empty(0, np.float32)
                events.extend(self._frame(frame, force_final=True))
            elif self.pending_rollover is not None:
                events.extend(self._complete_rollover(input_end=True))
            elif self.active:
                events.extend(self._finish_context("input_end"))
            self.pre_roll = np.empty(0, np.float32)
        return events

    def close(self) -> None:
        self.requests.clear()
        self.models.clear()
        gc.collect()
