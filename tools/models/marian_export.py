"""Offline Marian graph export, derived from MTranServer's Apache-2.0 converter.

Reference: src/converter/graph.ts at ea3e8b1523c70115d16f2d48027daaf9caeb3e8e.
See third_party/MTranServer-NOTICE and MTranServer-LICENSE.
"""

from __future__ import annotations
from pathlib import Path
import json
import struct
from typing import Any
import numpy as np
from npu_subtitles.translation.languages import MARIAN_PAIRS


def read_release(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read/dequantize Marian intgemm storage, without running neural inference."""
    import yaml

    data = path.read_bytes()
    offset = 0

    def take(size: int) -> memoryview:
        nonlocal offset
        if size < 0 or offset + size > len(data):
            raise ValueError(f"Truncated Marian model at {offset}")
        value = memoryview(data)[offset : offset + size]
        offset += size
        return value

    version, count = struct.unpack("<QQ", take(16))
    if version != 1 or count > 10000:
        raise ValueError("Unsupported Marian header")
    headers = [list(struct.unpack("<QQQQ", take(32))) for _ in range(count)]
    for header in headers:
        name = bytes(take(header[0]))
        if not name.endswith(b"\0"):
            raise ValueError("Unterminated Marian tensor name")
        header.append(name[:-1].decode("utf-8"))
    for header in headers:
        rank = header[2]
        if not 1 <= rank <= 8:
            raise ValueError("Unsupported Marian rank")
        dims = struct.unpack("<" + "i" * rank, take(rank * 4))
        if any(dim <= 0 for dim in dims):
            raise ValueError("Invalid Marian dimensions")
        header.append(dims)
    take(struct.unpack("<Q", take(8))[0])
    weights = {}
    config = None
    for _, kind, _, size, name, dims in headers:
        raw = take(size)
        n = int(np.prod(dims))
        if name == "special:model.yml" and kind == 0x101:
            config = yaml.safe_load(bytes(raw[:n]).rstrip(b"\0").decode("utf-8"))
            continue
        if kind == 0x404:
            value = np.frombuffer(raw, dtype="<f4", count=n).copy()
        elif kind in (0x4101, 0x4102):
            dtype = np.dtype("i1" if kind == 0x4101 else "<i2")
            multiplier = struct.unpack_from("<f", raw, n * dtype.itemsize)[0]
            if not np.isfinite(multiplier) or multiplier <= 0:
                raise ValueError("Invalid Marian quantization multiplier")
            value = np.frombuffer(raw, dtype=dtype, count=n).astype(np.float32)
            value *= np.float32(1.0 / multiplier)
            if "Wemb" not in name:
                if len(dims) != 2:
                    raise ValueError("Non-matrix intgemm tensor")
                value = value.reshape(dims[1], dims[0]).T.copy().reshape(-1)
        else:
            raise ValueError(f"Unsupported Marian tensor type: {kind:#x}")
        if name in weights:
            raise ValueError("Duplicate Marian tensor")
        weights[name] = value.reshape(dims)
    if config is None or offset != len(data):
        raise ValueError("Missing metadata or trailing bytes")
    required = {
        "type": "transformer",
        "dec-cell": "ssru",
        "transformer-decoder-autoreg": "rnn",
        "transformer-ffn-activation": "relu",
        "transformer-postprocess": "dan",
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"Unsupported architecture {key}={config.get(key)!r}")
    if (
        config.get("transformer-preprocess")
        or config.get("transformer-postprocess-top")
        or config.get("right-left")
        or config.get("transformer-train-position-embeddings")
        or config.get("transformer-rnn-projection")
        or config.get("transformer-no-projection")
        or config.get("transformer-tied-layers")
    ):
        raise ValueError("Unsupported Marian architecture options")
    if not (config.get("tied-embeddings") or config.get("tied-embeddings-all")):
        raise ValueError("Untied decoder projection is unsupported")
    return weights, config


class _Graph:
    def __init__(self, weights: dict, config: dict):
        from openvino import opset13 as op

        self.op, self.weights, self.config = op, weights, config
        self.dim = int(config["dim-emb"])
        self.heads = int(config["transformer-heads"])
        self.depth = int(config["dec-depth"])
        self.emb = "Wemb" if config.get("tied-embeddings-all") else "decoder_Wemb"
        self.constants: dict[str, Any] = {}

    def c(self, value, dtype=np.float32):
        return self.op.constant(np.asarray(value, dtype=dtype))

    def w(self, name):
        if name not in self.constants:
            value = self.weights[name]
            if name == "decoder_ff_logit_out_b":
                value = value.copy()
                value.reshape(-1)[1] = -10000.0  # Marian suppresses UNK id 1.
            self.constants[name] = self.c(value)
        return self.constants[name]

    def reshape(self, x, shape):
        return self.op.reshape(x, self.c(shape, np.int64), False)

    def transpose(self, x, axes):
        return self.op.transpose(x, self.c(axes, np.int64))

    def linear(self, x, prefix, bias=""):
        value = self.op.matmul(x, self.w(prefix), False, False)
        return self.op.add(value, self.w(bias)) if bias else value

    def norm(self, x, prefix):
        # Equivalent to converter InstanceNormalization over each embedding.
        mean = self.op.reduce_mean(x, self.c([-1], np.int64), True)
        delta = self.op.subtract(x, mean)
        variance = self.op.reduce_mean(self.op.multiply(delta, delta), self.c([-1], np.int64), True)
        normal = self.op.divide(delta, self.op.sqrt(self.op.add(variance, self.c(1e-6))))
        return self.op.add(
            self.op.multiply(normal, self.w(prefix + "_scale")), self.w(prefix + "_bias")
        )

    def heads_of(self, x, length):
        return self.transpose(
            self.reshape(x, [1, length, self.heads, self.dim // self.heads]), [0, 2, 1, 3]
        )

    def attention(self, x, prefix, k, v, mask, length):
        q = self.heads_of(self.linear(x, prefix + "_Wq", prefix + "_bq"), length)
        scores = self.op.multiply(
            self.op.matmul(q, k, False, False), self.c(1 / np.sqrt(self.dim // self.heads))
        )
        scores = self.op.add(scores, mask)
        probs = self.op.softmax(scores, -1)
        y = self.reshape(
            self.transpose(self.op.matmul(probs, v, False, False), [0, 2, 1, 3]),
            [1, 1, length, self.dim],
        )
        y = self.linear(y, prefix + "_Wo", prefix + "_bo")
        return self.norm(self.op.add(y, x), prefix + "_Wo_ln")

    def ffn(self, x, prefix):
        y = x
        depth = int(self.config["transformer-ffn-depth"])
        for layer in range(1, depth + 1):
            y = self.linear(y, f"{prefix}_W{layer}", f"{prefix}_b{layer}")
            if layer < depth:
                y = self.op.relu(y)
        return self.norm(self.op.add(y, x), prefix + "_ffn_ln")

    def positions(self, count):
        frequency = np.asarray(
            [1e-4 ** ((i % (self.dim // 2)) / (self.dim // 2 - 1)) for i in range(self.dim)],
            dtype=np.float32,
        )
        offsets = np.asarray(
            [(i // (self.dim // 2)) * np.pi / 2 for i in range(self.dim)], dtype=np.float32
        )
        return np.sin(np.arange(count, dtype=np.float32)[:, None] * frequency + offsets).astype(
            np.float32
        )


def convert_model(
    directory: Path, source_length: int = 128, max_target_length: int = 384, *, pair: str = "en-zh"
) -> None:
    import openvino as ov
    from openvino import opset13 as op

    if source_length < 8 or max_target_length < 8:
        raise ValueError("Static lengths must be at least 8")
    if pair not in MARIAN_PAIRS:
        raise ValueError("Unsupported Marian pair; use the fixed subtitle language set")
    pair_code = pair.replace("-", "")
    weights, config = read_release(directory / f"model.{pair_code}.intgemm.alphas.bin")
    print(json.dumps(config, ensure_ascii=False), flush=True)
    b = _Graph(weights, config)
    inputs = [
        op.parameter([source_length], np.int32, name="input_ids"),
        op.parameter([1, 1, 1, source_length], np.float32, name="source_mask"),
    ]
    source_emb = b.emb if config.get("tied-embeddings-all") else "encoder_Wemb"
    x = op.gather(b.w(source_emb), inputs[0], b.c(0, np.int64))
    x = b.reshape(x, [1, 1, source_length, b.dim])
    x = op.add(
        op.multiply(x, b.c(np.sqrt(b.dim))),
        b.c(b.positions(source_length).reshape(1, 1, source_length, b.dim)),
    )
    for layer in range(1, int(config["enc-depth"]) + 1):
        prefix = f"encoder_l{layer}_self"
        keys = b.transpose(
            b.heads_of(b.linear(x, prefix + "_Wk", prefix + "_bk"), source_length), [0, 1, 3, 2]
        )
        values = b.heads_of(b.linear(x, prefix + "_Wv", prefix + "_bv"), source_length)
        x = b.ffn(
            b.attention(x, prefix, keys, values, inputs[1], source_length), f"encoder_l{layer}_ffn"
        )
    outputs = []
    for layer in range(1, b.depth + 1):
        prefix = f"decoder_l{layer}_context"
        k = b.transpose(
            b.heads_of(b.linear(x, prefix + "_Wk", prefix + "_bk"), source_length), [0, 1, 3, 2]
        )
        v = b.heads_of(b.linear(x, prefix + "_Wv", prefix + "_bv"), source_length)
        for name, value in ((f"key_{layer}", k), (f"value_{layer}", v)):
            value.output(0).get_tensor().set_names({name})
            outputs.append(value)
    encoder = ov.Model(outputs, inputs, "mtran_encoder_static")
    ov.save_model(encoder, directory / "encoder.xml", compress_to_fp16=True)
    del encoder, inputs, outputs, b, x
    b = _Graph(weights, config)
    token = op.parameter([1], np.int32, name="previous_token")
    position = op.parameter([1], np.int32, name="position")
    first = op.parameter([1], np.float32, name="embedding_multiplier")
    mask = op.parameter([1, 1, 1, source_length], np.float32, name="source_mask")
    inputs = [token, position, first, mask]
    x = b.reshape(op.gather(b.w(b.emb), token, b.c(0, np.int64)), [1, 1, 1, b.dim])
    x = op.multiply(x, first)  # First decoder embedding is zero, not a BOS embedding.
    positions = op.gather(b.c(b.positions(max_target_length)), position, b.c(0, np.int64))
    x = op.add(op.multiply(x, b.c(np.sqrt(b.dim))), b.reshape(positions, [1, 1, 1, b.dim]))
    outputs = []
    for layer in range(1, b.depth + 1):
        keys = op.parameter(
            [1, b.heads, b.dim // b.heads, source_length], np.float32, name=f"key_{layer}"
        )
        values = op.parameter(
            [1, b.heads, source_length, b.dim // b.heads], np.float32, name=f"value_{layer}"
        )
        previous = op.parameter([1, 1, 1, b.dim], np.float32, name=f"cell_{layer}")
        inputs.extend([keys, values, previous])
        prefix = f"decoder_l{layer}_rnn"
        gate = op.sigmoid(b.linear(x, prefix + "_Wf", prefix + "_bf"))
        cell = op.add(
            op.multiply(gate, previous),
            op.multiply(op.subtract(b.c(1), gate), b.linear(x, prefix + "_W")),
        )
        cell.output(0).get_tensor().set_names({f"next_cell_{layer}"})
        outputs.append(cell)
        x = b.norm(op.add(op.relu(cell), x), prefix + "_ffn_ln")
        x = b.attention(x, f"decoder_l{layer}_context", keys, values, mask, 1)
        x = b.ffn(x, f"decoder_l{layer}_ffn")
    logits = op.add(
        op.matmul(b.reshape(x, [1, b.dim]), b.w(b.emb), False, True), b.w("decoder_ff_logit_out_b")
    )
    logits.output(0).get_tensor().set_names({"logits"})
    decoder = ov.Model([logits, *outputs], inputs, "mtran_decoder_step_static")
    ov.save_model(decoder, directory / "decoder.xml", compress_to_fp16=True)
    metadata = {
        "format": "mtran-npu-static-v1",
        "source_length": source_length,
        "max_target_length": max_target_length,
        "dim": b.dim,
        "heads": b.heads,
        "decoder_depth": b.depth,
        "encoder_depth": config["enc-depth"],
        "weight_format": "Mozilla intgemm dequantized to FP16 storage",
        "decoding": "greedy; host token loop; external SSRU state",
        "mtran_reference_commit": "ea3e8b1523c70115d16f2d48027daaf9caeb3e8e",
        "marian_config": config,
    }
    metadata.update(
        source_language=pair.split("-")[0],
        target_language="zh-Hans" if pair == "en-zh" else "en",
        source_vocab="srcvocab.enzh.spm" if pair == "en-zh" else f"vocab.{pair_code}.spm",
        target_vocab="trgvocab.enzh.spm" if pair == "en-zh" else f"vocab.{pair_code}.spm",
    )
    (directory / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Converted static encoder/decoder to {directory}; no neural inference performed.",
        flush=True,
    )
