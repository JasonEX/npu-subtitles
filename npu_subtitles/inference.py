"""Single NPU worker and explicitly verified model adapters."""

from __future__ import annotations

from npu_subtitles.devices import require_npu

from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, Callable
from npu_subtitles.memory import process_memory

from npu_subtitles.paths import ROOT, SAMPLE_RATE


@dataclass(frozen=True)
class InferenceSettings:
    asr_model: str
    translation_model: str
    cache_dir: str
    vad_threshold: float = 0.008
    vad_silence: float = 1.0
    asr_max_seconds: float = 12.0
    person_names: tuple[str, ...] = ()
    asr_backend: str = "moonshine"
    source_language: str = "en"
    japanese_translation_model: str | None = None


def create_backends(
    asr_model: str,
    translation_model: str,
    cache_dir: str,
    progress: Callable[[dict], None] | None = None,
    *,
    vad_threshold: float = 0.008,
    vad_silence: float = 1.0,
    asr_max_seconds: float = 12.0,
    person_names: tuple[str, ...] = (),
    asr_backend: str = "moonshine",
    source_language: str = "en",
    japanese_translation_model: str | None = None,
) -> tuple[Any, Any]:
    """An injectable factory for the two dedicated NPU adapters; no fallback."""
    from npu_subtitles.translation.marian import MTranNPUTranslator
    from npu_subtitles.captions.language import check_language

    check_language(source_language)
    if source_language == "ja" and (asr_backend != "granite" or person_names):
        raise ValueError("Japanese requires Granite and does not accept English person-name rules")
    if asr_backend == "moonshine":
        from npu_subtitles.asr.moonshine import MoonshineNPU

        adapter = MoonshineNPU
    elif asr_backend == "granite":
        from npu_subtitles.asr.granite import GraniteNPU

        adapter = GraniteNPU
    else:
        raise ValueError(f"Unknown ASR backend: {asr_backend}")
    asr = adapter(
        model_dir=asr_model,
        cache_dir=str(Path(cache_dir) / asr_backend),
        energy_threshold=vad_threshold,
        silence_seconds=vad_silence,
        max_segment_seconds=asr_max_seconds,
        **({"source_language": source_language} if asr_backend == "granite" else {}),
    )
    translator = None
    try:
        if progress is not None:
            progress(
                {
                    "kind": "startup",
                    "stage": "asr_loaded",
                    "memory": process_memory(),
                    "execution_devices": asr.execution_devices,
                    "asr_details": asr.details(),
                }
            )
        if source_language == "ja":
            from npu_subtitles.translation.japanese import JapaneseChineseNPUTranslator

            translator = JapaneseChineseNPUTranslator(
                japanese_model=japanese_translation_model or ROOT / "models/mtran-ja-en-npu",
                chinese_model=translation_model,
                cache_dir=Path(cache_dir) / "japanese-translation",
            )
        else:
            translator = MTranNPUTranslator(
                model_dir=translation_model,
                cache_dir=str(Path(cache_dir) / "mtran"),
                person_names=person_names,
            )
            if translator.source_language != "en" or translator.target_language != "zh-Hans":
                raise ValueError("English subtitles require an en -> zh-Hans translator")
        if progress is not None:
            progress({"kind": "startup", "stage": "translation_loaded", "memory": process_memory()})
    except BaseException:
        if translator is not None:
            translator.close()
        asr.close()
        raise
    return asr, translator


def inference_worker(
    connection: Any,
    settings: InferenceSettings,
    *,
    factory: Callable[..., tuple[Any, Any]] = create_backends,
    cancellation: Any = None,
) -> None:
    """One bounded worker serializes both NPU models; only one task is in flight."""
    asr = translator = None
    try:
        import numpy as np

        connection.send({"kind": "startup", "stage": "before_models", "memory": process_memory()})
        asr, translator = factory(**asdict(settings), progress=connection.send)
        asr_devices = asr.execution_devices
        translation_details = translator.details()
        translation_devices = translation_details.get("execution_devices", {})
        require_npu(asr_devices, "ASR")
        require_npu(translation_devices, "MTran")
        connection.send(
            {
                "kind": "ready",
                "asr_execution_devices": asr_devices,
                "asr_details": asr.details(),
                "translation_details": translation_details,
                "memory": process_memory(),
            }
        )
        generation = None
        next_audio_sample = None
        while True:
            task = connection.recv()
            kind = task["kind"]
            if kind == "stop":
                connection.send({"kind": "stopped", "memory": process_memory()})
                break
            if kind == "audio":
                if task["end"] > task["read_limit"] + 1e-7:
                    raise RuntimeError(
                        "Audio exceeds this playback snapshot's bounded read horizon"
                    )
                samples = np.frombuffer(task["pcm"], dtype="<f4").copy()
                expected = round((task["end"] - task["start"]) * SAMPLE_RATE)
                if samples.size != expected:
                    raise RuntimeError("Audio sample count does not match its declared interval")
                if generation != task["generation"]:
                    asr.reset(start_time=task["start"])
                    generation = task["generation"]
                    next_audio_sample = round(task["start"] * SAMPLE_RATE)
                if round(task["start"] * SAMPLE_RATE) != next_audio_sample:
                    raise RuntimeError("Noncontiguous audio requires an explicit timeline reset")
                updates = asr.accept_audio(samples, final=bool(task.get("final", False)))
                next_audio_sample = round(task["end"] * SAMPLE_RATE)
                # Only JSON-compatible scalars cross into the presentation loop.
                normalized = [
                    {
                        **event,
                        "id": int(event["id"]),
                        "revision": int(event["revision"]),
                        "text": " ".join(str(event["text"]).split()),
                        "start": float(event["start"]),
                        "end": float(event["end"]),
                        "is_final": bool(event["is_final"]),
                    }
                    for event in updates
                ]
                connection.send(
                    {
                        "kind": "asr",
                        "generation": generation,
                        "updates": normalized,
                        "audio_start": task["start"],
                        "audio_end": task["end"],
                        "seen_playback_position": task["seen_playback_position"],
                        "read_limit": task["read_limit"],
                        "flushed": bool(task.get("final", False)),
                        "memory": process_memory(),
                    }
                )
            elif kind == "translate":
                from npu_subtitles.translation.marian import TranslationExpired, TranslationRejected

                try:
                    cancelled = (
                        (lambda: cancellation.value != task["ticket"])
                        if cancellation is not None
                        else None
                    )
                    text = translator.translate(
                        task["text"], deadline=task.get("deadline"), cancelled=cancelled
                    )
                except TranslationExpired:
                    connection.send(
                        {
                            "kind": "translation_cancelled"
                            if translator.last_result.get("status") == "cancelled"
                            else "translation_expired",
                            "generation": task["generation"],
                            "id": task["id"],
                            "revision": task["revision"],
                            "source": task["text"],
                            "details": translator.last_result,
                            "memory": process_memory(),
                        }
                    )
                    continue
                except TranslationRejected as exc:
                    connection.send(
                        {
                            "kind": "translation_failed",
                            "generation": task["generation"],
                            "id": task["id"],
                            "revision": task["revision"],
                            "source": task["text"],
                            "error": str(exc),
                            "details": translator.last_result,
                            "memory": process_memory(),
                        }
                    )
                    continue
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError("MTran returned an empty translation")
                connection.send(
                    {
                        "kind": "translation",
                        "generation": task["generation"],
                        "id": task["id"],
                        "revision": task["revision"],
                        "source": task["text"],
                        "text": " ".join(text.split()),
                        "details": translator.last_result,
                        "memory": process_memory(),
                    }
                )
            else:
                raise ValueError(f"Unknown inference task: {kind}")
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as exc:
        try:
            connection.send({"kind": "error", "error": f"{type(exc).__name__}: {exc}"})
        except (OSError, EOFError):
            pass
    finally:
        for backend in (translator, asr):
            if backend is not None:
                try:
                    backend.close()
                except Exception:
                    pass
        connection.close()
