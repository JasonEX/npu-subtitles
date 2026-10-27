#!/usr/bin/env python3
"""Open a movie with automatically prepared, bounded-lookahead NPU subtitles.

Playback starts without a transcription step. Audio may lead playback by a small
configured window; caption preparation never grants permission to display early.
All neural graphs must execute on NPU. No CPU inference fallback is available.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import multiprocessing as mp
from pathlib import Path
import subprocess
import time
from typing import Any, Callable
import uuid

from npu_subtitles.captions.pipeline import CaptionPipeline
from npu_subtitles.inference import InferenceSettings, create_backends, inference_worker
from npu_subtitles.playback import AudioReader, CaptionDisplay, create_live_player
from npu_subtitles.timeline import MediaTimeline, SAMPLE_RATE
from npu_subtitles.media import executable, probe_media
from npu_subtitles.memory import power_status, process_memory

from npu_subtitles.ass import DEFAULT_FONT
from npu_subtitles.paths import ROOT, SPEECH_RMS_THRESHOLD


def parser(*, optional_video: bool = False) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    result.add_argument("video", type=Path, **({"nargs": "?"} if optional_video else {}))
    result.add_argument("--player", choices=["mpv", "mpc-hc"], default="mpv")
    result.add_argument(
        "--asr",
        choices=["moonshine", "granite"],
        default="moonshine",
        help="NPU speech recognition backend; Granite needs its prepared static model",
    )
    result.add_argument(
        "--language",
        dest="source_language",
        choices=["en", "ja"],
        default="en",
        help="Source speech language; Japanese requires --asr granite",
    )
    model = result.add_mutually_exclusive_group()
    model.add_argument("--moonshine", choices=["small", "medium"], default="small")
    model.add_argument("--asr-model", type=Path)
    result.add_argument("--translation-model", type=Path, default=ROOT / "models/mtran-en-zh-npu")
    result.add_argument(
        "--japanese-translation-model",
        type=Path,
        default=ROOT / "models/mtran-ja-en-npu",
        help="Japanese-to-English NPU stage; followed by --translation-model on the same deadline",
    )
    result.add_argument("--audio-track", type=int, default=0)
    result.add_argument("--start", type=float, default=0)
    result.add_argument("--duration", type=float)
    result.add_argument(
        "--lookahead-seconds",
        type=float,
        default=8,
        help="Automatic audio preparation ahead of playback, 0..12 seconds; no full-file generation",
    )
    result.add_argument("--chunk-seconds", type=float, default=0.8)
    result.add_argument(
        "--causal-margin", type=float, default=0.08, help="Safety margin at the read horizon"
    )
    result.add_argument(
        "--max-backlog",
        type=float,
        default=2,
        help="Explicitly reset an audio stream that falls this far behind",
    )
    result.add_argument("--vad-threshold", type=float, default=SPEECH_RMS_THRESHOLD)
    result.add_argument(
        "--vad-silence",
        type=float,
        default=1.0,
        help="Long silence ending ASR context, not subtitle sentence length",
    )
    result.add_argument(
        "--asr-max-seconds",
        type=float,
        default=12,
        help="Audio-window capacity, at most 12 seconds; Granite logical text can continue across windows",
    )
    result.add_argument(
        "--caption-max-lag",
        type=float,
        default=3,
        help="Admission budget after the estimated cue due time; not measured word alignment",
    )
    result.add_argument(
        "--caption-seconds",
        type=float,
        default=6,
        help="Base media-time retention; aligned Granite cues also account for bounded spoken duration",
    )
    result.add_argument(
        "--read-seconds", type=float, default=1.2, help="Minimum Chinese reading opportunity"
    )
    result.add_argument("--translation-budget", type=float, default=1.5)
    result.add_argument("--end-grace", type=float, default=3)
    result.add_argument("--person-name", action="append", default=[])
    result.add_argument("--timeout", type=float, default=600)
    result.add_argument("--sub-font", default=DEFAULT_FONT)
    result.add_argument("--sub-font-size", type=float, default=24)
    result.add_argument("--output-dir", type=Path, default=ROOT / "outputs/live")
    result.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/live-npu")
    result.add_argument("--mpv")
    result.add_argument("--mpc-hc")
    result.add_argument("--ffmpeg")
    result.add_argument("--ffprobe")
    return result


def validate(args: argparse.Namespace) -> tuple[Path, str, float, int]:
    for name in (
        "start",
        "lookahead_seconds",
        "causal_margin",
        "chunk_seconds",
        "max_backlog",
        "vad_silence",
        "asr_max_seconds",
        "caption_max_lag",
        "caption_seconds",
        "read_seconds",
        "translation_budget",
        "end_grace",
        "timeout",
        "sub_font_size",
    ):
        value = getattr(args, name)
        allow_zero = name in ("start", "lookahead_seconds", "causal_margin")
        if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        raise ValueError("--duration must be finite and positive")
    if not math.isfinite(args.vad_threshold) or not 0 <= args.vad_threshold <= 1:
        raise ValueError("--vad-threshold must be in 0..1")
    if args.lookahead_seconds > 12 or args.asr_max_seconds > 12:
        raise ValueError("Read lookahead and ASR audio-window capacity are bounded to 12 seconds")
    if args.chunk_seconds > args.max_backlog:
        raise ValueError("--chunk-seconds must not exceed --max-backlog")
    if args.read_seconds >= args.caption_seconds or args.caption_max_lag > args.caption_seconds:
        raise ValueError("Caption lifetime must accommodate reading time and admission budget")
    if not args.sub_font or any(c in args.sub_font for c in "\\{},\r\n\0"):
        raise ValueError("Invalid subtitle font")
    if args.asr == "granite" and args.moonshine != "small":
        raise ValueError("--moonshine selects Moonshine models; use --asr moonshine with it")
    if args.source_language == "ja" and args.asr != "granite":
        raise ValueError("Japanese live subtitles require --asr granite --language ja")
    if args.source_language == "ja" and args.person_name:
        raise ValueError("--person-name is an English text rule and cannot be used for Japanese")
    if args.asr_model is None:
        args.asr_model = (
            ROOT / "models/granite-speech-5.0-470m-turboctc"
            if args.asr == "granite"
            else ROOT / f"models/moonshine-streaming-{args.moonshine}"
        )
    video = args.video.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    duration, stream_index = probe_media(
        executable(args.ffprobe, "ffprobe"), video, args.audio_track
    )
    if args.start >= duration:
        raise ValueError("--start must precede the movie end")
    media_end = min(duration, args.start + args.duration) if args.duration is not None else duration
    return video, executable(args.ffmpeg, "ffmpeg"), media_end, stream_index


def run(
    args: argparse.Namespace,
    factory: Callable[..., tuple[Any, Any]] = create_backends,
    *,
    attachment: Any = None,
) -> int:
    video, ffmpeg, media_end, stream_index = validate(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex[:10]
    sidecar = args.output_dir.resolve() / f"{video.stem}.live.{session_id}.srt"
    log_path = sidecar.with_suffix(".jsonl")
    endpoint = rf"\\.\pipe\live-npu-{session_id}"
    context = mp.get_context("spawn")
    connection, child = context.Pipe()
    cancellation = context.Value("q", 0)
    settings = InferenceSettings(
        asr_model=str(args.asr_model.resolve()),
        translation_model=str(args.translation_model.resolve()),
        cache_dir=str(args.cache_dir.resolve()),
        vad_threshold=args.vad_threshold,
        vad_silence=args.vad_silence,
        asr_max_seconds=args.asr_max_seconds,
        person_names=tuple(args.person_name),
        asr_backend=args.asr,
        source_language=args.source_language,
        japanese_translation_model=str(args.japanese_translation_model.resolve()),
    )
    worker = context.Process(
        target=inference_worker,
        args=(child, settings),
        kwargs=dict(factory=factory, cancellation=cancellation),
        daemon=True,
    )
    timeline = MediaTimeline(
        args.start,
        media_end,
        lookahead=args.lookahead_seconds,
        safety_margin=args.causal_margin,
        max_backlog=args.max_backlog,
    )
    captions = CaptionPipeline(
        lookahead_seconds=args.lookahead_seconds,
        max_lag=args.caption_max_lag,
        max_hold=args.caption_seconds,
        min_read=args.read_seconds,
        translation_budget=args.translation_budget,
        source_language=args.source_language,
    )
    captions.reset(0)
    player = ipc = reader = display = None
    active = pending_audio = audio_future = None
    audio_future_epoch = None
    last_frame = None
    end_flushed = False
    ending_started = None
    observed_poll_intervals = deque(maxlen=24)
    chunk_samples = max(1, round(args.chunk_seconds * SAMPLE_RATE))
    log_file = log_path.open("w", encoding="utf-8")

    def log(event: str, **values: Any) -> None:
        log_file.write(
            json.dumps(
                {"event": event, "monotonic": time.monotonic(), **values},
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )
        log_file.flush()

    def refresh_caption_budget(
        now: float,
        position: float,
        paused: bool,
        speed: float,
        at_end: bool,
        completed_task: dict | None = None,
    ) -> None:
        if args.asr != "granite":
            return
        captions.update_preparation_budget(
            now,
            position,
            paused=paused and not at_end,
            speed=speed,
            completed_task=completed_task,
            poll_seconds=max(observed_poll_intervals, default=0.025),
            asr_default_seconds=args.chunk_seconds,
            worker=dict(
                kind=active["kind"] if active else None,
                elapsed=max(0.0, now - active["submitted_at"]) if active else 0.0,
                pending_asr=pending_audio is not None and (not paused or at_end),
                asr_urgent=pending_audio is not None and position - pending_audio["start"] > 0.75,
                last_task=last_task,
            ),
        )

    def reset_state(
        reason: str, position: float, now: float, *, move_timeline: bool = False
    ) -> None:
        nonlocal pending_audio, last_frame, end_flushed, ending_started
        if move_timeline:
            timeline.reset(position)
        cancellation.value += 1
        observed_poll_intervals.clear()
        pending_audio = None
        captions.reset(timeline.epoch)
        end_flushed = False
        ending_started = None
        if display is not None:
            display.clear()
        if last_frame is not None:
            log(
                "display_exited",
                key=last_frame[0],
                position=position,
                reason=reason,
                evidence="player_api_returned",
            )
        last_frame = None
        if reader is not None:
            reader.cancel_before(timeline.epoch)
        log("stream_reset", reason=reason, generation=timeline.epoch, position=position)

    try:
        print(f"Loading NPU models. Session log: {log_path}", flush=True)
        log(
            "session",
            video=str(video),
            player=args.player,
            audio_source="bounded_file_lookahead",
            lookahead_seconds=args.lookahead_seconds,
            full_file_preprocessing=False,
            power=power_status(),
            configuration=vars(args) | {"video": str(video)},
            architecture="media-time-caption-pipeline-v5",
            transcript_contract="sealed_occurrences"
            if args.asr == "granite"
            else "hypothesis_revisions",
        )
        worker.start()
        child.close()
        timeout = time.monotonic() + args.timeout
        while True:
            if attachment is not None and not attachment.current():
                log("attachment_ended_during_startup")
                return 0
            if time.monotonic() >= timeout:
                raise TimeoutError("NPU model loading timed out")
            if not connection.poll(0.1):
                if not worker.is_alive():
                    raise RuntimeError(f"NPU worker exited during startup ({worker.exitcode})")
                continue
            ready = connection.recv()
            if ready["kind"] == "startup":
                log("model_loading", **ready)
            elif ready["kind"] == "ready":
                log("models_ready", **ready)
                break
            else:
                raise RuntimeError(ready.get("error", "Unexpected startup response"))
        if attachment is None:
            player, ipc, audio_id = create_live_player(
                args, video, sidecar, media_end, stream_index, endpoint
            )
        else:
            player, ipc, audio_id = attachment.connect()
            # Model loading happens while the original player continues. Start
            # at its current position instead of processing an obsolete backlog.
            timeline.reset(float(ipc.get_optional("time-pos") or 0))
            captions.reset(timeline.epoch)
        reader = AudioReader(ffmpeg, video, args.audio_track)
        display = CaptionDisplay(
            ipc, args.player, sidecar, args.sub_font, args.sub_font_size, media_end
        )
        log(
            "player_ready",
            pid=player.pid,
            audio_id=audio_id,
            ffmpeg_stream_index=stream_index,
            generated_audio_seconds=0,
            captions_waited_for=False,
        )
        print(
            f"Playing with automatic {args.lookahead_seconds:g}s preparation window; all inference on NPU.",
            flush=True,
        )
        next_memory = next_track = 0.0
        last_task = None
        ipc.events.clear()
        while player.poll() is None:
            now = time.monotonic()
            raw_position = ipc.get_optional("time-pos")
            if raw_position is None:
                time.sleep(0.03)
                continue
            position = float(raw_position)
            paused, speed = bool(ipc.get("pause")), float(ipc.get("speed"))
            at_end = position >= media_end or bool(ipc.get("eof-reached"))
            seek_event = any(e.get("event") == "seek" for e in ipc.events)
            ipc.events.clear()
            previous_state = timeline.state
            update = timeline.observe(position, now, paused, speed, seek_event=seek_event)
            if update.reset_reason:
                reset_state(update.reset_reason, position, now)
            elif update.pause_changed:
                log(
                    "playback_paused" if paused else "playback_resumed",
                    generation=timeline.epoch,
                    position=position,
                )
            elif previous_state is not None and not paused and not previous_state.paused:
                # The nominal sleep is not the actual opportunity to commit.
                # Player IPC and coarse frame timestamps can advance by more
                # than 25 ms. Measure both clocks; never include a seek/pause.
                observed_poll_intervals.append(
                    max(
                        0.025,
                        now - previous_state.sampled_at,
                        (position - previous_state.position) / max(speed, 0.01),
                    )
                )
            if now >= next_memory:
                log(
                    "host_memory",
                    controller=process_memory(),
                    player=process_memory(player.pid),
                    power=power_status(),
                )
                next_memory = now + 2
            if now >= next_track:
                selected = ipc.get("aid")
                if selected != audio_id:
                    # Never recognize a different track from the one being heard.
                    raise RuntimeError(
                        "Audio track changed; reopen using the matching --audio-track"
                    )
                next_track = now + 1
            if at_end and ending_started is None:
                ending_started = now
            if ending_started is not None and now - ending_started >= args.end_grace:
                log(
                    "finished",
                    position=position,
                    reason="bounded_end_grace",
                    final_audio_flushed=end_flushed,
                    pending_captions=not captions.done,
                )
                break

            if audio_future is not None and audio_future.done():
                # An old read may have been cancelled, killed, or failed.
                # Its exception cannot terminate a different playback epoch.
                decoded = audio_future.result() if audio_future_epoch == timeline.epoch else None
                audio_future = None
                if decoded is not None:
                    pending_audio = decoded
                    log(
                        "audio_decoded",
                        generation=timeline.epoch,
                        audio_start=decoded["start"],
                        audio_end=decoded["end"],
                        seen_playback_position=decoded["seen_playback_position"],
                        read_limit=decoded["read_limit"],
                    )
                else:
                    log(
                        "discarded_audio",
                        generation=audio_future_epoch,
                        current_generation=timeline.epoch,
                    )
            if active is not None and connection.poll():
                message = connection.recv()
                completed = active
                active = None
                log("worker_result", task_seconds=now - completed["submitted_at"], **message)
                refresh_caption_budget(
                    now,
                    position,
                    paused,
                    speed,
                    at_end,
                    dict(kind=completed["kind"], seconds=now - completed["submitted_at"]),
                )
                if message["kind"] == "error":
                    raise RuntimeError(message["error"])
                if message["generation"] != timeline.epoch:
                    log(
                        "discarded_result",
                        kind=message["kind"],
                        result_generation=message["generation"],
                        current_generation=timeline.epoch,
                    )
                elif message["kind"] == "asr":
                    for event in message["updates"]:
                        if event["end"] > message["audio_end"] + 1e-6:
                            raise RuntimeError("ASR evidence extends beyond supplied audio")
                        captions.feed(event, timeline.epoch, now, position)
                    captions.audio_advanced(message["audio_end"], timeline.epoch, now, position)
                    if message.get("flushed"):
                        end_flushed = True
                        captions.finalize()
                elif message["kind"] == "translation":
                    if not captions.translated(message, timeline.epoch, now, position):
                        log(
                            "discarded_translation",
                            id=message["id"],
                            revision=message["revision"],
                            reason="version_or_admission",
                        )
                else:
                    captions.translation_failed(message, timeline.epoch, now, position)
                    log(
                        "discarded_translation",
                        id=message.get("id"),
                        revision=message.get("revision"),
                        reason=message["kind"],
                    )

            # Detect processing backlog, not the intentionally prepared lead.
            if active is None and not at_end:
                oldest = pending_audio["start"] if pending_audio else timeline.cursor / SAMPLE_RATE
                if timeline.audio_is_late(oldest):
                    log("audio_gap", dropped_from=oldest, dropped_to=position)
                    reset_state("audio_gap", position, now, move_timeline=True)
            refresh_caption_budget(now, position, paused, speed, at_end)
            frame = captions.frame(now, position, paused=paused and not at_end)
            signature = (
                (frame["key"], frame["text"], frame["chinese"], frame["deadline"])
                if frame
                else None
            )
            if signature != last_frame:
                if frame:
                    display.show(frame["text"], frame["chinese"], position, frame["deadline"])
                    if last_frame is not None and last_frame[0] != frame["key"]:
                        log(
                            "display_exited",
                            key=last_frame[0],
                            position=position,
                            reason="replaced",
                            replacement_key=frame["key"],
                            evidence="player_api_returned",
                        )
                    log(
                        "bilingual_displayed"
                        if frame["chinese"]
                        else "japanese_displayed"
                        if args.source_language == "ja"
                        else "english_displayed",
                        **{
                            **frame,
                            "position": position,
                            "generation": timeline.epoch,
                            "evidence": "player_api_returned",
                        },
                    )
                else:
                    display.clear()
                    log("display_cleared", position=position, reason="presentation_state")
                    if last_frame is not None:
                        log(
                            "display_exited",
                            key=last_frame[0],
                            position=position,
                            reason="cleared",
                            evidence="player_api_returned",
                        )
                last_frame = signature
            for notice in captions.notices:
                log(
                    "caption_state",
                    **{"generation": timeline.epoch, "position": position, **notice},
                )
            captions.notices.clear()
            if (
                active
                and active["kind"] == "translate"
                and active["key"] not in captions.valid_translation_keys()
            ):
                if cancellation.value == active["ticket"]:
                    cancellation.value += 1
                    log("translation_cancel_requested", key=active["key"])
            if active and now - active["submitted_at"] > args.timeout:
                raise TimeoutError(f"NPU {active['kind']} request timed out")
            if not worker.is_alive():
                raise RuntimeError("NPU worker exited; CPU fallback is disabled")

            if active is None:
                candidate = captions.translation_candidate(now, position, speed=speed)
                audio_urgent = (
                    pending_audio is not None and position - pending_audio["start"] > 0.75
                )
                if (
                    candidate
                    and not audio_urgent
                    and (last_task != "translate" or pending_audio is None)
                ):
                    deadline = timeline.task_deadline(
                        candidate["deadline"], now, args.translation_budget
                    )
                    if deadline > now:
                        cancellation.value += 1
                        task = {
                            "kind": "translate",
                            "generation": timeline.epoch,
                            "id": candidate["id"],
                            "revision": candidate["revision"],
                            "text": candidate["text"],
                            "deadline": deadline,
                            "ticket": cancellation.value,
                        }
                        connection.send(task)
                        captions.mark_submitted(candidate["key"])
                        active = dict(task, key=candidate["key"], submitted_at=now)
                        last_task = "translate"
                        log(
                            "translation_submitted",
                            generation=timeline.epoch,
                            position=position,
                            id=task["id"],
                            revision=task["revision"],
                            source=task["text"],
                            media_deadline=candidate["deadline"],
                            due_at=candidate["due_at"],
                            alignment_kind=candidate["alignment_kind"],
                        )
                if active is None and pending_audio is not None and (not paused or at_end):
                    connection.send(pending_audio)
                    active = {"kind": "audio", "submitted_at": now}
                    last_task = "audio"
                    log(
                        "audio_submitted",
                        generation=timeline.epoch,
                        audio_start=pending_audio["start"],
                        audio_end=pending_audio["end"],
                        read_limit=pending_audio["read_limit"],
                        seen_playback_position=pending_audio["seen_playback_position"],
                    )
                    pending_audio = None
                elif (
                    active is None
                    and pending_audio is None
                    and audio_future is None
                    and timeline.cursor >= timeline.end_sample
                    and not end_flushed
                ):
                    endpoint_time = timeline.cursor / SAMPLE_RATE
                    connection.send(
                        {
                            "kind": "audio",
                            "generation": timeline.epoch,
                            "start": endpoint_time,
                            "end": endpoint_time,
                            "seen_playback_position": position,
                            "read_limit": endpoint_time,
                            "pcm": b"",
                            "final": True,
                        }
                    )
                    active = {"kind": "audio", "submitted_at": now}
            if audio_future is None and pending_audio is None and not end_flushed:
                interval = timeline.reserve(chunk_samples, ending=at_end)
                if interval:
                    audio_future_epoch = timeline.epoch
                    audio_future = reader.submit(
                        *interval,
                        position,
                        timeline.epoch,
                        timeline.read_limit(ending=at_end) / SAMPLE_RATE,
                    )
            if (
                at_end
                and end_flushed
                and active is None
                and audio_future is None
                and pending_audio is None
                and captions.done
            ):
                log(
                    "finished",
                    position=position,
                    reason="fresh_captions_finished",
                    final_audio_flushed=True,
                )
                break
            time.sleep(0.1 if attachment is not None else 0.025)
        return 0
    except KeyboardInterrupt:
        log("interrupted")
        return 130
    except (BrokenPipeError, EOFError):
        if player is not None and player.poll() == 0:
            log("player_closed")
            return 0
        raise
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        if player is not None and player.poll() == 0:
            # A closed player or detached plugin session ends normally, but the
            # error that coincided with it must remain visible in the log.
            log("player_closed", error=error)
            return 0
        log("error", error=error)
        raise
    finally:
        cancellation.value += 1
        if reader is not None:
            reader.close()
        if ipc is not None:
            if attachment is None and player is not None and player.poll() is None:
                try:
                    ipc.command("quit")
                    player.wait(timeout=3)
                except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired):
                    if player.poll() is None:
                        player.terminate()
                        player.wait(timeout=3)
            if attachment is None:
                ipc.close()
        if worker.pid is not None:
            if worker.is_alive():
                try:
                    connection.send({"kind": "stop"})
                    until = time.monotonic() + 3
                    while worker.is_alive() and time.monotonic() < until:
                        if connection.poll(0.05):
                            log("worker_shutdown", **connection.recv())
                        worker.join(timeout=0.05)
                except (OSError, EOFError):
                    pass
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=3)
        connection.close()
        child.close()
        log_file.close()


def main() -> int:
    return run(parser().parse_args())


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
