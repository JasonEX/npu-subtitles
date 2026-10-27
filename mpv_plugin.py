"""On-demand helper for player/mpv/npu_subtitles.lua; never starts a player.

Only the player plugin launches this entry. Plain metadata routing is kept free
of inference imports. Every displayed result is scoped by the Lua session token.
"""

from __future__ import annotations

import os

# Apply before any NumPy/OpenVINO adapter import, including spawned workers.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import gc
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

from npu_subtitles.players.mpv import MpvIPC
from npu_subtitles.media import executable, hidden_options
from npu_subtitles.process import bind_process_tree, InferenceLease
from npu_subtitles.routing import (
    choose_route,
    language,
    parse_srt,
    text_language,
    track_language,
    SubtitleCue,
    BITMAP,
)
from npu_subtitles.translation.languages import available_subtitle_languages

ROOT = Path(__file__).resolve().parent


class Session:
    def __init__(self, args):
        self.args = args
        self.ipc = MpvIPC(args.ipc, timeout=2)
        self.pid = args.pid
        self.audio_id = self.ipc.get("aid")
        self.seek_serial = 0

    @property
    def events(self):
        return self.ipc.events

    def current(self):
        try:
            state = self.ipc.get_optional("user-data/npu-subtitles") or {}
            self.seek_serial = state.get("seek", 0)
            return state.get("token") == self.args.token and state.get("active", False)
        except (OSError, RuntimeError, TimeoutError):
            return False

    def poll(self):
        return None if self.current() else 0

    def connect(self):
        return self, self, self.audio_id

    def get(self, name):
        return self.ipc.get(name)

    def get_optional(self, name):
        return self.ipc.get_optional(name)

    def command(self, *command):
        if command[0] == "quit":
            raise RuntimeError("An attached helper cannot close the player")
        if command[0] == "osd-overlay":
            self.message("npu-overlay", command[3] if command[2] == "ass-events" else "")
        else:
            return self.ipc.command(*command)

    def message(self, name, value):
        if name == "npu-overlay":
            return self.ipc.command(
                "script-message-to",
                "npu_subtitles",
                name,
                self.args.token,
                str(self.seek_serial),
                str(value),
            )
        return self.ipc.command(
            "script-message-to", "npu_subtitles", name, self.args.token, str(value)
        )

    def status(self, value, **details):
        self.message("npu-status", value)
        print(
            json.dumps(
                dict(event=value, pid=os.getpid(), monotonic=time.monotonic(), **details),
                ensure_ascii=False,
            ),
            flush=True,
        )

    def clear(self):
        self.message("npu-overlay", "")

    def close(self):
        self.ipc.close()


def probe(args):
    result = subprocess.run(
        [
            executable(args.ffprobe, "ffprobe"),
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(args.video),
        ],
        capture_output=True,
        check=True,
        timeout=30,
        **hidden_options(),
    )
    return json.loads(result.stdout)


def extract_subtitle(args, track, metadata):
    if track.get("external"):
        source = Path(track["external-filename"])
        if not source.is_absolute():
            source = args.video.parent / source
        mapping = "0:s:0"
    else:
        source = args.video
        index = track.get("ff-index")
        matches = [
            s for s in metadata["streams"] if s["codec_type"] == "subtitle" and s["index"] == index
        ]
        if len(matches) != 1:
            raise RuntimeError("Subtitle stream cannot be mapped unambiguously to FFmpeg")
        mapping = f"0:{index}"
    result = subprocess.run(
        [
            executable(args.ffmpeg, "ffmpeg"),
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(source),
            "-map",
            mapping,
            "-vn",
            "-an",
            "-dn",
            "-f",
            "srt",
            "pipe:1",
        ],
        capture_output=True,
        timeout=60,
        **hidden_options(),
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[-1000:])
    if len(result.stdout) > 16 * 1024 * 1024:
        raise RuntimeError("Subtitle exceeds the 16 MiB text limit")
    return parse_srt(result.stdout.decode("utf-8-sig"))


def draw(session, text):
    from npu_subtitles.ass import ass_text

    session.message(
        "npu-overlay",
        "{\\an2\\pos(640,678)\\fnMicrosoft YaHei\\fs24\\bord1.6\\shad0.5}" + ass_text(text)
        if text
        else "",
    )


def subtitle_loop(session, route, cues):
    """One inference at a time; 30-second/24-cue preparation, media-time display."""
    from npu_subtitles.translation.subtitles import SubtitleTranslator

    cancelled = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="subtitle-npu")
    model_future = executor.submit(SubtitleTranslator, route.source)
    translator, future, pending = None, None, None
    ready, failed = {}, set()
    previous, shown = None, None
    try:
        session.status("loading-translation", route=asdict(route))
        while session.current():
            if not model_future.done():
                time.sleep(0.15)
                continue
            if translator is None:
                translator = model_future.result()
                session.status(
                    "subtitle_models_ready",
                    source=route.source,
                    execution_devices=dict(
                        en_zh=translator.chinese.execution_devices,
                        source_en=translator.pivot.execution_devices if translator.pivot else [],
                    ),
                )
                session.status("subtitle", route=asdict(route), cue_count=len(cues))
            position = session.get_optional("time-pos")
            if position is None:
                time.sleep(0.15)
                continue
            position = float(position) - float(session.get("sub-delay"))
            paused = bool(session.get("pause"))
            seek = any(e.get("event") == "seek" for e in session.ipc.events)
            session.ipc.events.clear()
            if seek or (
                previous is not None and (position < previous - 0.5 or position > previous + 3)
            ):
                cancelled.set()
                failed.clear()
                shown = None
                session.clear()
            previous = position
            if future is not None and future.done():
                try:
                    translated = future.result()
                    if not cancelled.is_set():
                        ready[pending] = translated
                        print(
                            json.dumps(
                                dict(event="subtitle_translated", index=pending, text=translated),
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                except Exception as exc:
                    if not cancelled.is_set():
                        failed.add(pending)
                        print(
                            json.dumps(
                                dict(event="subtitle_rejected", index=pending, error=str(exc))
                            ),
                            flush=True,
                        )
                future, pending = None, None
            # Keep only a bounded window in memory, including nearby rewind.
            ready = {
                i: value
                for i, value in ready.items()
                if cues[i].end >= position - 30 and cues[i].start <= position + 60
            }
            failed = {i for i in failed if cues[i].end >= position}
            current = "\n".join(
                ready[i]
                for i, cue in enumerate(cues)
                if cue.start <= position < cue.end and i in ready
            )
            if current != shown:
                draw(session, current)
                shown = current
            if future is None and not paused:
                candidates = [
                    i
                    for i, cue in enumerate(cues)
                    if cue.end > position and cue.start <= position + 30
                ][:24]
                candidate = next(
                    (i for i in candidates if i not in ready and i not in failed), None
                )
                if candidate is not None:
                    cancelled = threading.Event()
                    pending = candidate
                    future = executor.submit(
                        translator.translate, cues[pending].text, route.display, cancelled.is_set
                    )
            time.sleep(0.1 if not paused else 0.25)
    finally:
        cancelled.set()
        executor.shutdown(wait=True, cancel_futures=True)
        if translator is None and model_future.done() and not model_future.cancelled():
            try:
                translator = model_future.result()
            except Exception:
                pass
        if translator:
            translator.close()


def audio_mapping(session, metadata):
    tracks = session.get("track-list")
    selected = next((t for t in tracks if t.get("type") == "audio" and t.get("selected")), None)
    if selected is None or selected.get("external"):
        raise RuntimeError("No supported internal audio track selected")
    audio = [s for s in metadata["streams"] if s["codec_type"] == "audio"]
    indices = [i for i, s in enumerate(audio) if s["index"] == selected.get("ff-index")]
    if len(indices) != 1:
        raise RuntimeError("Selected audio cannot be mapped unambiguously to FFmpeg")
    return indices[0]


def audio_slice(args, track, start, duration):
    from npu_subtitles.playback import AudioReader

    if not hasattr(args, "_audio_reader"):
        args._audio_reader = AudioReader(executable(args.ffmpeg, "ffmpeg"), args.video, track)
    begin, end = round(start * 16000), round((start + duration) * 16000)
    return args._audio_reader.submit(begin, end, start, 0, end / 16000).result()["pcm"]


def detect_audio(session, track):
    import numpy as np
    from npu_subtitles.asr.sensevoice import SenseVoicePipeline
    from npu_subtitles.devices import require_npu

    session.status("detecting-language")
    model = SenseVoicePipeline(ROOT / "models/sensevoice-small-ov")
    require_npu(model.execution_devices, "Language identification")
    try:
        for attempt in range(3):
            if not session.current():
                return ""
            start = float(session.get_optional("time-pos") or 0)
            duration = min(6.0, float(session.get("duration")) - start)
            if duration <= 0.3:
                return ""
            pcm = np.frombuffer(audio_slice(session.args, track, start, duration), dtype="<f4")
            if np.sqrt(np.mean(pcm * pcm)) >= 0.008:
                result = model.generate(pcm, language="auto", return_timestamps=False)
                found = language(getattr(result, "detected_language", ""))
                score = getattr(result, "language_score", 0.0)
                session.status(
                    "language-probe",
                    detected=found,
                    score=score,
                    text=result.texts[0],
                    attempt=attempt,
                )
                if (
                    score >= 0.8
                    and len(result.texts[0].strip()) >= 3
                    and found in {"zh", "en", "ja"}
                ):
                    return found
                if score >= 0.8 and found and found not in {"zh", "en", "ja", "nospeech"}:
                    return found
            # Retry at later playback positions, not repeatedly on the same silence.
            until = time.monotonic() + 4
            while time.monotonic() < until and session.current():
                time.sleep(0.2)
        return ""
    finally:
        del model
        gc.collect()


def chinese_loop(session, track):
    import numpy as np
    from npu_subtitles.asr.sensevoice import SenseVoicePipeline
    from npu_subtitles.devices import require_npu

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chinese-npu")
    load = executor.submit(SenseVoicePipeline, ROOT / "models/sensevoice-small-ov")
    model, future, pending = None, None, None
    generation, previous, cursor = 0, None, None
    cues, shown = [], None

    def recognize(start, end):
        pcm = np.frombuffer(audio_slice(session.args, track, start, end - start), dtype="<f4")
        if not len(pcm) or np.sqrt(np.mean(pcm * pcm)) < 0.008:
            return [], end
        # Prefer an acoustic pause in the latter half of the granted window.
        # Never cut because of recognized words or a particular test sentence.
        width = 320  # 20 ms at 16 kHz
        frames = len(pcm) // width
        if frames:
            quiet = np.sqrt(np.mean(pcm[: frames * width].reshape(-1, width) ** 2, axis=1)) < 0.008
            run, cut = 0, None
            for i, value in enumerate(quiet):
                run = run + 1 if value else 0
                if run >= 10 and i >= frames // 2:
                    cut = (i - run // 2) * width
            if cut and cut >= 16000:
                pcm = pcm[:cut]
                end = start + len(pcm) / 16000
        result = model.generate(pcm, language="zh")
        return [
            SubtitleCue(start + c["start_ts"], start + c["end_ts"], c["text"])
            for c in result.chunks
        ], end

    try:
        session.status("loading-chinese-asr")
        while session.current():
            if not load.done():
                time.sleep(0.15)
                continue
            if model is None:
                model = load.result()
                require_npu(model.execution_devices, "Chinese ASR")
                session.status("asr-zh", execution_devices=model.execution_devices)
            position = session.get_optional("time-pos")
            if position is None:
                time.sleep(0.15)
                continue
            position = float(position)
            paused = bool(session.get("pause"))
            duration = float(session.get("duration"))
            seek = any(e.get("event") == "seek" for e in session.ipc.events)
            session.ipc.events.clear()
            if (
                cursor is None
                or seek
                or (previous is not None and (position < previous - 0.5 or position > previous + 3))
            ):
                generation += 1
                cues = []
                cursor = position
                shown = None
                session.clear()
            previous = position
            if future is not None and future.done():
                result, decoded_end = future.result()
                if pending == generation:
                    cursor = decoded_end
                    cues.extend(result)
                    print(
                        json.dumps(
                            dict(event="chinese_recognized", cues=[asdict(c) for c in result]),
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                future = None
            cues = [c for c in cues if c.end > position]
            current = "\n".join(c.text for c in cues if c.start <= position < c.end)
            if shown != current:
                draw(session, current)
                shown = current
            if future is None and not paused and cursor < duration and cursor < position + 4:
                cursor = max(cursor, position)
                end = min(cursor + 8, position + 8, duration)
                pending = generation
                future = executor.submit(recognize, cursor, end)
                cursor = end
            time.sleep(0.1 if not paused else 0.25)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ipc", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")
    args = parser.parse_args()
    folder = ROOT / "outputs/plugin"
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / f"{args.pid}-{args.token}.log"
    sys.stdout = sys.stderr = log.open("w", encoding="utf-8", buffering=1)
    bind_process_tree()
    session = Session(args)
    try:
        if not args.video.is_file() or not session.current():
            return 0
        session.status("routing", video=str(args.video))
        metadata = probe(args)
        tracks = session.get("track-list")
        selected = next((t for t in tracks if t.get("type") == "audio" and t.get("selected")), {})
        audio_language = track_language(selected)
        extracted = {}
        # Check unknown text tracks once, without importing an inference library.
        for t in tracks:
            if t.get("type") == "sub" and not track_language(t) and t.get("codec") not in BITMAP:
                try:
                    extracted[t["id"]] = extract_subtitle(args, t, metadata)
                    t["lang"] = text_language("\n".join(c.text for c in extracted[t["id"]][:100]))
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    print(
                        json.dumps(
                            dict(event="subtitle_probe_failed", track=t["id"], error=str(exc))
                        ),
                        flush=True,
                    )
        available = available_subtitle_languages(ROOT)
        route = choose_route(tracks, audio_language, available)
        if route.kind == "native":
            session.message("npu-select", route.track_id)
            session.status("native-zh")
            return 0
        with InferenceLease():
            if (
                selected
                and not audio_language
                and (
                    route.kind in {"none", "asr"}
                    or any(track_language(t) == "ja" for t in tracks if t.get("type") == "sub")
                )
            ):
                audio_language = detect_audio(session, audio_mapping(session, metadata))
                route = choose_route(tracks, audio_language, available)
            if not session.current():
                return 0
            if route.kind == "subtitle":
                track = next(t for t in tracks if t["type"] == "sub" and t["id"] == route.track_id)
                cues = extracted.get(route.track_id)
                if cues is None:
                    cues = extract_subtitle(args, track, metadata)
                if cues:
                    if text_language("\n".join(c.text for c in cues[:100]), route.source) == "zh":
                        session.message("npu-select", route.track_id)
                        session.status("native-zh")
                        return 0
                    subtitle_loop(session, route, cues)
                    return 0
                # An empty subtitle file is not a usable transcript.
                if selected and not audio_language:
                    audio_language = detect_audio(session, audio_mapping(session, metadata))
                route = choose_route([], audio_language, available)
            if route.kind == "asr":
                track = audio_mapping(session, metadata)
                if route.source == "zh":
                    chinese_loop(session, track)
                else:
                    import npu_subtitles.app as live_subtitles

                    live = live_subtitles.parser().parse_args(
                        [
                            str(args.video),
                            "--language",
                            route.source,
                            "--asr",
                            "granite" if route.source == "ja" else "moonshine",
                            "--audio-track",
                            str(track),
                        ]
                    )
                    live.ffmpeg = args.ffmpeg
                    live.ffprobe = args.ffprobe
                    session.status("asr-" + route.source, route=asdict(route))
                    live_subtitles.run(live, attachment=session)
            else:
                session.status("unsupported", reason=route.reason)
        return 0
    except Exception as exc:
        import traceback

        traceback.print_exc()
        if session.current():
            session.clear()
            session.status("error", error=str(exc))
        return 1
    finally:
        if hasattr(args, "_audio_reader"):
            args._audio_reader.close()
        if session.current():
            session.clear()
        session.close()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    raise SystemExit(main())
