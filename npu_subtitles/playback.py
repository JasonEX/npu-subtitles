"""Audio I/O and player rendering. Caption lifetime belongs to the controller."""

from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

from npu_subtitles.formats import Cue, atomic_srt
from npu_subtitles.players.mpv import MpvIPC
from npu_subtitles.media import executable, hidden_options

from npu_subtitles.paths import SAMPLE_RATE

OVERLAY_ID = 9531


class AudioReader:
    """Decode one bounded slice; the timeline grants each read horizon."""

    def __init__(self, ffmpeg: str, video: Path, audio_track: int):
        self.ffmpeg, self.video, self.audio_track = ffmpeg, video, audio_track
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="movie-audio"
        )
        self.lock = threading.Lock()
        self.process: subprocess.Popen | None = None
        self.process_generation = -1
        self.minimum_generation = 0
        self.closed = False

    def submit(
        self,
        start_sample: int,
        end_sample: int,
        seen_position: float,
        generation: int,
        read_limit: float,
    ) -> concurrent.futures.Future:
        if end_sample <= start_sample or end_sample / SAMPLE_RATE > read_limit + 1e-7:
            raise ValueError("Audio range exceeds the timeline's granted read horizon")
        return self.executor.submit(
            self._decode, start_sample, end_sample, seen_position, generation, read_limit
        )

    def _decode(
        self, start_sample: int, end_sample: int, seen: float, generation: int, read_limit: float
    ) -> dict:
        start, end = start_sample / SAMPLE_RATE, end_sample / SAMPLE_RATE
        # Some container/codec seeks begin at a packet slightly AFTER the
        # requested sample. Seek into a bounded past preroll, then trim on the
        # decoded clock. Preroll never reaches ASR or extends the future horizon.
        decode_start = max(0.0, start - 1.0)
        command = [
            self.ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-ss",
            f"{decode_start:.8f}",
            "-i",
            str(self.video),
            "-ss",
            f"{start - decode_start:.8f}",
            "-t",
            f"{end - start:.8f}",
            "-map",
            f"0:a:{self.audio_track}",
            "-vn",
            "-sn",
            "-dn",
            "-af",
            f"aresample={SAMPLE_RATE}:rematrix_maxval=1",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "f32le",
            "pipe:1",
        ]
        with self.lock:
            if self.closed:
                raise RuntimeError("Audio decoder was closed")
            if generation < self.minimum_generation:
                raise concurrent.futures.CancelledError("Audio belongs to a retired playback epoch")
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **hidden_options(),
            )
            self.process = process
            self.process_generation = generation
        try:
            pcm, stderr = process.communicate(timeout=30)
            if process.returncode:
                raise RuntimeError(f"ffmpeg: {stderr.decode(errors='replace')[-1500:]}")
            # Codec packets and resampling can produce a few extra samples.
            # Never send those samples to ASR, even when ffmpeg emits them.
            pcm = pcm[: (end_sample - start_sample) * 4]
            if len(pcm) != (end_sample - start_sample) * 4:
                raise RuntimeError(
                    f"Incomplete audio slice {start:.3f}–{end:.3f}; refusing time drift"
                )
            return {
                "kind": "audio",
                "generation": generation,
                "start": start,
                "end": end,
                "seen_playback_position": seen,
                "read_limit": read_limit,
                "pcm": pcm,
            }
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
            with self.lock:
                if self.process is process:
                    self.process = None

    def cancel_before(self, generation: int) -> None:
        """A seek must not wait for an obsolete ffmpeg slice to time out."""
        with self.lock:
            self.minimum_generation = generation
            if (
                self.process is not None
                and self.process_generation < generation
                and self.process.poll() is None
            ):
                self.process.kill()

    def close(self) -> None:
        with self.lock:
            self.closed = True
            if self.process is not None and self.process.poll() is None:
                self.process.kill()
        self.executor.shutdown(wait=True, cancel_futures=True)


class CaptionDisplay:
    """Render a controller-approved frame; no independent expiration policy."""

    def __init__(
        self, ipc: Any, player: str, sidecar: Path, font: str, font_size: float, media_end: float
    ):
        self.ipc, self.player, self.sidecar = ipc, player, sidecar
        self.font, self.font_size = font, font_size
        self.media_end = media_end
        self.expires_at: float | None = None
        self.text = ""

    def show(self, english: str, chinese: str, position: float, deadline: float) -> None:
        text = f"{chinese}\n{english}" if chinese else english
        if not text:
            self.clear()
            return
        self.text = text
        self.expires_at = deadline
        if self.player == "mpv":
            from npu_subtitles.ass import ass_text

            # osd-overlay uses libass events directly; text cannot inject tags.
            styled = (
                f"{{\\an2\\pos(640,678)\\fn{self.font}\\fs{self.font_size:g}"
                "\\bord1.6\\shad0.5\\1c&HFFFFFF&\\3c&H000000&}" + ass_text(text)
            )
            self.ipc.command(
                "osd-overlay", OVERLAY_ID, "ass-events", styled, 1280, 720, 0, False, False
            )
        else:
            # The file transports only this approved frame. It is never an
            # input transcript and contains no future prepared captions.
            atomic_srt(self.sidecar, [Cue(max(0, position - 0.05), self.expires_at, text)])
            self.ipc.command("sub-reload")

    def clear(self) -> None:
        self.text, self.expires_at = "", None
        if self.player == "mpv":
            self.ipc.command("osd-overlay", OVERLAY_ID, "none", "")
        else:
            # MPC-HC rejects empty subtitle documents. An invisible cue beyond
            # the movie preserves the track while clearing the current text.
            atomic_srt(
                self.sidecar, [Cue(self.media_end + 60, self.media_end + 61, "Live captions")]
            )
            self.ipc.command("sub-reload")


def create_live_player(
    args: argparse.Namespace,
    video: Path,
    sidecar: Path,
    media_end: float,
    stream_index: int,
    endpoint: str,
) -> tuple[Any, Any, int]:
    """Load the selected audio track and start immediately, with no caption wait."""
    if args.player == "mpc-hc":
        from npu_subtitles.players.mpc_hc import configure_audio, create_player

        player, ipc = create_player(args, video, sidecar, media_end)
        try:
            aid = configure_audio(ipc, args.audio_track, stream_index)
            # Existing adapter opens paused only while connecting/selecting
            # tracks. No audio has been recognized or translated at this point.
            ipc.command("set_property", "pause", False)
            return player, ipc, aid
        except BaseException:
            try:
                ipc.command("quit")
            finally:
                ipc.close()
            raise
    command = [
        executable(args.mpv, "mpv"),
        "--no-config",
        "--pause=no",
        "--keep-open=yes",
        "--hwdec=auto-safe",
        "--idle=no",
        "--sub-auto=no",
        "--sid=no",
        "--secondary-sid=no",
        "--input-default-bindings=yes",
        f"--input-ipc-server={endpoint}",
        f"--start={args.start}",
        "--force-window=yes",
        "--title=Live NPU bilingual captions",
    ]
    if args.duration is not None:
        command.append(f"--end={media_end}")
    command += ["--", str(video)]
    player = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **hidden_options(),
    )
    ipc = None
    try:
        ipc = MpvIPC(endpoint, timeout=10, process=player)
        deadline = time.monotonic() + 20
        while True:
            tracks = ipc.get("track-list") or []
            matching = [
                track
                for track in tracks
                if track.get("type") == "audio" and track.get("ff-index") == stream_index
            ]
            if len(matching) == 1:
                aid = matching[0]["id"]
                ipc.command("set_property", "aid", aid)
                if ipc.get("aid") != aid:
                    raise RuntimeError("mpv did not select the requested audio track")
                return player, ipc, aid
            if time.monotonic() >= deadline or player.poll() is not None:
                raise RuntimeError(
                    f"Cannot match ffmpeg audio stream {stream_index} to mpv: {tracks}"
                )
            time.sleep(0.05)
    except BaseException:
        if player.poll() is None:
            player.terminate()
            player.wait(timeout=3)
        if ipc is not None:
            ipc.close()
        raise
