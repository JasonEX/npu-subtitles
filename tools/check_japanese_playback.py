"""Real-player Japanese NPU acceptance/lifecycle check on fixed FLEURS audio."""

from __future__ import annotations
import argparse
import json
import subprocess
import time

from npu_subtitles.paths import ROOT
import numpy as np
import soundfile as sf
import npu_subtitles.app as live_subtitles
from npu_subtitles.media import executable


class ObservedIPC:
    def __init__(self, ipc, output, player, lifecycle):
        self.ipc, self.output, self.player, self.lifecycle = ipc, output, player, lifecycle
        self.step, self.resume_at, self.last_position = 0, 0.0, 0.0
        self.screenshots, self.capture_at = 0, None
        self.actions = []

    def __getattr__(self, name):
        return getattr(self.ipc, name)

    def command(self, *args):
        result = self.ipc.command(*args)
        if (
            self.player == "mpv"
            and args[0] == "osd-overlay"
            and args[2] == "ass-events"
            and self.screenshots < 3
        ):
            self.capture_at = time.monotonic() + 0.15
        return result

    def get_optional(self, name):
        value = self.ipc.get_optional(name)
        if name != "time-pos" or value is None:
            return value
        self.last_position = float(value)
        now = time.monotonic()
        if self.capture_at is not None and now >= self.capture_at:
            self.screenshots += 1
            self.capture_at = None
            self.ipc.command(
                "screenshot-to-file", str(self.output / f"caption-{self.screenshots}.png"), "window"
            )
        action = None
        if self.lifecycle:
            if self.step == 0 and value >= 8.0:
                self.ipc.command("set_property", "pause", True)
                self.resume_at, self.step, action = now + 1.5, 1, "pause"
            elif self.step == 1 and now >= self.resume_at:
                self.ipc.command("set_property", "pause", False)
                self.step, action = 2, "resume"
            elif self.step == 2 and value >= 11.0:
                self.ipc.command("seek", 20.0, "absolute+exact")
                self.step, action = 3, "seek_forward_20"
            elif self.step == 3 and value >= 23.0:
                self.ipc.command("seek", 4.0, "absolute+exact")
                self.step, action = 4, "seek_backward_4"
        if action:
            self.actions.append(dict(action=action, position=value, monotonic=now))
            (self.output / "actions.json").write_text(
                json.dumps(self.actions, indent=2), encoding="utf-8"
            )
            return self.ipc.get_optional(name)
        return value


def media(group, output):
    selection = json.loads(
        (ROOT / "outputs/validation/japanese-npu/fleurs/selection.json").read_text(encoding="utf-8")
    )
    segments = [np.zeros(16000, np.float32)]
    for row in selection["rows"]:
        if row["group"] != group:
            continue
        waveform, rate = sf.read(row["audio_path"], dtype="float32")
        if rate != 16000 or waveform.ndim != 1:
            raise ValueError("Invalid validation waveform")
        segments.extend([waveform, np.zeros(16000, np.float32)])
    segments.append(np.zeros(32000, np.float32))
    wav, video = output / f"{group}.wav", output / f"{group}.mkv"
    waveform = np.concatenate(segments)
    # Make the audio and 5-fps video end on the same exact clock. -shortest
    # alone can leave buffered video packets beyond the audio track's end.
    waveform = np.pad(waveform, (0, (-len(waveform)) % 3200))
    sf.write(wav, waveform, 16000, subtype="FLOAT")
    subprocess.run(
        [
            executable(None, "ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x18212b:s=1280x720:r=5",
            "-i",
            str(wav),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "30",
            "-c:a",
            "flac",
            "-t",
            f"{len(waveform) / 16000:.8f}",
            str(video),
        ],
        check=True,
    )
    return video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["preflight", "acceptance"], required=True)
    parser.add_argument("--player", choices=["mpv", "mpc-hc"], default="mpv")
    parser.add_argument("--lifecycle", action="store_true")
    args = parser.parse_args()
    output = (
        ROOT
        / "outputs/validation/japanese-npu"
        / f"{args.group}-{args.player}{'-lifecycle' if args.lifecycle else '-live'}"
    )
    output.mkdir(parents=True, exist_ok=True)
    video = media(args.group, output)
    original = live_subtitles.create_live_player

    def create(*positional, **keywords):
        player, ipc, aid = original(*positional, **keywords)
        return player, ObservedIPC(ipc, output, args.player, args.lifecycle), aid

    live_subtitles.create_live_player = create
    options = live_subtitles.parser().parse_args(
        [
            str(video),
            "--asr",
            "granite",
            "--language",
            "ja",
            "--player",
            args.player,
            "--output-dir",
            str(output),
            "--timeout",
            "180",
        ]
    )
    return live_subtitles.run(options)


if __name__ == "__main__":
    raise SystemExit(main())
