"""Check actual installed mpv plugin routing and user overrides on generated media."""

import json
import subprocess
import uuid

from npu_subtitles.paths import ROOT
from tools.validate_mpv_plugin import OUT, wait_for, fixtures
from npu_subtitles.players.mpv import MpvIPC
from npu_subtitles.media import hidden_options, executable


def main():
    fixtures()
    video = OUT / "audio-switch.mkv"
    if not video.exists():
        subprocess.run(
            [
                executable(None, "ffmpeg"),
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(OUT / "plain.mkv"),
                "-i",
                str(OUT / "english.srt"),
                "-i",
                str(OUT / "japanese.srt"),
                "-map",
                "0:v",
                "-map",
                "0:a",
                "-map",
                "0:a",
                "-map",
                "1",
                "-map",
                "2",
                "-c",
                "copy",
                "-metadata:s:a:0",
                "language=eng",
                "-metadata:s:a:1",
                "language=jpn",
                "-metadata:s:s:0",
                "language=eng",
                "-metadata:s:s:1",
                "language=jpn",
                str(video),
            ],
            check=True,
            **hidden_options(),
        )
    endpoint = rf"\\.\pipe\mpv-priority-{uuid.uuid4().hex[:8]}"
    player = subprocess.Popen(
        [
            executable(None, "mpv"),
            f"--input-ipc-server={endpoint}",
            "--vo=null",
            "--ao=null",
            "--pause=yes",
            "--idle=yes",
            str(video),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **hidden_options(),
    )
    ipc = None
    try:
        ipc = MpvIPC(endpoint, process=player)

        def state():
            return ipc.get_optional("user-data/npu-subtitles") or {}

        first = wait_for(lambda: s if (s := state()).get("status") == "subtitle" else None, 60)
        ipc.command("set_property", "aid", 2)
        second = wait_for(
            lambda: (
                s
                if (s := state()).get("status") == "subtitle" and s.get("token") != first["token"]
                else None
            ),
            60,
        )
        log = ROOT / "outputs/plugin" / f"{player.pid}-{second['token']}.log"
        entries = [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines()
            if line.startswith("{")
        ]
        route = next(r["route"] for r in entries if r.get("event") == "subtitle")
        assert route["source"] == "ja" and route["display"] == "ja", route
        ipc.command("set_property", "sid", 1)
        manual = wait_for(lambda: s if (s := state()).get("status") == "manual" else None)
        ipc.command("script-binding", "npu_subtitles/toggle-npu-subtitles")
        resumed = wait_for(lambda: s if (s := state()).get("status") == "subtitle" else None, 60)
        report = dict(
            status="passed",
            japanese_audio_prefers_japanese_subtitles=route,
            audio_change=second,
            manual_override=manual,
            resume_auto=resumed,
        )
        (OUT / "audio-switch-result.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report))
    finally:
        if ipc:
            try:
                ipc.command("quit")
            except (OSError, RuntimeError):
                pass
            ipc.close()
        if player.poll() is None:
            try:
                player.wait(timeout=5)
            except subprocess.TimeoutExpired:
                player.terminate()
                player.wait(timeout=5)


if __name__ == "__main__":
    main()
