"""Bounded real-mpv integration checks with generated, non-film fixtures."""

from __future__ import annotations
import argparse
import json
import subprocess
import time
import uuid

from npu_subtitles.paths import ROOT
from npu_subtitles.players.mpv import MpvIPC
from npu_subtitles.media import executable, hidden_options
from tools.install_mpv_plugin import install
from npu_subtitles.memory import process_memory

OUT = ROOT / "outputs/validation/mpv-plugin"


def wait_for(callback, timeout=45):
    until = time.monotonic() + timeout
    last = None
    while time.monotonic() < until:
        last = callback()
        if last:
            return last
        time.sleep(0.15)
    raise TimeoutError(f"Condition did not complete in {timeout}s; last={last!r}")


def fixtures():
    OUT.mkdir(parents=True, exist_ok=True)
    ffmpeg = executable(None, "ffmpeg")
    base = OUT / "plain.mkv"
    if not base.exists():
        subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=s=128x72:r=10",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=16000:cl=mono",
                "-t",
                "40",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-c:a",
                "flac",
                str(base),
            ],
            check=True,
            **hidden_options(),
        )
    paths = {}
    for name, lang, text in [
        ("english", "eng", "Please wait here. The train will arrive soon."),
        ("japanese", "jpn", "ここで待っていてください。電車はもうすぐ到着します。"),
        ("chinese", "zho", "请在这里等候，列车即将到站。"),
    ]:
        srt = OUT / f"{name}.srt"
        srt.write_text(
            f"1\n00:00:02,000 --> 00:00:09,000\n{text}\n\n2\n00:00:12,000 --> 00:00:30,000\n{text}\n",
            encoding="utf-8",
        )
        dest = OUT / f"{name}.mkv"
        if not dest.exists():
            subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    str(base),
                    "-i",
                    str(srt),
                    "-map",
                    "0",
                    "-map",
                    "1",
                    "-c",
                    "copy",
                    "-metadata:s:a:0",
                    "language=jpn" if name == "japanese" else "language=eng",
                    "-metadata:s:s:0",
                    f"language={lang}",
                    str(dest),
                ],
                check=True,
                **hidden_options(),
            )
        paths[name] = dest
    return paths


def run_case(name, video, config):
    endpoint = rf"\\.\pipe\mpv-plugin-test-{uuid.uuid4().hex[:10]}"
    log = OUT / f"{name}-mpv.log"
    command = [
        executable(None, "mpv"),
        f"--config-dir={config}",
        f"--input-ipc-server={endpoint}",
        "--vo=null",
        "--ao=null",
        "--pause=yes",
        "--idle=yes",
        "--keep-open=yes",
        f"--log-file={log}",
        str(video),
    ]
    player = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **hidden_options()
    )
    try:
        ipc = MpvIPC(endpoint, process=player)
    except BaseException:
        player.terminate()
        player.wait(timeout=5)
        raise
    report = dict(
        case=name, pid=player.pid, video=str(video), player_started_without_script_argument=True
    )
    try:

        def state():
            return ipc.get_optional("user-data/npu-subtitles") or {}

        ready = wait_for(
            lambda: (
                (s if s.get("status") in {"native-zh", "subtitle", "error", "manual"} else None)
                if (s := state())
                else None
            ),
            120,
        )
        report["state"] = ready
        if name == "chinese":
            assert ready["status"] == "native-zh", ready
            report["helper_logs"] = list(
                str(p) for p in (ROOT / "outputs/plugin").glob(f"{player.pid}-*.log")
            )
            assert not report["helper_logs"], "Chinese path started a helper"
            report["sid"] = ipc.get("sid")
        else:
            assert ready["status"] == "subtitle", ready
            ipc.command("seek", 12, "absolute+exact")
            ipc.command("set_property", "pause", False)
            helper_log = ROOT / "outputs/plugin" / f"{player.pid}-{ready['token']}.log"
            wait_for(
                lambda: (
                    helper_log.exists()
                    and "subtitle_translated" in helper_log.read_text(encoding="utf-8")
                ),
                30,
            )
            report["helper_log"] = str(helper_log)
            report["translated"] = True
            ipc.command("set_property", "pause", True)
            old = state()["token"]
            ipc.command("loadfile", str(OUT / "chinese.mkv"), "replace")
            final = wait_for(
                lambda: (
                    s
                    if (s := state()).get("status") == "native-zh" and s.get("token") != old
                    else None
                )
            )
            report["file_change"] = final
            ipc.command("script-message-to", "npu_subtitles", "npu-status", old, "obsolete-result")
            assert state()["status"] == "native-zh", "Obsolete helper changed the next file"
            report["obsolete_session_rejected"] = True
        report["player_memory"] = process_memory(player.pid)
        report["status"] = "passed"
    finally:
        try:
            ipc.command("quit")
        except (RuntimeError, OSError):
            pass
        ipc.close()
        player.wait(timeout=8)
    return report


def asr_fixture(lang):
    sources = {
        "en": OUT / "synthetic-en.wav",
        "ja": ROOT / "outputs/validation/japanese-npu/fleurs/test-0000.wav",
        "zh": OUT / "sensevoice-example-zh.mp3",
    }
    dest = OUT / f"asr-{lang}.mkv"
    if not dest.exists():
        subprocess.run(
            [
                executable(None, "ffmpeg"),
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=s=128x72:r=10",
                "-i",
                str(sources[lang]),
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-c:a",
                "flac",
                "-af",
                "apad=pad_dur=4",
                "-shortest",
                "-metadata:s:a:0",
                f"language={dict(en='eng', ja='jpn', zh='und')[lang]}",
                str(dest),
            ],
            check=True,
            **hidden_options(),
        )
    return dest


def run_asr(lang, config):
    video = asr_fixture(lang)
    endpoint = rf"\\.\pipe\mpv-plugin-asr-{uuid.uuid4().hex[:10]}"
    player = subprocess.Popen(
        [
            executable(None, "mpv"),
            f"--config-dir={config}",
            f"--input-ipc-server={endpoint}",
            "--vo=null",
            "--ao=null",
            "--pause=yes",
            "--idle=yes",
            "--keep-open=yes",
            f"--log-file={OUT / ('asr-' + lang + '-mpv.log')}",
            str(video),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **hidden_options(),
    )
    ipc = None
    started = time.time()
    report = dict(case="asr-" + lang, video=str(video), pid=player.pid)
    try:
        ipc = MpvIPC(endpoint, process=player)

        def state():
            return ipc.get_optional("user-data/npu-subtitles") or {}

        initial = wait_for(
            lambda: s if (s := state()).get("status") in {"asr-" + lang, "error"} else None, 90
        )
        assert initial["status"] == "asr-" + lang, initial
        helper_log = ROOT / "outputs/plugin" / f"{player.pid}-{initial['token']}.log"
        live_log = None
        if lang != "zh":

            def loaded():
                files = [
                    p
                    for p in (ROOT / "outputs/live").glob(f"{video.stem}.live.*.jsonl")
                    if p.stat().st_mtime >= started
                ]
                return next(
                    (p for p in files if "models_ready" in p.read_text(encoding="utf-8")), None
                )

            live_log = wait_for(loaded, 90)
        ipc.command("set_property", "pause", False)

        def output_ready():
            target = helper_log if lang == "zh" else live_log
            content = target.read_text(encoding="utf-8")
            return (
                content
                if ("chinese_recognized" if lang == "zh" else "bilingual_displayed") in content
                else None
            )

        evidence = wait_for(output_ready, 45)
        report["inference_output"] = True
        report["helper_log"] = str(helper_log)
        report["live_log"] = str(live_log) if live_log else None
        report["language_detected_on_npu"] = lang == "zh" and "language-probe" in evidence
        ipc.command("set_property", "pause", True)
        before = float(ipc.get("time-pos"))
        time.sleep(0.5)
        assert abs(float(ipc.get("time-pos")) - before) < 0.2
        ipc.command("seek", 1, "absolute+exact")
        ipc.command("set_property", "pause", False)
        time.sleep(0.5)
        ipc.command("loadfile", str(OUT / "chinese.mkv"), "replace")
        report["switch_to_chinese"] = wait_for(
            lambda: s if (s := state()).get("status") == "native-zh" else None
        )
        assert player.poll() is None, "Helper closed the original player"
        report["status"] = "passed"
    finally:
        if ipc:
            try:
                ipc.command("quit")
            except (OSError, RuntimeError):
                pass
            ipc.close()
        if player.poll() is None:
            try:
                player.wait(timeout=8)
            except subprocess.TimeoutExpired:
                player.terminate()
                player.wait(timeout=5)
    (OUT / f"asr-{lang}-result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", choices=["chinese", "english", "japanese", "all", "asr"], default="all"
    )
    args = parser.parse_args()
    paths = fixtures()
    config = OUT / "config"
    install(config)
    if args.case == "asr":
        for lang in ("en", "ja", "zh"):
            run_asr(lang, config)
        return 0
    reports = []
    for name in [args.case] if args.case != "all" else ["chinese", "english", "japanese"]:
        report = run_case(name, paths[name], config)
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        (OUT / f"{name}-result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
