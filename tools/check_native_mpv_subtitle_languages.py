"""Check installed original mpv against generated containers and frozen text.

No custom config directory or explicit script argument: exercises the installed
plugin. Null video/audio outputs test integration, not visual rendering quality.
"""

import json
import shutil
import subprocess
import time
import uuid

from npu_subtitles.paths import ROOT
from npu_subtitles.players.mpv import MpvIPC
from npu_subtitles.media import executable, hidden_options
from tools.validate_mpv_plugin import fixtures, wait_for, OUT as BASE
from tools.measure_mpv_plugin import cpu_seconds
from npu_subtitles.memory import process_memory
from npu_subtitles.translation.languages import EXTRA_SUBTITLE_LANGUAGES

OUT = ROOT / "outputs/validation/subtitle-languages"
ALIASES = dict(ko="kor", es="spa", fr="fre", de="ger", pt="pt-BR", it="ita", ru="rus")


def prepare():
    fixtures()
    cases = json.loads((OUT / "cases.json").read_text(encoding="utf-8"))
    folder = OUT / "player-fixtures"
    folder.mkdir(exist_ok=True)
    videos = {}
    for i, lang in enumerate(EXTRA_SUBTITLE_LANGUAGES):
        video = folder / f"generic-{lang}.mkv"
        srt = folder / f"generic-{lang}.{ALIASES[lang]}.srt"
        source = cases["rows"][i]["text"][lang]
        srt.write_text(f"1\n00:00:02,000 --> 00:00:38,000\n{source}\n", encoding="utf-8")
        external = lang in {"fr", "pt", "ru"}
        if external:
            shutil.copyfile(BASE / "plain.mkv", video)
        else:
            subprocess.run(
                [
                    executable(None, "ffmpeg"),
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(BASE / "plain.mkv"),
                    "-i",
                    str(srt),
                    "-map",
                    "0",
                    "-map",
                    "1",
                    "-c",
                    "copy",
                    "-metadata:s:a:0",
                    f"language={ALIASES[lang]}",
                    "-metadata:s:s:0",
                    f"language={ALIASES[lang]}",
                    str(video),
                ],
                check=True,
                **hidden_options(),
            )
            srt.unlink()  # internal case must not accidentally pass via a sidecar
        videos[lang] = (video, external, cases["rows"][i]["id"])
    return videos


def run(lang, video, external, row_id):
    endpoint = rf"\\.\pipe\mpv-languages-{uuid.uuid4().hex[:10]}"
    command = [
        executable(None, "mpv"),
        f"--input-ipc-server={endpoint}",
        "--vo=null",
        "--ao=null",
        "--pause=yes",
        "--idle=yes",
        str(video),
    ]
    player = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **hidden_options()
    )
    ipc = None
    report = dict(language=lang, external=external, row_id=row_id, command=command)
    try:
        ipc = MpvIPC(endpoint, process=player)
        state = lambda: ipc.get_optional("user-data/npu-subtitles") or {}
        ready = wait_for(
            lambda: (
                s if (s := state()).get("status") in {"subtitle", "error", "unsupported"} else None
            ),
            90,
        )
        assert ready["status"] == "subtitle", ready
        log = ROOT / "outputs/plugin" / f"{player.pid}-{ready['token']}.log"

        def events():
            return [
                json.loads(line)
                for line in log.read_text(encoding="utf-8").splitlines()
                if line.startswith("{")
            ]

        route = next(row["route"] for row in events() if row.get("event") == "subtitle")
        assert route["source"] == lang and route["display"] == "en", route
        selected = next(
            t for t in ipc.get("track-list") if t["type"] == "sub" and t["id"] == route["track_id"]
        )
        assert bool(selected.get("external")) == external, selected
        devices = next(
            row["execution_devices"]
            for row in events()
            if row.get("event") == "subtitle_models_ready"
        )
        assert set(devices) == {"en_zh", "source_en"}
        assert all(
            d and all(str(v).startswith("NPU") for v in d)
            for stage in devices.values()
            for d in stage.values()
        )
        ipc.command("seek", 3, "absolute+exact")
        ipc.command("set_property", "pause", False)
        translated = wait_for(
            lambda: next(
                (row for row in events() if row.get("event") == "subtitle_translated"), None
            ),
            30,
        )
        expected = next(
            row
            for row in json.loads((OUT / f"{lang}.json").read_text(encoding="utf-8"))["samples"]
            if row["id"] == row_id
        )
        assert translated["text"] == expected["chinese"] + "\n" + expected["english"]
        assert not any(row.get("event", "").startswith("asr-") for row in events())
        helper_pid = events()[0]["pid"]
        report.update(
            route=route,
            execution_devices=devices,
            translated=translated["text"],
            helper_log=str(log),
        )
        # One representative two-stage controller: bounded prefetch completed,
        # so these short samples are idle-control cost, not sustained translation.
        if lang == "ko":
            samples = []
            for paused in (False, True):
                ipc.command("set_property", "pause", paused)
                time.sleep(0.3)
                cpu = cpu_seconds(helper_pid)
                start = time.perf_counter()
                time.sleep(3)
                wall = time.perf_counter() - start
                samples.append(
                    dict(
                        paused=paused,
                        wall_seconds=wall,
                        helper_cpu_percent_one_core=100 * (cpu_seconds(helper_pid) - cpu) / wall,
                        helper_memory=process_memory(helper_pid),
                    )
                )
            report["control_samples"] = samples
        ipc.command("set_property", "pause", True)
        old = ready["token"]
        ipc.command("loadfile", str(BASE / "chinese.mkv"), "replace")
        final = wait_for(
            lambda: (
                s if (s := state()).get("status") == "native-zh" and s.get("token") != old else None
            )
        )
        ipc.command("script-message-to", "npu_subtitles", "npu-status", old, "obsolete")
        assert state()["status"] == "native-zh" and player.poll() is None
        report.update(status="passed", switch_to_chinese=final, obsolete_session_rejected=True)
        return report
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


def main():
    videos = prepare()
    rows = []
    for lang, (video, external, row_id) in videos.items():
        row = run(lang, video, external, row_id)
        rows.append(row)
        (OUT / "native-mpv.json").write_text(
            json.dumps(
                dict(
                    status="passed" if len(rows) == len(videos) else "partial",
                    method=__doc__,
                    rows=rows,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(dict(language=lang, status=row["status"], external=external)), flush=True)


if __name__ == "__main__":
    main()
