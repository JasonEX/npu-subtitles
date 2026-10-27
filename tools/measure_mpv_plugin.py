"""Short control-overhead measurement; null video/audio, not GPU or movie FPS."""

from __future__ import annotations
import ctypes
from ctypes import wintypes
import json
import os
import subprocess
import time
import uuid

from npu_subtitles.paths import ROOT
from tools.validate_mpv_plugin import OUT, wait_for, fixtures
from tools.install_mpv_plugin import install
from npu_subtitles.players.mpv import MpvIPC
from npu_subtitles.media import executable, hidden_options
from npu_subtitles.memory import process_memory


def cpu_seconds(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    times = [wintypes.FILETIME() for _ in range(4)]
    try:
        if not kernel.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
            raise ctypes.WinError(ctypes.get_last_error())
        return sum((t.dwHighDateTime << 32) + t.dwLowDateTime for t in times[2:]) / 1e7
    finally:
        kernel.CloseHandle(handle)


def measure(name, config):
    endpoint = rf"\\.\pipe\mpv-resource-test-{uuid.uuid4().hex[:10]}"
    video = OUT / ("english.mkv" if name == "translation" else "chinese.mkv")
    cmd = [
        executable(None, "mpv"),
        f"--config-dir={config}",
        f"--input-ipc-server={endpoint}",
        "--vo=null",
        "--ao=null",
        "--pause=yes",
        "--idle=yes",
        "--keep-open=yes",
    ]
    if name == "baseline":
        cmd.append("--load-scripts=no")
    cmd.append(str(video))
    player = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **hidden_options()
    )
    ipc = None
    try:
        ipc = MpvIPC(endpoint, process=player)
        wait_for(lambda: ipc.get_optional("time-pos") is not None)
        pids = [("player", player.pid)]
        if name != "baseline":
            ready = wait_for(
                lambda: (
                    s
                    if (s := ipc.get_optional("user-data/npu-subtitles") or {}).get("status")
                    in {"native-zh", "subtitle"}
                    else None
                ),
                60,
            )
            if name == "translation":
                log = ROOT / "outputs/plugin" / f"{player.pid}-{ready['token']}.log"
                row = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
                pids.append(("helper", row["pid"]))
        ipc.command("seek", 12, "absolute+exact")
        ipc.command("set_property", "pause", False)
        time.sleep(2)
        samples = []
        for paused in (False, True):
            ipc.command("set_property", "pause", paused)
            time.sleep(0.4)
            before = {pid: cpu_seconds(pid) for _, pid in pids}
            started = time.perf_counter()
            time.sleep(4)
            wall = time.perf_counter() - started
            samples.append(
                dict(
                    paused=paused,
                    wall_seconds=wall,
                    processes=[
                        dict(
                            role=role,
                            cpu_percent_one_core=(cpu_seconds(pid) - before[pid]) / wall * 100,
                            memory=process_memory(pid),
                        )
                        for role, pid in pids
                    ],
                )
            )
        return dict(case=name, samples=samples)
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


def main():
    fixtures()
    config = OUT / "config"
    install(config)
    rows = []
    for name in ("baseline", "native-chinese", "translation"):
        row = measure(name, config)
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = dict(
        method="4-second samples after warmup; generated 128x72 video; null audio/video outputs; "
        "translation measured after bounded prefetch, not sustained ASR or GPU rendering. CPU is percent of one logical core; "
        "do not add process working sets as unique physical memory.",
        logical_processors=os.cpu_count(),
        rows=rows,
    )
    (OUT / "control-overhead.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
