"""MPC-HC adapter for the shared media-time subtitle scheduler (Windows only).

Uses MPC-HC's documented /slave + WM_COPYDATA API, with the built-in Reload
Subtitles command after each atomic subtitle replacement. A session-local ASS
sidecar applies font size without changing the player's global style settings.
No web server, registry
edits, injected DLLs, or extra Python dependencies are needed.

Protocol: https://github.com/clsid2/mpc-hc/blob/2.8.4/src/mpc-hc/MpcApi.h
Reload command: resource.h ID_SUBTITLES_SUBITEM_START + 2 (2302).
"""

from __future__ import annotations

from npu_subtitles.ass import DEFAULT_FONT, write_ass

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any
import uuid

CMD_CONNECT = 0x50000000
CMD_STATE = 0x50000001
CMD_PLAYMODE = 0x50000002
CMD_NOWPLAYING = 0x50000003
CMD_SUBTITLETRACKS = 0x50000004
CMD_AUDIOTRACKS = 0x50000005
CMD_POSITION = 0x50000007
CMD_SEEK = 0x50000008
CMD_EOF = 0x50000009
CMD_VERSION = 0x5000000A
CMD_DISCONNECT = 0x5000000B
CMD_PLAY = 0xA0000004
CMD_PAUSE = 0xA0000005
CMD_SET_POSITION = 0xA0002000
CMD_SET_AUDIO = 0xA0002004
CMD_SET_SUBTITLE = 0xA0002005
CMD_GET_SUBTITLE = 0xA0003000
CMD_GET_AUDIO = 0xA0003001
CMD_GET_POSITION = 0xA0003004
CMD_GET_VERSION = 0xA0003006
CMD_QUIT = 0xA0004006
WM_COPYDATA = 0x004A
WM_COMMAND = 0x0111
RELOAD_SUBTITLES = 2302


def split_fields(value: str) -> list[str]:
    """The API escapes literal pipes, but not Windows path backslashes."""
    return [part.replace(r"\|", "|") for part in re.split(r"(?<!\\)\|", value)]


def parse_tracks(value: str, kind: str) -> list[dict]:
    parts = split_fields(value)
    if len(parts) == 1 and parts[0] in ("-1", "-2"):
        return []
    if len(parts) < 2:
        raise RuntimeError(f"Invalid MPC-HC {kind} track reply: {value!r}")
    try:
        selected = int(parts[-1])
    except ValueError as exc:
        raise RuntimeError(f"Invalid MPC-HC track selection: {value!r}") from exc
    if selected < -1 or selected >= len(parts) - 1:
        raise RuntimeError(f"Out-of-range MPC-HC track selection: {value!r}")
    return [
        {"type": kind, "id": index, "title": name, "selected": index == selected}
        for index, name in enumerate(parts[:-1])
    ]


def executable(value: str | None = None) -> str:
    candidates = (
        [value]
        if value
        else [
            shutil.which("mpc-hc64"),
            shutil.which("mpc-hc"),
            str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "MPC-HC/mpc-hc64.exe"),
            str(
                Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
                / "MPC-HC/mpc-hc.exe"
            ),
        ]
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
        if candidate and shutil.which(candidate):
            return str(shutil.which(candidate))
    raise RuntimeError("MPC-HC not found; supply --mpc-hc PATH")


class COPYDATASTRUCT(ctypes.Structure):
    _fields_ = [
        ("dwData", ctypes.c_size_t),
        ("cbData", wintypes.DWORD),
        ("lpData", ctypes.c_void_p),
    ]


class MpcHCIPC:
    """Small compatible adapter for the scheduler's existing mpv IPC calls.

    A hidden host window receives player notifications. It lives on the calling
    thread; pump() dispatches notifications during bounded waits and polling.
    All track selections are queried back from the real player.
    """

    def __init__(
        self,
        subtitle: Path,
        timeout: float = 5,
        *,
        font: str = DEFAULT_FONT,
        font_size: float = 28,
    ):
        if os.name != "nt":
            raise RuntimeError("MPC-HC integration requires Windows")
        self.source_subtitle = subtitle.resolve()
        self.subtitle = self.source_subtitle.with_suffix(".ass")
        self.font = font
        self.font_size = font_size
        self.timeout = timeout
        self.process: subprocess.Popen | None = None
        self.player_hwnd = 0
        self.hwnd = 0
        self.events: list[dict] = []
        self.responses: dict[int, str] = {}
        self.loaded = False
        self.paused = True
        self.eof = False
        self.disconnected = False
        self.reload_count = 0
        self.version = ""
        self._callback_error: Exception | None = None
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._window_proc_type = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
        )

        class WNDCLASS(ctypes.Structure):
            _fields_ = [
                ("style", wintypes.UINT),
                ("lpfnWndProc", self._window_proc_type),
                ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int),
                ("hInstance", wintypes.HINSTANCE),
                ("hIcon", wintypes.HICON),
                ("hCursor", wintypes.HANDLE),
                ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR),
                ("lpszClassName", wintypes.LPCWSTR),
            ]

        self.user32.DefWindowProcW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        self.user32.DefWindowProcW.restype = ctypes.c_ssize_t
        self.user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASS)]
        self.user32.RegisterClassW.restype = wintypes.ATOM
        self.user32.CreateWindowExW.argtypes = [
            wintypes.DWORD,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.HWND,
            wintypes.HMENU,
            wintypes.HINSTANCE,
            ctypes.c_void_p,
        ]
        self.user32.CreateWindowExW.restype = wintypes.HWND
        self.user32.SendMessageTimeoutW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
            wintypes.UINT,
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.user32.SendMessageTimeoutW.restype = ctypes.c_ssize_t
        self.user32.PeekMessageW.argtypes = [
            ctypes.POINTER(wintypes.MSG),
            wintypes.HWND,
            wintypes.UINT,
            wintypes.UINT,
            wintypes.UINT,
        ]
        self.user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
        self.user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
        self.user32.DispatchMessageW.restype = ctypes.c_ssize_t
        self.user32.DestroyWindow.argtypes = [wintypes.HWND]
        self.user32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
        self.user32.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        self.kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        self.instance = self.kernel32.GetModuleHandleW(None)
        self._callback = self._window_proc_type(self._window_proc)
        self.class_name = "NPUSubtitlesMpcHost_" + uuid.uuid4().hex
        wc = WNDCLASS(
            lpfnWndProc=self._callback, hInstance=self.instance, lpszClassName=self.class_name
        )
        if not self.user32.RegisterClassW(ctypes.byref(wc)):
            raise ctypes.WinError(ctypes.get_last_error())
        self.hwnd = self.user32.CreateWindowExW(
            0, self.class_name, "", 0, 0, 0, 0, 0, None, None, self.instance, None
        )
        if not self.hwnd:
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def _window_proc(self, hwnd: int, message: int, sender: int, parameter: int) -> int:
        if message == WM_COPYDATA:
            try:
                if self.process is not None:
                    sender_pid = wintypes.DWORD()
                    self.user32.GetWindowThreadProcessId(sender, ctypes.byref(sender_pid))
                    if sender_pid.value != self.process.pid:
                        return 0
                data = ctypes.cast(parameter, ctypes.POINTER(COPYDATASTRUCT)).contents
                if (
                    not data.lpData
                    or data.cbData < 2
                    or data.cbData > 1024 * 1024
                    or data.cbData % 2
                ):
                    return 0
                text = ctypes.string_at(data.lpData, data.cbData).decode("utf-16-le").rstrip("\0")
                self._notification(int(data.dwData), text, sender)
                return 1
            except Exception as exc:
                self._callback_error = exc
                return 0
        return self.user32.DefWindowProcW(hwnd, message, sender, parameter)

    def _notification(self, command: int, text: str, sender: int = 0) -> None:
        self.responses[command] = text
        if command == CMD_CONNECT:
            handle = int(text)
            if sender and handle != sender:
                raise RuntimeError("MPC-HC connection handle does not match its sender")
            self.player_hwnd = handle
        elif command == CMD_STATE:
            self.loaded = text == "2"
            if text == "4":
                raise RuntimeError("MPC-HC failed to load the movie")
        elif command == CMD_PLAYMODE:
            self.paused = text != "0"
        elif command == CMD_SEEK:
            self.eof = False
            self.events.append({"event": "seek", "position": float(text)})
            self.events = self.events[-100:]
        elif command == CMD_EOF:
            self.eof = True
        elif command == CMD_DISCONNECT:
            self.disconnected = True

    def pump(self) -> None:
        message = wintypes.MSG()
        while self.user32.PeekMessageW(ctypes.byref(message), self.hwnd, 0, 0, 1):
            self.user32.TranslateMessage(ctypes.byref(message))
            self.user32.DispatchMessageW(ctypes.byref(message))
        if self._callback_error:
            error, self._callback_error = self._callback_error, None
            raise error

    def wait_until(self, predicate: Any, description: str, timeout: float | None = None) -> None:
        deadline = time.monotonic() + (timeout if timeout is not None else self.timeout)
        while True:
            self.pump()
            if predicate():
                return
            if self.disconnected or (self.process is not None and self.process.poll() is not None):
                raise BrokenPipeError(f"MPC-HC closed while waiting for {description}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"MPC-HC timed out waiting for {description}")
            time.sleep(0.01)

    def _send_window(self, message: int, wparam: int, lparam: int) -> None:
        if not self.player_hwnd or self.disconnected:
            raise BrokenPipeError("MPC-HC is not connected")
        result = ctypes.c_size_t()
        if not self.user32.SendMessageTimeoutW(
            self.player_hwnd,
            message,
            wparam,
            lparam,
            2,
            int(self.timeout * 1000),
            ctypes.byref(result),
        ):
            raise TimeoutError(f"MPC-HC did not respond to window message {message:#x}")
        self.pump()

    def send(self, command: int, text: str = "") -> None:
        # Service notifications queued while Python decoded the next window;
        # MPC-HC sends them synchronously and may otherwise be waiting on us.
        self.pump()
        payload = ctypes.create_unicode_buffer(text)
        data = COPYDATASTRUCT(
            command, ctypes.sizeof(payload), ctypes.cast(payload, ctypes.c_void_p)
        )
        self._send_window(WM_COPYDATA, self.hwnd, ctypes.addressof(data))

    def query(self, command: int, response: int) -> str:
        self.responses.pop(response, None)
        self.send(command)
        self.wait_until(lambda: response in self.responses, f"API reply {response:#x}")
        return self.responses[response]

    def subtitle_tracks(self) -> list[dict]:
        tracks = parse_tracks(self.query(CMD_GET_SUBTITLE, CMD_SUBTITLETRACKS), "sub")
        for track in tracks:
            # MPC-HC can auto-load the transport SRT beside our rendered ASS.
            # Only the exact generated filename identifies the selected track.
            track["external"] = self.subtitle.name.casefold() in track["title"].casefold()
        return tracks

    def generated_subtitle(self) -> dict:
        candidates = [track for track in self.subtitle_tracks() if track["external"]]
        if len(candidates) != 1:
            raise RuntimeError(f"Cannot identify generated subtitle in MPC-HC tracks: {candidates}")
        return candidates[0]

    def select_subtitle(self, index: int) -> None:
        self.send(CMD_SET_SUBTITLE, str(index))
        selected = [track["id"] for track in self.subtitle_tracks() if track["selected"]]
        if selected != ([index] if index >= 0 else []):
            raise RuntimeError(f"MPC-HC did not select requested subtitle {index}: {selected}")

    def get(self, name: str) -> Any:
        self.pump()
        if name == "time-pos":
            return float(self.query(CMD_GET_POSITION, CMD_POSITION)) if self.loaded else None
        if name == "pause":
            return self.paused
        if name == "speed":
            # The slave API has no rate query. Seek notifications still provide
            # precise seek detection, and buffering uses actual media position.
            return 1.0
        if name == "eof-reached":
            return self.eof
        if name == "track-list":
            return (
                parse_tracks(self.query(CMD_GET_AUDIO, CMD_AUDIOTRACKS), "audio")
                + self.subtitle_tracks()
            )
        if name == "sid":
            return next(
                (track["id"] for track in self.subtitle_tracks() if track["selected"]), None
            )
        if name == "aid":
            return next(
                (
                    track["id"]
                    for track in parse_tracks(self.query(CMD_GET_AUDIO, CMD_AUDIOTRACKS), "audio")
                    if track["selected"]
                ),
                None,
            )
        raise ValueError(f"MPC-HC does not expose {name!r}")

    def get_optional(self, name: str) -> Any:
        return self.get(name)

    def command(self, *command: Any) -> Any:
        name, *arguments = command
        if name == "get_property":
            return self.get(arguments[0])
        if name == "set_property":
            property_name, value = arguments
            if property_name == "pause":
                self.send(CMD_PAUSE if value else CMD_PLAY)
                self.wait_until(lambda: self.paused == bool(value), "play/pause notification")
            elif property_name == "sid":
                self.select_subtitle(-1 if value in (None, "no") else int(value))
            elif property_name == "aid":
                self.send(CMD_SET_AUDIO, str(value))
            else:
                raise ValueError(f"Unsupported MPC-HC property {property_name!r}")
        elif name in ("sub-add", "sub-reload"):
            if name == "sub-add" and Path(arguments[0]).resolve() != self.source_subtitle:
                raise ValueError("MPC-HC can only reload this session's generated SRT")
            write_ass(self.source_subtitle, self.subtitle, self.font, self.font_size)
            self.pump()
            self._send_window(WM_COMMAND, RELOAD_SUBTITLES, 0)
            self.select_subtitle(self.generated_subtitle()["id"])
            self.reload_count += 1
        elif name == "seek":
            if len(arguments) > 1 and arguments[1] not in ("absolute", "absolute+exact"):
                raise ValueError("MPC-HC adapter accepts absolute seeks only")
            self.send(CMD_SET_POSITION, str(float(arguments[0])))
        elif name == "quit":
            self.send(CMD_QUIT)
            # MPC-HC synchronously sends CMD_DISCONNECT during shutdown. Keep
            # pumping until it arrives before callers wait on the process.
            self.wait_until(
                lambda: (
                    self.disconnected
                    or (self.process is not None and self.process.poll() is not None)
                ),
                "shutdown notification",
            )
        else:
            raise ValueError(f"Unsupported MPC-HC command {name!r}")

    def close(self) -> None:
        if self.hwnd:
            self.user32.DestroyWindow(self.hwnd)
            self.hwnd = 0
        if getattr(self, "class_name", None):
            self.user32.UnregisterClassW(self.class_name, self.instance)
            self.class_name = ""


def create_player(
    args: Any, video: Path, output: Path, media_end: float
) -> tuple[subprocess.Popen, MpcHCIPC]:
    from npu_subtitles.formats import Cue, atomic_srt
    from npu_subtitles.media import hidden_options

    # Empty SRT files are rejected by MPC-HC. This cue is outside the playable
    # range, and is replaced by the first real generated window.
    if not output.exists():
        atomic_srt(output, [Cue(media_end + 1, media_end + 2, "Subtitle buffer")])
    font = getattr(args, "sub_font", None) or DEFAULT_FONT
    font_size = getattr(args, "sub_font_size", None)
    font_size = 28 if font_size is None else font_size
    # Validate and produce the sidecar before creating a window or process.
    write_ass(output, output.with_suffix(".ass"), font, font_size)
    ipc = MpcHCIPC(output, font=font, font_size=font_size)
    player = None
    try:
        command = [
            executable(getattr(args, "mpc_hc", None)),
            "/new",
            "/open",
            "/slave",
            str(ipc.hwnd),
            "/start",
            str(round(args.start * 1000)),
            "/sub",
            str(ipc.subtitle),
            str(video),
        ]
        player = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **hidden_options(),
        )
        ipc.process = player
        ipc.wait_until(
            lambda: bool(ipc.player_hwnd) and ipc.loaded, "connection and movie load", timeout=30
        )
        ipc.version = ipc.query(CMD_GET_VERSION, CMD_VERSION)
        ipc.command("set_property", "pause", True)
        ipc.select_subtitle(ipc.generated_subtitle()["id"])
        ipc.events.clear()
        return player, ipc
    except BaseException:
        if player is not None and player.poll() is None:
            try:
                ipc.command("quit")
                player.wait(timeout=3)
            except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired):
                player.terminate()
                player.wait(timeout=3)
        ipc.close()
        raise


def configure_audio(ipc: MpcHCIPC, audio_track: int, stream_index: int) -> int:
    """Select an in-file audio stream by its ordered audio-only index.

    MPC-HC's API exposes audio-only indexes rather than ffmpeg global stream
    indexes. External audio files are never supplied by this launcher.
    """
    tracks = parse_tracks(ipc.query(CMD_GET_AUDIO, CMD_AUDIOTRACKS), "audio")
    if not 0 <= audio_track < len(tracks):
        raise RuntimeError(f"MPC-HC has {len(tracks)} audio tracks, requested a:{audio_track}")
    ipc.send(CMD_SET_AUDIO, str(audio_track))
    selected = [
        track["id"]
        for track in parse_tracks(ipc.query(CMD_GET_AUDIO, CMD_AUDIOTRACKS), "audio")
        if track["selected"]
    ]
    if selected != [audio_track]:
        raise RuntimeError(
            f"MPC-HC audio selection failed: requested {audio_track}, got {selected}"
        )
    return audio_track
