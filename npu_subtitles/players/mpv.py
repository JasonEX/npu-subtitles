"""Synchronous mpv IPC with bounded reads."""

from __future__ import annotations
import ctypes
import json
import os
import socket
import subprocess
import time
from typing import Any


def matching_response(message: dict, request_id: int) -> bool:
    return message.get("request_id") == request_id and "error" in message


class MpvCommandError(RuntimeError):
    """A structured mpv error response, distinct from local application errors."""

    def __init__(self, command: tuple[Any, ...], status: str):
        self.command = command
        self.status = status
        super().__init__(f"mpv {command[0]}: {status}")


class MpvIPC:
    """Small synchronous JSON IPC client with bounded reads and request IDs.

    Windows PeekNamedPipe avoids a blocking ReadFile when no bytes are available.
    Commands are small and one is outstanding at a time. No background reader
    thread or external Windows dependency is required.
    """

    def __init__(
        self, endpoint: str, timeout: float = 5.0, process: subprocess.Popen | None = None
    ):
        self.timeout = timeout
        self.request_id = 0
        self.buffer = b""
        self.events: list[dict] = []
        self.handle = None
        self.sock = None
        deadline = time.monotonic() + timeout
        if os.name == "nt":
            from ctypes import wintypes

            self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self.kernel.CreateFileW.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                ctypes.c_void_p,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            ]
            self.kernel.CreateFileW.restype = wintypes.HANDLE
            self.kernel.PeekNamedPipe.argtypes = [
                wintypes.HANDLE,
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.c_void_p,
                ctypes.POINTER(wintypes.DWORD),
                ctypes.c_void_p,
            ]
            self.kernel.ReadFile.argtypes = [
                wintypes.HANDLE,
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD),
                ctypes.c_void_p,
            ]
            self.kernel.WriteFile.argtypes = self.kernel.ReadFile.argtypes
            self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            last_error = 0
            while time.monotonic() < deadline:
                if process is not None and process.poll() is not None:
                    raise RuntimeError(
                        f"mpv exited before IPC connected (exit {process.returncode})"
                    )
                handle = self.kernel.CreateFileW(endpoint, 0xC0000000, 0, None, 3, 0, None)
                if handle != ctypes.c_void_p(-1).value:
                    self.handle = handle
                    break
                last_error = ctypes.get_last_error()
                time.sleep(0.05)
            if self.handle is None:
                raise TimeoutError(
                    f"mpv IPC did not become available at {endpoint} ({ctypes.WinError(last_error)})"
                )
        else:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            while True:
                if process is not None and process.poll() is not None:
                    self.sock.close()
                    raise RuntimeError(
                        f"mpv exited before IPC connected (exit {process.returncode})"
                    )
                try:
                    self.sock.connect(endpoint)
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.monotonic() >= deadline:
                        self.sock.close()
                        raise TimeoutError("mpv IPC did not become available")
                    time.sleep(0.05)

    def _write(self, data: bytes) -> None:
        if self.sock is not None:
            self.sock.settimeout(self.timeout)
            self.sock.sendall(data)
            return
        count = ctypes.c_ulong()
        if not self.kernel.WriteFile(
            self.handle, data, len(data), ctypes.byref(count), None
        ) or count.value != len(data):
            raise BrokenPipeError("Writing to mpv IPC failed")

    def _read(self, deadline: float) -> bytes:
        if self.sock is not None:
            self.sock.settimeout(max(0.001, deadline - time.monotonic()))
            data = self.sock.recv(65536)
            if not data:
                raise BrokenPipeError("mpv IPC closed")
            return data
        available = ctypes.c_ulong()
        while time.monotonic() < deadline:
            if not self.kernel.PeekNamedPipe(
                self.handle, None, 0, None, ctypes.byref(available), None
            ):
                raise BrokenPipeError("mpv IPC closed")
            if available.value:
                buffer = ctypes.create_string_buffer(min(65536, available.value))
                count = ctypes.c_ulong()
                if not self.kernel.ReadFile(
                    self.handle, buffer, len(buffer), ctypes.byref(count), None
                ):
                    raise BrokenPipeError("Reading mpv IPC failed")
                return buffer.raw[: count.value]
            time.sleep(0.01)
        raise TimeoutError("Timed out waiting for mpv IPC")

    def command(self, *command: Any) -> Any:
        self.request_id += 1
        current_id = self.request_id
        self._write(
            (json.dumps({"command": list(command), "request_id": current_id}) + "\n").encode()
        )
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if b"\n" not in self.buffer:
                self.buffer += self._read(deadline)
                continue
            line, self.buffer = self.buffer.split(b"\n", 1)
            if not line:
                continue
            message = json.loads(line)
            if matching_response(message, current_id):
                if message["error"] != "success":
                    raise MpvCommandError(command, message["error"])
                return message.get("data")
            if "event" in message:
                self.events.append(message)
                self.events = self.events[-100:]
        raise TimeoutError(f"Timed out waiting for mpv {command[0]}")

    def get(self, name: str) -> Any:
        return self.command("get_property", name)

    def get_optional(self, name: str) -> Any:
        """Return None only for a temporarily unavailable property, e.g. during seek."""
        try:
            return self.get(name)
        except MpvCommandError as exc:
            if exc.status == "property unavailable":
                return None
            raise

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
