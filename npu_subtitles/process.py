"""Windows lifetime and resource boundaries for the native player helper."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os

_job = None


def bind_process_tree() -> None:
    """Kill helper descendants if mpv aborts its subprocess or exits.

    The non-inheritable job handle stays open until this helper exits. The
    helper joins its own job, and spawned inference/FFmpeg children inherit
    membership, not the handle. No daemon, process scanning or elevated access.
    """
    global _job
    if os.name != "nt" or _job is not None:
        return

    class Basic(ctypes.Structure):
        _fields_ = [
            ("per_process", ctypes.c_int64),
            ("per_job", ctypes.c_int64),
            ("flags", wintypes.DWORD),
            ("minimum", ctypes.c_size_t),
            ("maximum", ctypes.c_size_t),
            ("active", wintypes.DWORD),
            ("affinity", ctypes.c_size_t),
            ("priority", wintypes.DWORD),
            ("scheduling", wintypes.DWORD),
        ]

    class IO(ctypes.Structure):
        _fields_ = [
            (n, ctypes.c_uint64)
            for n in ("read", "write", "other", "read_bytes", "write_bytes", "other_bytes")
        ]

    class Extended(ctypes.Structure):
        _fields_ = [("basic", Basic), ("io", IO)] + [
            (n, ctypes.c_size_t)
            for n in ("process_memory", "job_memory", "peak_process", "peak_job")
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateJobObjectW(None, None)
    limits = Extended()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if (
        not handle
        or not kernel.SetInformationJobObject(
            handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        )
        or not kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess())
    ):
        error = ctypes.get_last_error()
        if handle:
            kernel.CloseHandle(handle)
        raise ctypes.WinError(error)
    _job = handle


class InferenceLease:
    """Nonblocking per-user-session exclusion: no competing NPU queues."""

    def __enter__(self):
        self.handle = None
        if os.name == "nt":
            self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self.kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
            self.kernel.CreateMutexW.restype = wintypes.HANDLE
            self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            self.kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
            self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            self.handle = self.kernel.CreateMutexW(None, False, r"Local\NPUSubtitlesPluginNPU")
            if not self.handle:
                raise ctypes.WinError(ctypes.get_last_error())
            status = self.kernel.WaitForSingleObject(self.handle, 0)
            if status not in (0, 0x80):
                self.kernel.CloseHandle(self.handle)
                self.handle = None
                raise RuntimeError("Another automatic subtitle session is using the NPU")
        return self

    def __exit__(self, *args):
        if self.handle:
            self.kernel.ReleaseMutex(self.handle)
            self.kernel.CloseHandle(self.handle)
