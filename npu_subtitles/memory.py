"""Resident and private process memory, without loading inference libraries."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os


class _Counters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
        (name, ctypes.c_size_t)
        for name in (
            "PeakWorkingSetSize",
            "WorkingSetSize",
            "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage",
            "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage",
            "PagefileUsage",
            "PeakPagefileUsage",
            "PrivateUsage",
        )
    ]


class _PowerStatus(ctypes.Structure):
    _fields_ = [
        (name, wintypes.BYTE)
        for name in ("ACLineStatus", "BatteryFlag", "BatteryLifePercent", "SystemStatusFlag")
    ]
    _fields_ += [(name, wintypes.DWORD) for name in ("BatteryLifeTime", "BatteryFullLifeTime")]


def power_status() -> dict:
    """Read the current power source so AC runs are not labeled as battery runs."""
    if os.name != "nt":
        return {"available": False, "line_power": "unknown"}
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetSystemPowerStatus.argtypes = [ctypes.POINTER(_PowerStatus)]
    kernel.GetSystemPowerStatus.restype = wintypes.BOOL
    status = _PowerStatus()
    if not kernel.GetSystemPowerStatus(ctypes.byref(status)):
        return {"available": False, "line_power": "unknown", "error": ctypes.get_last_error()}
    return {
        "available": True,
        "line_power": {0: "battery", 1: "ac"}.get(status.ACLineStatus, "unknown"),
        "battery_percent": None if status.BatteryLifePercent == 255 else status.BatteryLifePercent,
        "battery_saver": bool(status.SystemStatusFlag),
    }


def process_memory(pid: int | None = None) -> dict:
    """Windows process-only bytes; resident pages include shared/mapped pages.

    Private commit is not additional resident memory. Neither measurement
    separately attributes driver/device allocations or another player process.
    OS peak counters include native inference even when Python cannot sample.
    """
    pid = os.getpid() if pid is None else pid
    if os.name != "nt":
        return {"pid": pid, "available": False}
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_Counters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = _Counters()
    counters.cb = ctypes.sizeof(counters)
    own_process = pid == os.getpid()
    handle = (
        kernel.GetCurrentProcess()
        if own_process
        else kernel.OpenProcess(0x0400 | 0x0010, False, pid)
    )
    if not handle:
        return {"pid": pid, "available": False, "error": ctypes.get_last_error()}
    try:
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return {"pid": pid, "available": False, "error": ctypes.get_last_error()}
    finally:
        if not own_process:
            kernel.CloseHandle(handle)
    return {
        "pid": pid,
        "available": True,
        "working_set_bytes": counters.WorkingSetSize,
        "private_committed_bytes": counters.PrivateUsage,
        "os_peak_working_set_bytes": counters.PeakWorkingSetSize,
        "os_peak_committed_bytes": counters.PeakPagefileUsage,
    }
