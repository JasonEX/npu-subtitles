"""Validate actual neural-graph execution devices."""

from typing import Any


def require_npu(report: Any, label: str) -> None:
    """Require actual compiled-graph device reports, not a requested-device flag."""
    devices: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            devices.append(value)
        else:
            raise RuntimeError(f"Invalid {label} execution-device report: {value!r}")

    visit(report)
    if not devices or any(device.split(".", 1)[0] != "NPU" for device in devices):
        raise RuntimeError(f"{label} must execute every neural graph on NPU; reported {report!r}")
