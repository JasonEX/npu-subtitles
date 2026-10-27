#!/usr/bin/env python3
import openvino as ov

core = ov.Core()
print("OpenVINO:", ov.__version__)
print("Devices:", core.available_devices)
for d in core.available_devices:
    try:
        name = core.get_property(d, "FULL_DEVICE_NAME")
        print(f"  {d}: {name}")
    except Exception as e:
        print(f"  {d}: (no name) {e}")
