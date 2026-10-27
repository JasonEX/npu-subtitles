"""Install only this project's native mpv script and options; preserve mpv.conf."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil

from npu_subtitles.paths import ROOT
from npu_subtitles.media import executable

MARKER = "# Managed by NPU Subtitles native mpv plugin"


def default_config(mpv: Path) -> Path:
    if os.environ.get("MPV_HOME"):
        return Path(os.environ["MPV_HOME"]).expanduser().resolve()
    portable = mpv.resolve().parent / "portable_config"
    if portable.is_dir():
        return portable
    return Path(os.environ["APPDATA"]) / "mpv"


def install(config: Path, *, uninstall=False):
    config = config.resolve()
    script = config / "scripts/npu_subtitles.lua"
    options = config / "script-opts/npu_subtitles.conf"
    manifest = config / "npu-subtitles-install.json"
    source = ROOT / "player/mpv/npu_subtitles.lua"
    if manifest.exists():
        owned = json.loads(manifest.read_text(encoding="utf-8"))
        if owned.get("project") != str(ROOT):
            raise RuntimeError("This config is managed by another project checkout")
    else:
        if script.exists() or options.exists():
            raise RuntimeError(
                "Existing files have no project ownership manifest; refusing overwrite"
            )
    if uninstall:
        if not manifest.exists():
            return
        for path in (script, options, manifest):
            if not path.resolve().is_relative_to(config):
                raise RuntimeError("Installation path escaped the selected configuration directory")
            path.unlink(missing_ok=True)
        return
    python = ROOT / ".venv/Scripts/pythonw.exe"
    if not python.is_file():
        raise FileNotFoundError("Prepare the project Python environment before installing")
    values = dict(root=ROOT.as_posix(), python=python.as_posix(), enabled="yes")
    for name in ("ffmpeg", "ffprobe"):
        value = shutil.which(name)
        if not value:
            raise FileNotFoundError(f"{name} is required")
        values[name] = Path(value).as_posix()
    if any("\n" in value or "\r" in value for value in values.values()):
        raise ValueError("Configuration paths cannot contain newlines")
    script.parent.mkdir(parents=True, exist_ok=True)
    options.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, script)
    options.write_text(
        MARKER + "\n" + "\n".join(f"{k}={v}" for k, v in values.items()) + "\n", encoding="utf-8"
    )
    manifest.write_text(
        json.dumps(dict(project=str(ROOT), files=[str(script), str(options)]), indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpv", type=Path, help="mpv executable; defaults to PATH")
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--uninstall", action="store_true")
    args = parser.parse_args()
    config = args.config_dir or default_config(
        Path(executable(str(args.mpv) if args.mpv else None, "mpv"))
    )
    install(config, uninstall=args.uninstall)
    print(("Removed" if args.uninstall else "Installed") + ": " + str(config.resolve()))


if __name__ == "__main__":
    main()
