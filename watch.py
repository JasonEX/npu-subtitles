"""Open one movie with the existing NPU live-caption pipeline.

Run directly through uv, drag a movie onto watch.cmd, or double-click
the command file to choose a movie. There is no separate full-file transcription
step; bounded preparation is owned by the live pipeline.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import multiprocessing as mp
from pathlib import Path
import sys
import traceback
from typing import TextIO
import uuid

ROOT = Path(__file__).resolve().parent
# The Windows launcher recognizes this as an error already reported in a dialog.
REPORTED_ERROR = 3


class _Tee:
    def __init__(self, console: TextIO | None, log: TextIO):
        self.console, self.log = console, log

    def write(self, text: str) -> int:
        self.log.write(text)
        self.log.flush()
        if self.console is not None:
            try:
                self.console.write(text)
            except UnicodeEncodeError:
                encoding = self.console.encoding or "utf-8"
                self.console.write(text.encode(encoding, errors="replace").decode(encoding))
        return len(text)

    def flush(self) -> None:
        self.log.flush()
        if self.console is not None:
            self.console.flush()

    def isatty(self) -> bool:
        return False


def parser() -> argparse.ArgumentParser:
    from npu_subtitles.app import parser as live_parser

    result = live_parser(optional_video=True)
    result.prog = "watch.py"
    result.description = "打开视频并实时生成 NPU 双语字幕；未指定视频时弹出文件选择框。"

    def argument_error(message: str) -> None:
        raise ValueError(f"启动参数有误：{message}")

    result.error = argument_error
    return result


def choose_video() -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        selected = filedialog.askopenfilename(
            parent=root,
            title="选择视频 — NPU 实时双语字幕",
            filetypes=[
                ("视频文件", "*.mkv *.mp4 *.avi *.mov *.wmv *.m4v *.webm *.ts *.m2ts *.mpg *.mpeg"),
                ("所有文件", "*.*"),
            ],
        )
        return Path(selected) if selected else None
    finally:
        root.destroy()


def show_error(error: BaseException, log_path: Path | None) -> None:
    message = f"无法启动或继续实时字幕：\n\n{error}"
    if log_path is not None:
        message += f"\n\n详细日志已保留：\n{log_path}"
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        try:
            messagebox.showerror("NPU 实时字幕", message, parent=root)
        finally:
            root.destroy()
    except Exception:
        # The console still carries the error if Tk itself is unavailable.
        if sys.stderr is not None:
            print(message, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    log_path = None
    try:
        log_directory = ROOT / "outputs/launch"
        log_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = log_directory / f"watch-{timestamp}-{uuid.uuid4().hex[:8]}.log"
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
                try:
                    args = parser().parse_args(argv)
                    if args.video is None:
                        args.video = choose_video()
                        if args.video is None:
                            print("已取消选择视频。", flush=True)
                            return 0
                    args.video = args.video.expanduser().resolve()
                    if not args.video.is_file():
                        raise FileNotFoundError(f"找不到视频文件：{args.video}")
                    print(f"启动日志：{log_path}", flush=True)
                    print(f"播放器：{args.player}；视频：{args.video}", flush=True)
                    from npu_subtitles.app import run

                    code = run(args)
                    if code not in (0, 130):
                        raise RuntimeError(f"实时字幕退出，状态码 {code}")
                    return code
                except KeyboardInterrupt:
                    print("已停止实时字幕。", flush=True)
                    return 130
                except SystemExit as exc:
                    # argparse --help is a normal exit and must not open Tk.
                    if exc.code in (None, 0):
                        return 0
                    raise RuntimeError(f"启动参数解析失败，状态码 {exc.code}") from exc
                except Exception:
                    traceback.print_exc()
                    raise
    except Exception as exc:
        show_error(exc, log_path)
        return REPORTED_ERROR


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
