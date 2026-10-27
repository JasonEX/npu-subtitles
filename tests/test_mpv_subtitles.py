"""Current subtitle format, mpv IPC and process-lifetime regressions."""

import json
from pathlib import Path
import tempfile
import unittest
from npu_subtitles.formats import Cue, atomic_srt, merge_cues, render_srt, timestamp
from npu_subtitles.players.mpv import (
    MpvCommandError,
    MpvIPC,
    matching_response,
)


class SubtitleTests(unittest.TestCase):
    def test_overlap_dedup_does_not_drop_later_repeated_dialogue(self):
        cues = [Cue(20, 24, "Yes!"), Cue(23, 26, "yes"), Cue(35, 37, "Yes!")]
        self.assertEqual(merge_cues(cues), [Cue(20, 26, "Yes!"), Cue(35, 37, "Yes!")])

    def test_timestamp_rounding_carries_minutes(self):
        self.assertEqual(timestamp(59.9996), "00:01:00,000")
        self.assertEqual(timestamp(3661.125), "01:01:01,125")

    def test_srt_orders_segments_and_atomic_write_is_utf8(self):
        cues = [Cue(302, 303, "世界"), Cue(300, 301, "Hello")]
        expected = (
            "1\n00:05:00,000 --> 00:05:01,000\nHello\n\n2\n00:05:02,000 --> 00:05:03,000\n世界\n"
        )
        self.assertEqual(render_srt(cues), expected)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "live.srt"
            atomic_srt(path, cues)
            self.assertEqual(path.read_text(encoding="utf-8"), expected)
            atomic_srt(path, [])
            self.assertEqual(path.read_text(encoding="utf-8"), "")
            self.assertFalse(path.with_suffix(".srt.tmp").exists())


class FakeIPC(MpvIPC):
    def __init__(self, messages):
        self.timeout = 1
        self.request_id = 0
        self.buffer = b""
        self.events = []
        self.sent = []
        self.incoming = iter(messages)

    def _write(self, data):
        self.sent.append(json.loads(data))

    def _read(self, deadline):
        return next(self.incoming)


class IPCTests(unittest.TestCase):
    def test_events_and_old_responses_are_not_command_results(self):
        ipc = FakeIPC(
            [
                b'{"event":"seek"}\n{"request_id":99,"error":"success","data":"old"}\n',
                b'{"request_id":1,"error":"success","data":',
                b"300.25}\n",
            ]
        )
        self.assertEqual(ipc.get("time-pos"), 300.25)
        self.assertEqual(ipc.events, [{"event": "seek"}])
        self.assertEqual(ipc.sent, [{"command": ["get_property", "time-pos"], "request_id": 1}])

    def test_command_failure_is_reported(self):
        ipc = FakeIPC([b'{"request_id":1,"error":"property unavailable"}\n'])
        with self.assertRaisesRegex(MpvCommandError, "property unavailable") as caught:
            ipc.get("time-pos")
        self.assertEqual(caught.exception.status, "property unavailable")
        self.assertEqual(caught.exception.command, ("get_property", "time-pos"))

    def test_optional_time_is_unavailable_during_seek_then_recovers(self):
        ipc = FakeIPC(
            [
                b'{"request_id":1,"error":"property unavailable"}\n',
                b'{"request_id":2,"error":"success","data":300.0}\n',
            ]
        )
        self.assertIsNone(ipc.get_optional("time-pos"))
        self.assertEqual(ipc.get_optional("time-pos"), 300.0)

    def test_optional_property_does_not_hide_other_player_errors(self):
        ipc = FakeIPC([b'{"request_id":1,"error":"property not found"}\n'])
        with self.assertRaisesRegex(MpvCommandError, "property not found"):
            ipc.get_optional("time-pos")

    def test_response_requires_matching_id_and_status(self):
        self.assertFalse(matching_response({"event": "seek", "request_id": 2}, 2))
        self.assertFalse(matching_response({"request_id": 1, "error": "success"}, 2))
        self.assertTrue(matching_response({"request_id": 2, "error": "success"}, 2))
