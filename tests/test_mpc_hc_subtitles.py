"""MPC-HC protocol and failure-path checks without launching a desktop player."""

from pathlib import Path
import unittest
from unittest.mock import Mock

from npu_subtitles.players.mpc_hc import (
    CMD_CONNECT,
    CMD_DISCONNECT,
    CMD_EOF,
    CMD_GET_POSITION,
    CMD_PLAYMODE,
    CMD_POSITION,
    CMD_SEEK,
    CMD_SET_AUDIO,
    CMD_STATE,
    MpcHCIPC,
    configure_audio,
    parse_tracks,
    split_fields,
)


def adapter():
    ipc = object.__new__(MpcHCIPC)
    ipc.source_subtitle = Path("generated-session.srt").resolve()
    ipc.subtitle = ipc.source_subtitle.with_suffix(".ass")
    ipc.font = "Microsoft YaHei"
    ipc.font_size = 28
    ipc.responses = {}
    ipc.events = []
    ipc.loaded = False
    ipc.paused = True
    ipc.eof = False
    ipc.disconnected = False
    ipc.player_hwnd = 0
    return ipc


class MpcHCProtocolTests(unittest.TestCase):
    def test_escaped_track_name_and_selection(self):
        tracks = parse_tracks(r"English \| SDH|[Local] generated-session.srt|1", "sub")
        self.assertEqual(tracks[0]["title"], "English | SDH")
        self.assertFalse(tracks[0]["selected"])
        self.assertTrue(tracks[1]["selected"])
        self.assertEqual(split_fields(r"C:\Movies\film.mkv|120"), [r"C:\Movies\film.mkv", "120"])

    def test_no_tracks_disabled_tracks_and_malformed_reply(self):
        self.assertEqual(parse_tracks("-2", "audio"), [])
        self.assertEqual(parse_tracks("-1", "sub"), [])
        self.assertFalse(parse_tracks("English|-1", "sub")[0]["selected"])
        for invalid in ("English", "English|garbage", "English|2", "English|-2"):
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                parse_tracks(invalid, "sub")

    def test_notifications_keep_seek_eof_and_user_pause_in_sync(self):
        ipc = adapter()
        ipc._notification(CMD_STATE, "2")
        ipc._notification(CMD_PLAYMODE, "0")
        self.assertTrue(ipc.loaded)
        self.assertFalse(ipc.paused)
        ipc._notification(CMD_PLAYMODE, "1")
        ipc._notification(CMD_EOF, "")
        self.assertTrue(ipc.paused)
        self.assertTrue(ipc.eof)
        ipc._notification(CMD_SEEK, "63.125")
        self.assertFalse(ipc.eof)
        self.assertEqual(ipc.events, [{"event": "seek", "position": 63.125}])
        ipc._notification(CMD_DISCONNECT, "")
        self.assertTrue(ipc.disconnected)

    def test_rejects_wrong_connection_handle_and_failed_load(self):
        ipc = adapter()
        with self.assertRaises(RuntimeError):
            ipc._notification(CMD_CONNECT, "42", sender=43)
        with self.assertRaises(RuntimeError):
            ipc._notification(CMD_STATE, "4")

    def test_query_discards_stale_position_before_request(self):
        ipc = adapter()
        ipc.responses[CMD_POSITION] = "old"

        def send(command):
            self.assertEqual(command, CMD_GET_POSITION)
            self.assertNotIn(CMD_POSITION, ipc.responses)
            ipc._notification(CMD_POSITION, "63.500")

        ipc.send = send
        ipc.wait_until = lambda predicate, description: self.assertTrue(predicate())
        self.assertEqual(ipc.query(CMD_GET_POSITION, CMD_POSITION), "63.500")

    def test_generated_track_must_exist_and_be_unique(self):
        ipc = adapter()
        ipc.query = Mock(return_value="English|[Local] generated-session.ass|0")
        self.assertEqual(ipc.generated_subtitle()["id"], 1)
        ipc.query.return_value = "English|0"
        with self.assertRaises(RuntimeError):
            ipc.generated_subtitle()
        ipc.query.return_value = "generated-session.ass|generated-session.ass|0"
        with self.assertRaises(RuntimeError):
            ipc.generated_subtitle()

    def test_auto_loaded_transport_srt_does_not_ambigously_match_generated_ass(self):
        ipc = adapter()
        ipc.query = Mock(
            return_value="[Local] generated-session.ass|[Local] generated-session.srt|0"
        )
        self.assertEqual(ipc.generated_subtitle()["id"], 0)
        self.assertEqual([track["external"] for track in ipc.subtitle_tracks()], [True, False])

    def test_selection_requires_player_acknowledgement(self):
        ipc = adapter()
        ipc.send = Mock()
        ipc.subtitle_tracks = Mock(
            return_value=[{"id": 0, "selected": True}, {"id": 1, "selected": False}]
        )
        with self.assertRaises(RuntimeError):
            ipc.select_subtitle(1)
        ipc.subtitle_tracks.return_value[0]["selected"] = False
        ipc.subtitle_tracks.return_value[1]["selected"] = True
        ipc.select_subtitle(1)

    def test_audio_selection_uses_audio_index_and_verifies_result(self):
        ipc = adapter()
        ipc.send = Mock()
        ipc.query = Mock(side_effect=["English|Spanish|0", "English|Spanish|1"])
        self.assertEqual(configure_audio(ipc, 1, 7), 1)
        ipc.send.assert_called_once_with(CMD_SET_AUDIO, "1")
        ipc.query = Mock(return_value="English|Spanish|0")
        with self.assertRaises(RuntimeError):
            configure_audio(ipc, 1, 7)
        with self.assertRaises(RuntimeError):
            configure_audio(ipc, 2, 7)

    def test_sub_add_rejects_other_files(self):
        ipc = adapter()
        with self.assertRaises(ValueError):
            ipc.command("sub-add", "another-session.srt", "select")


if __name__ == "__main__":
    unittest.main()
