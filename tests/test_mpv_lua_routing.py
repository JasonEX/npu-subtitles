"""The mpv Lua frontend copies a few routing.py constants; keep the copies equal."""

from pathlib import Path
import re
import unittest

from npu_subtitles.ass import OVERLAY_RES_X, OVERLAY_RES_Y
from npu_subtitles.routing import ALIASES, CHINESE_TITLE_HINTS, SUFFIX_TAGS, TITLE_HINTS

LUA = Path(__file__).resolve().parents[1] / "player/mpv/npu_subtitles.lua"


def lua_table(source: str, name: str) -> dict:
    body = re.search(rf"^local {name} = \{{\n(.*?)^\}}", source, re.S | re.M).group(1)
    return {
        key: True if value == "true" else value.strip("'")
        for key, value in re.findall(r"\['([^']*)'\]=('[^']*'|true)", body)
    }


class LuaRoutingMirrorTests(unittest.TestCase):
    def test_lua_constants_equal_python(self):
        source = LUA.read_text(encoding="utf-8")
        self.assertEqual(lua_table(source, "aliases"), ALIASES)
        self.assertEqual(
            lua_table(source, "chinese_title_hints"), dict.fromkeys(CHINESE_TITLE_HINTS, True)
        )
        self.assertEqual(lua_table(source, "title_hints"), TITLE_HINTS)
        self.assertEqual(lua_table(source, "suffix_tags"), dict.fromkeys(SUFFIX_TAGS, True))
        self.assertIn(
            f"overlay.res_x, overlay.res_y, overlay.z = {OVERLAY_RES_X}, {OVERLAY_RES_Y}, 0",
            source,
        )


if __name__ == "__main__":
    unittest.main()
