import unittest
import tempfile
from pathlib import Path
from npu_subtitles.routing import choose_route, language, parse_srt, text_language, track_language
from npu_subtitles.translation.languages import available_subtitle_languages


def sub(index, lang, **kw):
    return dict(id=index, type="sub", lang=lang, codec="subrip", **kw)


class RoutingTests(unittest.TestCase):
    def test_priority_matrix(self):
        cases = [
            ([sub(1, "eng"), sub(2, "zh-Hant")], "ja", ("native", "zh", 2)),
            ([sub(1, "eng")], "en", ("subtitle", "en", 1)),
            ([sub(1, "jpn")], "en", ("subtitle", "en", 1)),
            ([sub(1, "eng"), sub(2, "jpn")], "ja", ("subtitle", "ja", 2)),
            ([sub(1, "eng")], "ja", ("subtitle", "en", 1)),
            ([], "zho", ("asr", "zh", None)),
            ([], "en", ("asr", "en", None)),
            ([], "ja", ("asr", "ja", None)),
            ([sub(1, "fr")], "fr", ("subtitle", "en", 1)),
            ([sub(1, "fr")], "en", ("subtitle", "en", 1)),
            ([], "", ("none", "", None)),
        ]
        for tracks, audio, expected in cases:
            with self.subTest(audio=audio, tracks=tracks):
                route = choose_route(tracks, audio)
                self.assertEqual((route.kind, route.display, route.track_id), expected)

    def test_bitmap_chinese_native_other_bitmap_uses_audio(self):
        self.assertEqual(
            choose_route([dict(sub(1, "zh"), codec="hdmv_pgs_subtitle")], "en").kind, "native"
        )
        self.assertEqual(
            choose_route([dict(sub(1, "en"), codec="hdmv_pgs_subtitle")], "en").kind, "asr"
        )

    def test_aliases_and_bilingual_titles(self):
        for alias in ["zh-Hant", "zh_TW", "chi", "zho", "chs", "cht"]:
            self.assertEqual(language(alias), "zh")
        self.assertEqual(track_language(sub(1, "eng", title="中英双语")), "zh")
        self.assertEqual(
            track_language(sub(1, "", **{"external-filename": r"C:\Videos\test.jpn.srt"})), "ja"
        )

    def test_additional_languages_and_no_new_asr(self):
        for alias, expected in [
            ("kor", "ko"),
            ("spa", "es"),
            ("fra", "fr"),
            ("fre", "fr"),
            ("ger", "de"),
            ("deu", "de"),
            ("pt-BR", "pt"),
            ("por", "pt"),
            ("ita", "it"),
            ("rus", "ru"),
        ]:
            self.assertEqual(language(alias), expected)
            self.assertEqual(choose_route([sub(1, alias)], "ja").source, expected)
            self.assertEqual(choose_route([], alias).kind, "none")
        self.assertEqual(choose_route([sub(1, "fr"), sub(2, "jpn")], "ja").display, "ja")
        self.assertEqual(choose_route([sub(1, "fr"), sub(2, "eng")], "fr").source, "en")
        self.assertEqual(choose_route([sub(1, "fr"), sub(2, "ita")], "it").source, "it")
        self.assertEqual(choose_route([sub(1, "ar")], "ar").kind, "none")
        self.assertEqual(language("hin"), "hi")
        self.assertEqual(choose_route([sub(1, "hin")], "hi").kind, "none")

    def test_missing_model_uses_available_route_and_filename_suffix_is_safe(self):
        self.assertEqual(choose_route([sub(1, "fr")], "en", {"en", "ja"}).kind, "asr")
        self.assertEqual(choose_route([sub(1, "fr")], "fr", {"en", "ja"}).kind, "none")
        for name, lang in [
            ("movie.pt-BR.srt", "pt"),
            ("movie.kor.sdh.srt", "ko"),
            ("It.2017.srt", ""),
            ("The.French.Connection.srt", ""),
        ]:
            self.assertEqual(track_language(sub(1, "", **{"external-filename": name})), lang)

    def test_incomplete_or_failed_preparation_is_not_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "models/mtran-fr-en-npu"
            directory.mkdir(parents=True)
            for name in (
                "encoder.xml",
                "encoder.bin",
                "decoder.xml",
                "decoder.bin",
                "config.json",
                "vocab.fren.spm",
            ):
                (directory / name).touch()
            marker = directory / "npu-validation.json"
            self.assertNotIn("fr", available_subtitle_languages(root))
            for contents in ("{", '{"status":"failed"}'):
                marker.write_text(contents, encoding="utf-8")
                self.assertNotIn("fr", available_subtitle_languages(root))
            marker.write_text('{"status":"passed"}', encoding="utf-8")
            self.assertIn("fr", available_subtitle_languages(root))
            (directory / "decoder.bin").unlink()
            self.assertNotIn("fr", available_subtitle_languages(root))

    def test_untagged_text_requires_clear_language_evidence(self):
        self.assertEqual(text_language("여기에서 잠시 기다려 주세요."), "ko")
        self.assertEqual(
            text_language(
                "Nous ne sommes pas dans cette maison mais vous pouvez nous attendre avec les autres."
            ),
            "fr",
        )
        self.assertEqual(
            text_language("Wir haben eine Frage für Sie und wir wissen nicht ob das richtig ist."),
            "de",
        )
        self.assertEqual(
            text_language("Вы знаете что это было очень важно для меня и они также были здесь."),
            "ru",
        )
        self.assertEqual(
            text_language("यह एक उदाहरण है और हम इस विषय पर भी बात कर रहे हैं क्योंकि यह ज़रूरी है।"), "hi"
        )
        self.assertEqual(text_language("Bonjour merci thank you danke gracias"), "")

    def test_unknown_latin_is_not_assumed_english(self):
        self.assertEqual(text_language("Bonjour, merci beaucoup."), "")
        self.assertEqual(text_language("明日は一緒に図書館へ行きましょう。"), "ja")
        self.assertEqual(text_language("我们明天一起去附近的图书馆看书吧。"), "zh")
        self.assertEqual(text_language("漢字", "ja"), "ja")
        self.assertEqual(
            text_language(
                "Please wait here because the train will arrive soon and you have enough time."
            ),
            "en",
        )
        bilingual = "明日は図書館に行きましょう。\n我们明天一起去附近的图书馆看书吧。\n" * 3
        self.assertEqual(text_language(bilingual, "ja"), "zh")

    def test_srt_keeps_overlap_duplicates_and_escapes_no_commands(self):
        cues = parse_srt(
            "1\n00:00:01,200 --> 00:00:03,200\n<i>Hello &amp; goodbye.</i>\n\n2\n00:00:02,000 --> 00:00:04,000\nHello &amp; goodbye.\n"
        )
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0].start, 1.2)
        self.assertEqual(cues[0].end, 3.2)
        self.assertEqual(cues[0].text, "Hello & goodbye.")


if __name__ == "__main__":
    unittest.main()
