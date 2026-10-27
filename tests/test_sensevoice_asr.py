"""CTC and subtitle regressions independent of model downloads/NPU hardware."""

import unittest

import numpy as np

from npu_subtitles.asr.sensevoice import (
    SenseVoicePipeline,
    TokenSpan,
    ctc_spans,
    extract_features,
    subtitle_chunks,
)


def logits(labels):
    result = np.zeros((1, len(labels), max(labels) + 1), dtype=np.float32)
    result[0, np.arange(len(labels)), labels] = 1
    return result


class SenseVoiceTests(unittest.TestCase):
    def test_ctc_preserves_repeated_word_separated_by_blank(self):
        spans = ctc_spans(logits([4, 4, 4, 4, 1, 1, 0, 1, 2, 2]), 10, 1)
        self.assertEqual([span.token for span in spans], [1, 1, 2])
        self.assertAlmostEqual(spans[0].start, 0)
        self.assertAlmostEqual(spans[0].end, 0.09)
        self.assertAlmostEqual(spans[1].start, 0.15)

    def test_padding_and_metadata_do_not_become_subtitle(self):
        spans = ctc_spans(logits([4, 4, 4, 4, 1, 0, 2, 3]), 6, 0.5)
        self.assertEqual([span.token for span in spans], [1])
        self.assertEqual(ctc_spans(logits([4, 4, 4, 4, 0, 0]), 6, 0.5), [])

    def test_subtitles_preserve_dr_and_have_no_overlap_or_lone_punctuation(self):
        pieces = {1: "This is Dr.", 2: " Smith.", 3: " Next.", 4: "."}
        decode = lambda ids: "".join(pieces[i] for i in ids).strip()
        spans = [
            TokenSpan(1, 0, 1),
            TokenSpan(2, 1.1, 2),
            TokenSpan(3, 2.1, 3),
            TokenSpan(4, 9, 9.1),
        ]
        chunks = subtitle_chunks(spans, decode, 10)
        self.assertEqual([c["text"] for c in chunks], ["This is Dr. Smith.", "Next."])
        self.assertLessEqual(chunks[0]["end_ts"], chunks[1]["start_ts"])

    def test_pause_separates_cues_and_clamps_audio_end(self):
        decode = lambda ids: " ".join({1: "Hello", 2: "again"}[i] for i in ids)
        chunks = subtitle_chunks([TokenSpan(1, 1, 1.1), TokenSpan(2, 4, 4.1)], decode, 4.15)
        self.assertEqual(len(chunks), 2)
        self.assertAlmostEqual(chunks[1]["end_ts"], 4.15)

    def test_30s_features_fit_static_window_and_short_audio_is_finite(self):
        cmvn = (np.zeros(560, np.float32), np.ones(560, np.float32))
        features = extract_features(np.zeros(30 * 16000, np.float32), cmvn)
        self.assertEqual(features.shape, (500, 560))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(extract_features(np.zeros(10, np.float32), cmvn).shape, (1, 560))

    def test_invalid_audio_or_language_is_rejected_before_inference(self):
        pipe = object.__new__(SenseVoicePipeline)
        for audio in ([float("nan")], [[0]], np.zeros(480001)):
            with self.subTest(audio_type=type(audio)), self.assertRaises(ValueError):
                pipe.generate(audio)
        with self.assertRaises(ValueError):
            pipe.generate([0], language="fr")
        self.assertEqual(pipe.generate([], language="<|en|>").chunks, [])
        self.assertEqual(pipe.generate([0, 0]).texts, [""])


if __name__ == "__main__":
    unittest.main()
