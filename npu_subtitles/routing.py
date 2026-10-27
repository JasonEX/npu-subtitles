"""Model-free, deterministic routing for the native mpv plugin."""

from __future__ import annotations

from dataclasses import dataclass
import html
import re
import unicodedata
from npu_subtitles.translation.languages import SUBTITLE_LANGUAGES

ALIASES = {
    "zh": "zh",
    "zho": "zh",
    "chi": "zh",
    "chs": "zh",
    "cht": "zh",
    "cmn": "zh",
    "yue": "zh",
    "chinese": "zh",
    "mandarin": "zh",
    "en": "en",
    "eng": "en",
    "english": "en",
    "ja": "ja",
    "jpn": "ja",
    "japanese": "ja",
    "ko": "ko",
    "kor": "ko",
    "korean": "ko",
    "한국어": "ko",
    "es": "es",
    "spa": "es",
    "spanish": "es",
    "español": "es",
    "fr": "fr",
    "fra": "fr",
    "fre": "fr",
    "french": "fr",
    "français": "fr",
    "de": "de",
    "deu": "de",
    "ger": "de",
    "german": "de",
    "deutsch": "de",
    "pt": "pt",
    "por": "pt",
    "portuguese": "pt",
    "português": "pt",
    "it": "it",
    "ita": "it",
    "italian": "it",
    "italiano": "it",
    "ru": "ru",
    "rus": "ru",
    "russian": "ru",
    "русский": "ru",
    "hi": "hi",
    "hin": "hi",
    "hindi": "hi",
    "हिन्दी": "hi",
    "हिंदी": "hi",
}
BITMAP = {"hdmv_pgs_subtitle", "pgs", "dvd_subtitle", "dvb_subtitle", "xsub"}
# Conservative function-word evidence for untagged text. No vocabulary model,
# per-video phrases, inference imports, or guesses from Latin/Cyrillic alone.
LANGUAGE_WORDS = {
    "en": set(
        "the and that this with your you have what would could should there they their from about because please where which will were was are not here".split()
    ),
    "es": set(
        "que los las una por para pero con del usted ustedes nosotros está son esto fue más muy también como sus sin tiene hay quiero".split()
    ),
    "fr": set(
        "le les une vous nous pour avec est que pas dans mais cette ce sont être aux des qui sur plus elle il du au ne je tu".split()
    ),
    "de": set(
        "die der das und nicht ist sind ich sie wir für mit den dem ein eine auf dass aber auch von zu haben werden wird du es im".split()
    ),
    "pt": set(
        "que não uma para por com os as você vocês nós está são isso foi mais dos das pelo pela muito também este esta seu sua".split()
    ),
    "it": set(
        "che non una gli della delle dei per con sono questo questa è siamo noi voi loro anche dopo come più nella dallo ma il lo un hai era".split()
    ),
    "ru": set(
        "что это как его она они для было были или когда чтобы если только меня тебя есть также который которые свои себя будет очень".split()
    ),
    "hi": set(
        "है हैं में कि और यह इस नहीं को से लिए एक था थी थे लेकिन हो कर भी पर आप हम उनके किया क्योंकि".split()
    ),
}


def language(value: str | None) -> str:
    raw = (value or "").strip().lower().replace("_", "-")
    return ALIASES.get(raw.split("-")[0], raw if raw not in {"", "und", "unknown"} else "")


def text_language(text: str, declared: str = "") -> str:
    """Conservative script check; Latin alone never proves English."""
    declared = language(declared)
    if declared == "zh":
        return "zh"
    kana = len(re.findall(r"[\u3040-\u30ff]", text))
    hangul = len(re.findall(r"[\uac00-\ud7af]", text))
    han = len(re.findall(r"[\u3400-\u9fff]", text))
    chinese_lines = [
        line
        for line in text.splitlines()
        if len(re.findall(r"[\u3400-\u9fff]", line)) >= 12
        and not re.search(r"[\u3040-\u30ff]", line)
    ]
    if (kana or hangul) and len(chinese_lines) >= 3:
        return "zh"  # substantial Chinese lines alongside Japanese, not kanji alone
    if kana:
        # Do not misclassify ordinary Japanese kanji as a Chinese translation.
        return "ja" if not declared else declared
    if hangul >= 6 and not declared:
        return "ko"
    if han >= 12 and declared != "ja":
        return "zh"
    if not declared:
        words = re.findall(
            r"[\u0900-\u0963\u0971-\u097f]+|[^\W\d_]+",
            unicodedata.normalize("NFC", text.casefold()),
        )
        evidence = sorted(
            ((len(set(words) & anchors), lang) for lang, anchors in LANGUAGE_WORDS.items()),
            reverse=True,
        )
        if len(words) >= 12 and evidence[0][0] >= 4 and evidence[0][0] >= evidence[1][0] + 3:
            return evidence[0][1]
    return declared


def track_language(track: dict) -> str:
    declared = language(track.get("lang"))
    label = str(track.get("title", ""))
    if re.search(r"中文|汉语|漢語|简体|簡體|繁体|繁體|中英|英中|日中|中日", label):
        return "zh"
    if declared:
        return declared
    if re.search(r"日本語|日语|日語", label):
        return "ja"
    for hint, code in [
        ("韩语", "ko"),
        ("韓語", "ko"),
        ("西班牙语", "es"),
        ("西班牙語", "es"),
        ("法语", "fr"),
        ("法語", "fr"),
        ("德语", "de"),
        ("德語", "de"),
        ("葡萄牙语", "pt"),
        ("葡萄牙語", "pt"),
        ("意大利语", "it"),
        ("義大利語", "it"),
        ("俄语", "ru"),
        ("俄語", "ru"),
        ("印地语", "hi"),
        ("印地語", "hi"),
    ]:
        if hint in label:
            return code
    for token in re.split(r"[\s.\[\]()_-]+", label.lower()):
        if token in ALIASES:
            return ALIASES[token]
    # Recognize a language suffix, not a language word in the movie's title
    # (e.g. "It.2017.srt" and "The.French.Connection.srt" are not language tags).
    name = str(track.get("external-filename", "")).rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    tokens = re.split(r"[\s.\[\]()_-]+", name.rsplit(".", 1)[0].lower())
    for token in reversed(tokens[1:]):
        if token in ALIASES:
            return ALIASES[token]
        if token not in {
            "",
            "sdh",
            "cc",
            "forced",
            "default",
            "hans",
            "hant",
            "cn",
            "tw",
            "br",
            "us",
            "gb",
        }:
            break
    return ""


@dataclass(frozen=True)
class Route:
    kind: str
    source: str = ""
    display: str = ""
    track_id: int | None = None
    reason: str = ""


def choose_route(
    tracks: list[dict], audio_language: str, available_sources: set[str] | None = None
) -> Route:
    audio_language = language(audio_language)
    available = set(SUBTITLE_LANGUAGES) if available_sources is None else available_sources
    subs = [t for t in tracks if t.get("type") == "sub" and not t.get("npu-generated")]
    subs.sort(
        key=lambda t: (
            not bool(t.get("default")),
            bool(t.get("forced")),
            bool(t.get("hearing-impaired")),
            int(t["id"]),
        )
    )
    for t in subs:
        if track_language(t) == "zh":
            return Route("native", "zh", "zh", int(t["id"]))
    text = [t for t in subs if t.get("codec") not in BITMAP and track_language(t) in available]
    if audio_language == "ja":
        for t in text:
            if track_language(t) == "ja":
                return Route("subtitle", "ja", "ja", int(t["id"]))
    for t in text:
        if track_language(t) == "en":
            return Route("subtitle", "en", "en", int(t["id"]))
    other = sorted(text, key=lambda t: track_language(t) != audio_language)
    if other:
        t = other[0]
        return Route("subtitle", track_language(t), "en", int(t["id"]))
    if audio_language in {"zh", "en", "ja"}:
        return Route("asr", audio_language, audio_language)
    return Route("none", reason="No supported subtitle or confidently identified audio language")


@dataclass(frozen=True)
class SubtitleCue:
    start: float
    end: float
    text: str


def parse_srt(text: str) -> list[SubtitleCue]:
    """Parse FFmpeg-normalized SRT; keep timings, order and repeated dialogue."""
    stamp = r"(\d+):(\d{2}):(\d{2})[,.](\d{3})"
    result = []

    def seconds(groups):
        h, m, s, ms = map(int, groups)
        return h * 3600 + m * 60 + s + ms / 1000

    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        match = re.search(stamp + r"\s*-->\s*" + stamp + r"[^\n]*\n(.*)", block, re.S)
        if not match:
            continue
        start, end = seconds(match.groups()[:4]), seconds(match.groups()[4:8])
        body = html.unescape(re.sub(r"<[^>]+>|\{\\[^}]*\}", "", match.group(9)))
        body = "\n".join(line.strip() for line in body.splitlines() if line.strip())
        if end > start >= 0 and body:
            result.append(SubtitleCue(start, end, body))
    return sorted(result, key=lambda cue: (cue.start, cue.end))
