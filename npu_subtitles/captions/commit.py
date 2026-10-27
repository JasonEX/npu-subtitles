"""Commit Moonshine observations into stable caption segments."""

from __future__ import annotations
from collections import OrderedDict
from difflib import SequenceMatcher
import math


def token(word):
    return "".join(c for c in word.casefold() if c.isalnum()) or word.casefold()


def finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


class CueCommitter:
    """Two bounded contexts; only advancing hypotheses confirm immutable spans."""

    ABBREVIATIONS = {
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "sr.",
        "jr.",
        "st.",
        "vs.",
        "etc.",
        "e.g.",
        "i.e.",
        "no.",
        "fig.",
    }

    def __init__(self, max_words=18):
        self.max_words = max_words
        self.hard_words = max(max_words, min(24, max_words + 6))
        self.reset()

    def reset(self):
        self.contexts = OrderedDict()
        self.newest_start = -math.inf
        self.notices = []
        self.revisions = []
        self.pending_tail = None

    @classmethod
    def sentence_end(cls, word):
        word = word.rstrip("\"'”’)]}")
        # ASR uses an ellipsis or a trailing dash for an unfinished phrase.
        # Neither is a sentence boundary, including at a context capacity limit.
        if word.endswith(("..", "…", "-", "–", "—")):
            return False
        if word.endswith(("?", "!", ";")):
            return True
        if not word.endswith(".") or word.casefold() in cls.ABBREVIATIONS:
            return False
        return not all(len(p) == 1 and p.isalpha() for p in word[:-1].split("."))

    @staticmethod
    def _map_boundary(old, new, consumed, *, consume_insertions):
        if not consumed:
            return 0
        mapped = min(consumed, len(new))
        for tag, a, b, c, d in SequenceMatcher(
            None, list(map(token, old)), list(map(token, new)), autojunk=False
        ).get_opcodes():
            if tag == "insert" and a == consumed:
                return d if consume_insertions and a != len(old) else c
            if a <= consumed < b:
                return min(d, c + consumed - a)
            if consumed == b and tag != "insert":
                tail_growth = tag == "replace" and b == len(old) and d == len(new) and d - c > b - a
                mapped = c + b - a if tail_growth else d
        return min(mapped, len(new))

    @staticmethod
    def map_consumed(old, new, consumed):
        """Map an already consumed prefix, excluding replay of new insertions."""
        return CueCommitter._map_boundary(old, new, consumed, consume_insertions=True)

    @staticmethod
    def map_span_boundary(old, new, boundary):
        """Map both ends of displayed spans with the same left insertion bias.

        A new word at a shared boundary belongs to the following caption: the
        previous end stops before it and the following start includes it. A
        consumption cursor has different ownership semantics and cannot safely
        map a mutable caption start.
        """
        return CueCommitter._map_boundary(old, new, boundary, consume_insertions=False)

    def release_tail(self, reason):
        if self.pending_tail is None:
            return []
        tail = dict(self.pending_tail["proposal"], pending_release_reason=reason)
        self.pending_tail = None
        self.notices.append(
            dict(
                reason="capacity_tail_released",
                release_reason=reason,
                source_id=tail["source_id"],
                text=tail["text"],
            )
        )
        return [tail]

    def pending_progress(self, identity, start, encoded):
        """A held tail can wait for one neighboring context, never a third."""
        pending = self.pending_tail
        if pending is None or identity == pending["proposal"]["source_id"]:
            return []
        if pending["next_source_id"] is None:
            pending["next_source_id"] = identity
            if start - pending["proposal"]["source_end"] > 0.6 + 1e-6:
                return self.release_tail("context_gap")
        elif pending["next_source_id"] != identity:
            return self.release_tail("next_context_exhausted")
        if encoded - pending["proposal"]["source_end"] >= 4.0 - 1e-6:
            return self.release_tail("encoded_wait_limit")
        return []

    def advance_audio(self, processed_until):
        # Silence may produce no hypotheses or activity events at all. Every
        # completed audio task still advances this bounded continuation wait.
        if (
            self.pending_tail is not None
            and processed_until - self.pending_tail["proposal"]["source_end"] >= 4.0 - 1e-6
        ):
            return self.release_tail("processed_audio_wait_limit")
        return []

    @staticmethod
    def source_part(proposal, frozen):
        keys = (
            "source_id",
            "source_revision",
            "word_start",
            "word_end",
            "text",
            "source_start",
            "source_end",
            "context_start",
            "word_observed_until",
        )
        return dict(
            {key: proposal[key] for key in keys},
            frozen=frozen,
            word_timestamps=proposal.get("word_timestamps", []),
            timing_kind=proposal.get("timing_kind"),
        )

    def bridge_tail(self, ctx, proposals, advanced, final):
        pending = self.pending_tail
        if pending is None or pending["next_source_id"] != ctx["id"]:
            return proposals
        if advanced and ctx["words"]:
            pending["hypotheses"] += 1
        # An explicit separate turn marker is stronger than the opportunity to
        # join a capacity fragment. Do not infer speaker identity from words.
        if ctx["words"] and ctx["words"][0] in {"-", "–", "—"}:
            return self.release_tail("explicit_turn_marker") + proposals
        if proposals:
            head, tail = proposals[0], pending["proposal"]
            prefix = tail["text"].rstrip().rstrip(".…").rstrip()
            if (
                head["word_start"] != 0
                or not prefix
                or len(prefix.split()) + len(head["text"].split()) > self.hard_words
            ):
                return self.release_tail("bridge_span_limit") + proposals
            observations = tail["word_observed_until"] + head["word_observed_until"]
            if len(tail["word_observed_until"]) != len(tail["text"].split()) or len(
                head["word_observed_until"]
            ) != len(head["text"].split()):
                observations = []
            spans = tail.get("word_timestamps", []) + head.get("word_timestamps", [])
            if (
                tail.get("timing_kind") != "ctc_emission_spans"
                or head.get("timing_kind") != "ctc_emission_spans"
                or len(spans) != len(prefix.split()) + len(head["text"].split())
            ):
                spans = []
            joined = dict(
                head,
                prefix_text=prefix,
                source_text=head["text"],
                text=f"{prefix} {head['text']}",
                source_parts=[self.source_part(tail, True), self.source_part(head, False)],
                context_start=tail["context_start"],
                source_start=tail["source_start"],
                word_observed_until=observations,
                commit_reason="capacity_bridge",
                word_timestamps=spans,
                timing_kind="ctc_emission_spans" if spans else None,
                alignment_kind="capacity_bridge_estimated",
                boundary_evidence=dict(
                    head["boundary_evidence"], capacity_prefix=tail["boundary_evidence"]
                ),
            )
            self.pending_tail = None
            self.notices.append(
                dict(
                    reason="capacity_tail_bridged",
                    source_id=head["source_id"],
                    prefix_source_id=tail["source_id"],
                    text=joined["text"],
                )
            )
            return [joined] + proposals[1:]
        if pending["hypotheses"] >= 2:
            return self.release_tail("hypothesis_wait_limit")
        if final:
            return self.release_tail("next_context_ended")
        return proposals

    def hold_capacity_tail(self, ctx, proposals, final):
        if not final or ctx["flush_reason"] != "capacity_limit" or not proposals:
            return proposals
        tail = proposals[-1]
        terminal = tail["text"].rstrip().rstrip("\"'”’)]}")
        if terminal.endswith(("?", "!")):
            # An explicit question/exclamation boundary is stronger than a
            # capacity continuation hint. Keep it independent of the next turn.
            return proposals
        # Only a new, unsupported last boundary is tentative. Earlier complete
        # sentences remain independently available. A bridge never chains on.
        supported = (
            tail["word_end"] <= ctx["stable"]
            and tail["word_end"] in ctx["previous_sentence_boundaries"]
        )
        if (
            tail["source_id"] != ctx["id"]
            or tail["word_end"] != len(ctx["words"])
            or supported
            or tail.get("prefix_text")
            or tail.get("reopened_caption")
        ):
            return proposals
        released = self.release_tail("replaced_by_next_capacity_tail")
        self.pending_tail = dict(proposal=tail, next_source_id=None, hypotheses=0)
        self.notices.append(
            dict(
                reason="capacity_tail_pending",
                source_id=tail["source_id"],
                text=tail["text"],
                source_end=tail["source_end"],
            )
        )
        return released + proposals[:-1]

    def split(self, ctx, allow_unconfirmed=False, close_tail=False):
        if ctx.get("withdrawn"):
            return []
        words = ctx["words"]
        confirmed = len(words) if allow_unconfirmed else ctx["stable"]
        cursor, result = ctx["consumed"], []
        current_boundaries = ctx.get("current_sentence_boundaries", {})
        previous_boundaries = ctx.get("previous_sentence_boundaries", {})
        soft_boundaries = ctx.get("soft_sentence_boundaries", {})
        while cursor < confirmed:
            limit = min(confirmed, cursor + self.hard_words)
            target = min(confirmed, cursor + self.max_words)
            # A normal partial boundary must remain in the current hypothesis.
            # It either survives two advancing decodes, or has stable right
            # context. A disappearing trailing period alone cannot split a
            # short sentence before its object arrives.
            ends = [
                i + 1
                for i in range(cursor, limit)
                if i + 1 in current_boundaries
                and (allow_unconfirmed or i + 1 in previous_boundaries or i + 1 < confirmed)
            ]
            soft = None
            if ends:
                end = ends[0]
                reason = (
                    "confirmed_sentence"
                    if allow_unconfirmed
                    else "persistent_sentence_boundary"
                    if end in previous_boundaries
                    else "internal_stable_sentence_boundary"
                )
            elif len(words) - cursor > self.max_words:
                commas = [
                    i + 1
                    for i in range(cursor + 5, target)
                    if words[i].rstrip("\"'”’)]}").endswith(",")
                ]
                old_ends = [b for b in soft_boundaries if cursor < b <= target]
                if commas:
                    end, reason = commas[-1], "bounded_stable_clause"
                elif old_ends:
                    end, reason = max(old_ends), "long_clause_observed_boundary"
                    soft = soft_boundaries[end]
                elif limit - cursor >= self.hard_words:
                    end, reason = limit, "hard_word_limit"
                elif close_tail:
                    end, reason = limit, ctx["flush_reason"]
                else:
                    break
            elif close_tail:
                end, reason = limit, ctx["flush_reason"]
            else:
                break
            text = " ".join(words[cursor:end])
            if any(c.isalnum() for c in text):
                result.append(
                    dict(
                        text=text,
                        source_id=ctx["id"],
                        source_revision=ctx["revision"],
                        word_start=cursor,
                        word_end=end,
                        source_start=ctx["last_commit_bound"],
                        context_start=ctx["start"],
                        word_observed_until=ctx["word_observed_until"][cursor:end]
                        if ctx.get("word_observed_until")
                        else [],
                        word_timestamps=ctx["word_timestamps"][cursor:end]
                        if ctx.get("word_timestamps")
                        else [],
                        timing_kind=ctx.get("timing_kind"),
                        segmentation_profile=ctx.get("segmentation_profile"),
                        source_end=ctx["encoded_until"],
                        received_until=ctx["received_until"],
                        observed_complete_until=(
                            soft["observed_until"]
                            if soft
                            else ctx["confirmed_observation_until"]
                            if end <= ctx["stable"]
                            else ctx["encoded_until"]
                        ),
                        evidence_until=ctx["evidence_until"],
                        commit_reason=reason,
                        boundary_evidence={
                            "current": dict(
                                word_end=end,
                                sentence_end=end in current_boundaries,
                                token=words[end - 1],
                                encoded_until=ctx["encoded_until"],
                            ),
                            "previous": dict(
                                word_end=end,
                                sentence_end=end in previous_boundaries,
                                token=previous_boundaries.get(end),
                                encoded_until=ctx["confirmed_observation_until"],
                            ),
                            "soft": soft,
                            "ctc": None,
                        },
                        incomplete=bool(ctx.get("context_incomplete")) and end == len(words),
                        alignment_kind="region_bound_estimated",
                    )
                )
                if ctx.get("reopened_caption"):
                    reopened = ctx.pop("reopened_caption")
                    floor = reopened["observation_floor"]
                    # An invalidated short tail is no longer evidence that
                    # its replacement words were available at the old time.
                    # This is a new observation bound, not acoustic alignment.
                    result[-1]["word_observed_until"] = [
                        max(floor, value) for value in result[-1]["word_observed_until"]
                    ]
                    result[-1]["observed_complete_until"] = max(
                        floor, result[-1]["observed_complete_until"]
                    )
                    result[-1]["reopened_observation_floor"] = floor
                    result[-1]["reopened_caption"] = reopened
            cursor = end
        ctx["consumed"] = cursor
        if result:
            ctx["last_commit_bound"] = ctx["encoded_until"]
        return result

    def feed(self, event):
        identity, start = int(event["id"]), finite(event["start"], "ASR start")
        end = finite(event["end"], "ASR end")
        encoded = finite(event.get("encoded_until", end), "encoded_until")
        received = finite(
            event.get("received_until", event.get("observed_audio_time", end)), "received_until"
        )
        if encoded > received + 1e-5:
            raise ValueError("Encoded audio extends beyond received audio")
        ctx = self.contexts.get(identity)
        if ctx is None:
            if start < self.newest_start - 1e-5:
                self.notices.append(dict(reason="old_context_ignored", source_id=identity))
                return []
            self.newest_start = max(start, self.newest_start)
            ctx = dict(
                id=identity,
                start=start,
                words=[],
                revision=-1,
                stable=0,
                consumed=0,
                encoded_until=start,
                received_until=start,
                evidence_until=start,
                hypothesis_until=-math.inf,
                last_commit_bound=start,
                pause_boundary=None,
                pause_committed=-math.inf,
                flush_reason="context_end",
            )
            self.contexts[identity] = ctx
            while len(self.contexts) > 2:
                self.contexts.popitem(last=False)
        if event.get("pause_detected") and event.get("paused_at") is not None:
            pause = finite(event["paused_at"], "paused_at")
            if pause > ctx["pause_committed"]:
                ctx["pause_boundary"] = pause
        released = self.pending_progress(identity, start, encoded)
        if event.get("event_kind", "hypothesis") != "hypothesis":
            return released
        revision = int(event["revision"])
        if revision < ctx["revision"]:
            return released
        advanced = (
            bool(event.get("hypothesis_advanced", True))
            and encoded > ctx["hypothesis_until"] + 1e-6
        )
        final = bool(event.get("is_final"))
        if not advanced and not final:
            return released
        words = str(event["text"]).split()
        if len(words) > 2048:
            raise ValueError("ASR context exceeds bounded text budget")
        if not words:
            ctx["stable"], ctx["revision"] = 0, revision
            ctx["withdrawn"] = True
            ctx["current_sentence_boundaries"] = {}
            ctx["previous_sentence_boundaries"] = {}
            ctx["soft_sentence_boundaries"] = {}
            ctx["word_observed_until"] = []
            ctx["word_timestamps"] = []
            self.revisions.append(
                dict(source_id=identity, source_revision=revision, old=ctx["words"], new=[])
            )
            self.notices.append(dict(reason="uncommitted_hypothesis_withdrawn", source_id=identity))
            if (
                final
                and self.pending_tail is not None
                and self.pending_tail["next_source_id"] == identity
            ):
                released.extend(self.release_tail("next_context_ended_empty"))
            return released
        old, common = ctx["words"], 0
        consumed = self.map_consumed(old, words, ctx["consumed"])
        if old:
            self.revisions.append(
                dict(source_id=identity, source_revision=revision, old=old, new=words)
            )
        if advanced and not ctx.get("withdrawn"):
            for before, after in zip(old, words):
                if token(before) != token(after):
                    break
                common += 1
        if list(map(token, old[: ctx["consumed"]])) != list(map(token, words[:consumed])):
            self.notices.append(dict(reason="committed_prefix_revision", source_id=identity))
        previous_encoded = ctx["hypothesis_until"]
        previous_boundaries = (
            ctx.get("current_sentence_boundaries", {})
            if advanced and not ctx.get("withdrawn")
            else {}
        )
        current_boundaries = {
            i + 1: word for i, word in enumerate(words) if self.sentence_end(word)
        }
        # Keep only still-stable, unconsumed historical boundaries. These are
        # bounded suggestions for a long clause, never semantic sentence ends.
        soft_boundaries = {
            b: value
            for b, value in ctx.get("soft_sentence_boundaries", {}).items()
            if consumed < b <= common
        }
        for b, word in previous_boundaries.items():
            if consumed < b <= common and b not in current_boundaries:
                soft_boundaries.setdefault(
                    b,
                    dict(
                        word_end=b,
                        token=word,
                        observed_until=previous_encoded,
                        kind="historical_nonellipsis_sentence_end",
                    ),
                )
        soft_boundaries = dict(list(sorted(soft_boundaries.items()))[-64:])
        observed = event.get("word_first_seen") or []
        observed = (
            [finite(value, "word observation") for value in observed]
            if len(observed) == len(words)
            else []
        )
        ctx.update(
            words=words,
            consumed=consumed,
            stable=common,
            revision=revision,
            withdrawn=False,
            current_sentence_boundaries=current_boundaries,
            previous_sentence_boundaries=previous_boundaries,
            soft_sentence_boundaries=soft_boundaries,
            word_observed_until=observed,
            word_timestamps=[],
            timing_kind=None,
            segmentation_profile=event.get("segmentation_profile"),
            confirmed_observation_until=previous_encoded
            if math.isfinite(previous_encoded)
            else encoded,
            encoded_until=encoded,
            received_until=received,
            evidence_until=finite(event.get("evidence_until", end), "evidence_until"),
            hypothesis_until=max(ctx["hypothesis_until"], encoded),
        )
        pause = ctx["pause_boundary"]
        flush_pause = pause is not None and encoded >= pause - 1e-5
        ctx["flush_reason"] = (
            str(event.get("context_end_reason") or event.get("final_reason") or "context_end")
            if final
            else "observed_pause"
        )
        ctx["context_incomplete"] = (
            bool(event.get("continuation_expected")) or ctx["flush_reason"] == "capacity_limit"
        )
        # A short energy pause is evidence, not permission to freeze an
        # unfinished phrase. Partial commits need punctuation or max_words.
        result = self.split(ctx, allow_unconfirmed=final, close_tail=final)
        if flush_pause and (result or ctx["consumed"] >= len(words)):
            ctx["pause_committed"], ctx["pause_boundary"] = pause, None
        result = self.bridge_tail(ctx, result, advanced, final)
        result = self.hold_capacity_tail(ctx, result, final)
        return released + result

    def finalize(self):
        result = self.release_tail("input_end")
        for ctx in self.contexts.values():
            ctx["flush_reason"] = "input_end"
            result.extend(self.split(ctx, allow_unconfirmed=True, close_tail=True))
        return result
