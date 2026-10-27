"""Assign sealed ASR occurrences to immutable, bounded subtitle segments.

An ASR snapshot can revise its mutable suffix; it cannot revise a segment that
already owns sealed occurrences. Physical windows and logical capacity rolls
are transport boundaries, not sentence boundaries. This module never translates
or displays text and never maps ownership through mutable string positions.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import math

from npu_subtitles.captions.timing import CueTiming
from npu_subtitles.captions.language import (
    JAPANESE_MAX_CHARACTERS,
    JapaneseBoundaries,
    check_language,
    join_units,
    japanese_clause_end,
    japanese_sentence_end,
    reading_units,
)


class GraniteSentenceAssembler:
    PROFILE = "granite_ctc_v3"
    RIGHT_CONTEXT = 0.32
    SHORT_TAIL_CONTEXT = 0.8
    CLAUSE_GAP = 0.24
    TURN_GAP = 1.0
    MAX_CONTEXT_WORDS = 1024

    def __init__(
        self,
        sentence_end,
        max_words=18,
        max_hold=6.0,
        max_lag=3.0,
        min_read=1.2,
        source_language="en",
    ):
        self.source_language = check_language(source_language)
        self.sentence_end = japanese_sentence_end if source_language == "ja" else sentence_end
        self.max_words, self.max_hold = int(max_words), float(max_hold)
        self.max_units = JAPANESE_MAX_CHARACTERS if source_language == "ja" else self.max_words
        self.japanese_boundaries = JapaneseBoundaries() if source_language == "ja" else None
        self.max_lag, self.min_read = float(max_lag), float(min_read)
        self.reset(0)

    def reset(self, epoch):
        self.epoch, self.next_sentence = int(epoch), 0
        self.contexts = OrderedDict()
        self.pending, self.mutable, self.notices = [], [], []
        self.active_id = None
        self.sealed_until = self.encoded_until = self.received_until = -math.inf
        self.last_word_end = -math.inf
        self.continuation_from = None
        self.capacity_tail = None

    @staticmethod
    def number(value, name):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    def _validate(self, event):
        if event.get("source_language", "en") != self.source_language:
            raise ValueError("ASR source language does not match the sentence assembler")
        identity = int(event["id"])
        transcript = event["transcript"]
        revision = int(transcript["revision"])
        sealed = transcript["sealed_words"]
        mutable = transcript["mutable_words"]
        if len(sealed) + len(mutable) > self.MAX_CONTEXT_WORDS:
            raise ValueError("Transcript exceeds the bounded logical context")
        encoded = self.number(event.get("encoded_until", event["end"]), "encoded_until")
        received = self.number(event.get("received_until", encoded), "received_until")
        frontier = self.number(transcript["sealed_until"], "sealed_until")
        revision_from = self.number(transcript["revision_from"], "revision_from")
        if (
            frontier > encoded + 1e-5
            or encoded > received + 1e-5
            or revision_from > frontier + 1e-5
        ):
            raise ValueError("Transcript frontiers exceed their audio evidence")
        previous = self.contexts.get(identity)
        if self.active_id is not None and identity < self.active_id:
            return None
        if previous:
            if revision < previous["revision"]:
                return None
            if frontier < previous["sealed_until"] - 1e-5:
                raise ValueError("The sealed frontier moved backward")
            if revision_from < previous["sealed_until"] - 1e-5 and revision > previous["revision"]:
                raise ValueError("A revision reaches inside the sealed prefix")
            if sealed[: len(previous["words"])] != previous["words"]:
                raise ValueError("Sealed occurrences were rewritten or removed")
        elif self.active_id is not None and identity <= self.active_id:
            return None
        elif self.active_id is not None and not self.contexts[self.active_id]["closed"]:
            raise ValueError("A new logical context replaced an unclosed context")
        ids, last_start = set(), -math.inf
        for index, word in enumerate(sealed + mutable):
            word_id = str(word["occurrence_id"])
            if not word_id.startswith(f"ctx:{identity}:word:") or word_id in ids:
                raise ValueError("Occurrence identity is duplicate or belongs to another context")
            ids.add(word_id)
            start = self.number(word["start"], "word start")
            end = self.number(word["end"], "word end")
            if (
                not str(word["text"]).strip()
                or start < last_start - 1e-5
                or end < start
                or end > encoded + 1e-5
            ):
                raise ValueError("Word ranges are empty, unordered, or outside the encoded audio")
            if index < len(sealed) and end > frontier + 1e-5:
                raise ValueError("A sealed occurrence exceeds the sealed frontier")
            if index >= len(sealed) and start < frontier - 1e-5:
                raise ValueError("A mutable occurrence reaches inside the sealed frontier")
            last_start = start
        new_words = sealed[len(previous["words"]) if previous else 0 :]
        if new_words and float(new_words[0]["start"]) < self.last_word_end - 1e-5:
            raise ValueError("New sealed occurrences overlap already accepted audio")
        closure = (
            event.get("closure_reason")
            or event.get("context_end_reason")
            or event.get("final_reason")
        )
        closed = bool(event.get("is_final"))
        if closed and mutable:
            raise ValueError("A closed transcript still contains mutable occurrences")
        return dict(
            identity=identity,
            revision=revision,
            sealed=sealed,
            mutable=mutable,
            new_words=new_words,
            frontier=frontier,
            encoded=encoded,
            received=received,
            revision_from=revision_from,
            closed=closed,
            closure=closure,
        )

    def feed(self, event, now, position, budget=None):
        try:
            update = self._validate(event)
        except (KeyError, TypeError, ValueError) as exc:
            # A broken ownership contract cannot safely produce fresh cards.
            # The live runner records this explicit error and owns cleanup;
            # silently disabling captions until a seek would look like success.
            transcript = event.get("transcript") or {}
            raise RuntimeError(
                "sealed_source_contract_violation: "
                f"source_id={event.get('id')}, revision={transcript.get('revision')}, "
                f"sealed_until={transcript.get('sealed_until')}: {exc}"
            ) from exc
        if update is None:
            self.notices.append(dict(reason="stale_transcript_ignored", source_id=event["id"]))
            return []
        identity = update["identity"]
        old = self.contexts.get(identity)
        first_index = len(old["words"]) if old else 0
        for index, word in enumerate(update["new_words"], first_index):
            uncertain = [
                deepcopy(span)
                for span in event["transcript"].get("uncertain_intervals", [])
                if float(span["start"]) < float(word["end"])
                and float(span["end"]) > float(word["start"])
            ]
            self.pending.append(
                dict(
                    deepcopy(word),
                    source_id=identity,
                    source_revision=int(event["revision"]),
                    transcript_revision=update["revision"],
                    source_word_index=index,
                    sealed_at=now,
                    sealed_audio_until=update["frontier"],
                    uncertain_intervals=uncertain,
                )
            )
        if update["new_words"]:
            self.last_word_end = max(float(word["end"]) for word in update["new_words"])
        self.active_id = identity
        self.contexts[identity] = dict(
            words=deepcopy(update["sealed"]),
            revision=update["revision"],
            sealed_until=update["frontier"],
            closed=update["closed"],
        )
        while len(self.contexts) > 2:
            self.contexts.popitem(last=False)
        self.mutable = deepcopy(update["mutable"])
        # A new logical context may overlap old *silence* through pre-roll.
        # Word ownership, checked above, must still advance without overlap.
        self.sealed_until = max(self.sealed_until, update["frontier"])
        self.encoded_until = max(self.encoded_until, update["encoded"])
        self.received_until = max(self.received_until, update["received"])
        if update["closed"] and update["closure"] == "capacity_limit" and self.pending:
            self.capacity_tail = self.pending[-1]["occurrence_id"]
        self.notices.append(
            dict(
                reason="sentence_draft_updated",
                source_id=identity,
                transcript_revision=update["revision"],
                revision_from=update["revision_from"],
                sealed_until=update["frontier"],
                encoded_until=update["encoded"],
                pending_sealed_words=len(self.pending),
                mutable_words=len(self.mutable),
                newly_sealed_word_ids=[word["occurrence_id"] for word in update["new_words"]],
                closure_reason=update["closure"] if update["closed"] else None,
            )
        )
        force = (
            update["closure"]
            if update["closed"] and update["closure"] in {"long_silence", "input_end"}
            else None
        )
        return self._drain(now, position, budget, force)

    def _next_word(self, end):
        if end < len(self.pending):
            return self.pending[end]
        return self.mutable[0] if self.mutable else None

    @classmethod
    def _japanese_reading_cut(cls, words, limit, safe_cuts):
        candidates = []
        for end in range(1, limit + 1):
            word = words[end - 1]
            gap = float(words[end]["start"]) - float(word["end"]) if end < len(words) else 0.0
            if end in safe_cuts and (
                japanese_clause_end(str(word["text"])) or gap >= cls.CLAUSE_GAP
            ):
                candidates.append(end)
        return candidates[-1] if candidates else max(1, limit)

    def _boundary(self, force):
        limit = min(self.max_words, len(self.pending))
        japanese_cuts = None
        observed_japanese_cut = None
        if self.source_language == "ja":
            # Include the observed mutable suffix only to locate a word end.
            # It never grants ownership of unsealed characters.
            texts = [word["text"] for word in self.pending + self.mutable]
            ends = self.japanese_boundaries.ends(join_units(texts, "ja"))
            japanese_cuts = {
                index
                for index in range(1, len(self.pending) + 1)
                if len(join_units(texts[:index], "ja")) in ends
            }
            limit, used = 0, 0
            for word in self.pending:
                size = reading_units(word["text"], "ja")
                if limit and used + size > self.max_units:
                    break
                limit, used = limit + 1, used + size
            safe = [end for end in japanese_cuts if end <= limit]
            if safe and (limit < len(self.pending) or used >= self.max_units):
                limit = max(safe)
            # A long next compound may make the eventual cut fall BEFORE the
            # character cap. Use the already observed continuation to plan that
            # same cut now, once its prefix is sealed; waiting for the following
            # compound to seal can make this shorter prefix's deadline expire.
            observed = self.pending + self.mutable
            observed_limit, observed_used = 0, 0
            for word in observed:
                size = reading_units(word["text"], "ja")
                if observed_limit and (
                    observed_used + size > self.max_units
                    or float(word["end"]) - float(observed[0]["start"]) > self.max_hold
                ):
                    break
                observed_limit, observed_used = observed_limit + 1, observed_used + size
            if observed_limit < len(observed):
                all_cuts = [
                    index
                    for index in range(1, observed_limit + 1)
                    if len(join_units(texts[:index], "ja")) in ends
                ]
                target = max(all_cuts) if all_cuts else observed_limit
                target = self._japanese_reading_cut(observed, target, all_cuts)
                if 0 < target <= len(self.pending):
                    observed_japanese_cut = target
                    limit = min(limit, target)
        start = float(self.pending[0]["start"])
        offset = start
        for end in range(1, limit + 1):
            word = self.pending[end - 1]
            offset = max(offset, float(word["end"]))
            if end > 1 and offset - start > self.max_hold:
                safe = (
                    [cut for cut in japanese_cuts if cut < end] if japanese_cuts is not None else []
                )
                limit = max(safe) if safe else end - 1
                break
            following = self._next_word(end)
            gap = float(following["start"]) - offset if following else None
            sealed_right = self.sealed_until - offset
            observed_right = self.encoded_until - offset
            next_state = (
                ("sealed" if end < len(self.pending) else "mutable") if following else "none"
            )
            terminal = str(word["text"]).rstrip("\"'”’)]}")
            capacity_tail = word["occurrence_id"] == self.capacity_tail and not force
            # Ownership C<=S already holds. A sealed sentence needs received
            # right context A, not the next sentence's words to become sealed.
            sufficient = bool(force) or observed_right >= self.RIGHT_CONTEXT - 1e-6
            if self.source_language == "ja":
                terminal = terminal.rstrip("」』）】〉》］")
            weak = end <= 2 and not terminal.endswith(("?", "!", "？", "！"))
            weak_ready = (
                not weak
                or gap is not None
                and gap >= self.CLAUSE_GAP - 1e-6
                or gap is None
                and sealed_right >= self.SHORT_TAIL_CONTEXT - 1e-6
            )
            evidence = dict(
                end_occurrence_id=word["occurrence_id"],
                sealed_until=self.sealed_until,
                encoded_until=self.encoded_until,
                lexical_end=offset,
                observed_right_context=observed_right,
                sealed_right_context=sealed_right,
                next_word_gap=gap,
                next_word_state=next_state,
                gap_basis="next_lexical_observation" if following else "sealed_blank",
            )
            if (
                sufficient
                and not capacity_tail
                and self.sentence_end(terminal)
                and (force or weak_ready)
            ):
                return end, "sealed_sentence_boundary", False, evidence
            if (
                sufficient
                and (japanese_cuts is None or end in japanese_cuts)
                and (
                    gap is not None
                    and gap >= self.TURN_GAP - 1e-6
                    or gap is None
                    and sealed_right >= self.TURN_GAP - 1e-6
                )
            ):
                return end, "sealed_source_gap", False, evidence
        capacity_following = self._next_word(limit)
        capacity_end = max(float(word["end"]) for word in self.pending[:limit])
        at_capacity = (
            reading_units(join_units((word["text"] for word in self.pending[:limit]), "ja"), "ja")
            >= self.max_units
            if self.source_language == "ja"
            else limit == self.max_words
        )
        observed_continuation = observed_japanese_cut is not None or (
            len(self.pending) == limit
            and at_capacity
            and capacity_following is not None
            and float(capacity_following["start"]) >= capacity_end - 1e-6
        )
        overflow = len(self.pending) > limit or observed_continuation
        if overflow:
            # One reading budget governs both ordinary long sentences and
            # capacity continuations; physical windows never supply a cut.
            # Once full, observed following words prove continuation without
            # needing to own them. The cut still claims only sealed words.
            candidates = []
            for end in range(1, limit + 1):
                word = self.pending[end - 1]
                following = self._next_word(end)
                gap = float(following["start"]) - float(word["end"]) if following else 0.0
                if self.source_language == "ja":
                    if end in japanese_cuts and (
                        japanese_clause_end(str(word["text"])) or gap >= self.CLAUSE_GAP
                    ):
                        candidates.append(end)
                    continue
                if str(word["text"]).rstrip("\"'”’)]}").endswith(",") or gap >= self.CLAUSE_GAP:
                    candidates.append(end)
            end = (
                self._japanese_reading_cut(self.pending + self.mutable, limit, japanese_cuts)
                if japanese_cuts is not None
                else candidates[-1]
                if candidates
                else max(1, limit)
            )
            return (
                end,
                "sealed_reading_budget",
                True,
                dict(
                    max_words=self.max_words,
                    **(
                        {
                            "max_source_characters": self.max_units,
                            "text_boundary_basis": "sudachi_c_morphemes_with_attached_function_words",
                            "forced_inside_morpheme": end not in japanese_cuts,
                        }
                        if japanese_cuts is not None
                        else {}
                    ),
                    max_source_seconds=self.max_hold,
                    end_occurrence_id=self.pending[end - 1]["occurrence_id"],
                    capacity_trigger="observed_continuation_at_capacity"
                    if observed_continuation
                    else "sealed_overflow",
                    capacity_end=capacity_end,
                    next_word_state=("sealed" if limit < len(self.pending) else "mutable")
                    if capacity_following
                    else "none",
                    next_word_start=float(capacity_following["start"])
                    if capacity_following
                    else None,
                ),
            )
        if force:
            return len(self.pending), force, False, dict(closure_reason=force)
        return None

    def _proposal(self, words, reason, continuation, evidence, now, position, budget):
        self.next_sentence += 1
        sentence_id = f"epoch:{self.epoch}:sentence:{self.next_sentence}"
        ranges = []
        for word in words:
            if not ranges or ranges[-1]["source_id"] != word["source_id"]:
                ranges.append(
                    dict(
                        source_id=word["source_id"],
                        start=float(word["start"]),
                        end=float(word["end"]),
                        word_ids=[],
                    )
                )
            ranges[-1]["end"] = max(ranges[-1]["end"], float(word["end"]))
            ranges[-1]["word_ids"].append(word["occurrence_id"])
        proposal = dict(
            text=join_units((str(word["text"]) for word in words), self.source_language),
            source_language=self.source_language,
            sentence_id=sentence_id,
            word_ids=[word["occurrence_id"] for word in words],
            source_ownership="sealed_occurrence_ids",
            source_ranges=ranges,
            word_evidence=[
                dict(
                    occurrence_id=word["occurrence_id"],
                    confidence=word.get("confidence"),
                    sources=word.get("sources", []),
                    uncertain_intervals=word["uncertain_intervals"],
                )
                for word in words
            ],
            source_id=words[0]["source_id"],
            source_revision=max(word["source_revision"] for word in words),
            word_start=0,
            word_end=len(words),
            word_index_basis="segment_local_not_asr_cursor",
            source_start=float(words[0]["start"]),
            source_end=max(float(word["end"]) for word in words),
            context_start=float(words[0]["start"]),
            word_timestamps=[
                dict(start=float(word["start"]), end=float(word["end"])) for word in words
            ],
            word_observed_until=[word["sealed_audio_until"] for word in words],
            timing_kind="ctc_emission_spans",
            alignment_kind="ctc_emission_spans",
            segmentation_profile=self.PROFILE,
            received_until=self.received_until,
            evidence_until=self.encoded_until,
            observed_complete_until=self.encoded_until,
            commit_reason=reason,
            boundary_reason=reason,
            boundary_evidence=evidence,
            continuation=continuation,
            continuation_from=self.continuation_from,
            sealed_until_at_commit=self.sealed_until,
            committed_at=now,
            commit_playback_position=position,
            source_first_sealed_at=words[0]["sealed_at"],
            source_last_sealed_at=max(word["sealed_at"] for word in words),
        )
        timing = CueTiming.estimate(proposal, self.min_read, self.max_hold)
        proposal["committed_schedule"] = dict(
            due_at=timing["estimated_start"],
            **CueTiming.schedule(
                proposal,
                timing,
                timing["estimated_start"],
                self.min_read,
                self.max_hold,
                self.max_lag,
            ),
        )
        proposal["preparation_budget"] = (
            CueTiming.preparation(proposal["committed_schedule"], position, budget)
            if budget is not None
            else None
        )
        self.continuation_from = sentence_id if continuation else None
        self.notices.append(
            dict(
                reason="sentence_committed",
                **{
                    name: proposal[name]
                    for name in (
                        "sentence_id",
                        "word_ids",
                        "source_id",
                        "source_ranges",
                        "source_start",
                        "source_end",
                        "text",
                        "source_ownership",
                        "boundary_reason",
                        "boundary_evidence",
                        "continuation",
                        "continuation_from",
                        "sealed_until_at_commit",
                        "committed_at",
                        "commit_playback_position",
                        "committed_schedule",
                        "preparation_budget",
                    )
                },
            )
        )
        return proposal

    def _drain(self, now, position, budget, force=None):
        proposals = []
        while self.pending:
            boundary = self._boundary(force)
            if boundary is None:
                # Tick before the service budget is spent, rather than at the
                # NMT deadline. A pause cannot force a media-time continuation.
                if budget is None or budget["paused"]:
                    break
                start = float(self.pending[0]["start"])
                end = max(float(word["end"]) for word in self.pending)
                estimate = dict(estimated_end=end, alignment_kind="ctc_emission_spans")
                schedule = CueTiming.schedule(
                    dict(
                        segmentation_profile=self.PROFILE,
                        source_language=self.source_language,
                        text=join_units(
                            (word["text"] for word in self.pending), self.source_language
                        ),
                    ),
                    estimate,
                    start,
                    self.min_read,
                    self.max_hold,
                    self.max_lag,
                )
                preparation = CueTiming.preparation(schedule, position, budget)
                if position < preparation["latest_commit"]:
                    break
                boundary = (len(self.pending), "sealed_preparation_budget", True, preparation)
            end, reason, continuation, evidence = boundary
            words, self.pending = self.pending[:end], self.pending[end:]
            proposal = self._proposal(words, reason, continuation, evidence, now, position, budget)
            if any(character.isalnum() for character in proposal["text"]):
                proposals.append(proposal)
            else:
                self.notices.append(
                    dict(
                        reason="nonlexical_segment_retired",
                        sentence_id=proposal["sentence_id"],
                        word_ids=proposal["word_ids"],
                    )
                )
        return proposals

    def advance_audio(self, processed_until, now, position, budget=None):
        # Processing progress is not permission to invent a sealed frontier.
        return self._drain(now, position, budget)

    def tick(self, now, position, budget):
        """Advance preparation even when no new ASR result has arrived."""
        return self._drain(now, position, budget)

    def finalize(self, now, position, budget=None):
        if self.mutable:
            raise RuntimeError(
                "sealed_source_contract_violation: "
                f"source_id={self.active_id}, sealed_until={self.sealed_until}: "
                "input ended without closing the mutable transcript"
            )
        return self._drain(now, position, budget, "input_end")
