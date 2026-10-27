"""Bounded acoustic transcript state, independent of inference and subtitles.

A physical view is an observation of audio, not an append operation on text.
The newest view replaces the complete still-revisable path that it covers.
Agreement across advancing observations determines when words become immutable;
an explicit retention deadline bounds disagreement instead of inventing a seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class ObservationWord:
    text: str
    start: float
    end: float
    confidence: float
    token_ids: tuple[int, ...]
    token_start: int = -1
    token_end: int = -1
    token_pieces: tuple[str, ...] = ()
    starts_word: bool | None = None
    ends_word: bool | None = None

    def record(self) -> dict[str, Any]:
        return dict(
            text=self.text,
            start=self.start,
            end=self.end,
            confidence=self.confidence,
            token_ids=list(self.token_ids),
            token_start=self.token_start,
            token_end=self.token_end,
            token_pieces=list(self.token_pieces),
            starts_word=self.starts_word,
            ends_word=self.ends_word,
        )


@dataclass(frozen=True)
class WindowObservation:
    window_id: int
    revision: int
    audio_start: float
    audio_end: float
    words: tuple[ObservationWord, ...]
    tokens: tuple[dict[str, Any], ...] = ()

    def source(self) -> dict[str, Any]:
        return dict(
            window_id=self.window_id,
            revision=self.revision,
            audio_start=self.audio_start,
            audio_end=self.audio_end,
        )

    def record(self) -> dict[str, Any]:
        return dict(
            self.source(),
            words=[word.record() for word in self.words],
            tokens=[dict(token) for token in self.tokens],
        )


def _lexical_key(word: dict[str, Any]) -> tuple:
    # BPE segmentation, casing and attached punctuation may change without a
    # different spoken occurrence. Token IDs remain source-path evidence;
    # occurrence identity requires this lexical key AND acoustic correspondence.
    return tuple(char.casefold() for char in word["text"] if char.isalnum())


def _matches(old: list[dict], new: list[dict], tolerance: float = 0.16) -> list[tuple[int, int]]:
    """One-to-one matches of the same lexical emission on the original clock.

    Distinct repetitions remain distinct. The two-frame tolerance is deliberately
    narrower than a typical repeated spoken word; sequence similarity alone is
    never sufficient to consume an occurrence.
    """
    n, m = len(old), len(new)
    # A view is immutable throughout this comparison. Compute lexical identity
    # once per word rather than rebuilding it in every dynamic-programming cell.
    old_keys = [_lexical_key(word) for word in old]
    new_keys = [_lexical_key(word) for word in new]
    scores = [[0.0] * (m + 1) for _ in range(n + 1)]
    actions = [bytearray(m + 1) for _ in range(n + 1)]
    for i, a in enumerate(old, 1):
        for j, b in enumerate(new, 1):
            value, action = (
                (scores[i - 1][j], 1)
                if scores[i - 1][j] >= scores[i][j - 1]
                else (scores[i][j - 1], 2)
            )
            delta = abs((a["start"] + a["end"] - b["start"] - b["end"]) / 2)
            if old_keys[i - 1] == new_keys[j - 1] and delta <= tolerance + 1e-7:
                matched = scores[i - 1][j - 1] + 2.0 - delta / tolerance
                if matched >= value:
                    value, action = matched, 3
            scores[i][j], actions[i][j] = value, action
    pairs, i, j = [], n, m
    while i and j:
        action = actions[i][j]
        if action == 3:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif action == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


class AcousticTranscriptLedger:
    """One immutable prefix and one replaceable, time-bounded candidate path.

    Only the two newest physical views, and two advancing observations per view,
    are retained. No waveform, logits, alternative full-context transcript, or
    neural state is stored here. The owner closes logical contexts separately.
    """

    def __init__(
        self,
        context_id: int,
        start: float,
        *,
        revision_seconds: float = 4.0,
        retention_seconds: float = 6.0,
        confidence_floor: float = 0.35,
    ):
        if not 0 < revision_seconds <= retention_seconds:
            raise ValueError("Transcript revision budget must fit its retention budget")
        self.context_id = context_id
        self.start = self.sealed_until = self.encoded_until = float(start)
        self.revision_seconds = float(revision_seconds)
        self.retention_seconds = float(retention_seconds)
        self.confidence_floor = float(confidence_floor)
        self.revision = self._next_word_id = 0
        self._sealed: list[dict[str, Any]] = []
        self._mutable: list[dict[str, Any]] = []
        self._views: dict[int, list[WindowObservation]] = {}
        self._uncertain: list[dict[str, Any]] = []
        self._snapshot = self._make_snapshot(self.start)
        self.selection: dict[str, Any] = {}

    @staticmethod
    def _copy_word(word: dict) -> dict:
        return dict(
            word,
            token_ids=list(word["token_ids"]),
            token_pieces=list(word.get("token_pieces", ())),
            sources=[dict(source) for source in word["sources"]],
        )

    def _make_snapshot(self, revision_from: float) -> dict[str, Any]:
        return dict(
            revision=self.revision,
            sealed_until=self.sealed_until,
            revision_from=revision_from,
            sealed_words=[self._copy_word(word) for word in self._sealed],
            mutable_words=[self._copy_word(word) for word in self._mutable],
            uncertain_intervals=[dict(item) for item in self._uncertain],
        )

    def snapshot(self) -> dict[str, Any]:
        # The returned value is a detached publication, never internal ledger
        # state. Consumers may keep it while a later hypothesis is processed.
        snapshot = self._snapshot
        return dict(
            snapshot,
            sealed_words=[self._copy_word(word) for word in snapshot["sealed_words"]],
            mutable_words=[self._copy_word(word) for word in snapshot["mutable_words"]],
            uncertain_intervals=[dict(item) for item in snapshot["uncertain_intervals"]],
        )

    def _remember(self, observation: WindowObservation) -> bool:
        history = self._views.get(observation.window_id, [])
        if history and observation.audio_end <= history[-1].audio_end + 1e-7:
            return False  # A final repeat of the same audio is not another vote.
        self._views[observation.window_id] = (history + [observation])[-2:]
        while len(self._views) > 2:
            oldest = min(self._views)
            del self._views[oldest]
        return True

    def _evidence(self, words: list[dict]) -> None:
        supports: list[list[dict]] = [[] for _ in words]
        confidences: list[list[float]] = [[] for _ in words]
        for history in self._views.values():
            for observation in history:
                observed = [word.record() for word in observation.words]
                for i, j in _matches(words, observed):
                    # Seal the exact displayed spelling and punctuation only
                    # after they recur. Lexical-only identity is still used to
                    # retain occurrence IDs in the mutable path.
                    if words[i]["text"] == observed[j]["text"]:
                        supports[i].append(observation.source())
                        confidences[i].append(observed[j]["confidence"])
        for word, evidence, confidence in zip(words, supports, confidences):
            word["sources"] = evidence[-4:]
            word["support_count"] = len(evidence)
            word["reliable_support_count"] = sum(
                value >= self.confidence_floor for value in confidence
            )
            word["stable"] = (
                word["reliable_support_count"] >= 2
                and word["confidence"] >= self.confidence_floor
                and not word.get("boundary_uncertain", False)
            )

    def _source_path_blocks(
        self,
        observation: WindowObservation,
        raw: list[dict],
        witness: list[dict],
        claimed: set[str],
    ) -> tuple[set[int], list[dict], list[dict], list[dict]]:
        """Compare local unmatched blocks in two actual observation paths.

        The old path is never reconstructed from mixed ledger words. An empty
        old block is an insertion opportunity, not an owned prefix. Neighbours
        establish correspondence; only nonempty, already-owned old occurrences
        can reject a competing new block at the same acoustic location.
        """
        consumed = {j for i, j in _matches(witness, raw) if witness[i]["occurrence_id"] in claimed}
        previous_views = [view for view in self._views if view < observation.window_id]
        if not previous_views:
            return consumed, [], [], []
        previous = self._views[max(previous_views)][-1]
        old = [word.record() for word in previous.words]
        occurrences = {j: witness[i] for i, j in _matches(witness, old)}
        ownership = {j: word for j, word in occurrences.items() if word["occurrence_id"] in claimed}
        pairs = _matches(old, raw, tolerance=0.48)
        offsets = {
            (i, j): (raw[j]["start"] + raw[j]["end"] - old[i]["start"] - old[i]["end"]) / 2
            for i, j in pairs
        }
        trusted, correspondence = [], []
        for i, j in pairs:
            shift = offsets[i, j]
            neighbours = [
                (k, l)
                for k, l in pairs
                if k != i
                and _lexical_key(old[k]) != _lexical_key(old[i])
                and abs(old[k]["start"] - old[i]["start"]) <= 1.5
                and abs(offsets[k, l] - shift) <= 0.16 + 1e-7
            ]
            if abs(shift) <= 0.16 + 1e-7 or neighbours:
                trusted.append((i, j))
                if i in ownership:
                    consumed.add(j)
                    correspondence.append(
                        dict(
                            occurrence_id=ownership[i]["occurrence_id"],
                            observed_text=raw[j]["text"],
                            observed_start=raw[j]["start"],
                            offset_seconds=shift,
                            accepted=True,
                            old_source=previous.source(),
                            new_source=observation.source(),
                            supporting_neighbours=[
                                dict(
                                    old_index=k,
                                    raw_index=l,
                                    old_text=old[k]["text"],
                                    observed_text=raw[l]["text"],
                                    offset_seconds=offsets[k, l],
                                )
                                for k, l in neighbours
                            ],
                            reason="same_source_path_ordered_occurrence",
                        )
                    )
        decisions, retained_paths = [], []
        boundaries = [(-1, -1)] + trusted + [(len(old), len(raw))]
        for (left_old, left_new), (right_old, right_new) in zip(boundaries, boundaries[1:]):
            old_indices = list(range(left_old + 1, right_old))
            owned = [i for i in old_indices if i in ownership]
            incoming = list(range(left_new + 1, right_new))
            if not owned or not incoming:
                continue
            evidence = []
            for i in owned:
                # A following sentence is not an acoustic witness that an
                # unfamiliar word replaced the previous sentence's final word.
                # Punctuation here limits substitution evidence; it never forces
                # a subtitle boundary or delays a candidate's normal sealing.
                nearby = [
                    (k, l)
                    for k, l in trusted
                    if abs(old[k]["start"] - old[i]["start"]) <= 1.5
                    and not any(
                        word["text"].rstrip("\"'’”)]}").endswith((".", "?", "!"))
                        for word in old[min(i, k) : max(i, k)]
                    )
                ]
                if len({_lexical_key(old[k]) for k, _ in nearby}) < 2:
                    continue
                anchor = min(nearby, key=lambda pair: abs(old[pair[0]]["start"] - old[i]["start"]))
                shift = offsets[anchor]
                support = [
                    (k, l)
                    for k, l in nearby
                    if _lexical_key(old[k]) != _lexical_key(old[anchor[0]])
                    and abs(offsets[k, l] - shift) <= 0.16 + 1e-7
                ]
                if not support:
                    continue
                old_center = (old[i]["start"] + old[i]["end"]) / 2
                competing = [
                    j
                    for j in incoming
                    if abs((raw[j]["start"] + raw[j]["end"]) / 2 - old_center - shift)
                    <= 0.16 + 1e-7
                ]
                if competing:
                    evidence.append(
                        dict(
                            old_index=i,
                            old_occurrence_id=ownership[i]["occurrence_id"],
                            competing_raw_indices=competing,
                            anchors=[
                                dict(
                                    old_index=k,
                                    raw_index=l,
                                    old_text=old[k]["text"],
                                    new_text=raw[l]["text"],
                                    offset_seconds=offsets[k, l],
                                )
                                for k, l in [anchor] + support
                            ],
                        )
                    )
            if not evidence:
                continue
            # A correspondence has no authority over audio that the old view
            # never received, or over distant positive evidence in the same
            # syntactic anchor gap. Do not clip such a block and combine half
            # of each path: keep the incoming block and report the unresolved
            # source choice. The .48s extent is the existing correspondence
            # search bound, not an added observation/waiting budget.
            shared_start = max(previous.audio_start, observation.audio_start)
            shared_end = min(previous.audio_end, observation.audio_end)
            shifts = [anchor["offset_seconds"] for item in evidence for anchor in item["anchors"]]
            related_old = [item["old_index"] for item in evidence]
            local_start = max(
                shared_start, min(old[i]["start"] for i in related_old) + min(shifts) - 0.48
            )
            local_end = min(
                shared_end, max(old[i]["end"] for i in related_old) + max(shifts) + 0.48
            )
            outside = [
                j
                for j in incoming
                if raw[j]["start"] < local_start - 1e-7 or raw[j]["end"] > local_end + 1e-7
            ]
            if outside:
                decisions.append(
                    dict(
                        kind="preserve_new_cross_domain_path",
                        incoming_raw_indices=incoming,
                        outside_supported_domain=outside,
                        shared_start=shared_start,
                        shared_end=shared_end,
                        supported_start=local_start,
                        supported_end=local_end,
                        old_source=previous.source(),
                        new_source=observation.source(),
                        evidence=evidence,
                        reason="whole_candidate_not_safely_owned_by_old_local_evidence",
                    )
                )
                continue
            # The complete unmatched candidate block is the transaction. If a
            # one-word old path competes with a two-word new path, taking just
            # half of the new path would fabricate a third, unobserved phrase.
            consumed.update(incoming)
            retained_ids = []
            for i in old_indices:
                word = old[i]
                if word["start"] < self.sealed_until - 1e-7:
                    continue
                previous_word = occurrences.get(i)
                if previous_word is not None:
                    retained_paths.append(previous_word)
                    retained_ids.append(previous_word["occurrence_id"])
                else:
                    restored = dict(
                        word,
                        path_window_id=previous.window_id,
                        occurrence_id=f"ctx:{self.context_id}:word:{self._next_word_id}",
                        sources=[],
                    )
                    self._next_word_id += 1
                    retained_paths.append(restored)
                    retained_ids.append(restored["occurrence_id"])
            decisions.append(
                dict(
                    kind="select_complete_source_path_block",
                    retained_old_ids=retained_ids,
                    owned_old_ids=[ownership[i]["occurrence_id"] for i in owned],
                    rejected_raw_indices=incoming,
                    old_indices=old_indices,
                    old_words=[old[i] for i in old_indices],
                    new_words=[raw[j] for j in incoming],
                    old_source=previous.source(),
                    new_source=observation.source(),
                    shared_start=shared_start,
                    shared_end=shared_end,
                    supported_start=local_start,
                    supported_end=local_end,
                    left_anchor=[left_old, left_new],
                    right_anchor=[right_old, right_new],
                    evidence=evidence,
                    reason="atomic_nonempty_alternative_paths_between_ordered_anchors",
                )
            )
        return consumed, correspondence, decisions, retained_paths

    def _complete_word_alternative(
        self, observation: WindowObservation, raw: list[dict], witness: list[dict]
    ) -> tuple[list[dict], set[int], list[dict]]:
        """Use an observed whole path instead of its cut first-word subpath.

        Missing a start marker alone is not rejection evidence. A replacement
        requires a proper CTC subpath at the same original token times, inside
        a previous complete word with at least two reliable observations.
        """
        if not raw or raw[0].get("starts_word") is not False:
            return [], set(), []
        fragment = raw[0]
        new_ids = fragment["token_ids"]
        start, end = fragment.get("token_start", -1), fragment.get("token_end", -1)
        if not new_ids or start < 0 or end <= start or end > len(observation.tokens):
            return [], set(), []
        new_tokens = observation.tokens[start:end]
        for word in witness:
            if (
                word.get("starts_word") is not True
                or word.get("ends_word") is not True
                or word["path_window_id"] >= observation.window_id
                or word.get("reliable_support_count", 0) < 2
            ):
                continue
            old_ids = word["token_ids"]
            for offset in range(1, len(old_ids) - len(new_ids) + 1):
                if old_ids[offset : offset + len(new_ids)] != new_ids:
                    continue
                source_evidence = []
                for previous in self._views.get(word["path_window_id"], []):
                    for observed in previous.words:
                        if (
                            list(observed.token_ids) != old_ids
                            or observed.token_start < 0
                            or abs(
                                (observed.start + observed.end - word["start"] - word["end"]) / 2
                            )
                            > 0.16 + 1e-7
                        ):
                            continue
                        a = observed.token_start + offset
                        old_tokens = previous.tokens[a : a + len(new_tokens)]
                        if len(old_tokens) != len(new_tokens):
                            continue
                        deltas = [
                            abs((x["start"] + x["end"] - y["start"] - y["end"]) / 2)
                            for x, y in zip(old_tokens, new_tokens)
                        ]
                        if all(delta <= 0.16 + 1e-7 for delta in deltas):
                            source_evidence.append(
                                dict(source=previous.source(), token_time_deltas=deltas)
                            )
                if source_evidence:
                    return (
                        [word],
                        {0},
                        [
                            dict(
                                kind="retain_complete_word_path",
                                retained_old_ids=[word["occurrence_id"]],
                                rejected_raw_indices=[0],
                                old_token_ids=old_ids,
                                new_token_ids=new_ids,
                                proper_subpath_offset=offset,
                                old_start=word["start"],
                                old_end=word["end"],
                                source_evidence=source_evidence,
                                reason="first_word_internal_BPE_subpath_of_supported_complete_word",
                            )
                        ],
                    )
        return [], set(), []

    def _select_path(self, observation: WindowObservation) -> list[dict]:
        old = self._mutable
        raw = [word.record() for word in observation.words]
        coverage = max(self.sealed_until, observation.audio_start)
        # Emission times are not phoneme onset times. Near a view's cut left
        # edge, absence of a word is not valid negative evidence: the beginning
        # of its sound may have been cut away. Retain the previous whole-word
        # path there. Positive evidence in an uncovered gap is still usable.
        negative_coverage = max(self.sealed_until, observation.audio_start + 0.32)
        prefix = [
            word
            for word in old
            if word["start"] < coverage - 1e-7
            or (
                word["start"] < negative_coverage - 1e-7
                and word["path_window_id"] < observation.window_id
            )
        ]
        if prefix:
            coverage = max(coverage, prefix[-1]["end"])
        sealed_in_view = [
            word for word in self._sealed if word["end"] >= observation.audio_start - 0.48
        ]
        witness = sealed_in_view + old
        claimed = {word["occurrence_id"] for word in sealed_in_view + prefix}
        consumed, correspondence, block_decisions, source_alternatives = self._source_path_blocks(
            observation, raw, witness, claimed
        )
        complete_alternatives, fragments, lexical_decisions = self._complete_word_alternative(
            observation, raw, witness
        )
        sealed_ids = {word["occurrence_id"] for word in self._sealed}
        retained = {word["occurrence_id"]: word for word in prefix}
        for word in complete_alternatives + source_alternatives:
            if word["occurrence_id"] not in sealed_ids:
                retained[word["occurrence_id"]] = word
        rejected = []
        candidate = []
        for j, word in enumerate(raw):
            if j in fragments:
                rejected.append(
                    dict(word, raw_index=j, reason="internal_subpath_replaced_by_complete_word")
                )
                continue
            if j in consumed or word["end"] <= self.sealed_until + 1e-7:
                continue
            if word["start"] < self.sealed_until - 1e-7:
                rejected.append(dict(word, reason="crosses_immutable_or_unobserved_word_boundary"))
                continue
            candidate.append(word)
        # A positive complete candidate can replace an unsealed old boundary
        # word. Never sort overlapping alternatives into a fabricated sentence.
        # Conversely, a proved complete-word substitution above owns its entire
        # acoustic word and excludes a competing fragment at the same location.
        exclusive = []
        complete_ids = {word["occurrence_id"] for word in complete_alternatives}
        for word in candidate:
            overlaps = [
                old_word
                for old_word in retained.values()
                if word["start"] < old_word["end"] - 1e-7 and word["end"] > old_word["start"] + 1e-7
            ]
            if not overlaps:
                exclusive.append(word)
                continue
            old_complete = any(old_word["occurrence_id"] in complete_ids for old_word in overlaps)
            if word.get("starts_word") is True and not old_complete:
                replaced = [old_word["occurrence_id"] for old_word in overlaps]
                for identity in replaced:
                    del retained[identity]
                lexical_decisions.append(
                    dict(
                        kind="replace_mutable_complete_path",
                        retired_old_ids=replaced,
                        new_text=word["text"],
                        new_token_ids=word["token_ids"],
                        start=word["start"],
                        end=word["end"],
                        reason="mutually_exclusive_positive_word_paths",
                    )
                )
                exclusive.append(word)
            else:
                rejected.append(
                    dict(
                        word,
                        reason="overlapping_alternative_to_retained_complete_path",
                        retained_old_ids=[old_word["occurrence_id"] for old_word in overlaps],
                    )
                )
        candidate = exclusive
        # Selection is an entire path over the available audio, including an
        # empty path. Thus additions, substitutions, retractions and repeats
        # use exactly the same operation. Matching cannot freeze an old gap.
        revisable = [word for word in old if word["occurrence_id"] not in retained]
        replacements = {j: revisable[i] for i, j in _matches(revisable, candidate)}
        for j, word in enumerate(candidate):
            word["path_window_id"] = observation.window_id
            previous = replacements.get(j)
            if previous is not None:
                word["occurrence_id"] = previous["occurrence_id"]
            else:
                word["occurrence_id"] = f"ctx:{self.context_id}:word:{self._next_word_id}"
                self._next_word_id += 1
        selected = sorted(
            [self._copy_word(word) for word in retained.values()] + candidate,
            key=lambda word: (word["start"], word["end"]),
        )
        self._evidence(selected)
        self.selection = dict(
            kind="replace_revisable_acoustic_path",
            source=observation.source(),
            replacement_start=coverage,
            replacement_end=observation.audio_end,
            previous_occurrences=[word["occurrence_id"] for word in old if word not in prefix],
            selected_occurrences=[word["occurrence_id"] for word in candidate],
            matched_occurrences=len(replacements),
            boundary_rejections=rejected,
            boundary_correspondences=correspondence,
            candidate_block_decisions=block_decisions,
            lexical_path_decisions=lexical_decisions,
            retained_old_ids=list(retained),
            negative_evidence_starts=negative_coverage,
            reason="newest_received_context_is_candidate; advancing_agreement_controls_publication",
            retained_uncovered_words=len(prefix),
        )
        return selected

    def _mark_uncertain(self, start: float, end: float, reason: str) -> None:
        if (
            self._uncertain
            and self._uncertain[-1]["reason"] == reason
            and start <= self._uncertain[-1]["end"] + 1e-7
        ):
            self._uncertain[-1]["end"] = max(end, self._uncertain[-1]["end"])
        else:
            self._uncertain.append(dict(start=start, end=end, reason=reason))

    def _seal(self, final_reason: str | None) -> None:
        mature = self.encoded_until - self.revision_seconds
        deadline = self.encoded_until - self.retention_seconds
        count = 0
        while count < len(self._mutable):
            end = count + 1
            group_end = self._mutable[count]["end"]
            while end < len(self._mutable) and self._mutable[end]["start"] < group_end - 1e-7:
                group_end = max(group_end, self._mutable[end]["end"])
                end += 1
            group = self._mutable[count:end]
            confident = all(word["stable"] for word in group) and group_end <= mature + 1e-7
            expired = group[0]["start"] < deadline - 1e-7
            if not (confident or expired or final_reason):
                break
            if not confident:
                self._mark_uncertain(
                    group[0]["start"],
                    group_end,
                    "context_closed_before_confirmation" if final_reason else "retention_deadline",
                )
            self._sealed.extend(self._copy_word(word) for word in group)
            self.sealed_until = max(self.sealed_until, group_end)
            count = end
        self._mutable = self._mutable[count:]
        if final_reason:
            self.sealed_until = self.encoded_until
        else:
            # Quiet audio may advance the time frontier, but never through an
            # unresolved word. Advancing observations, not activity events,
            # confirm this blank part of the selected path.
            last_history = self._views[max(self._views)] if self._views else []
            quiet_limit = mature if len(last_history) >= 2 else deadline
            if self._mutable:
                quiet_limit = min(quiet_limit, self._mutable[0]["start"])
            self.sealed_until = max(self.sealed_until, quiet_limit, self.start)
        self.selection.update(
            mature_until=max(self.start, mature),
            retention_deadline=max(self.start, deadline),
            sealed_until=self.sealed_until,
            closure_reason=final_reason,
        )

    def observe(
        self, observation: WindowObservation, *, final_reason: str | None = None
    ) -> dict[str, Any]:
        if final_reason not in (None, "input_end", "long_silence", "capacity_limit"):
            raise ValueError(f"Unknown transcript closure: {final_reason}")
        if observation.audio_end < self.encoded_until - 1e-7:
            raise ValueError("Transcript observations cannot move backwards; reset the epoch first")
        if (
            not isfinite(observation.audio_start)
            or not isfinite(observation.audio_end)
            or observation.audio_start < self.start - 1e-7
            or observation.audio_end < observation.audio_start
        ):
            raise ValueError("Observation coverage must be finite and inside this context")
        if self._views and observation.window_id < max(self._views):
            raise ValueError("Physical observations must arrive in source order")
        previous_end = observation.audio_start
        for word in observation.words:
            if (
                not all(isfinite(value) for value in (word.start, word.end, word.confidence))
                or not 0 <= word.confidence <= 1
                or word.start < previous_end - 1e-7
                or word.end < word.start
                or word.end > observation.audio_end + 1e-7
            ):
                raise ValueError("Word evidence must be ordered and inside its original audio view")
            previous_end = word.start
        revision_from = self.sealed_until
        advanced = self._remember(observation)
        self.encoded_until = observation.audio_end
        if advanced:
            self._mutable = self._select_path(observation)
        if advanced or final_reason:
            self._seal(final_reason)
            self.revision += 1
            self._snapshot = self._make_snapshot(revision_from)
        return self.snapshot()
