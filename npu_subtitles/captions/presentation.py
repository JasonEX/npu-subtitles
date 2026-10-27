"""Own caption admission, display lifetime, and retirement."""

from __future__ import annotations
from collections import OrderedDict, deque
from difflib import SequenceMatcher
import math
from npu_subtitles.captions.timing import CueTiming
from npu_subtitles.captions.sentences import GraniteSentenceAssembler
from npu_subtitles.captions.commit import CueCommitter, finite, token


class PresentationController:
    """Current and successor, with at most eight live/prepared cues in total."""

    CAPACITY = 8

    def __init__(self, lookahead_seconds, max_lag, max_hold, min_read):
        self.lookahead_seconds, self.max_lag = lookahead_seconds, max_lag
        self.max_hold, self.min_read = max_hold, min_read
        self.reset(0)

    def reset(self, epoch):
        self.epoch, self.next_id = epoch, 0
        self.cues, self.current_key, self.notices = OrderedDict(), None, []
        self.durations = deque(maxlen=24)
        self.last_now, self.speed, self.paused_since = 0.0, 1.0, None
        self.paused_read_remaining = None
        self.last_due = -math.inf
        self.translation_reserve_seconds = 0.0

    @staticmethod
    def ownership(cue):
        if cue.get("segmentation_profile") == GraniteSentenceAssembler.PROFILE:
            return dict(sentence_id=cue["sentence_id"], word_ids=cue["word_ids"])
        return {}

    @staticmethod
    def presentation_eligible(cue, *, reserve=False):
        """One v3 readiness rule for both display and reading reservations."""
        if cue.get("segmentation_profile") == GraniteSentenceAssembler.PROFILE:
            return (
                cue["status"] == "english_only"
                or cue["status"] == "translated"
                and bool(cue["chinese"])
            )
        # Keep Moonshine's established early-English display and reservations.
        return cue["status"] == "translated" if reserve else True

    def english_fallback(self, cue, reason):
        cue["status"] = "english_only"
        self.notices.append(dict(reason=reason, key=cue["key"]))
        if self.ownership(cue):
            self.notices.append(
                dict(
                    reason=(
                        "source_fallback_entered"
                        if cue.get("source_language") == "ja"
                        else "english_fallback_entered"
                    ),
                    fallback_reason=reason,
                    key=cue["key"],
                    **self.ownership(cue),
                    admission_deadline=cue["admission_deadline"],
                    expires_at=cue["expires_at"],
                )
            )

    def drop(self, key, reason, **details):
        cue = self.cues.pop(key, None)
        if cue:
            self.notices.append(
                dict(
                    reason=reason,
                    key=key,
                    source_id=cue["source_id"],
                    text=cue["text"],
                    **self.ownership(cue),
                    **details,
                )
            )
        if self.current_key == key:
            self.current_key = None

    def source_revised(self, revision, now, position):
        """Replace only live text versions, preserving all media deadlines."""
        self.prune(position)
        old, new = revision["old"], revision["new"]
        operations = SequenceMatcher(
            None, list(map(token, old)), list(map(token, new)), autojunk=False
        ).get_opcodes()
        for key, cue in list(self.cues.items()):
            if cue.get("segmentation_profile") == GraniteSentenceAssembler.PROFILE:
                # The sentence assembler already assigned immutable sealed
                # occurrences. Later ASR strings have no authority here.
                continue
            if cue["source_id"] != revision["source_id"]:
                continue
            # Once a bilingual card is on screen its display version is fixed.
            # Re-decoding a longer ASR context must not clear Chinese, flicker
            # between synonyms, or restart the reader's hold on the same card.
            # Unshown preparation remains revisable; seek/expiry still retires
            # the frozen card normally. No later hypothesis can resurrect it.
            if cue["shown_at"] is not None and cue["chinese"]:
                self.notices.append(
                    dict(
                        reason="shown_revision_frozen",
                        key=key,
                        source_id=cue["source_id"],
                        text=cue["text"],
                        source_revision=revision["source_revision"],
                    )
                )
                continue
            # A prepared pair may use its one presentation grace after NMT
            # admission has closed. Replacing English now would destroy a
            # usable pair with no remaining opportunity to translate it again.
            if cue["chinese"] and position >= cue["admission_deadline"]:
                self.notices.append(
                    dict(
                        reason="prepared_revision_frozen",
                        key=key,
                        source_id=cue["source_id"],
                        text=cue["text"],
                        source_revision=revision["source_revision"],
                    )
                )
                continue
            if not new:
                self.drop(key, "source_revision_withdrawn")
                continue
            start = CueCommitter.map_span_boundary(old, new, cue["word_start"])
            end = CueCommitter.map_span_boundary(old, new, cue["word_end"])
            source_text = cue.get("source_text", cue["text"])
            prefix = cue.get("prefix_text", "")
            updated = list(map(token, new[start:end]))
            if updated == list(map(token, source_text.split())):
                cue["word_start"], cue["word_end"] = start, end
                if cue.get("source_parts"):
                    cue["source_parts"][-1].update(
                        word_start=start, word_end=end, source_revision=revision["source_revision"]
                    )
                continue
            # A rewrite that crosses a cue boundary gives no reliable way to
            # assign its new words to this cue versus its neighbor.
            crossing = any(
                tag in {"replace", "delete"}
                and (a < cue["word_start"] < b or a < cue["word_end"] < b)
                for tag, a, b, c, d in operations
            )
            old_span_matches = list(map(token, old[cue["word_start"] : cue["word_end"]])) == list(
                map(token, source_text.split())
            )
            if (
                crossing
                or not old_span_matches
                or not 0 <= start < end <= len(new)
                or end - start + len(prefix.split()) > getattr(self, "max_words", 18)
            ):
                self.drop(key, "source_revision_span_uncertain")
                continue
            revised_key = (self.epoch, cue["id"], cue["revision"] + 1)
            revised_source = " ".join(new[start:end])
            replacement = dict(
                cue,
                key=revised_key,
                revision=cue["revision"] + 1,
                source_revision=revision["source_revision"],
                text=f"{prefix} {revised_source}".strip(),
                word_start=start,
                word_end=end,
                chinese="",
                status="ready",
                submitted_at=None,
                latest_start=cue["admission_deadline"],
                presentation_grace=False,
            )
            if prefix:
                replacement["source_text"] = revised_source
                replacement["source_parts"] = [dict(part) for part in cue["source_parts"]]
                replacement["source_parts"][-1].update(
                    text=revised_source,
                    word_start=start,
                    word_end=end,
                    source_revision=revision["source_revision"],
                )
                replacement["source_parts"][-1].update(word_timestamps=[], timing_kind=None)
            if self.current_key == key:
                self.current_key = revised_key
                replacement["read_until_wall"] = now + self.min_read
                if self.paused_since is not None:
                    self.paused_read_remaining = self.min_read
            # Replace the key in place, retaining source-order scheduling.
            self.cues = OrderedDict(
                (revised_key, replacement) if existing == key else (existing, value)
                for existing, value in self.cues.items()
            )
            self.notices.append(
                dict(
                    reason="source_revision_replaced",
                    old_key=key,
                    key=revised_key,
                    source_id=cue["source_id"],
                    old_text=cue["text"],
                    text=replacement["text"],
                    due_at=cue["due_at"],
                    admission_deadline=cue["admission_deadline"],
                    expires_at=cue["expires_at"],
                )
            )

    def prune(self, position):
        for key, cue in list(self.cues.items()):
            if (
                self.ownership(cue)
                and cue["status"] in {"ready", "inflight"}
                and position >= cue["admission_deadline"]
            ):
                self.english_fallback(cue, "translation_admission_expired")
            limit = (
                cue["expires_at"]
                if key == self.current_key
                else cue["latest_start"]
                if cue["status"] == "translated"
                else cue["admission_deadline"]
            )
            if position >= limit:
                self.drop(key, "presentation_expired")

    def presentation_start_limit(self, cue):
        """One bounded start grace for a translation completed on time."""
        read_media = self.min_read * self.speed
        return min(
            cue["expires_at"] - read_media,
            cue["admission_deadline"] + self.min_read,
            max(cue["admission_deadline"], cue["estimated_end"] + 0.5 * self.min_read),
        )

    def earliest_presentation_start(self, key, now, position):
        """Reserve reading time already owed before this source-ordered cue."""
        if key == self.current_key:
            return position
        cursor = position
        current = self.cues.get(self.current_key)
        if current:
            remaining = (
                self.paused_read_remaining
                if self.paused_since is not None
                else max(0.0, (current["read_until_wall"] or now) - now)
            )
            cursor += (remaining or 0.0) * self.speed
        for previous_key, previous in self.cues.items():
            if previous_key == key:
                return max(cursor, previous["due_at"])
            if previous_key != self.current_key and self.presentation_eligible(
                previous, reserve=True
            ):
                cursor = max(cursor, previous["due_at"]) + self.min_read * self.speed
        return math.inf

    def admit(self, proposals, now, position):
        self.prune(position)
        for proposal in proposals:
            proposal = dict(proposal)
            reopened = proposal.pop("reopened_caption", None)
            if proposal["received_until"] > position + self.lookahead_seconds + 1e-4:
                self.notices.append(
                    dict(
                        reason="outside_audio_lookahead",
                        source_id=proposal["source_id"],
                        **self.ownership(proposal),
                    )
                )
                continue
            timing = CueTiming.estimate(proposal, self.min_read, self.max_hold)
            # Confirmation may arrive several sentences at once. Each cue has
            # its own estimated media interval; reading holds never shift the
            # due time of an entire context's remaining dialogue into the future.
            if proposal.get("segmentation_profile") == GraniteSentenceAssembler.PROFILE:
                schedule = dict(proposal["committed_schedule"])
                due, expiry, admission = (
                    schedule.pop(name) for name in ("due_at", "expires_at", "admission_deadline")
                )
            elif reopened:
                due, expiry, admission = (
                    reopened[name] for name in ("due_at", "expires_at", "admission_deadline")
                )
                # A corrected onset can move later; neither translation nor
                # absolute retention gains time from a revision.
                due = max(due, timing["estimated_start"])
                schedule = dict(schedule_basis="reopened_original_deadlines")
            else:
                due = max(self.last_due, timing["estimated_start"])
                schedule = CueTiming.schedule(
                    proposal, timing, due, self.min_read, self.max_hold, self.max_lag
                )
                expiry = schedule.pop("expires_at")
                admission = schedule.pop("admission_deadline")
            self.last_due = max(self.last_due, due)
            if due >= admission or position >= admission:
                self.notices.append(
                    dict(
                        reason="committed_cue_too_late",
                        source_id=proposal["source_id"],
                        text=proposal["text"],
                        due_at=due,
                        admission_deadline=admission,
                        position=position,
                        **self.ownership(proposal),
                        **timing,
                    )
                )
                continue
            if len(self.cues) >= self.CAPACITY:
                self.notices.append(
                    dict(
                        reason="preparation_capacity",
                        source_id=proposal["source_id"],
                        text=proposal["text"],
                        **self.ownership(proposal),
                    )
                )
                continue
            if reopened:
                identity, revision = reopened["id"], reopened["revision"] + 1
            else:
                self.next_id += 1
                identity, revision = self.next_id, 1
            key = (self.epoch, identity, revision)
            self.cues[key] = dict(
                proposal | timing | schedule,
                key=key,
                id=identity,
                revision=revision,
                due_at=due,
                admission_deadline=admission,
                expires_at=expiry,
                chinese="",
                latest_start=admission,
                presentation_grace=False,
                status="ready",
                submitted_at=None,
                shown_at=None,
                read_until_wall=None,
            )
            if proposal.get("segmentation_profile") == GraniteSentenceAssembler.PROFILE:
                self.cues[key]["admitted_at"] = now
                self.notices.append(
                    dict(
                        reason="sentence_admitted",
                        key=key,
                        **self.ownership(proposal),
                        text=proposal["text"],
                        admitted_at=now,
                        due_at=due,
                        admission_deadline=admission,
                        expires_at=expiry,
                    )
                )
            if reopened:
                self.notices.append(
                    dict(
                        reason="source_tail_recommitted",
                        key=key,
                        source_id=proposal["source_id"],
                        text=proposal["text"],
                        old_due_at=reopened["due_at"],
                        due_at=due,
                        observation_floor=reopened["observation_floor"],
                        admission_deadline=admission,
                        expires_at=expiry,
                    )
                )

    def advance(self, now, position):
        self.last_now = now
        self.prune(position)
        if self.paused_since is not None:
            return
        current = self.cues.get(self.current_key)
        next_cue = next(
            (
                cue
                for key, cue in self.cues.items()
                if key != self.current_key
                and cue["due_at"] <= position
                and self.presentation_eligible(cue)
            ),
            None,
        )
        if current:
            read = now >= (current["read_until_wall"] or math.inf)
            no_chinese_left = not current["chinese"] and position >= current["admission_deadline"]
            complete = bool(current["chinese"]) or current["status"] == "english_only"
            if read and ((next_cue is not None and complete) or no_chinese_left):
                self.drop(current["key"], "presentation_retired")
                current = None
        if current is None and next_cue:
            self.current_key = next_cue["key"]
            next_cue["shown_at"], next_cue["read_until_wall"] = now, now + self.min_read
            self.notices.append(
                dict(
                    reason="presentation_started",
                    key=next_cue["key"],
                    source_id=next_cue["source_id"],
                    **self.ownership(next_cue),
                )
            )

    def translation_candidate(self, now, position, speed=1):
        self.speed = max(0.01, finite(speed, "playback speed"))
        self.advance(now, position)
        if any(cue["status"] == "inflight" for cue in self.cues.values()):
            return None
        current = self.cues.get(self.current_key)
        ordered = ([current] if current else []) + [
            cue for key, cue in self.cues.items() if key != self.current_key
        ]
        duration = max(self.durations, default=0.35)
        advance = 0.0 if self.paused_since is not None else self.speed
        for cue in ordered:
            if cue["status"] != "ready":
                continue
            service = (
                max(duration, self.translation_reserve_seconds) if self.ownership(cue) else duration
            )
            if (
                position + service * advance >= cue["admission_deadline"]
                or position + (service + self.min_read) * advance >= cue["expires_at"]
            ):
                self.english_fallback(cue, "translation_admission_missed")
                continue
            return dict(cue, deadline=cue["admission_deadline"])
        return None

    def mark_submitted(self, key):
        cue = self.cues.get(tuple(key))
        if cue is None or cue["status"] != "ready":
            raise ValueError("Translation submission no longer matches a live ready cue")
        cue.update(status="inflight", submitted_at=self.last_now)
        if self.ownership(cue):
            self.notices.append(
                dict(
                    reason="sentence_translation_submitted",
                    key=cue["key"],
                    **self.ownership(cue),
                    submitted_at=self.last_now,
                )
            )

    def translated(self, message, epoch, now, position):
        self.prune(position)
        key = (epoch, int(message["id"]), int(message["revision"]))
        cue = self.cues.get(key)
        if (
            epoch != self.epoch
            or int(message.get("generation", epoch)) != epoch
            or cue is None
            or cue["text"] != message["source"]
            or cue["status"] != "inflight"
        ):
            self.notices.append(dict(reason="translation_version_discarded", key=key))
            return False
        self.durations.append(max(0.0, now - cue["submitted_at"]))
        advance = 0.0 if self.paused_since is not None else self.speed
        if (
            position >= cue["admission_deadline"]
            or position + self.min_read * advance >= cue["expires_at"]
        ):
            self.english_fallback(cue, "translation_arrived_too_late")
            return False
        chinese = str(message["text"]).strip()
        if not chinese:
            raise ValueError("A successful translation cannot be empty")
        latest_start = self.presentation_start_limit(cue)
        earliest_start = self.earliest_presentation_start(key, now, position)
        if key != self.current_key and earliest_start >= latest_start:
            self.drop(
                key,
                "presentation_infeasible",
                earliest_start=earliest_start,
                latest_start=latest_start,
                admission_deadline=cue["admission_deadline"],
            )
            return False
        cue.update(
            chinese=chinese,
            status="translated",
            latest_start=latest_start,
            presentation_grace=latest_start > cue["admission_deadline"],
        )
        if self.ownership(cue):
            cue["translation_completed_at"] = now
            self.notices.append(
                dict(
                    reason="bilingual_prepared",
                    key=key,
                    **self.ownership(cue),
                    text=cue["text"],
                    chinese=chinese,
                    prepared_at=now,
                    due_at=cue["due_at"],
                    admission_deadline=cue["admission_deadline"],
                    expires_at=cue["expires_at"],
                )
            )
        if key == self.current_key:
            cue["read_until_wall"] = now + self.min_read
            if self.paused_since is not None:
                self.paused_read_remaining = self.min_read
        return True

    def translation_failed(self, message, epoch, now, position):
        self.prune(position)
        key = (epoch, int(message["id"]), int(message["revision"]))
        cue = self.cues.get(key)
        if (
            epoch != self.epoch
            or int(message.get("generation", epoch)) != epoch
            or cue is None
            or cue["status"] != "inflight"
            or message.get("source", cue["text"]) != cue["text"]
        ):
            self.notices.append(dict(reason="translation_failure_version_discarded", key=key))
            return False
        self.english_fallback(cue, "translation_failed")
        return True

    def frame(self, now, position, paused=False):
        if paused and self.paused_since is None:
            self.paused_since = now
            current = self.cues.get(self.current_key)
            self.paused_read_remaining = (
                max(0.0, (current["read_until_wall"] or now) - now) if current else None
            )
        elif not paused and self.paused_since is not None:
            current = self.cues.get(self.current_key)
            if current and current["read_until_wall"] is not None:
                current["read_until_wall"] = now + (self.paused_read_remaining or 0.0)
            self.paused_since = None
            self.paused_read_remaining = None
        self.advance(now, position)
        cue = self.cues.get(self.current_key)
        return (
            dict(
                cue,
                start=cue["source_start"],
                end=cue["source_end"],
                deadline=cue["expires_at"],
                is_final=True,
                translation_ready=cue["status"] == "ready",
            )
            if cue
            else None
        )
