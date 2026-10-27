"""Connect recognition commitments, translation budgets, and presentation."""

from __future__ import annotations
from collections import deque
from difflib import SequenceMatcher
import math
from npu_subtitles.captions.commit import CueCommitter, finite, token
from npu_subtitles.captions.presentation import PresentationController
from npu_subtitles.captions.sentences import GraniteSentenceAssembler
from npu_subtitles.captions.language import check_language


class CaptionPipeline:
    def __init__(
        self,
        lookahead_seconds=8,
        max_lag=3,
        max_hold=6,
        min_read=1.2,
        max_words=18,
        translation_budget=1.5,
        source_language="en",
    ):
        self.source_language = check_language(source_language)
        if (
            any(
                not math.isfinite(v) or v < 0
                for v in (lookahead_seconds, max_lag, max_hold, min_read)
            )
            or min_read <= 0
            or max_lag <= 0
        ):
            raise ValueError("Invalid caption timing parameters")
        if max_hold < max_lag or max_hold <= min_read or max_words < 2:
            raise ValueError("Require max_hold >= max_lag, max_hold > min_read, max_words >= 2")
        self.configured_translation_budget = finite(translation_budget, "translation budget")
        if self.configured_translation_budget <= 0:
            raise ValueError("Translation budget must be positive")
        self.worker_costs = {kind: deque(maxlen=24) for kind in ("audio", "translate")}
        self.committer = CueCommitter(int(max_words))
        self.sentences = GraniteSentenceAssembler(
            CueCommitter.sentence_end,
            int(max_words),
            max_hold,
            max_lag,
            min_read,
            source_language=source_language,
        )
        self.presentation = PresentationController(lookahead_seconds, max_lag, max_hold, min_read)
        self.presentation.max_words = self.committer.hard_words
        self.reset(0)

    def reset(self, epoch):
        self.epoch = int(epoch)
        self.committer.reset()
        self.sentences.reset(self.epoch)
        self.presentation.reset(self.epoch)
        self.source_mode = None
        self.preparation_budget = None
        self.notices, self.finalized = [], False
        self.last_now = self.last_position = 0.0

    def collect_notices(self):
        for component in (self.committer, self.sentences, self.presentation):
            self.notices.extend(component.notices)
            component.notices.clear()

    def update_preparation_budget(
        self,
        now,
        position,
        *,
        paused=False,
        speed=1.0,
        worker=None,
        completed_task=None,
        poll_seconds=0.025,
        asr_default_seconds=0.8,
    ):
        """Budget one serial NPU worker before the next translation can finish.

        The runner reports its actual worker/pending-ASR state. This method is
        the single owner of cost estimates and dispatch reserves; it does not
        choose or submit tasks. All service estimates are wall-clock seconds.
        """
        if self.source_mode == "hypothesis_revisions":
            return
        worker = worker or {}
        if completed_task and completed_task["kind"] in self.worker_costs:
            self.worker_costs[completed_task["kind"]].append(
                max(0.0, finite(completed_task["seconds"], "task cost"))
            )
        poll = max(0.001, finite(poll_seconds, "frame polling interval"))
        asr = max(
            self.worker_costs["audio"], default=finite(asr_default_seconds, "initial ASR reserve")
        )
        translation = max(
            [
                self.configured_translation_budget,
                *self.worker_costs["translate"],
                *self.presentation.durations,
            ]
        )
        kind = worker.get("kind")
        estimate = asr if kind == "audio" else translation
        elapsed = max(0.0, finite(worker.get("elapsed", 0.0), "worker elapsed time"))
        remaining = max(poll, estimate - elapsed) if kind in self.worker_costs else 0.0
        must_run_asr = (
            bool(worker.get("pending_asr"))
            and not paused
            and (
                bool(worker.get("asr_urgent"))
                or worker.get("last_task") == "translate"
                or kind == "translate"
            )
        )
        queued = sum(
            cue["status"] == "ready" and position < cue["admission_deadline"]
            for cue in self.presentation.cues.values()
        )
        dispatch = remaining + (asr if must_run_asr else 0.0) + queued * translation + 2 * poll
        self.preparation_budget = dict(
            updated_at=now,
            observed_playback_position=position,
            paused=bool(paused),
            playback_speed=max(0.01, finite(speed, "playback speed")),
            translation_reserve_seconds=translation,
            dispatch_reserve_seconds=dispatch,
            worker_kind=kind,
            worker_elapsed_seconds=elapsed,
            worker_remaining_seconds=remaining,
            worker_cost_estimate_seconds=estimate if kind in self.worker_costs else 0.0,
            worker_estimate_exceeded=kind in self.worker_costs and elapsed > estimate,
            observed_asr_seconds=asr,
            required_asr_seconds=asr if must_run_asr else 0.0,
            queued_translation_count=queued,
            queued_translation_seconds=queued * translation,
            frame_poll_reserve_seconds=2 * poll,
            reserve_basis="configured_or_observed_nmt_plus_serial_worker_dispatch",
        )
        self.presentation.translation_reserve_seconds = translation
        self.presentation.speed = self.preparation_budget["playback_speed"]

    def reopen_unshown_tail(self, event, position):
        """Undo only a live, unshown tail boundary that this revision removed.

        The ASR cursor is rewound before normal splitting. A bounded context
        carries the old identity and absolute deadlines until a replacement
        is actually complete, so an unstable rewrite cannot create a fresh
        two-word cue or regain time by being recommitted.
        """
        if event.get("event_kind", "hypothesis") != "hypothesis" or not event.get("text"):
            return
        ctx = self.committer.contexts.get(int(event["id"]))
        if ctx is None or ctx.get("reopened_caption") or ctx.get("withdrawn"):
            return
        encoded = float(event.get("encoded_until", event["end"]))
        if int(event["revision"]) < ctx["revision"] or not (
            event.get("is_final")
            or event.get("hypothesis_advanced", True)
            and encoded > ctx["hypothesis_until"] + 1e-6
        ):
            return
        old, new = ctx["words"], str(event["text"]).split()
        if not old or not self.committer.sentence_end(old[-1]):
            return
        self.presentation.prune(position)
        candidates = [
            cue
            for cue in self.presentation.cues.values()
            if cue["source_id"] == ctx["id"] and cue["word_end"] == len(old)
        ]
        if len(candidates) != 1:
            return
        cue = candidates[0]
        if (
            cue["shown_at"] is not None
            or cue.get("prefix_text")
            or position >= cue["admission_deadline"]
        ):
            return
        start, end = cue["word_start"], cue["word_end"]
        source = cue.get("source_text", cue["text"])
        if list(map(token, old[start:end])) != list(map(token, source.split())):
            return
        mapped_end = self.committer.map_consumed(old, new, end)
        if not 0 < mapped_end <= len(new) or self.committer.sentence_end(new[mapped_end - 1]):
            return
        operations = SequenceMatcher(
            None, list(map(token, old)), list(map(token, new)), autojunk=False
        ).get_opcodes()
        if any(tag in {"replace", "delete"} and a < start < b for tag, a, b, c, d in operations):
            return
        keys = ("id", "revision", "due_at", "admission_deadline", "expires_at")
        ctx["reopened_caption"] = {name: cue[name] for name in keys if name in cue}
        evidence = finite(event.get("evidence_until", event["end"]), "reopened evidence")
        received = finite(event.get("received_until", evidence), "reopened received audio")
        observed = finite(event.get("observed_audio_time", evidence), "reopened observation")
        ctx["reopened_caption"]["observation_floor"] = min(observed, evidence, received)
        ctx["consumed"], ctx["last_commit_bound"] = start, cue["source_start"]
        self.presentation.drop(
            cue["key"], "source_tail_reopened", source_revision=int(event["revision"])
        )

    def feed(self, event, epoch, now, position):
        self.last_now, self.last_position = now, finite(position, "playback position")
        if epoch != self.epoch:
            self.notices.append(dict(reason="asr_epoch_discarded", epoch=epoch))
            return
        mode = (
            "sealed_occurrences"
            if event.get("segmentation_profile") == GraniteSentenceAssembler.PROFILE
            else "hypothesis_revisions"
        )
        if self.source_mode is not None and mode != self.source_mode:
            raise ValueError("ASR ownership contract changed within a playback epoch")
        self.source_mode = mode
        if mode == "sealed_occurrences":
            proposals = self.sentences.feed(event, now, position, budget=self.preparation_budget)
            self.presentation.admit(proposals, now, position)
            self.collect_notices()
            return
        self.reopen_unshown_tail(event, position)
        proposals = self.committer.feed(event)
        for revision in self.committer.revisions:
            self.presentation.source_revised(revision, now, position)
        self.committer.revisions.clear()
        self.presentation.admit(proposals, now, position)
        self.collect_notices()

    def translation_candidate(self, now, position, speed=1):
        self.last_now, self.last_position = now, position
        result = self.presentation.translation_candidate(now, position, speed)
        self.collect_notices()
        return result

    def mark_submitted(self, key):
        self.presentation.mark_submitted(tuple(key))

    def audio_advanced(self, processed_until, epoch, now, position):
        if epoch != self.epoch:
            return
        self.last_now, self.last_position = now, position
        processed_until = finite(processed_until, "processed audio progress")
        released = (
            self.sentences.advance_audio(
                processed_until, now, position, budget=self.preparation_budget
            )
            if self.source_mode == "sealed_occurrences"
            else self.committer.advance_audio(processed_until)
        )
        self.presentation.admit(released, now, position)
        self.collect_notices()

    def translated(self, message, epoch, now, position):
        result = self.presentation.translated(message, epoch, now, position)
        self.collect_notices()
        return result

    def translation_failed(self, message, epoch, now, position):
        result = self.presentation.translation_failed(message, epoch, now, position)
        self.collect_notices()
        return result

    def frame(self, now, position, paused=False):
        self.last_now, self.last_position = now, position
        if self.source_mode == "sealed_occurrences":
            if self.preparation_budget is None:
                self.update_preparation_budget(
                    now, position, paused=paused, speed=self.presentation.speed
                )
            budget = dict(self.preparation_budget, paused=bool(paused))
            self.presentation.admit(self.sentences.tick(now, position, budget), now, position)
        result = self.presentation.frame(now, position, paused)
        self.collect_notices()
        return result

    def finalize(self):
        if not self.finalized:
            proposals = (
                self.sentences.finalize(self.last_now, self.last_position, self.preparation_budget)
                if self.source_mode == "sealed_occurrences"
                else self.committer.finalize()
            )
            self.presentation.admit(proposals, self.last_now, self.last_position)
            self.finalized = True
        self.collect_notices()

    def valid_translation_keys(self):
        return {
            key
            for key, cue in self.presentation.cues.items()
            if cue["status"] in {"ready", "inflight"}
        }

    @property
    def done(self):
        return self.finalized and not self.presentation.cues
