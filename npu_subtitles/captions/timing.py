"""Estimate cue intervals independently of hypothesis confirmation time.

Moonshine does not expose acoustic word alignment. Word observation times only
bound when text became available. A bounded speech-duration estimate lets a
confirmed cue use lookahead without scheduling all sentences at context EOF.
Granite may provide CTC emission spans instead. These are decoded frame positions,
not forced acoustic alignment. The time basis remains explicit in diagnostics.
"""

from __future__ import annotations

import math
from npu_subtitles.captions.language import (
    JAPANESE_CHARACTERS_PER_SECOND,
    JAPANESE_MAX_CHARACTERS,
    reading_units,
)


class CueTiming:
    WORDS_PER_SECOND = 3.2
    OBSERVATION_LEAD = 0.4

    @staticmethod
    def preparation(schedule: dict, position: float, budget: dict) -> dict:
        """Translate one wall-time service budget onto the media clock.

        Admission is a fixed source deadline. Latest commit is a planning
        point before it, allowing dispatch and NMT to finish inside admission.
        Pausing freezes media time; it does not renew the source deadline.
        """
        rate = 0.0 if budget["paused"] else budget["playback_speed"]
        reserve = budget["translation_reserve_seconds"] + budget["dispatch_reserve_seconds"]
        admission = schedule["admission_deadline"]
        return dict(
            budget,
            admission_deadline=admission,
            latest_commit=admission - rate * reserve,
            expected_ready_position=position + rate * reserve,
            remaining_admission_seconds=admission - position,
        )

    @classmethod
    def schedule(
        cls,
        proposal: dict,
        timing: dict,
        due: float,
        min_read: float,
        max_hold: float,
        max_lag: float,
    ) -> dict:
        """Fix useful deadlines from source time when a cue is first admitted.

        A long CTC-backed line may still be spoken after onset+max_lag. Its
        first admission budget can account for that span, but neither queue
        delay nor translation arrival can renew it. Retention remains bounded
        by source duration and the caption's word budget.
        """
        if (
            proposal.get("segmentation_profile") == "granite_ctc_v3"
            and timing.get("alignment_kind") == "ctc_emission_spans"
        ):
            language = proposal.get("source_language", "en")
            words = reading_units(proposal["text"], language)
            rate = JAPANESE_CHARACTERS_PER_SECOND if language == "ja" else cls.WORDS_PER_SECOND
            cap = JAPANESE_MAX_CHARACTERS if language == "ja" else 24
            duration_cap = max_hold + min(words, cap) / rate
            source_end = min(max(due, timing["estimated_end"]), due + duration_cap)
            expiry = max(due + max_hold, source_end + min_read)
            admission = max(due + max_lag, min(source_end + 0.5 * min_read, due + max_hold))
            return dict(
                expires_at=expiry,
                admission_deadline=min(admission, expiry - min_read),
                schedule_basis="bounded_ctc_source_interval",
                schedule_source_end=source_end,
                source_duration_capped=source_end < timing["estimated_end"] - 1e-6,
            )
        expiry = due + max_hold
        return dict(
            expires_at=expiry,
            admission_deadline=min(due + max_lag, expiry - min_read),
            schedule_basis="estimated_onset_budget",
        )

    @classmethod
    def estimate(cls, proposal: dict, min_read: float, max_hold: float) -> dict:
        context_start = float(proposal.get("context_start", proposal["source_start"]))
        available = float(proposal.get("evidence_until", proposal["source_end"]))
        observations = proposal.get("word_observed_until", [])
        valid = [
            float(t)
            for t in observations
            if isinstance(t, (int, float))
            and math.isfinite(t)
            and context_start <= t <= available + 1e-5
        ]
        complete = (
            max(valid) if valid else float(proposal.get("observed_complete_until", available))
        )
        complete = min(available, max(context_start, complete))
        spans = proposal.get("word_timestamps") or []
        if proposal.get("timing_kind") == "ctc_emission_spans" and spans:
            # The committer validates full word coverage and encoded bounds.
            # Observation time describes availability, not the spoken onset.
            start, end = float(spans[0]["start"]), max(float(span["end"]) for span in spans)
            return dict(
                estimated_start=start,
                estimated_end=end,
                observation_complete_until=complete,
                estimated_speech_seconds=end - start,
                alignment_kind="ctc_emission_spans",
                timing_source_revision=proposal["source_revision"],
            )
        word_count = sum(any(c.isalnum() for c in word) for word in proposal["text"].split())
        end = max(context_start, complete - cls.OBSERVATION_LEAD)
        # Only the first group of newly observed words anchors the onset. A
        # missing last word supplied by a much later final must not move the
        # entire sentence to that final's time. Rewrites retain observation age.
        first = valid[0] if valid else complete
        first_count = (
            next((i for i, value in enumerate(valid) if abs(value - first) > 1e-5), len(valid))
            if valid
            else word_count
        )
        first_duration = max(min_read, first_count / cls.WORDS_PER_SECOND)
        start = max(context_start, first - cls.OBSERVATION_LEAD - first_duration)
        if not proposal.get("prefix_text"):
            start = max(start, end - (max_hold - min_read))
        start = min(end, start)
        return dict(
            estimated_start=start,
            estimated_end=end,
            observation_complete_until=complete,
            estimated_speech_seconds=end - start,
            alignment_kind=(
                "capacity_bridge_observation_estimated"
                if proposal.get("prefix_text")
                else "word_observation_duration_estimated"
            ),
        )
