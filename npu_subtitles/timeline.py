"""Sample-accurate read bounds and playback discontinuities, without model policy."""

from __future__ import annotations

from dataclasses import dataclass
import math

from npu_subtitles.paths import SAMPLE_RATE


@dataclass(frozen=True)
class PlaybackState:
    epoch: int
    position: float
    sampled_at: float
    paused: bool
    speed: float


@dataclass(frozen=True)
class TimelineUpdate:
    state: PlaybackState
    reset_reason: str | None = None
    pause_changed: bool = False


class MediaTimeline:
    """Pause preserves continuity; seeks and explicit audio gaps start a new epoch.

    A bounded lookahead is a read permission, never a subtitle display timestamp.
    The read cursor is integer samples; wall time never advances the media clock.
    """

    def __init__(
        self,
        start: float,
        end: float,
        *,
        lookahead: float = 8.0,
        safety_margin: float = 0.08,
        max_backlog: float = 2.0,
    ):
        if not all(math.isfinite(v) for v in (start, end, lookahead, safety_margin, max_backlog)):
            raise ValueError("Timeline values must be finite")
        if (
            start < 0
            or end <= start
            or not 0 <= lookahead <= 12
            or safety_margin < 0
            or max_backlog <= 0
        ):
            raise ValueError("Invalid media interval or read budget")
        self.start, self.end = start, end
        self.end_sample = math.floor(end * SAMPLE_RATE)
        self.lookahead, self.safety_margin = lookahead, safety_margin
        self.max_backlog = max_backlog
        self.epoch = 0
        self.cursor = round(start * SAMPLE_RATE)
        self.state: PlaybackState | None = None

    def observe(
        self, position: float, now: float, paused: bool, speed: float, *, seek_event: bool = False
    ) -> TimelineUpdate:
        if not all(math.isfinite(v) for v in (position, now, speed)) or speed <= 0:
            raise ValueError("Invalid playback clock")
        previous = self.state
        changed = previous is not None and paused != previous.paused
        reason = None
        if previous is not None:
            elapsed = max(0.0, now - previous.sampled_at)
            delta = position - previous.position
            # A player can stall without reporting pause (startup, buffering,
            # window changes). Moving less than wall time predicts is never
            # sufficient evidence of a seek. Detect only impossible forward
            # motion, actual backwards motion, or explicit player events.
            jumped = delta > elapsed * max(speed, previous.speed) + 0.8
            if delta < -0.15 or jumped or (seek_event and abs(delta) > 0.08):
                reason = "seek"
                self.reset(position)
        self.state = PlaybackState(self.epoch, position, now, paused, speed)
        return TimelineUpdate(self.state, reason, changed)

    def reset(self, position: float) -> None:
        self.epoch += 1
        self.cursor = max(0, math.floor(position * SAMPLE_RATE))

    def read_limit(self, *, ending: bool = False) -> int:
        if self.state is None:
            return self.cursor
        allowed = (
            self.end
            if ending
            else min(self.end, self.state.position + self.lookahead - self.safety_margin)
        )
        return max(0, math.floor(allowed * SAMPLE_RATE))

    def reserve(self, samples: int, *, ending: bool = False) -> tuple[int, int] | None:
        if self.state is None or (self.state.paused and not ending):
            return None
        limit = self.read_limit(ending=ending)
        count = min(samples, limit - self.cursor)
        if count <= 0 or (count < samples and not ending and limit < self.end_sample):
            return None
        start = self.cursor
        self.cursor += count
        return start, self.cursor

    def audio_is_late(self, oldest_unprocessed: float) -> bool:
        return bool(
            self.state
            and not self.state.paused
            and self.state.position - oldest_unprocessed > self.max_backlog
        )

    def task_deadline(self, media_deadline: float, now: float, budget: float) -> float:
        if self.state is None:
            return now
        remaining = (
            budget
            if self.state.paused
            else max(0.0, (media_deadline - self.state.position) / self.state.speed)
        )
        return now + min(budget, remaining)
