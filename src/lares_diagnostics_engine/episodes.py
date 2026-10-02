"""Episode pipeline: repeated observations of one fault fold into episodes.

Two layers, kept apart: the per-bucket evidence rows are how a course of
events is reconstructed; the episode is the unit that reports, notifications
and verdicts address. The pipeline is pure — observations in, episodes plus
notification events out — and is recomputed from history on each run, so a
redeploy cannot corrupt or lose state.

Severity is the quantile of the fault's own score distribution, promoted one
tier by duration — there is no global ladder. An episode ends after a few
quiet runs, never at the first, so a flickering fault stays one incident.
Each episode emits at most three notification events: appearing, escalating,
ending. The escalation budget spends at the first rise; later worsening
rides the severity written to the bus, never a new notification. The numeric
tiers are the delivery contract in `severity`. An event that reached a row
becomes an `EpisodeEvent`, the pointer the bus carries.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from itertools import groupby
from operator import attrgetter
from typing import Literal

from .severity import CLEAR, CRITICAL, INFO, WARNING


class EventKind(StrEnum):
    APPEARED = "appeared"
    ESCALATED = "escalated"
    ENDED = "ended"


# What a subject names, for everyone who has to resolve it against the KNX
# catalog. `ref` is the group address that resolves: the subject itself for
# a channel, a channel that stands for the room or plant otherwise — never
# the channel a room fault measured, which compares two of them.
EntityKind = Literal["channel", "room", "plant"]


@dataclass(frozen=True, slots=True)
class Entity:
    """What an observation's subject names, and the address that finds it."""

    kind: EntityKind
    ref: str | None = None


@dataclass(frozen=True, slots=True)
class Observation:
    """One per-bucket observation of a firing fault on one subject. The score
    is the fault's own magnitude in its declared unit; it is compared only
    against this fault's history, never across faults.
    """

    subject: str
    time: datetime
    score: float
    value: float | None = None
    # Set by the kinds whose subject is not a group address; the rest leave
    # it None and are read as the channel their subject names.
    entity: Entity | None = None


@dataclass(frozen=True, slots=True)
class EpisodePolicy:
    """How observations fold: the run cadence, how many quiet runs end an
    episode, when duration promotes a tier, and which quantiles of the
    fault's own history mean warning and critical.
    """

    bucket: timedelta = timedelta(hours=1)
    quiet_runs: int = 3
    promote_after_runs: int = 4
    warning_quantile: float = 0.99
    critical_quantile: float = 0.999

    def __post_init__(self) -> None:
        if self.bucket <= timedelta(0):
            raise ValueError("bucket must be a positive interval")
        if self.quiet_runs < 1:
            # Ending on the first quiet run shatters flickering faults.
            raise ValueError("quiet_runs must be at least 1")
        if self.promote_after_runs < 1:
            raise ValueError("promote_after_runs must be at least 1")
        if not (0 < self.warning_quantile < self.critical_quantile < 1):
            raise ValueError("quantiles must satisfy 0 < warning < critical < 1")

    @property
    def max_gap(self) -> timedelta:
        """Largest observation gap that still continues an episode: quiet_runs
        missed runs, back on the next.
        """
        return (self.quiet_runs + 1) * self.bucket


@dataclass(frozen=True, slots=True)
class Cutoffs:
    """Score cutoffs from the fault's own distribution: above warning is
    rare, above critical is worse than the fault has been all quarter.
    """

    warning: float
    critical: float


@dataclass(frozen=True, slots=True)
class EvidenceRow:
    """One per-bucket evidence row under an episode: the observation plus the
    severity it carried at that bucket. The rows in order are the episode's
    severity trajectory.
    """

    time: datetime
    score: float
    severity: int
    value: float | None = None


@dataclass(frozen=True, slots=True)
class NotificationEvent:
    kind: EventKind
    time: datetime
    severity: int


@dataclass(frozen=True, slots=True)
class EpisodeEvent:
    """A notification event as the database recorded it: the row's own id and
    the fault and subject it belongs to, beside the event. That is what makes
    it addressable from outside the engine, and it is exactly what goes on
    `episode.<kind>` — a pointer, never the evidence. Whoever wants the
    sentence, the parameters or the observations fetches the episode by id.
    """

    episode_id: int
    fault: str
    subject: str
    kind: EventKind
    time: datetime
    severity: int


@dataclass(frozen=True, slots=True)
class Episode:
    """One incident: when it started, when it was last seen, how bad it got,
    with the evidence rows that formed it and at most three notification
    events. `ended_at` is the run boundary after which no observation can
    continue the episode, decided on the first run past it.
    """

    fault: str
    subject: str
    started_at: datetime
    last_seen_at: datetime
    ended_at: datetime | None
    severity: int
    peak_score: float
    evidence: tuple[EvidenceRow, ...]
    events: tuple[NotificationEvent, ...]
    entity: Entity | None = None


def severity_cutoffs(
    history_scores: Sequence[float], policy: EpisodePolicy
) -> Cutoffs | None:
    """Cutoffs from the fault's own distribution. None without history: with
    nothing to be rare against, everything is info — the global ladder this
    replaces is not a fallback.
    """
    if not history_scores:
        return None
    ordered = sorted(history_scores)
    return Cutoffs(
        warning=_quantile(ordered, policy.warning_quantile),
        critical=_quantile(ordered, policy.critical_quantile),
    )


def fold_observations(
    fault: str,
    observations: Iterable[Observation],
    history_scores: Sequence[float],
    policy: EpisodePolicy,
    now: datetime,
) -> tuple[Episode, ...]:
    """The one seam: a fault's observations in, its episodes with their
    notification events out. Pure and deterministic — `now` only decides
    which episodes are still open.
    """
    cutoffs = severity_cutoffs(history_scores, policy)
    episodes: list[Episode] = []
    for subject, per_subject in groupby(
        sorted(observations, key=attrgetter("subject", "time")), key=attrgetter("subject")
    ):
        for run in _split_runs(list(per_subject), policy.max_gap):
            episodes.append(_build_episode(fault, subject, run, cutoffs, policy, now))
    return tuple(episodes)


def _split_runs(
    ordered: list[Observation], max_gap: timedelta
) -> Iterable[list[Observation]]:
    """Split one subject's time-ordered observations wherever the gap exceeds
    what quiet_runs tolerates.
    """
    run: list[Observation] = []
    for obs in ordered:
        if run and obs.time - run[-1].time > max_gap:
            yield run
            run = []
        run.append(obs)
    if run:
        yield run


def _build_episode(
    fault: str,
    subject: str,
    run: list[Observation],
    cutoffs: Cutoffs | None,
    policy: EpisodePolicy,
    now: datetime,
) -> Episode:
    started_at = run[0].time
    last_seen_at = run[-1].time

    evidence = tuple(
        EvidenceRow(
            time=obs.time,
            score=obs.score,
            severity=_tier(obs, started_at, cutoffs, policy),
            value=obs.value,
        )
        for obs in run
    )

    # An observation at exactly close_time still continues the episode, so the
    # end is decidable only on the first run strictly past it.
    close_time = last_seen_at + policy.max_gap
    ended_at = close_time if now > close_time else None

    events = [NotificationEvent(EventKind.APPEARED, started_at, evidence[0].severity)]
    escalation = next((e for e in evidence if e.severity > evidence[0].severity), None)
    if escalation is not None:
        events.append(NotificationEvent(EventKind.ESCALATED, escalation.time, escalation.severity))
    if ended_at is not None:
        events.append(NotificationEvent(EventKind.ENDED, ended_at, CLEAR))

    return Episode(
        entity=run[0].entity,
        fault=fault,
        subject=subject,
        started_at=started_at,
        last_seen_at=last_seen_at,
        ended_at=ended_at,
        severity=max(e.severity for e in evidence),
        peak_score=max(obs.score for obs in run),
        evidence=evidence,
        events=tuple(events),
    )


def _tier(
    obs: Observation,
    started_at: datetime,
    cutoffs: Cutoffs | None,
    policy: EpisodePolicy,
) -> int:
    if cutoffs is None:
        base = INFO
    elif obs.score > cutoffs.critical:
        base = CRITICAL
    elif obs.score > cutoffs.warning:
        base = WARNING
    else:
        base = INFO
    # Duration distinguishes a shower from a leak better than magnitude does.
    promoted = obs.time - started_at >= policy.promote_after_runs * policy.bucket
    return min(base + (1 if promoted else 0), CRITICAL)


def _quantile(ordered: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an ascending sequence."""
    position = (len(ordered) - 1) * q
    lower = int(position)
    if lower + 1 >= len(ordered):
        return ordered[-1]
    fraction = position - lower
    return ordered[lower] + (ordered[lower + 1] - ordered[lower]) * fraction
