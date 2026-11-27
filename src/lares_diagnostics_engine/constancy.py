"""Constancy measurement: the `constancy` kind of the fault list.

A channel keeps sending and delivers the same value hour after hour — the
producer works, the register behind it is dead. `15/1/22`, the L2 voltage
of the energy meter, read exactly 0.0 for 717 hourly buckets while its two
sibling phases carried real voltages, and looked perfectly healthy to every
measurement that only asked whether something arrived.

Kept apart from silence deliberately: the arithmetic is different (the
spread of a value against the time since the last bucket), the parameters
are different, and so is what broke. What the silence kind drops as a dead
register is exactly what this one reports.

Constancy is measured over the *extremes* of each hourly bucket, never its
average: an hour that swung and averaged back is a living register, and an
hour whose own min and max differ by more than the declared band is not
constant at all. A run ends where the value leaves the band — measured over
the whole run, so a slow walk is drift rather than constancy — and where a
bucket is missing, because an hour the channel said nothing in is silence's
business and cannot be claimed here.

Two parameters, both in the channel's own terms: `constant_hours`, how long
one value may stand, and `same_within`, what counts as the same value in
that channel's unit. Every bucket a run stands past the limit is an
observation whose score is the run length in units of that limit — the
fault's declared unit.

Like silence, time is the aggregate's own frontier, and the delivery is per
main group — on this fault's own address there, not silence's: the address
says roughly where, the payload names the exact channels and what each is
stuck at. The per-group rule itself is shared with silence in `groups`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from operator import attrgetter
from typing import TYPE_CHECKING, Any

from .episodes import Observation
from .groups import GroupPublish, group_plan, main_group
from .logging_setup import get_logger
from .reconcile import Measured, Window
from .silence import BUCKET, Channel, resolve_scope

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import psycopg
    from psycopg.rows import DictRow

    from .episode_store import OpenEpisodeRow
    from .episodes import Episode
    from .faults import Fault
    from .reconcile import Plan

log = get_logger(__name__)

# One reading in the tail is a value, not a constancy: the least the
# candidate filter can read a standing run off.
MIN_TAIL_BUCKETS = 2


@dataclass(frozen=True, slots=True)
class Reading:
    """One hourly bucket as constancy reads it: the extremes the channel
    showed inside that hour, never the average it landed on.
    """

    bucket: datetime
    low: float
    high: float

    @property
    def swing(self) -> float:
        return self.high - self.low


@dataclass(frozen=True, slots=True)
class ConstantRun:
    """One stretch a channel held its value: first and last bucket, and the
    band it stayed inside. A bucket covers its full hour, so a run of one
    bucket is already an hour long.
    """

    start: datetime
    end: datetime
    low: float
    high: float

    @property
    def duration(self) -> timedelta:
        return self.end - self.start + BUCKET

    @property
    def value(self) -> float:
        """The value it settled on — the floor of the band it held, which
        with the usual `same_within` of zero is the value itself."""
        return self.low

    def extended_by(self, reading: Reading, same_within: float) -> ConstantRun | None:
        """This run with the reading added, or None where the reading ends
        it instead: a bucket missing in between, or a value that takes the
        run's whole spread outside the band. The band is measured over the
        run rather than against the predecessor, so half a degree an hour is
        a walk the drift kind owns, not a stuck register.
        """
        if reading.bucket - self.end > BUCKET:
            return None
        low, high = min(self.low, reading.low), max(self.high, reading.high)
        if high - low > same_within:
            return None
        return ConstantRun(start=self.start, end=reading.bucket, low=low, high=high)


@dataclass(frozen=True, slots=True)
class ConstancyState:
    """Whether the channel is standing on one value at the frontier, and on
    which — what the publish payload names beside the severity."""

    channel: Channel
    run: ConstantRun | None = None

    @property
    def stuck_hours(self) -> float | None:
        return self.run.duration / BUCKET if self.run is not None else None

    @property
    def value(self) -> float | None:
        return self.run.value if self.run is not None else None


@dataclass(frozen=True, slots=True)
class WindowStats:
    """What one cheap query says about a channel: how far it was measured at
    all, and how far its value moved over the tail a standing constancy
    would have to cover. Enough to rule most channels out without touching
    their bucket series.
    """

    ga: str
    last_bucket: datetime
    tail_buckets: int
    tail_low: float | None
    tail_high: float | None

    def held_the_band(self, same_within: float) -> bool:
        """Whether the tail shows the channel standing on one value: enough
        readings to say so at all, and none of them outside the band. No
        tail is not constancy — nothing measured it.
        """
        if self.tail_buckets < MIN_TAIL_BUCKETS:
            return False
        if self.tail_low is None or self.tail_high is None:
            return False
        return self.tail_high - self.tail_low <= same_within


def constant_runs(readings: Sequence[Reading], *, same_within: float) -> tuple[ConstantRun, ...]:
    """The stretches the channel held one value, in time order.

    A bucket whose own extremes already leave the band belongs to no run at
    all — the register moved inside that hour, which is life — and it
    separates the runs on either side. Where a run ends is
    `ConstantRun.extended_by`.
    """
    runs: list[ConstantRun] = []
    open_run: ConstantRun | None = None
    for reading in sorted(readings, key=attrgetter("bucket")):
        moved = reading.swing > same_within
        extended = (
            None if moved or open_run is None else open_run.extended_by(reading, same_within)
        )
        if extended is not None:
            open_run = extended
            continue
        if open_run is not None:
            runs.append(open_run)
        open_run = (
            None
            if moved
            else ConstantRun(
                start=reading.bucket, end=reading.bucket, low=reading.low, high=reading.high
            )
        )
    if open_run is not None:
        runs.append(open_run)
    return tuple(runs)


def constancy_observations(
    ga: str, runs: Sequence[ConstantRun], max_constant: timedelta
) -> list[Observation]:
    """One observation per bucket a run stands past the declared limit, for
    the episode pipeline. The score is the run length so far in units of
    that limit — the fault's declared unit; the value is that length in
    hours. A bucket covers its full hour, so a run's first bucket already
    counts as one.
    """
    observations: list[Observation] = []
    for run in runs:
        t = run.start
        while t <= run.end:
            elapsed = t - run.start + BUCKET
            if elapsed > max_constant:
                observations.append(
                    Observation(
                        subject=ga, time=t, score=elapsed / max_constant, value=elapsed / BUCKET
                    )
                )
            t += BUCKET
    return observations


def classify(
    channel: Channel, runs: Sequence[ConstantRun], window: Window
) -> ConstancyState:
    """The run the channel is standing in now, if its last bucket still
    reaches the frontier — a channel that has since gone quiet is unmeasured
    rather than recovered, and silence owns it.
    """
    if runs and window.reaches(runs[-1].end):
        return ConstancyState(channel, run=runs[-1])
    return ConstancyState(channel)


def candidates(
    channels: Sequence[Channel],
    stats_by_ga: Mapping[str, WindowStats],
    open_subjects: frozenset[str],
    *,
    same_within: float,
) -> list[Channel]:
    """Channels whose bucket series is worth fetching: those whose value did
    not leave the band anywhere in the tail a standing constancy would have
    to cover, and those carrying an open episode — whose recovery is only
    visible in the series. Everything else is ruled out off the stats query
    alone, including channels the window holds no data for at all.
    """
    kept: list[Channel] = []
    for channel in channels:
        stats = stats_by_ga.get(channel.ga)
        if stats is None:
            continue
        if channel.ga in open_subjects or stats.held_the_band(same_within):
            kept.append(channel)
    return kept


def window_stats(
    conn: psycopg.Connection[DictRow],
    gas: Sequence[str],
    window_start: datetime,
    tail_start: datetime,
) -> dict[str, WindowStats]:
    """One query for the whole scope: how far each channel was measured, and
    how far its value moved over the tail."""
    rows = conn.execute(
        """
        SELECT ga,
               max(bucket) AS last_bucket,
               count(*) FILTER (WHERE bucket >= %(tail)s) AS tail_buckets,
               min(min_value) FILTER (WHERE bucket >= %(tail)s) AS tail_low,
               max(max_value) FILTER (WHERE bucket >= %(tail)s) AS tail_high
        FROM knx_1h
        WHERE ga = ANY(%(gas)s) AND bucket >= %(start)s
        GROUP BY ga
        """,
        {"gas": list(gas), "start": window_start, "tail": tail_start},
    ).fetchall()
    return {
        r["ga"]: WindowStats(
            ga=r["ga"],
            last_bucket=r["last_bucket"],
            tail_buckets=r["tail_buckets"],
            tail_low=None if r["tail_low"] is None else float(r["tail_low"]),
            tail_high=None if r["tail_high"] is None else float(r["tail_high"]),
        )
        for r in rows
    }


def readings(
    conn: psycopg.Connection[DictRow], gas: Sequence[str], window_start: datetime
) -> dict[str, list[Reading]]:
    """The candidates' hourly extremes over the window — a handful of frozen
    channels, not the whole scope."""
    rows = conn.execute(
        """
        SELECT ga, bucket, min_value, max_value FROM knx_1h
        WHERE ga = ANY(%(gas)s) AND bucket >= %(start)s
        ORDER BY ga, bucket
        """,
        {"gas": list(gas), "start": window_start},
    ).fetchall()
    series: dict[str, list[Reading]] = {}
    for row in rows:
        series.setdefault(row["ga"], []).append(
            Reading(
                bucket=row["bucket"],
                low=float(row["min_value"]),
                high=float(row["max_value"]),
            )
        )
    return series


def measure(
    conn: psycopg.Connection[DictRow],
    fault: Fault,
    window: Window,
    open_rows: Sequence[OpenEpisodeRow],
) -> Measured[ConstancyState]:
    """The kind's whole measurement: the scope resolved where the catalog
    lives, most channels ruled out off one cheap stats query, the hourly
    extremes fetched only for the candidates, and the run walk over those.
    A dry run measures exactly what the real run would.
    """
    max_constant = timedelta(hours=float(fault.parameters["constant_hours"]))
    same_within = float(fault.parameters["same_within"])

    channels = resolve_scope(conn, fault.channel_scope())
    stats_by_ga = window_stats(
        conn, [c.ga for c in channels], window.start, window.frontier - max_constant
    )
    open_subjects = frozenset(row.subject for row in open_rows)
    frozen = candidates(channels, stats_by_ga, open_subjects, same_within=same_within)
    series = readings(conn, [c.ga for c in frozen], window.start)

    states: dict[str, ConstancyState] = {}
    observations: list[Observation] = []
    for channel in frozen:
        runs = constant_runs(series.get(channel.ga, []), same_within=same_within)
        states[channel.ga] = classify(channel, runs, window)
        observations.extend(constancy_observations(channel.ga, runs, max_constant))

    # A channel with no bucket near the frontier was not measured — which is
    # not the same as recovered, so its episode stays open and the silence
    # fault reports the quiet. Channels the window holds nothing for at all
    # are the never-sent symmetry addresses, and they land here too.
    dataless = frozenset(
        channel.ga
        for channel in channels
        if channel.ga not in stats_by_ga
        or not window.reaches(stats_by_ga[channel.ga].last_bucket)
    )
    return Measured(
        states=states,
        observations=tuple(observations),
        dataless=dataless,
        record={
            "channels": len(channels),
            "candidates": len(frozen),
            "stuck": sum(
                1
                for state in states.values()
                if state.run is not None and state.run.duration > max_constant
            ),
        },
        labels={c.ga: c.name for c in channels},
    )


@dataclass(frozen=True, slots=True)
class ChannelReport:
    """One stuck channel inside a group publish — the payload names the
    exact channel, since when it has not moved, and the value it is stuck
    at.
    """

    ga: str
    name: str
    stuck_since: datetime | None
    severity: int
    stuck_hours: float | None
    value: float | None

    @property
    def magnitude(self) -> float | None:
        """How bad this channel is, in the fault's declared unit — how long
        it has stood, which orders the group's list."""
        return self.stuck_hours


def report_for(
    subject: str, severity: int, state: ConstancyState | None, _episode: Episode | None
) -> ChannelReport:
    """One channel's line in the group publish. A subject held open for want
    of data has no state this run, so its report names the address and the
    tier it still carries, and nothing it cannot know. The episode the
    protocol offers says nothing this state does not: the run carries how
    long the channel has stood.
    """
    if state is None:
        return ChannelReport(
            ga=subject,
            name=subject,
            stuck_since=None,
            severity=severity,
            stuck_hours=None,
            value=None,
        )
    return ChannelReport(
        ga=subject,
        name=state.channel.name,
        stuck_since=state.run.start if state.run is not None else None,
        severity=severity,
        stuck_hours=state.stuck_hours,
        value=state.value,
    )


def plan_run(
    *,
    episodes: Sequence[Episode],
    open_rows: Sequence[OpenEpisodeRow],
    measured: Measured[ConstancyState],
    frontier: datetime,
) -> Plan[GroupPublish[ChannelReport]]:
    """The shared per-main-group delivery, with constancy's own channel
    reports."""
    return group_plan(
        episodes=episodes,
        open_rows=open_rows,
        measured=measured,
        frontier=frontier,
        report_for=report_for,
        group_of=main_group,
        publish_for=GroupPublish,
    )


def group_payload(publish: GroupPublish[ChannelReport]) -> dict[str, Any]:
    """What this kind says on the bus: how many channels in the group stand
    on one value, which, and what each is stuck at — fields and wire names
    in one place."""
    return {
        "stuck_channels": len(publish.channels),
        "channels": [
            {
                "ga": report.ga,
                "name": report.name,
                "stuck_since": report.stuck_since,
                "severity": report.severity,
                "stuck_hours": report.stuck_hours,
                "value": report.value,
            }
            for report in publish.channels
        ],
    }
