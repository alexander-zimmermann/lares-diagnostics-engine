"""Deviation measurement: the `deviation` kind of the fault list.

A value sits too far under the reference declared for it. The reference is
swappable by declaration, and what a fault declares it as decides which of
the kind's two shapes it runs in:

* **rooms against a channel** — the fault names a channel pattern for the
  reference and the optional gate, and each room names the channel it is
  measured on;
* **a daily yield against a named model** — the fault names an
  `expectation` instead, and the plant's day is compared against the kWh
  that model says it should have made.

Both compare a measured value against an expected one and score the gap in
units of the gap the fault allows; only where the two numbers come from
differs.

## Rooms against a channel

A room at least its declared gap under the setpoint while the valve is open
past the gate threshold, for at least the declared hours. The reference and
the optional gate follow a uniform naming rule across rooms, which is what
makes a pattern the honest way to write them, while each room names the
channel it is measured on. The two marry into triples — strictly both
ways, like the duration kind's device limits.

Values come from the hourly aggregate as a dense series: KNX channels are
state, not samples — a setpoint that sent once holds until changed — so
each role's last seen hourly average is carried forward across silent
buckets, deliberately without a staleness bound (a role dying mid-window
freezes the room's picture; the silence fault owns the dead channel). A
room measures only from the bucket where every role has appeared inside
the window; a room where one never does is `dataless`, must not
self-clear, and is warned about. Several value channels per room (the
sensor-less halls measure via their BWM heads) are averaged; a value
channel constant at zero for a day or more is a dead register, not a cold
room, and drops out before measurement.

Consecutive cold buckets form a run — a warm hour restarts the clock — and
every bucket from `min_hours` onward becomes an observation whose score is
the gap in units of the room's declared threshold, the fault's declared
unit. Like silence and duration, time is the aggregate's own frontier.

## A daily yield against a named model

Low PV production is nearly always the weather, so the only shortfall worth
a notification is one against what *this* weather was expected to give: the
plant's whole-day yield against the day's forecast, both in kWh. The
expectation is named in the fault entry and looked up in `EXPECTATIONS` —
the computation is handed two numbers per day and never learns which model
produced the second, so a sibling inverter or a clear-sky model is a line
in the file and a function here, and nothing else moves.

Whole days, not hours: a cloud passing at noon is not a fault, and the
counter arithmetic over a full day is exact where an hourly ratio is noise.
The day's production is still read off the hourly closes of each
inverter's lifetime counter, because that counter is not monotonic — an
inverter has come back from a reboot 12.8 kWh lower and counted on from
there — so a day is the sum of the plausible rises between consecutive
closes, not its last close minus the previous day's: a drop is a re-base,
a rise past what the plane can physically make in the hours it spans is a
bogus reading, and neither is production or loss. What a plane can make
comes from the site file (`site.yaml`, what the house is), never from a
fault parameter. A day is judged only once the aggregate has moved past
it — the frontier here is the newest *complete* day, so the engine speaks
about yesterday and never about a day still in progress — and only when
enough was expected of it for a percentage to mean anything. A day that
expected almost nothing (deep overcast, or a forecast that never arrived)
is unmeasured rather than quiet, which is what keeps a dead forecast job
from clearing a standing fault. Days are UTC days: nothing is produced
across a UTC midnight at this latitude, so a local day holds the same
yield and needs a timezone nobody has to agree on.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Protocol

from . import forecast_solar
from .episodes import EpisodePolicy, Observation
from .faults import DeviationExpectation, Roles
from .logging_setup import get_logger
from .reconcile import Measured, Window
from .runs import split_runs
from .silence import BUCKET, DEAD_MIN_BUCKETS, hourly_averages, like_match, resolve_scope
from .slug import entity_slug

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import psycopg
    from psycopg.rows import DictRow

    from .episode_store import OpenEpisodeRow
    from .faults import Fault, RoomRule
    from .silence import Channel
    from .site import Plane, Site

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Room:
    """One monitored room: its channel triple and declared threshold. The
    slug doubles as episode subject and NATS entity — the writer rules pin
    `fault.<fault>.<slug>` to the room's fault address."""

    label: str
    slug: str
    value_gas: tuple[str, ...]
    reference_ga: str
    gate_ga: str | None
    min_gap: float

    @property
    def gas(self) -> tuple[str, ...]:
        """Every channel the room measures with — the fetch list and the
        carry keys must enumerate the roles identically."""
        gate = (self.gate_ga,) if self.gate_ga is not None else ()
        return (*self.value_gas, self.reference_ga, *gate)


@dataclass(frozen=True, slots=True)
class RoomBucket:
    """One measurable hour of a room: every role has a (carried) value."""

    time: datetime
    value: float
    reference: float
    gate: float | None

    @property
    def gap(self) -> float:
        return self.reference - self.value


@dataclass(frozen=True, slots=True)
class RoomState:
    """Whether the room is cold at the frontier, since when, and the
    numbers behind it — what the publish payload names beside the severity."""

    room: Room
    cold_since: datetime | None = None
    gap: float | None = None
    value: float | None = None
    reference: float | None = None
    gate: float | None = None


def resolve_rooms(
    channels: Sequence[Channel], rules: Sequence[RoomRule], roles: Roles
) -> list[Room]:
    """Marry the declared rooms to the scoped channels, strictly both ways:
    every room resolves one reference (and gate, where declared) and at
    least one value channel; every scoped channel belongs to exactly one
    room and role. Every problem is reported at once.
    """
    problems: list[str] = []
    rooms: list[Room] = []
    claimed: dict[str, str] = {}
    slugs: dict[str, str] = {}
    for rule in rules:
        slug = entity_slug(rule.match)
        if slug in slugs:
            problems.append(
                f"rooms {slugs[slug]!r} and {rule.match!r} share the slug {slug!r}"
            )
            continue
        slugs[slug] = rule.match
        mine = [c for c in channels if f".{rule.match}." in c.name]
        by_role: dict[str, list[Channel]] = {"value": [], "reference": [], "gate": []}
        for channel in mine:
            if channel.ga in claimed:
                problems.append(
                    f"channel {channel.ga} matched by {claimed[channel.ga]!r} "
                    f"and {rule.match!r}"
                )
                continue
            claimed[channel.ga] = rule.match
            hits = [
                role
                for role, pattern in (
                    ("value", rule.value),
                    ("reference", roles.reference),
                    ("gate", roles.gate),
                )
                if pattern is not None and like_match(pattern, channel.name)
            ]
            if len(hits) != 1:
                problems.append(
                    f"room {rule.match!r}: channel {channel.ga} ({channel.name}) "
                    + ("matches no role" if not hits else f"matches roles {', '.join(hits)}")
                )
                continue
            by_role[hits[0]].append(channel)

        broken = False
        for role, pattern in (("reference", roles.reference), ("gate", roles.gate)):
            if pattern is None:
                continue
            if len(by_role[role]) != 1:
                gas = ", ".join(c.ga for c in by_role[role])
                problems.append(
                    f"room {rule.match!r}: role {role} matches no channel"
                    if not by_role[role]
                    else f"room {rule.match!r}: role {role} matches "
                    f"{len(by_role[role])} channels: {gas}"
                )
                broken = True
        if not by_role["value"]:
            problems.append(f"room {rule.match!r}: role value matches no channel")
            broken = True
        if broken:
            continue

        rooms.append(
            Room(
                label=rule.match,
                slug=slug,
                value_gas=tuple(sorted(c.ga for c in by_role["value"])),
                reference_ga=by_role["reference"][0].ga,
                gate_ga=by_role["gate"][0].ga if roles.gate is not None else None,
                min_gap=rule.min_gap_k,
            )
        )
    problems.extend(
        f"channel {c.ga} ({c.name}) belongs to no declared room"
        for c in channels
        if c.ga not in claimed
    )
    if problems:
        raise ValueError("room rules do not fit the scope: " + "; ".join(problems))
    return sorted(rooms, key=lambda r: r.label)


def dead_value_gas(
    rooms: Sequence[Room], by_ga: Mapping[str, Mapping[datetime, float]]
) -> list[str]:
    """Value channels constant at zero over a day or more of buckets: a dead
    register reading 0.0 would score as a huge gap, so it drops out before
    measurement — the silence fault owns reporting it. Only the value role
    is checked: a closed valve legitimately sits at 0 % for weeks."""
    value_gas = {ga for room in rooms for ga in room.value_gas}
    return [
        ga
        for ga in sorted(value_gas)
        if (buckets := by_ga.get(ga))
        and len(buckets) >= DEAD_MIN_BUCKETS
        and all(v == 0.0 for v in buckets.values())
    ]


def room_series(
    room: Room,
    by_ga: Mapping[str, Mapping[datetime, float]],
    window_start: datetime,
    frontier: datetime,
) -> list[RoomBucket]:
    """The room's dense hourly series: each role's last seen hourly average,
    carried forward across silent buckets. Buckets before every role has
    appeared are unmeasurable and skipped; a room with none is dataless."""
    carry: dict[str, float] = {}
    buckets: list[RoomBucket] = []
    gas = room.gas
    t = window_start
    while t <= frontier:
        for ga in gas:
            value = by_ga.get(ga, {}).get(t)
            if value is not None:
                carry[ga] = value
        values = [carry[ga] for ga in room.value_gas if ga in carry]
        reference = carry.get(room.reference_ga)
        gate = carry.get(room.gate_ga) if room.gate_ga is not None else None
        if values and reference is not None and (room.gate_ga is None or gate is not None):
            buckets.append(
                RoomBucket(
                    time=t, value=sum(values) / len(values), reference=reference, gate=gate
                )
            )
        t += BUCKET
    return buckets


def cold_buckets(
    room: Room, buckets: Sequence[RoomBucket], gate_min: float | None
) -> list[RoomBucket]:
    """The buckets where the fault condition stands: the gap at or past the
    room's threshold while the gate (where declared) sits strictly above
    its minimum."""
    return [
        b
        for b in buckets
        if b.gap >= room.min_gap
        and (gate_min is None or (b.gate is not None and b.gate > gate_min))
    ]


def deviation_observations(
    room: Room, cold: Sequence[RoomBucket], min_hours: float
) -> list[Observation]:
    """One observation per bucket once a cold run has stood `min_hours`, for
    the episode pipeline. A run is consecutive cold buckets — a warm hour
    restarts the clock. The score is the gap in units of the room's own
    threshold — the fault's declared unit; the value is the gap itself. A
    bucket covers its full hour, so a run's first bucket already counts as
    one.
    """
    observations: list[Observation] = []
    run_start: datetime | None = None
    previous: datetime | None = None
    for bucket in cold:
        if run_start is None or previous is None or bucket.time - previous > BUCKET:
            run_start = bucket.time
        previous = bucket.time
        if bucket.time - run_start + BUCKET >= min_hours * BUCKET:
            observations.append(
                Observation(
                    subject=room.slug,
                    time=bucket.time,
                    score=bucket.gap / room.min_gap,
                    value=bucket.gap,
                )
            )
    return observations


def classify(room: Room, cold: Sequence[RoomBucket], frontier: datetime) -> RoomState:
    """The room's current cold stretch, if one reaches the frontier — what
    the publish payload names alongside the severity."""
    runs = split_runs((b.time for b in cold), BUCKET)
    if runs and runs[-1].end == frontier:
        latest = max(cold, key=lambda b: b.time)
        return RoomState(
            room=room,
            cold_since=runs[-1].start,
            gap=latest.gap,
            value=latest.value,
            reference=latest.reference,
            gate=latest.gate,
        )
    return RoomState(room=room)


@dataclass(frozen=True, slots=True)
class RoomPublish:
    """One room whose severity or openness moved — the payload names the
    room and its numbers; the writer rule carries only the severity to the
    room's anomaly address."""

    slug: str
    severity: int
    room: str
    cold_since: datetime | None = None
    gap: float | None = None
    value: float | None = None
    reference: float | None = None
    gate: float | None = None
    min_gap: float | None = None

    @property
    def subject(self) -> str:
        return self.slug

    @property
    def entity(self) -> str:
        # The slug is already a NATS-safe token — the room's identity here,
        # in the episode subject and on the bus is one string.
        return self.slug


def publish_for(subject: str, severity: int, state: RoomState | None) -> RoomPublish:
    # A room that left the scope while its row was open still gets its
    # clear; the payload then only names the slug.
    if state is None:
        return RoomPublish(slug=subject, severity=severity, room=subject)
    return RoomPublish(
        slug=subject,
        severity=severity,
        room=state.room.label,
        cold_since=state.cold_since,
        gap=state.gap,
        value=state.value,
        reference=state.reference,
        gate=state.gate,
        min_gap=state.room.min_gap,
    )


def payload(publish: RoomPublish) -> dict[str, Any]:
    """What this kind says on the bus: the gap, and the reference and gate
    behind it — fields and wire names in one place."""
    return {
        "room": publish.room,
        "cold_since": publish.cold_since,
        "gap": publish.gap,
        "value": publish.value,
        "reference": publish.reference,
        "gate": publish.gate,
        "min_gap": publish.min_gap,
    }


def measure(
    conn: psycopg.Connection[DictRow],
    fault: Fault,
    window: Window,
    _open_rows: Sequence[OpenEpisodeRow],
) -> Measured[RoomState]:
    """The kind's whole measurement: the declared rooms married to the
    scope, then the dense series, the cold buckets and the observations per
    room.
    """
    if not isinstance(fault.roles, Roles):
        raise ValueError(f"fault {fault.name}: the deviation kind needs declared roles")
    min_hours = float(fault.parameters["min_hours"])
    gate_min = fault.parameters.get("gate_min_pct")

    channels = resolve_scope(conn, fault.channel_scope())
    # A room without its channels or a channel without its room fails the
    # run loudly — never a silently unmeasured room.
    rooms = resolve_rooms(channels, fault.rooms, fault.roles)
    series = hourly_averages(conn, [ga for room in rooms for ga in room.gas], window.start)

    dead = dead_value_gas(rooms, series)
    if dead:
        # A dead register reading 0.0 would score as a huge gap; the silence
        # fault owns reporting the channel itself.
        log.info("scope_drops", fault=fault.name, dead=len(dead), dead_channels=dead)
        for ga in dead:
            del series[ga]

    states: dict[str, RoomState] = {}
    observations: list[Observation] = []
    dataless: set[str] = set()
    for room in rooms:
        buckets = room_series(room, series, window.start, window.frontier)
        if not window.reaches(buckets[-1].time if buckets else None):
            dataless.add(room.slug)
        cold = cold_buckets(room, buckets, gate_min)
        states[room.slug] = classify(room, cold, window.frontier)
        observations.extend(deviation_observations(room, cold, min_hours))

    return Measured(
        states=states,
        observations=tuple(observations),
        dataless=frozenset(dataless),
        record={
            "rooms": len(rooms),
            "cold": sum(1 for s in states.values() if s.cold_since is not None),
        },
    )


DAY = timedelta(days=1)

# Days fold, not hours. One quiet day between two short ones is the
# weather, so it must not split the incident; three quiet days in a row is
# a recovery. Promotion counts days too: a fourth short day in a row is
# worse news than the first, whatever the shortfall reads.
YIELD_POLICY = EpisodePolicy(bucket=DAY, quiet_runs=1, promote_after_runs=3)

# How far back this shape can be measured at all: the counters reach a year,
# but the expectation it compares them against is a stored forecast, and
# `mcp_forecasts` keeps 90 days. Past that there is nothing to be short of.
YIELD_HISTORY = timedelta(days=90)

# The one subject. The forecast models both roof planes in a single curve,
# so there is nothing to fan out over; the plant's own anomaly address
# carries it, and the writer rule pins the bare `fault.<fault>` subject
# — a 1:1 subject with no entity token, like the volume watchdog's.
PLANT = "pv"

# The forecast is a sampled power curve. Two samples further apart than
# this are not one interval but a hole in it — the night between two days
# above all, which must never be integrated across.
MAX_FORECAST_STEP = timedelta(hours=1)


def utc_day(moment: datetime) -> datetime:
    """The UTC day a moment falls in — the same bucket `time_bucket` cuts on
    the SQL side, whatever timezone the session renders a timestamptz in."""
    return moment.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


@dataclass(frozen=True, slots=True)
class YieldDay:
    """One judged day of the plant: what it produced, and what the day's
    declared expectation said it would, both in kWh."""

    day: datetime
    actual_kwh: float
    expected_kwh: float

    @property
    def shortfall_pct(self) -> float:
        """How far under the expectation the day came in, in percent of it."""
        return 100.0 * (1.0 - self.actual_kwh / self.expected_kwh)


@dataclass(frozen=True, slots=True)
class YieldState:
    """The plant at the frontier: the newest judged day and how it came in
    against its expectation, plus the start of the short stretch if one is
    still standing — what the publish payload names beside the severity.
    The numbers are reported whether or not the day was short: an episode
    stays open for a few quiet days, and "yesterday made its forecast" is
    what a human needs to read in that window."""

    expectation: DeviationExpectation
    min_shortfall_pct: float
    day: datetime | None = None
    actual_kwh: float | None = None
    expected_kwh: float | None = None
    shortfall_pct: float | None = None
    short_since: datetime | None = None


class Expectation(Protocol):
    """Where a daily-yield fault's expectation comes from: the kWh the plant
    should have produced on each day since `window_start`. The fault names
    one, `EXPECTATIONS` resolves it, and nothing downstream knows which was
    named — a sibling inverter or a clear-sky model is another function
    here and a different line in the fault file.
    """

    def __call__(
        self, conn: psycopg.Connection[DictRow], window_start: datetime
    ) -> dict[datetime, float]: ...


def yield_frontier(conn: psycopg.Connection[DictRow]) -> datetime | None:
    """The newest *complete* day of the inverter aggregate — the 'now' a
    daily-yield fault is measured against. Truncating the aggregate's own
    frontier and stepping back one day is what makes "complete" decidable
    without a clock: the engine speaks about yesterday and never about a
    day still in progress.
    """
    row = conn.execute(
        "SELECT time_bucket(INTERVAL '1 day', max(bucket)) - INTERVAL '1 day' AS frontier "
        "FROM solaredge_inverter_1h"
    ).fetchone()
    return row["frontier"] if row else None


def hourly_closes(
    conn: psycopg.Connection[DictRow], window_start: datetime, inverter_ids: Iterable[int]
) -> dict[int, dict[datetime, float]]:
    """Each declared inverter's lifetime energy counter, in Wh, as it stood
    at the close of every hourly bucket since `window_start`. A counter is
    exact and cheap where a power integral is neither; hourly and not daily
    closes, because the counter is not monotonic and only the steps between
    closes can tell production from a re-base."""
    rows = conn.execute(
        """
        SELECT bucket, inverter_id, energytotal_last AS wh
        FROM solaredge_inverter_1h
        WHERE bucket >= %(start)s
          AND inverter_id = ANY(%(inverters)s)
          AND energytotal_last IS NOT NULL
        ORDER BY inverter_id, bucket
        """,
        {"start": window_start, "inverters": list(inverter_ids)},
    ).fetchall()
    closes: dict[int, dict[datetime, float]] = {}
    for row in rows:
        closes.setdefault(int(row["inverter_id"]), {})[row["bucket"]] = float(row["wh"])
    return closes


@dataclass(frozen=True, slots=True)
class InverterYield:
    """One inverter's production, credited from its hourly counter closes:
    kWh per judged UTC day, and how many steps between closes were
    declined — drops (the counter re-based) and rises past what the plane
    can physically make (a bogus reading) — so a noisy counter is visible
    without a query."""

    kwh: Mapping[datetime, float]
    drops: int
    over_bound: int


def credit_rises(closes: Mapping[datetime, float], bound_wh_per_hour: float) -> InverterYield:
    """One inverter's production per UTC day, out of its lifetime counter
    at every hourly close. The counter is not monotonic: an inverter has
    come back from a reboot 12.8 kWh *lower* and counted on from there, so
    a day is not its last close minus the previous day's, but the sum of
    the plausible rises between consecutive closes. Each step is judged on
    its own: a rise of at most the plane's peak power times the hours the
    step spans is production, credited to the day of the later close; a
    drop is a re-base — neither production nor loss — and the next step is
    measured from the lower value; a rise past the bound is a bogus reading
    (a 0 followed by the true counter) and is declined as well.

    A day is judged only if the inverter also closed on the day before: an
    inverter that powers down at night has no reading until it is already
    producing, so its first daylight hour lives in the step from the
    previous day's last close, and a day whose predecessor is missing would
    be credited with the whole gap.
    """
    energy: dict[datetime, float] = defaultdict(float)
    drops = over_bound = 0
    for (start, before), (end, after) in pairwise(sorted(closes.items())):
        rise = after - before
        if rise < 0:
            drops += 1
            continue
        if rise > bound_wh_per_hour * ((end - start) / BUCKET):
            over_bound += 1
            continue
        energy[utc_day(end)] += rise / 1000.0
    days = {utc_day(moment) for moment in closes}
    judged = sorted(day for day in days if day - DAY in days)
    return InverterYield(
        kwh={day: energy[day] for day in judged}, drops=drops, over_bound=over_bound
    )


def inverter_yields(
    closes: Mapping[int, Mapping[datetime, float]], planes: Iterable[Plane]
) -> dict[int, InverterYield]:
    """Every declared inverter's production, by inverter id. The plant is
    what the site file says it is: each plane names the inverter it feeds
    and, in its kWp, the most that inverter's counter can rise in an hour —
    a property of the plant, never a fault parameter. A counter the
    aggregate carries for an inverter no plane names is not the plant's
    production, and a declared inverter without closes has no judged day
    and no declined step, so a silent plane still shows up in the record.
    """
    return {
        plane.inverter_id: credit_rises(closes.get(plane.inverter_id, {}), plane.kwp * 1000.0)
        for plane in planes
    }


def daily_yield(yields: Mapping[int, InverterYield]) -> dict[datetime, float]:
    """What the plant produced per day, in kWh: the inverters' judged days
    summed. A day one inverter could not judge still counts the others'
    production, as the per-inverter shape already left that inverter out.
    """
    energy: dict[datetime, float] = defaultdict(float)
    for inverter in yields.values():
        for day, kwh in inverter.kwh.items():
            energy[day] += kwh
    return {day: energy[day] for day in sorted(energy)}


def forecast_curve(
    conn: psycopg.Connection[DictRow], window_start: datetime
) -> list[tuple[datetime, float]]:
    """The stored forecast power curve since `window_start`, in watts. Each
    row is the last forecast written for that instant, so a past day carries
    the best-informed forecast that day itself produced — which is what
    makes the expectation weather-adjusted rather than a week-old guess.
    """
    rows = conn.execute(
        """
        SELECT forecast_for, forecast_value FROM mcp_forecasts
        WHERE source = %(source)s AND metric = %(metric)s AND model = %(model)s
          AND forecast_for >= %(start)s AND forecast_value IS NOT NULL
        ORDER BY forecast_for
        """,
        {
            # The identity of these rows belongs to the job that writes
            # them; naming it here again would let a rename pass silently.
            "source": forecast_solar.SOURCE,
            "metric": forecast_solar.METRIC,
            "model": forecast_solar.MODEL,
            "start": window_start,
        },
    ).fetchall()
    return [(row["forecast_for"], float(row["forecast_value"])) for row in rows]


def daily_energy(curve: Sequence[tuple[datetime, float]]) -> dict[datetime, float]:
    """A sampled power curve in watts integrated into kWh per UTC day,
    trapezoidally between consecutive samples and only where they are one
    interval apart. The curve carries daylight points alone, so the stretch
    from one sunset to the next sunrise is not an interval to integrate but
    the night between two days. `MAX_FORECAST_STEP` is where one becomes
    the other: it spans a single missing sample at either resolution the
    job may have stored, and nothing wider — so a gap of hours lowers the
    expectation rather than bridging it with a straight line through
    midday, which is the conservative direction, a lower expectation being
    harder to fall short of. An interval is credited to the day it starts
    in; at this latitude none of them straddles a UTC midnight.
    """
    energy: dict[datetime, float] = defaultdict(float)
    for (start, left), (end, right) in pairwise(curve):
        step = end - start
        if step <= timedelta(0) or step > MAX_FORECAST_STEP:
            continue
        hours = step / timedelta(hours=1)
        energy[utc_day(start)] += (left + right) / 2.0 * hours / 1000.0
    return dict(energy)


def forecast_solar_expectation(
    conn: psycopg.Connection[DictRow], window_start: datetime
) -> dict[datetime, float]:
    """The weather-adjusted expectation: the forecast.solar power curve the
    forecast job stores, integrated per day."""
    return daily_energy(forecast_curve(conn, window_start))


EXPECTATIONS: Mapping[DeviationExpectation, Expectation] = {
    DeviationExpectation.FORECAST_SOLAR: forecast_solar_expectation,
}


def judged_days(
    actual: Mapping[datetime, float],
    expected: Mapping[datetime, float],
    *,
    min_expected_kwh: float,
    frontier: datetime,
) -> list[YieldDay]:
    """The complete days both sides know about that expected enough for a
    percentage of it to mean anything. A day that expected almost nothing —
    deep overcast, or a forecast that never arrived — is not a quiet day but
    an unmeasured one, and is left out rather than scored.
    """
    return [
        YieldDay(day=day, actual_kwh=actual[day], expected_kwh=expected[day])
        for day in sorted(expected)
        if day <= frontier and day in actual and expected[day] >= min_expected_kwh
    ]


def short_days(days: Sequence[YieldDay], min_shortfall_pct: float) -> list[YieldDay]:
    """The judged days that came in at or past the declared shortfall."""
    return [d for d in days if d.shortfall_pct >= min_shortfall_pct]


def yield_observations(
    days: Sequence[YieldDay], min_shortfall_pct: float
) -> list[Observation]:
    """One observation per short day, for the episode pipeline. The score is
    the shortfall in units of the declared minimum — the fault's declared
    unit; the value is the shortfall itself. Unlike the room shape there is
    no run to stand first: a day is already the duration, and consecutive
    short days fold into one incident behind the episode seam.
    """
    return [
        Observation(
            subject=PLANT,
            time=d.day,
            score=d.shortfall_pct / min_shortfall_pct,
            value=d.shortfall_pct,
        )
        for d in short_days(days, min_shortfall_pct)
    ]


def classify_yield(
    days: Sequence[YieldDay],
    min_shortfall_pct: float,
    expectation: DeviationExpectation,
    frontier: datetime,
) -> YieldState:
    """The newest judged day and, if a short stretch still reaches the
    frontier, when it began — what the publish payload names alongside the
    severity. The day's numbers are reported either way: an episode stays
    open across a few quiet days, and a severity published in that window
    must still say what the last measured day actually did.
    """
    runs = split_runs((d.day for d in short_days(days, min_shortfall_pct)), DAY)
    state = YieldState(
        expectation=expectation,
        min_shortfall_pct=min_shortfall_pct,
        short_since=runs[-1].start if runs and runs[-1].end == frontier else None,
    )
    if not days:
        # Nothing judged in the whole window: no day to report on.
        return state
    latest = days[-1]
    return replace(
        state,
        day=latest.day,
        actual_kwh=latest.actual_kwh,
        expected_kwh=latest.expected_kwh,
        shortfall_pct=latest.shortfall_pct,
    )


@dataclass(frozen=True, slots=True)
class PlantPublish:
    """The plant when its severity moved — the payload names the day's yield
    against what was expected of it and which model said so; the writer rule
    carries only the severity to the PV anomaly address."""

    severity: int
    state: YieldState

    @property
    def subject(self) -> str:
        return PLANT

    @property
    def entity(self) -> None:
        # One plant, one declared address: a 1:1 subject, no entity token.
        return None


def publish_for_plant(_subject: str, severity: int, state: YieldState | None) -> PlantPublish:
    if state is None:
        # The plant is the fault's whole scope and the measurement always
        # states it, so a publish without a state is a wiring error, never
        # a subject that left the scope.
        raise ValueError("plant publish without a measured state")
    return PlantPublish(severity=severity, state=state)


def payload_yield(publish: PlantPublish) -> dict[str, Any]:
    """What the daily-yield shape says on the bus: the day the severity was
    earned on, and the expectation it fell short of — named, so Basalte's
    e-mail says what the shortfall was measured against."""
    state = publish.state
    return {
        "expectation": state.expectation,
        "day": state.day,
        "actual_kwh": state.actual_kwh,
        "expected_kwh": state.expected_kwh,
        "shortfall_pct": state.shortfall_pct,
        "min_shortfall_pct": state.min_shortfall_pct,
        "short_since": state.short_since,
    }


def measure_yield(
    conn: psycopg.Connection[DictRow],
    fault: Fault,
    window: Window,
    _open_rows: Sequence[OpenEpisodeRow],
    *,
    site: Site,
) -> Measured[YieldState]:
    """The daily-yield shape's whole measurement: what the plant produced
    per day against what its declared expectation says it should have. The
    site says which inverters the plant has and what each can make in an
    hour; the fault says only what counts as short.
    """
    expectation = fault.expectation
    if expectation is None:
        # The named expectation is what routed the fault here at all; this
        # is the measurement's side of the same contract.
        raise ValueError(f"fault {fault.name}: the daily-yield shape needs a named expectation")
    expected_kwh = EXPECTATIONS.get(expectation)
    if expected_kwh is None:
        # A name the schema accepts but nobody wrote a provider for: it
        # fails its own fault loudly rather than reporting a quiet plant.
        raise ValueError(f"fault {fault.name}: no expectation is wired up for {expectation}")
    min_shortfall = float(fault.parameters["min_shortfall_pct"])
    min_expected = float(fault.parameters["min_expected_kwh"])

    closes = hourly_closes(conn, window.start, [plane.inverter_id for plane in site.planes])
    yields = inverter_yields(closes, site.planes)
    days = judged_days(
        daily_yield(yields),
        expected_kwh(conn, window.start),
        min_expected_kwh=min_expected,
        frontier=window.frontier,
    )
    state = classify_yield(days, min_shortfall, expectation, window.frontier)
    # No judged day near the frontier means nobody could tell a recovery
    # from a dark week or a dead forecast job; the episode must not clear.
    reaches = window.reaches(days[-1].day if days else None)

    return Measured(
        states={PLANT: state},
        observations=tuple(yield_observations(days, min_shortfall)),
        # Nobody could tell a recovery from a dark week or a dead forecast
        # job, so the episode must not clear itself on the silence.
        dataless=frozenset() if reaches else frozenset({PLANT}),
        record={
            "expectation": str(expectation),
            "judged_days": len(days),
            "short_days": len(short_days(days, min_shortfall)),
            "short_since": state.short_since.isoformat() if state.short_since else None,
            # The steps declined per inverter: a noisy counter should be
            # visible in the log without a query.
            "ignored_steps": {
                str(inverter_id): {"drops": inverter.drops, "over_bound": inverter.over_bound}
                for inverter_id, inverter in yields.items()
            },
        },
    )
