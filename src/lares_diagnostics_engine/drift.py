"""Drift measurement: the `drift` kind of the fault list.

Something a device does sits persistently above the healthy level declared
for it. The measurement is a tabular CUSUM (Page 1954) with a **pinned**
reference: the healthy value is declared, never derived from history, so a
fault that has stood for months cannot quietly become the new normal. A
trailing reference would be worse than useless here — against a rolling
window a linear ramp scores the same whatever its slope (#1593), which is
why the z-score this replaces could never have seen a creeping standby.

Three signals walk that same CUSUM, and the fault file says which:

* **standby** — the device's idle draw in mA. A relay that no longer opens,
  not a savings topic.
* **duty_cycle** — the share of the day a compressor runs, in percent. An
  iced-up evaporator buys the same cold with far more running, months
  before anything in the freezer thaws.
* **recovery** — the heat-recovery efficiency of an air exchanger, in
  percent of the extract–outdoor gradient. The one signal that walks
  *downward*: fouling lowers what the exchanger can do, so the deficit
  below healthy accumulates instead of the excess above it.

Three steps, each its own function:

1. **The level** — the trailing window's reading of the signal. Standby is
   the *lowest* hourly floor of the trailing day (`standby_floors`): an
   appliance's hourly floor is its idle draw only in the hours it does not
   run. Duty cycle is the share of the trailing day the compressor actually
   drew above the declared on-threshold (`duty_cycles`), with door events
   cut out first. A window too thinly covered to hold a reading yields no
   sample.
2. **The accumulation** — every hour the level sits more than the declared
   rise above healthy adds its excess to a budget; an hour back inside the
   band starts the count over (`accumulate`). The rise is a floor, not a
   noise band: a permanent step smaller than it never accumulates, however
   long it stands, which is what keeps the sentence's "more than 40 mA"
   honest — the Vorratsraum dehumidifier's real +10 mA step would otherwise
   fire after a few months of standing still.
3. **The observations** — once the budget is spent, every further hour is
   an observation for the episode pipeline (`drift_observations`). The
   score is the excess in units of the declared rise, so magnitude and
   persistence stay separate: the budget decides *whether* it fires, the
   score says *how far* past the line it sits.

State is two floats and is recomputed from the aggregate on every run,
never persisted — a redeploy cannot corrupt or lose it, and replaying a
window is the same run twice. Time is the aggregate's own frontier, as
everywhere else, so a stalled materialization freezes the picture instead
of clearing episodes nobody fixed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from math import ceil
from typing import TYPE_CHECKING, Any

from .episodes import Observation
from .faults import ExchangerRoles
from .reconcile import Measured, Window, measurement_reaches
from .runs import split_runs
from .silence import BUCKET, hourly_averages, like_match, pair_by_match, resolve_scope
from .slug import entity_slug

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    import psycopg
    from psycopg.rows import DictRow

    from .episode_store import OpenEpisodeRow
    from .faults import DeviceReference, Fault
    from .silence import Channel


@dataclass(frozen=True, slots=True)
class Device:
    """One monitored appliance: its Stromwert channel and the healthy level
    declared for it, in the signal's own unit."""

    ga: str
    name: str
    label: str
    healthy: float


# A telegram's value stands until the next one, but no longer than this: a
# longer gap is the bridge missing, not a compressor running for hours, and
# silence is its own fault.
MAX_HOLD = BUCKET
# How far back the duty-cycle signal can be measured: it reads the bus
# archive itself rather than an aggregate, and this is what a scheduled run
# reads of it every hour. A longer window would pull more raw telegrams than
# anything in production does, on a database with a gigabyte of memory.
ARCHIVE_HISTORY = timedelta(days=30)
# An hour telegrams account for less than half of is a delivery gap, not a
# reading.
MIN_COVERAGE = 0.5
# ... and one with less than a minute of idle in it counts as never idle:
# every appliance channel sends cyclically even standing still, so a truly
# running hour reads as a clean 1.0.
IDLE_TOLERANCE = timedelta(minutes=1)


@dataclass(frozen=True, slots=True)
class OnTime:
    """One hour of a device's on-time: how much of the hour it drew above the
    on threshold, and how much of the hour telegrams accounted for at all.

    Time, not telegram counts: KNX sends on change, so a running compressor
    emits hundreds of telegrams an hour and an idle one a handful. A share
    of counts would read a freezer running half the day as running almost
    all of it — the pinned reference would then be a number about telegram
    rates instead of about the day.
    """

    time: datetime
    on: timedelta
    total: timedelta

    @property
    def covered(self) -> bool:
        return self.total >= MIN_COVERAGE * BUCKET

    @property
    def saturated(self) -> bool:
        """The compressor never stopped inside this hour — which ordinary
        cycling, at 25 to 46 minutes a run, never does."""
        return self.covered and self.total - self.on <= IDLE_TOLERANCE


@dataclass(frozen=True, slots=True)
class Step:
    """One bucket of the CUSUM walk: the level there, how far it sits above
    healthy, the budget spent so far, and when the current accumulation
    began.
    """

    time: datetime
    level: float
    excess: float
    budget_used: float
    since: datetime | None = None


@dataclass(frozen=True, slots=True)
class DeviceState:
    """The device at the frontier: what its level reads now, how far that is
    above healthy, and since when it has been climbing."""

    device: Device
    level: float | None = None
    excess: float | None = None
    rising_since: datetime | None = None


def resolve_devices(
    channels: Sequence[Channel], references: Sequence[DeviceReference]
) -> list[Device]:
    """The declared references married to the scoped channels — never a
    silently unmeasured device."""
    return [
        Device(
            ga=channel.ga,
            name=channel.name,
            label=reference.match,
            healthy=reference.healthy,
        )
        for channel, reference in pair_by_match(channels, references, noun="reference")
    ]


def min_window_samples(window: timedelta, fraction: float) -> int:
    """How many hourly buckets a trailing `window` must carry before its
    reading counts: the fault declares the fraction, the aggregate's hourly
    grid turns it into a count.
    """
    return ceil(fraction * (window / BUCKET))


def trailing_windows(
    times: Sequence[datetime], *, window: timedelta, min_samples: int
) -> Iterator[tuple[datetime, slice]]:
    """Each bucket's trailing `window` as a slice into the series, including
    the bucket itself — the shape both signals read their level off.

    A window carrying fewer than `min_samples` buckets yields nothing at
    all: the device may simply have run through the day, and a reading taken
    off three hours is not one. That also swallows the first hours of any
    series, which have no day behind them yet.
    """
    start = 0
    for end, time in enumerate(times):
        while times[start] <= time - window:
            start += 1
        if end - start + 1 >= min_samples:
            yield time, slice(start, end + 1)


def standby_floors(
    buckets: Sequence[tuple[datetime, float]],
    *,
    window: timedelta,
    min_samples: int,
) -> list[tuple[datetime, float]]:
    """The device's standby valley at each bucket: the lowest hourly floor
    in the trailing `window`.
    """
    times = [time for time, _ in buckets]
    return [
        (time, min(v for _, v in buckets[span]))
        for time, span in trailing_windows(times, window=window, min_samples=min_samples)
    ]


def door_hours(buckets: Sequence[OnTime], *, door_run: timedelta) -> frozenset[datetime]:
    """The hours a door-open event covers: at least `door_run` in a row in
    which the compressor never dropped below the on-threshold.

    A door left open is the one thing that looks like an iced evaporator and
    is not — both make the compressor run far more than it should — so it is
    cut out of the series rather than scored. They are told apart by shape,
    not by duration alone: ordinary cycling leaves an idle sample in every
    hour even when a warm load stretches its runs, while a door ajar has the
    compressor running through hours on end. That is also why the hourly
    grid suffices here, where the runtime fault could not use it at all.

    Those events are Basalte's fault to report, on the door addresses it
    owns; here they only have to stop counting as ice.
    """
    saturated = [b.time for b in buckets if b.saturated]
    hours: set[datetime] = set()
    for run in split_runs(saturated, BUCKET):
        if run.duration < door_run:
            continue
        time = run.start
        while time <= run.end:
            hours.add(time)
            time += BUCKET
    return frozenset(hours)


def duty_cycles(
    buckets: Sequence[OnTime],
    *,
    window: timedelta,
    min_samples: int,
    excluded: frozenset[datetime],
) -> list[tuple[datetime, float]]:
    """The compressor's duty cycle at each covered bucket, in percent: its
    on-time over the trailing `window`, counting only hours that are neither
    a delivery gap nor part of a door event.

    The series stays hourly even across a door event — the level is read
    *at* every covered hour, only never *from* an excluded one. A hole would
    be worse than a wrong number here: an observation gap longer than the
    episode policy's tolerance splits one incident into one episode a day,
    and a freezer whose door is used every afternoon would clear and re-fire
    its address daily instead of standing as the one situation it is.

    Leaving the door hours out of the coverage too is what keeps that
    honest: a day that spent most of itself with the door open drops out
    under `min_samples` instead of reporting a reading taken around it.
    """
    kept = [b for b in buckets if b.covered and b.time not in excluded]
    levels: list[tuple[datetime, float]] = []
    first, past = 0, 0
    for bucket in buckets:
        if not bucket.covered:
            continue
        while past < len(kept) and kept[past].time <= bucket.time:
            past += 1
        while first < past and kept[first].time <= bucket.time - window:
            first += 1
        span = kept[first:past]
        if len(span) < min_samples:
            continue
        on = sum((b.on for b in span), timedelta())
        total = sum((b.total for b in span), timedelta())
        levels.append((bucket.time, 100.0 * (on / total)))
    return levels


def reaches_frontier(
    levels: Sequence[tuple[datetime, float]], *, frontier: datetime, max_gap: timedelta
) -> bool:
    """The shared `dataless` test read off the level series, which is where
    a drift measurement ends: a device that ran through the last day, or
    sent too thinly to hold a reading, is unmeasured rather than well.
    """
    return measurement_reaches(
        levels[-1][0] if levels else None, frontier=frontier, max_gap=max_gap
    )


def accumulate(
    levels: Sequence[tuple[datetime, float]], *, healthy: float, rise: float
) -> list[Step]:
    """The CUSUM walk over the levels: `S = S + (excess - rise)` while the
    level sits more than `rise` above `healthy`, back to zero the moment it
    returns into the band.

    Zeroing on return is what keeps an episode honest about the present: a
    plain CUSUM decays only at `rise` per hour, so a repaired device would
    go on reporting for days on nothing but accumulated history. The cost is
    that a fault flickering in and out of the band never fills its budget —
    which is the reading the sentence asks for, since such a device is not
    *persistently* high.

    Because the level is a trailing window, one in-band sample means the
    device really did read healthy at some point in the last day, and the
    count is right to start over: the budget only ever runs while the device
    did not reach healthy once in a whole day. The flip side is that a
    single low reading shadows the next 24 h, so a stuck relay that briefly
    drops out is reported a day later, not never.

    An hour without a level contributes nothing: the budget counts hours the
    device was measurably high, never hours nobody looked.
    """
    trace: list[Step] = []
    budget_used = 0.0
    since: datetime | None = None
    for time, level in levels:
        excess = level - healthy
        if excess <= rise:
            budget_used = 0.0
            since = None
        else:
            budget_used += excess - rise
            since = since or time
        trace.append(
            Step(
                time=time,
                level=level,
                excess=excess,
                budget_used=budget_used,
                since=since,
            )
        )
    return trace


def drift_observations(
    subject: str, trace: Sequence[Step], *, rise: float, budget: float
) -> list[Observation]:
    """One observation per hour the spent budget stands past the declared
    one, for the episode pipeline. The score is the excess in units of the
    declared rise — the fault's own unit — and the value is that excess in
    the signal's unit, the number a human acts on.
    """
    return [
        Observation(subject=subject, time=s.time, score=s.excess / rise, value=s.excess)
        for s in trace
        if s.budget_used > budget
    ]


def classify(device: Device, trace: Sequence[Step], *, frontier: datetime) -> DeviceState:
    """What the device reads now — the payload's side of the severity. A
    trace that does not reach the frontier says nothing about now.

    `rising_since` is bounded by the replay window: a drift older than the
    lookback reports the window's own start, which advances with it. The
    episode's `started_at` in the database is the stable onset.
    """
    if not trace or trace[-1].time != frontier:
        return DeviceState(device)
    last = trace[-1]
    return DeviceState(
        device=device,
        level=last.level,
        excess=last.excess,
        rising_since=last.since,
    )


@dataclass(frozen=True, slots=True)
class DevicePublish:
    """One device whose severity moved — the payload names the device, what
    it reads, what it should read, and since when it has been high; the
    writer rule carries only the severity to the device's fault address.
    """

    ga: str
    severity: int
    device: str
    name: str
    level: float | None
    healthy: float | None
    excess: float | None
    rising_since: datetime | None

    @property
    def subject(self) -> str:
        return self.ga

    @property
    def entity(self) -> str:
        # Raw GA; the NATS adapter owns the dialect and slugs it there.
        return self.ga


def publish_for(subject: str, severity: int, state: DeviceState | None) -> DevicePublish:
    # A device that left the scope while its row was open still gets its
    # clear; the payload then only names the address.
    if state is None:
        return DevicePublish(
            ga=subject,
            severity=severity,
            device=subject,
            name=subject,
            level=None,
            healthy=None,
            excess=None,
            rising_since=None,
        )
    return DevicePublish(
        ga=subject,
        severity=severity,
        device=state.device.label,
        name=state.device.name,
        level=state.level,
        healthy=state.device.healthy,
        excess=state.excess,
        rising_since=state.rising_since,
    )


def payload_standby(publish: DevicePublish) -> dict[str, Any]:
    """What the standby signal says on the bus: what the device idles at
    against what it should — fields and wire names in one place."""
    return {
        "device": publish.device,
        "ga": publish.ga,
        "name": publish.name,
        "standby_ma": publish.level,
        "healthy_ma": publish.healthy,
        "excess_ma": publish.excess,
        "rising_since": publish.rising_since,
    }


def payload_duty_cycle(publish: DevicePublish) -> dict[str, Any]:
    """What the duty-cycle signal says on the bus: how much of the day the
    compressor runs against how much it should — the mail's number for
    "the freezer is icing up"."""
    return {
        "device": publish.device,
        "ga": publish.ga,
        "name": publish.name,
        "duty_pct": publish.level,
        "healthy_pct": publish.healthy,
        "excess_pct": publish.excess,
        "rising_since": publish.rising_since,
    }


def hourly_floors(
    conn: psycopg.Connection[DictRow], gas: Sequence[str], window_start: datetime
) -> dict[str, list[tuple[datetime, float]]]:
    """The devices' hourly idle floors over the window, one query for the
    whole scope — 21 appliances, not 2500 channels."""
    rows = conn.execute(
        """
        SELECT ga, bucket, min(idle_floor) AS idle_floor FROM knx_appliance_1h
        WHERE ga = ANY(%(gas)s) AND bucket >= %(start)s AND idle_floor IS NOT NULL
        -- One row per hour: the aggregate also groups by knx_name, so a
        -- channel renamed mid-hour would otherwise be counted twice.
        GROUP BY ga, bucket
        ORDER BY ga, bucket
        """,
        {"gas": list(gas), "start": window_start},
    ).fetchall()
    series: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    for row in rows:
        series[row["ga"]].append((row["bucket"], float(row["idle_floor"])))
    return dict(series)


def hourly_on_time(
    conn: psycopg.Connection[DictRow], gas: Sequence[str], window: Window, threshold: float
) -> dict[str, list[OnTime]]:
    """The devices' hourly on-time over the window, from the bus archive.

    The appliance aggregate cannot answer this: it counts telegrams, and a
    running compressor sends two orders of magnitude more of them than an
    idle one. So the archive is read the way KNX means it — each telegram's
    value stands until the next — and the holds are summed per hour. The
    query is indexed on `ga` (the compression segment) and bounded on both
    sides, and it aggregates in the database: what comes back is one row per
    hour per device, not the telegrams themselves.

    Buckets past the aggregate's frontier are left out, so this fault ends
    its window where its siblings end theirs.
    """
    rows = conn.execute(
        """
        WITH held AS (
            SELECT ga, time, value,
                   least(
                       lead(time) OVER (PARTITION BY ga ORDER BY time) - time,
                       %(max_hold)s
                   ) AS hold
            FROM knx
            WHERE ga = ANY(%(gas)s) AND time >= %(start)s AND time < %(end)s
        )
        SELECT ga, time_bucket('1 hour', time) AS bucket,
               coalesce(sum(hold) FILTER (WHERE value > %(threshold)s), '0'::interval)
                   AS on_time,
               sum(hold) AS total_time
        FROM held WHERE hold IS NOT NULL
        GROUP BY ga, bucket
        ORDER BY ga, bucket
        """,
        {
            "gas": list(gas),
            "start": window.start,
            "end": window.frontier + BUCKET,
            "max_hold": MAX_HOLD,
            "threshold": threshold,
        },
    ).fetchall()
    series: dict[str, list[OnTime]] = defaultdict(list)
    for row in rows:
        series[row["ga"]].append(
            OnTime(time=row["bucket"], on=row["on_time"], total=row["total_time"])
        )
    return dict(series)


def _trailing_window(fault: Fault, window: Window) -> tuple[timedelta, int]:
    """The declared trailing window as an interval and a bucket count."""
    trailing = timedelta(hours=float(fault.parameters["window_hours"]))
    if trailing > window.lookback:
        # A window the lookback cannot cover measures nothing, forever, and
        # would pin every open episode open. Fail the way a bad reference does.
        raise ValueError(
            f"fault {fault.name}: window_hours exceeds the {window.lookback.days}-day lookback"
        )
    return trailing, min_window_samples(
        trailing, float(fault.parameters["min_window_fraction"])
    )


def _devices(conn: psycopg.Connection[DictRow], fault: Fault) -> list[Device]:
    # A reference without a channel or a channel without a reference fails
    # the run loudly — never a silently unmeasured device.
    return resolve_devices(resolve_scope(conn, fault.channel_scope()), fault.references)


def _walk(
    devices: Sequence[Device],
    levels: Mapping[str, Sequence[tuple[datetime, float]]],
    *,
    window: Window,
    rise: float,
    budget: float,
    record: Mapping[str, Any],
) -> Measured[DeviceState]:
    """The part both signals share: CUSUM walk, state and observations per
    device, once the levels are read."""
    states: dict[str, DeviceState] = {}
    observations: list[Observation] = []
    dataless: set[str] = set()
    for device in devices:
        series = levels.get(device.ga, ())
        if not reaches_frontier(
            series, frontier=window.frontier, max_gap=window.policy.max_gap
        ):
            dataless.add(device.ga)
        trace = accumulate(series, healthy=device.healthy, rise=rise)
        states[device.ga] = classify(device, trace, frontier=window.frontier)
        observations.extend(drift_observations(device.ga, trace, rise=rise, budget=budget))

    return Measured(
        states=states,
        observations=tuple(observations),
        dataless=frozenset(dataless),
        record={
            "devices": len(devices),
            "high": sum(1 for s in states.values() if s.excess is not None and s.excess > rise),
            **record,
        },
        labels={d.ga: d.label for d in devices},
    )


def measure_standby(
    conn: psycopg.Connection[DictRow],
    fault: Fault,
    window: Window,
    _open_rows: Sequence[OpenEpisodeRow],
) -> Measured[DeviceState]:
    """The standby signal: the declared references married to the scope,
    then valley, CUSUM walk and observations per device.
    """
    trailing, min_samples = _trailing_window(fault, window)
    devices = _devices(conn, fault)
    series = hourly_floors(conn, [d.ga for d in devices], window.start)
    levels = {
        device.ga: standby_floors(
            series.get(device.ga, []), window=trailing, min_samples=min_samples
        )
        for device in devices
    }
    return _walk(
        devices,
        levels,
        window=window,
        rise=float(fault.parameters["rise_ma"]),
        budget=float(fault.parameters["budget_ma_h"]),
        record={},
    )


def measure_duty_cycle(
    conn: psycopg.Connection[DictRow],
    fault: Fault,
    window: Window,
    _open_rows: Sequence[OpenEpisodeRow],
) -> Measured[DeviceState]:
    """The duty-cycle signal: the same walk over the share of the day the
    compressor runs, with door events cut out of the series first.
    """
    trailing, min_samples = _trailing_window(fault, window)
    door_run = timedelta(hours=float(fault.parameters["door_run_hours"]))
    devices = _devices(conn, fault)
    series = hourly_on_time(
        conn, [d.ga for d in devices], window, float(fault.parameters["on_ma"])
    )
    doors = {
        device.ga: door_hours(series.get(device.ga, []), door_run=door_run)
        for device in devices
    }
    levels = {
        device.ga: duty_cycles(
            series.get(device.ga, []),
            window=trailing,
            min_samples=min_samples,
            excluded=doors[device.ga],
        )
        for device in devices
    }
    return _walk(
        devices,
        levels,
        window=window,
        rise=float(fault.parameters["rise_pct"]),
        budget=float(fault.parameters["budget_pct_h"]),
        # The hours cut out as door events — the run record says how much of
        # the window the fault deliberately did not look at. A device that
        # loses its whole window to them lands in `dataless`, where a
        # compressor running through days on end belongs: at this
        # resolution it is indistinguishable from a door standing open,
        # and Basalte owns that one.
        record={"door_hours": sum(len(hours) for hours in doors.values())},
    )


@dataclass(frozen=True, slots=True)
class Exchanger:
    """The declared heat exchanger: its three air-temperature channels and
    the healthy recovery efficiency declared for it, in percent of the
    extract–outdoor gradient."""

    label: str
    outdoor_ga: str
    extract_ga: str
    supply_ga: str
    healthy: float

    @property
    def slug(self) -> str:
        """Episode subject and NATS entity in one: the writer rules pin
        `fault.<fault>.<slug>` to the exchanger's fault address."""
        return entity_slug(self.label)

    @property
    def gas(self) -> tuple[str, str, str]:
        return (self.outdoor_ga, self.extract_ga, self.supply_ga)


@dataclass(frozen=True, slots=True)
class ExchangerState:
    """The exchanger at the frontier: what it recovers now, how far that is
    below healthy, and since when it has been falling."""

    exchanger: Exchanger
    efficiency: float | None = None
    deficit: float | None = None
    falling_since: datetime | None = None


def resolve_exchanger(
    channels: Sequence[Channel],
    references: Sequence[DeviceReference],
    roles: ExchangerRoles,
) -> Exchanger:
    """The declared roles married to the scoped channels, strictly both
    ways: every role resolves exactly one channel, every scoped channel
    belongs to exactly one role, and exactly one reference names the
    exchanger. Every problem is reported at once.
    """
    problems: list[str] = []
    if len(references) != 1:
        problems.append(
            f"{len(references)} references declared — "
            f"a recovery fault declares exactly one exchanger"
        )
    by_role: dict[str, list[Channel]] = {}
    claimed: dict[str, str] = {}
    for role, pattern in (
        ("outdoor", roles.outdoor),
        ("extract", roles.extract),
        ("supply", roles.supply),
    ):
        hits = [c for c in channels if like_match(pattern, c.name)]
        by_role[role] = hits
        if len(hits) != 1:
            gas = ", ".join(c.ga for c in hits)
            problems.append(
                f"role {role} matches no channel in scope"
                if not hits
                else f"role {role} matches {len(hits)} channels: {gas}"
            )
        for channel in hits:
            if channel.ga in claimed:
                problems.append(
                    f"channel {channel.ga} matched by roles {claimed[channel.ga]} and {role}"
                )
            claimed[channel.ga] = role
    problems.extend(
        f"channel {c.ga} ({c.name}) matches no role"
        for c in channels
        if c.ga not in claimed
    )
    if problems:
        raise ValueError("exchanger roles do not fit the scope: " + "; ".join(problems))
    return Exchanger(
        label=references[0].match,
        outdoor_ga=by_role["outdoor"][0].ga,
        extract_ga=by_role["extract"][0].ga,
        supply_ga=by_role["supply"][0].ga,
        healthy=references[0].healthy,
    )


def efficiency_series(
    exchanger: Exchanger,
    by_ga: Mapping[str, Mapping[datetime, float]],
    window_start: datetime,
    frontier: datetime,
    *,
    min_delta: float,
) -> list[tuple[datetime, float]]:
    """The exchanger's hourly recovery efficiency: how much of the
    extract–outdoor gradient the supply air keeps, in percent.

    Each role's last seen hourly average is carried forward across silent
    buckets — KNX channels are state, not samples — deliberately without a
    staleness bound: a role dying mid-window freezes the picture, and the
    silence fault owns the dead channel. Hours where the gradient is inside
    `min_delta` measure nothing: the quotient would be noise over noise,
    and in summer that is every hour of the day.
    """
    carry: dict[str, float] = {}
    etas: list[tuple[datetime, float]] = []
    t = window_start
    while t <= frontier:
        for ga in exchanger.gas:
            value = by_ga.get(ga, {}).get(t)
            if value is not None:
                carry[ga] = value
        outdoor = carry.get(exchanger.outdoor_ga)
        extract = carry.get(exchanger.extract_ga)
        supply = carry.get(exchanger.supply_ga)
        if outdoor is not None and extract is not None and supply is not None:
            delta = extract - outdoor
            if delta >= min_delta:
                etas.append((t, 100.0 * (supply - outdoor) / delta))
        t += BUCKET
    return etas


def capability_levels(
    etas: Sequence[tuple[datetime, float]],
    *,
    window_start: datetime,
    frontier: datetime,
    window: timedelta,
    min_samples: int,
) -> list[tuple[datetime, float]]:
    """The exchanger's capability at each bucket: the best valid hour in the
    trailing `window` — the mirror of the standby valley. Bypass, defrost
    and boost transients can only lower single hours; fouling lowers the
    best one, which is why the maximum is the honest reading of what the
    exchanger can still do.

    The level is read *at* every bucket of the hourly grid but only *from*
    the valid hours, so a stretch of small gradients inside the window does
    not split one decay into an episode per cold night. A window carrying
    fewer than `min_samples` valid hours yields nothing at all — a summer
    week is unmeasured, not healthy.
    """
    levels: list[tuple[datetime, float]] = []
    first, past = 0, 0
    t = window_start
    while t <= frontier:
        while past < len(etas) and etas[past][0] <= t:
            past += 1
        while first < past and etas[first][0] <= t - window:
            first += 1
        span = etas[first:past]
        if len(span) >= min_samples:
            levels.append((t, max(v for _, v in span)))
        t += BUCKET
    return levels


def accumulate_fall(
    levels: Sequence[tuple[datetime, float]], *, healthy: float, fall: float
) -> list[Step]:
    """The CUSUM walk downward: a decay is a rise of the negated series, so
    the same walk runs upside down and `excess` reads as the deficit below
    healthy. Everything `accumulate` guarantees — the floor, the reset on
    returning into the band, hours nobody looked contributing nothing —
    holds here word for word, with "above" read as "below".
    """
    trace = accumulate([(t, -v) for t, v in levels], healthy=-healthy, rise=fall)
    return [replace(step, level=-step.level) for step in trace]


def classify_exchanger(
    exchanger: Exchanger, trace: Sequence[Step], *, frontier: datetime
) -> ExchangerState:
    """What the exchanger reads now — the payload's side of the severity. A
    trace that does not reach the frontier says nothing about now.

    `falling_since` is bounded by the replay window, like the devices'
    `rising_since`; the episode's `started_at` in the database is the
    stable onset.
    """
    if not trace or trace[-1].time != frontier:
        return ExchangerState(exchanger)
    last = trace[-1]
    return ExchangerState(
        exchanger=exchanger,
        efficiency=last.level,
        deficit=last.excess,
        falling_since=last.since,
    )


@dataclass(frozen=True, slots=True)
class ExchangerPublish:
    """The exchanger when its severity moved — the payload names what it
    recovers, what it should, and since when it has been falling; the
    writer rule carries only the severity to the fault address."""

    slug: str
    severity: int
    exchanger: str
    efficiency: float | None
    healthy: float | None
    deficit: float | None
    falling_since: datetime | None

    @property
    def subject(self) -> str:
        return self.slug

    @property
    def entity(self) -> str:
        return self.slug


def publish_for_exchanger(
    subject: str, severity: int, state: ExchangerState | None
) -> ExchangerPublish:
    # An exchanger that left the scope while its row was open still gets
    # its clear; the payload then only names the subject.
    if state is None:
        return ExchangerPublish(
            slug=subject,
            severity=severity,
            exchanger=subject,
            efficiency=None,
            healthy=None,
            deficit=None,
            falling_since=None,
        )
    return ExchangerPublish(
        slug=subject,
        severity=severity,
        exchanger=state.exchanger.label,
        efficiency=state.efficiency,
        healthy=state.exchanger.healthy,
        deficit=state.deficit,
        falling_since=state.falling_since,
    )


def payload_recovery(publish: ExchangerPublish) -> dict[str, Any]:
    """What the recovery signal says on the bus: what the exchanger recovers
    against what it should — the number behind "the supply air is
    approaching the outdoor air"."""
    return {
        "exchanger": publish.exchanger,
        "efficiency_pct": publish.efficiency,
        "healthy_pct": publish.healthy,
        "deficit_pct": publish.deficit,
        "falling_since": publish.falling_since,
    }


def measure_recovery(
    conn: psycopg.Connection[DictRow],
    fault: Fault,
    window: Window,
    _open_rows: Sequence[OpenEpisodeRow],
) -> Measured[ExchangerState]:
    """The recovery signal: the declared roles married to the scope, then
    the hourly efficiency, the capability, and the downward walk — one
    subject, the exchanger itself.
    """
    if not isinstance(fault.roles, ExchangerRoles):
        raise ValueError(f"fault {fault.name}: the recovery signal needs declared air roles")
    trailing, min_samples = _trailing_window(fault, window)
    fall = float(fault.parameters["fall_pct"])
    budget = float(fault.parameters["budget_pct_h"])

    exchanger = resolve_exchanger(
        resolve_scope(conn, fault.channel_scope()), fault.references, fault.roles
    )
    series = hourly_averages(conn, exchanger.gas, window.start)
    etas = efficiency_series(
        exchanger,
        series,
        window.start,
        window.frontier,
        min_delta=float(fault.parameters["min_delta_k"]),
    )
    levels = capability_levels(
        etas,
        window_start=window.start,
        frontier=window.frontier,
        window=trailing,
        min_samples=min_samples,
    )
    trace = accumulate_fall(levels, healthy=exchanger.healthy, fall=fall)

    dataless: set[str] = set()
    if not reaches_frontier(levels, frontier=window.frontier, max_gap=window.policy.max_gap):
        dataless.add(exchanger.slug)
    state = classify_exchanger(exchanger, trace, frontier=window.frontier)

    return Measured(
        states={exchanger.slug: state},
        observations=tuple(
            drift_observations(exchanger.slug, trace, rise=fall, budget=budget)
        ),
        dataless=frozenset(dataless),
        record={
            "valid_hours": len(etas),
            "measured_hours": len(levels),
            "falling": int(state.deficit is not None and state.deficit > fall),
        },
        labels={exchanger.slug: exchanger.label},
    )
