"""Consistency measurement: the `consistency` kind of the fault list.

A summary channel should equal the OR of the member channels it stands for,
and does not. `3/0/15`, the Büro's light status, stayed on from 2026-10-23
17:33 until a Basalte restart four days later, while both lights in the room
reported off — and the agent asked whether a light was on anywhere in the
house answered "yes, in the Büro".

Which members a summary stands for is the ETS building tree, read off the
catalog's `space_path`: every member whose space lies at or beneath the
summary's own. A room status stands for the room's lights, a floor status
for every light on the floor, the building's for all of them — one rule at
every level, and no room named in the fault file. Members are always the
member channels themselves, never the summaries below: Basalte computes a
floor's status from the lights, so a hung room status must not make its
floor look wrong too. A channel both roles match is a summary, which also
lets the history be back-tested under the names it was recorded with.

Values are each hour's last value, not its average — a light on for ten
minutes is off when the hour closes, and so is the status that followed
it — carried forward across the hours a channel said nothing in, starting
from the last value it stood on before the window. A status speaks only
when it changes, so its silence is its last word, not missing data. A
summary saying off is contradicted only by a member known to be on; one
saying on only by every member known to be off. So the one hour that
cannot be judged is a summary saying on while no member is known to be on
and one has never been heard from — and a summary with no judged hour is
dataless.

One parameter, `min_hours`: how long a disagreement may stand. One hour
always may — a light switched just before the hour closes leaves its status
a bucket behind. Every hour past it is an observation whose score is the
run length in units of `min_hours`, the fault's declared unit.

Time is the aggregate's frontier, as for the channel kinds. Delivery is on
one declared address for the whole house, through the group rule in
`groups` with the house as the only group: the worst open summary's tier,
re-published whenever that tier or the set of open summaries moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .episodes import Observation
from .faults import SummaryRoles
from .groups import group_plan
from .reconcile import Measured
from .runs import split_runs
from .silence import BUCKET, like_match

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import psycopg
    from psycopg.rows import DictRow

    from .episode_store import OpenEpisodeRow
    from .episodes import Episode
    from .faults import Fault
    from .reconcile import Plan, Window

# The one group every summary is delivered in: the whole house, on one
# declared address.
HOUSE = "house"


@dataclass(frozen=True, slots=True)
class PlacedChannel:
    """One catalog channel with its place in the building tree: the spaces
    from the top down to the one its ETS function sits in. None where the
    catalog cannot place it."""

    ga: str
    name: str
    space_path: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class Summary:
    """One summary channel and the members it stands for — none where its
    space holds no member, which leaves it unmeasured."""

    channel: PlacedChannel
    space_path: tuple[str, ...]
    members: tuple[PlacedChannel, ...]


def resolve_summaries(
    channels: Sequence[PlacedChannel], roles: SummaryRoles
) -> tuple[Summary, ...]:
    """Every summary the roles name, with the members beneath its space,
    both ordered by group address.

    A channel the catalog cannot place fails the run, every one named at
    once: a summary without a place would go unmeasured, a member without
    one would count for no summary, and the repair is a link in ETS.
    """
    summaries = sorted(
        (c for c in channels if like_match(roles.summary, c.name)), key=lambda c: c.ga
    )
    summary_gas = {c.ga for c in summaries}
    members = sorted(
        (c for c in channels if c.ga not in summary_gas and like_match(roles.members, c.name)),
        key=lambda c: c.ga,
    )
    paths = {c.ga: c.space_path for c in (*summaries, *members) if c.space_path is not None}
    unplaced = [c for c in (*summaries, *members) if c.ga not in paths]
    if unplaced:
        named = "; ".join(f"{c.ga} ({c.name})" for c in sorted(unplaced, key=lambda c: c.ga))
        raise ValueError(f"channels without a place in the building tree: {named}")
    return tuple(
        Summary(
            channel=summary,
            space_path=paths[summary.ga],
            members=tuple(m for m in members if _beneath(paths[m.ga], paths[summary.ga])),
        )
        for summary in summaries
    )


def _beneath(path: tuple[str, ...], space: tuple[str, ...]) -> bool:
    """Whether a space is that one or lies inside it."""
    return path[: len(space)] == space


@dataclass(frozen=True, slots=True)
class Agreement:
    """One measurable hour of a summary: what it said, and how many of its
    members said on — each its last value, carried across silent hours."""

    time: datetime
    summary_on: bool
    members_on: int

    @property
    def disagrees(self) -> bool:
        return self.summary_on != (self.members_on > 0)


@dataclass(frozen=True, slots=True)
class ConsistencyState:
    """Whether the summary disagrees with its members at the frontier, since
    when, and what each side says — what the publish names beside the
    severity."""

    summary: Summary
    since: datetime | None = None
    hours: float | None = None
    summary_on: bool | None = None
    members_on: int | None = None


def placed_channels(conn: psycopg.Connection[DictRow], roles: SummaryRoles) -> list[PlacedChannel]:
    """The channels either role names, with their place in the building
    tree — resolved where the catalog lives, never a hand-written list."""
    rows = conn.execute(
        """
        SELECT ga, name, space_path FROM ga_catalog
        WHERE name LIKE %(summary)s OR name LIKE %(members)s
        ORDER BY ga
        """,
        {"summary": roles.summary, "members": roles.members},
    ).fetchall()
    return [
        PlacedChannel(
            ga=r["ga"],
            name=r["name"],
            space_path=tuple(r["space_path"]) if r["space_path"] is not None else None,
        )
        for r in rows
    ]


def hourly_last_values(
    conn: psycopg.Connection[DictRow], gas: Sequence[str], window_start: datetime
) -> dict[str, dict[datetime, float]]:
    """The value each channel stood on at the end of every hour it sent in —
    the end of the hour, not its average: a light on for ten minutes is off
    by the bucket's close, and so is the status that followed it."""
    rows = conn.execute(
        """
        SELECT ga, bucket, last_value FROM knx_1h
        WHERE ga = ANY(%(gas)s) AND bucket >= %(start)s
        ORDER BY ga, bucket
        """,
        {"gas": list(gas), "start": window_start},
    ).fetchall()
    series: dict[str, dict[datetime, float]] = {}
    for row in rows:
        series.setdefault(row["ga"], {})[row["bucket"]] = float(row["last_value"])
    return series


def last_values_before(
    conn: psycopg.Connection[DictRow], gas: Sequence[str], window_start: datetime
) -> dict[str, float]:
    """The value each channel last stood on before the window opened — what
    a light switched on five weeks ago and silent since still says."""
    rows = conn.execute(
        """
        SELECT DISTINCT ON (ga) ga, last_value FROM knx_1h
        WHERE ga = ANY(%(gas)s) AND bucket < %(start)s
        ORDER BY ga, bucket DESC
        """,
        {"gas": list(gas), "start": window_start},
    ).fetchall()
    return {row["ga"]: float(row["last_value"]) for row in rows}


def agreement_series(
    summary: Summary,
    by_ga: Mapping[str, Mapping[datetime, float]],
    before: Mapping[str, float],
    window_start: datetime,
    frontier: datetime,
) -> list[Agreement]:
    """The summary's dense hourly series: its own last value and its
    members', each starting from where it stood before the window and
    carried forward across silent hours. An hour is judged once the summary
    and at least one member are known — except a summary saying on while no
    member is known to be on and another has never been heard from: that
    every member is off needs all of them known."""
    gas = (summary.channel.ga, *(m.ga for m in summary.members))
    carry = {ga: before[ga] for ga in gas if ga in before}
    series: list[Agreement] = []
    t = window_start
    while t <= frontier:
        for ga in gas:
            value = by_ga.get(ga, {}).get(t)
            if value is not None:
                carry[ga] = value
        known = [carry[m.ga] for m in summary.members if m.ga in carry]
        if summary.channel.ga in carry and known:
            summary_on = carry[summary.channel.ga] != 0
            members_on = sum(1 for v in known if v != 0)
            unheard = len(known) < len(summary.members)
            if not (summary_on and members_on == 0 and unheard):
                series.append(Agreement(time=t, summary_on=summary_on, members_on=members_on))
        t += BUCKET
    return series


def consistency_observations(
    summary: Summary, disagreeing: Sequence[Agreement], min_hours: float
) -> list[Observation]:
    """One observation per hour once a disagreement has stood `min_hours`,
    for the episode pipeline. The score is how long it has stood in units
    of `min_hours` — the fault's declared unit; the value is that length in
    hours. An hour that agrees restarts the clock, and a bucket covers its
    full hour, so a run's first bucket already counts as one."""
    observations: list[Observation] = []
    for run in split_runs((a.time for a in disagreeing), BUCKET):
        t = run.start
        while t <= run.end:
            elapsed = (t - run.start + BUCKET) / BUCKET
            if elapsed >= min_hours:
                observations.append(
                    Observation(
                        subject=summary.channel.ga,
                        time=t,
                        score=elapsed / min_hours,
                        value=elapsed,
                    )
                )
            t += BUCKET
    return observations


def classify(summary: Summary, series: Sequence[Agreement], frontier: datetime) -> ConsistencyState:
    """The disagreement the summary stands in at the frontier, if any —
    what the publish payload names alongside the severity."""
    if not series or series[-1].time != frontier:
        return ConsistencyState(summary)
    latest = series[-1]
    runs = split_runs((a.time for a in series if a.disagrees), BUCKET)
    run = runs[-1] if latest.disagrees and runs else None
    return ConsistencyState(
        summary,
        since=run.start if run is not None else None,
        hours=run.duration / BUCKET if run is not None else None,
        summary_on=latest.summary_on,
        members_on=latest.members_on,
    )


def measure(
    conn: psycopg.Connection[DictRow],
    fault: Fault,
    window: Window,
    _open_rows: Sequence[OpenEpisodeRow],
) -> Measured[ConsistencyState]:
    """The kind's whole measurement: both roles resolved where the catalog
    lives, every summary married to the members beneath it, then the dense
    series, the disagreeing hours and the observations per summary."""
    if not isinstance(fault.roles, SummaryRoles):
        raise ValueError(f"fault {fault.name}: the consistency kind needs its summary roles")
    min_hours = float(fault.parameters["min_hours"])

    summaries = resolve_summaries(placed_channels(conn, fault.roles), fault.roles)
    measurable = [s for s in summaries if s.members]
    gas = sorted({ga for s in measurable for ga in (s.channel.ga, *(m.ga for m in s.members))})
    series = hourly_last_values(conn, gas, window.start)
    before = last_values_before(conn, gas, window.start)

    states: dict[str, ConsistencyState] = {}
    observations: list[Observation] = []
    dataless: set[str] = set()
    for summary in measurable:
        agreements = agreement_series(summary, series, before, window.start, window.frontier)
        if not window.reaches(agreements[-1].time if agreements else None):
            dataless.add(summary.channel.ga)
        disagreeing = [a for a in agreements if a.disagrees]
        states[summary.channel.ga] = classify(summary, agreements, window.frontier)
        observations.extend(consistency_observations(summary, disagreeing, min_hours))

    return Measured(
        states=states,
        observations=tuple(observations),
        dataless=frozenset(dataless),
        record={
            "summaries": len(summaries),
            "unmeasured": [s.channel.name for s in summaries if not s.members],
            "dataless": sorted(states[ga].summary.channel.name for ga in dataless),
            "disagreeing": sum(1 for s in states.values() if s.since is not None),
        },
        labels={s.channel.ga: s.channel.name for s in summaries},
    )


@dataclass(frozen=True, slots=True)
class SummaryReport:
    """One disagreeing summary inside the house-wide publish — which status,
    in which space, since when, and what each side says."""

    ga: str
    name: str
    space: str | None
    severity: int
    since: datetime | None
    hours: float | None
    summary_on: bool | None
    members_on: int | None
    member_count: int | None

    @property
    def magnitude(self) -> float | None:
        """How bad this summary is, in the fault's declared unit — how long
        it has disagreed, which orders the publish's list."""
        return self.hours


def report_for(
    subject: str, severity: int, state: ConsistencyState | None, _episode: Episode | None
) -> SummaryReport:
    """One summary's line in the house-wide publish. A subject held open for
    want of data has no state this run, so its report names the address and
    the tier it still carries, and nothing it cannot know."""
    if state is None:
        return SummaryReport(
            ga=subject,
            name=subject,
            space=None,
            severity=severity,
            since=None,
            hours=None,
            summary_on=None,
            members_on=None,
            member_count=None,
        )
    return SummaryReport(
        ga=subject,
        name=state.summary.channel.name,
        space=state.summary.space_path[-1],
        severity=severity,
        since=state.since,
        hours=state.hours,
        summary_on=state.summary_on,
        members_on=state.members_on,
        member_count=len(state.summary.members),
    )


@dataclass(frozen=True, slots=True)
class HousePublish:
    """The house-wide severity — the worst of every open summary — and the
    summaries that earned it, worst first."""

    severity: int
    summaries: tuple[SummaryReport, ...]

    @property
    def subject(self) -> str:
        return HOUSE

    @property
    def entity(self) -> None:
        # One declared address: a 1:1 subject with no entity token.
        return None


def plan_run(
    *,
    episodes: Sequence[Episode],
    open_rows: Sequence[OpenEpisodeRow],
    measured: Measured[ConsistencyState],
    frontier: datetime,
) -> Plan[HousePublish]:
    """The shared group delivery with the whole house as its one group: it
    publishes when the worst open summary's tier moved or the set of open
    summaries changed — one summary ending never clears another."""
    return group_plan(
        episodes=episodes,
        open_rows=open_rows,
        measured=measured,
        frontier=frontier,
        report_for=report_for,
        group_of=_house,
        publish_for=_house_publish,
    )


def _house(_subject: str) -> str:
    return HOUSE


def _house_publish(_group: str, severity: int, reports: tuple[SummaryReport, ...]) -> HousePublish:
    return HousePublish(severity, reports)


def payload(publish: HousePublish) -> dict[str, Any]:
    """What this kind says on the bus: how many summaries disagree with
    their members, which, and what each side says — fields and wire names
    in one place."""
    return {
        "disagreeing_summaries": len(publish.summaries),
        "summaries": [
            {
                "ga": report.ga,
                "name": report.name,
                "space": report.space,
                "severity": report.severity,
                "since": report.since,
                "hours": report.hours,
                "summary_on": report.summary_on,
                "members_on": report.members_on,
                "member_count": report.member_count,
            }
            for report in publish.summaries
        ],
    }
