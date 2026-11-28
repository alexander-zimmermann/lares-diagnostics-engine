"""Consistency tests — the `consistency` kind against an invented building.

Each test hands the real computation what the catalog and `knx_1h` would:
channels placed in the ETS building tree, and hourly last values. What is
asserted is only what comes out — which members a summary stands for, the
observations for the episode pipeline, the state the payload is shaped
from, and the one house-wide publish. The spec fixture is the Büro hang of
2026-10-23: the room status stayed on for days after both lights went off.
No cluster, no live database.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

import pytest

from lares_diagnostics_engine.consistency import (
    ConsistencyState,
    HousePublish,
    PlacedChannel,
    Summary,
    measure,
    plan_run,
    resolve_summaries,
)
from lares_diagnostics_engine.episode_store import OpenEpisodeRow
from lares_diagnostics_engine.episodes import (
    Episode,
    EpisodePolicy,
    EventKind,
    EvidenceRow,
    NotificationEvent,
)
from lares_diagnostics_engine.faults import SummaryRoles, parse_entry
from lares_diagnostics_engine.reconcile import Measured, Plan, Window

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import psycopg
    from psycopg.rows import DictRow

_HOUR = timedelta(hours=1)
_FRONTIER = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)

_GEBAEUDE = ("Anwesen", "Gebäude")
_EG = (*_GEBAEUDE, "Erdgeschoss (EG)")
_OG = (*_GEBAEUDE, "Obergeschoss (OG)")
_BUERO = (*_EG, "Büro (E3)")
_WOHNZIMMER = (*_EG, "Wohnzimmer (E4)")
_FLUR_OG = (*_OG, "Flur (O1)")
_GARTEN = ("Anwesen", "Garten")

_ROLES = SummaryRoles(
    summary="Allgemein.%.Beleuchtung-Status", members="Beleuchtung.%.Ein/Aus-Status"
)

_HOUSE_STATUS = PlacedChannel("0/0/183", "Allgemein.Zentral.Gebäude.Beleuchtung-Status", _GEBAEUDE)
_EG_STATUS = PlacedChannel("0/0/113", "Allgemein.Zentral.EG.Beleuchtung-Status", _EG)
_GARDEN_STATUS = PlacedChannel("0/0/203", "Allgemein.Zentral.Garten.Beleuchtung-Status", _GARTEN)
_BUERO_STATUS = PlacedChannel("0/2/43", "Allgemein.EG.Büro.Beleuchtung-Status", _BUERO)
_SPOTS = PlacedChannel("3/2/62", "Beleuchtung.Gebäude.EG.Büro.Spots.Ein/Aus-Status", _BUERO)
_STRIPE = PlacedChannel("3/2/72", "Beleuchtung.Gebäude.EG.Büro.Stripe.Ein/Aus-Status", _BUERO)
_SOFA = PlacedChannel(
    "4/2/30", "Beleuchtung.Einrichtung.EG.Wohnzimmer.Sofa.Ein/Aus-Status", _WOHNZIMMER
)
_FLUR_LIGHT = PlacedChannel(
    "3/3/10", "Beleuchtung.Gebäude.OG.Flur.Downlight.Ein/Aus-Status", _FLUR_OG
)

_BUILDING = [
    _HOUSE_STATUS,
    _EG_STATUS,
    _GARDEN_STATUS,
    _BUERO_STATUS,
    _SPOTS,
    _STRIPE,
    _SOFA,
    _FLUR_LIGHT,
]


def _members(channels: list[PlacedChannel], roles: SummaryRoles = _ROLES) -> dict[str, list[str]]:
    return {s.channel.ga: [m.ga for m in s.members] for s in resolve_summaries(channels, roles)}


class TestResolveSummaries:
    def test_a_summary_stands_for_every_member_beneath_its_space(self) -> None:
        # The room for its own lights, the floor for every room on it, the
        # building for every floor — one rule at each level of the tree.
        assert _members(_BUILDING) == {
            "0/0/113": ["3/2/62", "3/2/72", "4/2/30"],
            "0/0/183": ["3/2/62", "3/2/72", "3/3/10", "4/2/30"],
            "0/0/203": [],
            "0/2/43": ["3/2/62", "3/2/72"],
        }

    def test_a_channel_both_roles_match_is_a_summary_never_a_member(self) -> None:
        # Before the status moves to Allgemein, summaries and lights share a
        # naming pattern; the room status must not become a floor's light.
        before = SummaryRoles(
            summary="Beleuchtung.Gebäude.Zentral.%.Ein/Aus-Status",
            members="Beleuchtung.%.Ein/Aus-Status",
        )
        floor = PlacedChannel("3/0/111", "Beleuchtung.Gebäude.Zentral.EG.Ein/Aus-Status", _EG)
        room = PlacedChannel("3/0/15", "Beleuchtung.Gebäude.Zentral.EG.Büro.Ein/Aus-Status", _BUERO)
        assert _members([floor, room, _SPOTS, _STRIPE], before) == {
            "3/0/111": ["3/2/62", "3/2/72"],
            "3/0/15": ["3/2/62", "3/2/72"],
        }

    def test_a_channel_the_catalog_cannot_place_fails_naming_every_one(self) -> None:
        # A summary without a place would go unmeasured, a member without
        # one would count for no summary — either is a blind spot, and the
        # repair is a link in ETS, so the run names all of them at once.
        unplaced_status = PlacedChannel("0/1/3", "Allgemein.KG.Flur.Beleuchtung-Status", None)
        unplaced_light = PlacedChannel(
            "3/1/5", "Beleuchtung.Gebäude.KG.Flur.Spots.Ein/Aus-Status", None
        )
        with pytest.raises(ValueError, match=r"0/1/3 .*3/1/5 "):
            resolve_summaries([*_BUILDING, unplaced_status, unplaced_light], _ROLES)


# --- measurement --------------------------------------------------------------

# The two statements the measurement runs, each named by a fragment only it
# carries: a fake that guessed would answer the wrong query with plausible
# rows.
_CATALOG = "FROM ga_catalog"
_LAST_VALUES = "SELECT ga, bucket, last_value FROM knx_1h"
_LAST_BEFORE = "SELECT DISTINCT ON (ga) ga, last_value FROM knx_1h"

_ENTRY = {
    "name": "light_status_consistency",
    "sentence": "Ein Lichtstatus widerspricht seit mindestens 2 h seinen Leuchten.",
    "unit": "× der erlaubten Dauer",
    "kind": "consistency",
    "parameters": {"min_hours": 2},
    "roles": {"summary": _ROLES.summary, "members": _ROLES.members},
    "target": {"ga": "0/0/253"},
}


@dataclass(frozen=True, slots=True)
class _Result:
    rows: Sequence[Mapping[str, Any]]

    def fetchall(self) -> list[Mapping[str, Any]]:
        return list(self.rows)


class _Conn:
    """Enough of a connection for the measurement: each statement answered
    with the rows registered for the fragment that names it, and anything
    unregistered a failure rather than an empty answer."""

    def __init__(self, rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        self._rows = rows

    def execute(self, sql: str, _params: Mapping[str, Any] | None = None) -> _Result:
        matches = [
            rows for fragment, rows in self._rows.items() if fragment in " ".join(sql.split())
        ]
        if len(matches) != 1:
            raise AssertionError(f"{len(matches)} fixtures match this statement: {sql}")
        return _Result(matches[0])


def _catalog(*channels: PlacedChannel) -> list[dict[str, Any]]:
    return [
        {
            "ga": c.ga,
            "name": c.name,
            "space_path": list(c.space_path) if c.space_path is not None else None,
        }
        for c in channels
    ]


def _switched(ga: str, *changes: tuple[int, float]) -> list[dict[str, Any]]:
    """The hourly last values a channel sent: (hours before the frontier,
    value) per bucket it said anything in — silent hours carry no row."""
    return [
        {"ga": ga, "bucket": _FRONTIER - hours * _HOUR, "last_value": value}
        for hours, value in changes
    ]


def _before(ga: str, value: float) -> dict[str, Any]:
    """The value a channel last stood on before the window opened."""
    return {"ga": ga, "last_value": value}


def _measure(
    catalog: Sequence[PlacedChannel],
    *series: list[dict[str, Any]],
    before: Sequence[dict[str, Any]] = (),
) -> Any:
    conn = _Conn(
        {
            _CATALOG: _catalog(*catalog),
            _LAST_VALUES: [r for s in series for r in s],
            _LAST_BEFORE: before,
        }
    )
    window = Window(
        start=_FRONTIER - timedelta(days=30), frontier=_FRONTIER, policy=EpisodePolicy()
    )
    return measure(cast("psycopg.Connection[DictRow]", conn), parse_entry(_ENTRY), window, ())


class TestMeasure:
    def test_a_status_left_on_after_its_lights_went_off_fires_once_it_stood_min_hours(
        self,
    ) -> None:
        # The Büro hang: the Stripe goes off at the Sentido, the room status
        # never follows. Six hours of the status saying on, both lights off.
        measured = _measure(
            [_BUERO_STATUS, _SPOTS, _STRIPE],
            _switched("0/2/43", (10, 1.0)),
            _switched("3/2/62", (10, 0.0)),
            _switched("3/2/72", (10, 1.0), (5, 0.0)),
        )
        scores = [(o.time, o.score) for o in measured.observations]
        assert scores == [(_FRONTIER - h * _HOUR, (6 - h) / 2) for h in range(4, -1, -1)]
        assert {o.subject for o in measured.observations} == {"0/2/43"}
        state = measured.states["0/2/43"]
        assert (state.since, state.hours) == (_FRONTIER - 5 * _HOUR, 6.0)
        assert state.summary_on is True
        assert state.members_on == 0
        assert measured.dataless == frozenset()

    def test_a_status_left_off_while_a_light_is_on_fires_too(self) -> None:
        # The other direction: the light came on, the status never did.
        measured = _measure(
            [_BUERO_STATUS, _SPOTS, _STRIPE],
            _switched("0/2/43", (10, 0.0)),
            _switched("3/2/62", (10, 0.0), (2, 1.0)),
            _switched("3/2/72", (10, 0.0)),
        )
        assert [o.time for o in measured.observations] == [_FRONTIER - _HOUR, _FRONTIER]
        state = measured.states["0/2/43"]
        assert (state.since, state.summary_on, state.members_on) == (
            _FRONTIER - 2 * _HOUR,
            False,
            1,
        )

    def test_a_status_that_follows_into_the_next_hour_does_not_fire(self) -> None:
        # The light goes off just before the hour closes, the status a moment
        # later: one bucket disagrees, which is the aggregate, not a hang.
        measured = _measure(
            [_BUERO_STATUS, _SPOTS],
            _switched("0/2/43", (10, 1.0), (3, 0.0)),
            _switched("3/2/62", (10, 1.0), (4, 0.0)),
        )
        assert measured.observations == ()
        assert measured.states["0/2/43"].since is None

    def test_a_status_whose_lights_never_appeared_is_not_measured_but_dataless(self) -> None:
        # Nothing can be said before a member has spoken: no observation,
        # and an open episode must stay open rather than clear on silence.
        measured = _measure([_BUERO_STATUS, _SPOTS], _switched("0/2/43", (10, 1.0)))
        assert measured.observations == ()
        assert measured.dataless == frozenset({"0/2/43"})
        # Not a warning every hour for a store room nobody entered — but the
        # run record names it.
        assert measured.record["dataless"] == ["Allgemein.EG.Büro.Beleuchtung-Status"]

    def test_a_light_on_since_before_the_window_still_counts_as_on(self) -> None:
        # The Stripe was switched on five weeks ago and has said nothing
        # since; the status that says on is right, not hung.
        measured = _measure(
            [_BUERO_STATUS, _SPOTS, _STRIPE],
            _switched("0/2/43", (10, 1.0)),
            _switched("3/2/62", (10, 0.0)),
            before=[_before("3/2/72", 1.0)],
        )
        assert measured.observations == ()
        assert measured.states["0/2/43"].members_on == 1

    def test_all_off_is_only_concluded_once_every_member_has_spoken(self) -> None:
        # The Stripe has never sent anything the aggregate holds: whether
        # every light is off cannot be said, so the status saying on is not
        # judged — but a member known to be on would settle it.
        measured = _measure(
            [_BUERO_STATUS, _SPOTS, _STRIPE],
            _switched("0/2/43", (10, 1.0)),
            _switched("3/2/62", (10, 0.0)),
        )
        assert measured.observations == ()
        assert measured.dataless == frozenset({"0/2/43"})

    def test_a_status_saying_off_agrees_once_no_member_is_known_to_be_on(self) -> None:
        # Off is contradicted only by a light known to be on: once the Spots
        # go off, the unheard Stripe cannot keep the disagreement alive, and
        # the hour is judged agreeing rather than left unmeasured.
        measured = _measure(
            [_BUERO_STATUS, _SPOTS, _STRIPE],
            _switched("0/2/43", (10, 0.0)),
            _switched("3/2/62", (10, 1.0), (4, 0.0)),
        )
        assert [o.time for o in measured.observations] == [
            _FRONTIER - h * _HOUR for h in (9, 8, 7, 6, 5)
        ]
        assert measured.states["0/2/43"].since is None
        assert measured.dataless == frozenset()

    def test_a_hung_room_status_does_not_make_its_floor_disagree(self) -> None:
        # The floor follows the lights, the room status beneath it hangs: the
        # floor is compared with the lights, so only the room fires.
        measured = _measure(
            [_EG_STATUS, _BUERO_STATUS, _SPOTS, _STRIPE],
            _switched("0/0/113", (10, 1.0), (5, 0.0)),
            _switched("0/2/43", (10, 1.0)),
            _switched("3/2/62", (10, 0.0)),
            _switched("3/2/72", (10, 1.0), (5, 0.0)),
        )
        assert {o.subject for o in measured.observations} == {"0/2/43"}
        assert measured.states["0/0/113"].since is None

    def test_a_status_with_no_member_beneath_it_is_counted_not_measured(self) -> None:
        # The garden status has no light in its own part of the tree: no
        # subject at all, and the run record names it.
        measured = _measure(
            [_GARDEN_STATUS, _BUERO_STATUS, _SPOTS],
            _switched("0/0/203", (10, 1.0)),
            _switched("0/2/43", (10, 0.0)),
            _switched("3/2/62", (10, 0.0)),
        )
        assert set(measured.states) == {"0/2/43"}
        assert measured.dataless == frozenset()
        assert measured.record["unmeasured"] == ["Allgemein.Zentral.Garten.Beleuchtung-Status"]


# --- delivery -----------------------------------------------------------------

_SINCE = _FRONTIER - 6 * _HOUR


def _episode(subject: str, severity: int, *, ended: bool = False) -> Episode:
    events = [NotificationEvent(EventKind.APPEARED, _SINCE, severity)]
    if ended:
        events.append(NotificationEvent(EventKind.ENDED, _FRONTIER, 0))
    return Episode(
        fault="light_status_consistency",
        subject=subject,
        started_at=_SINCE,
        last_seen_at=_SINCE + _HOUR,
        ended_at=_FRONTIER if ended else None,
        severity=severity,
        peak_score=1.0,
        evidence=(EvidenceRow(time=_SINCE, score=1.0, severity=severity, value=2.0),),
        events=tuple(events),
    )


def _disagreeing(summary: PlacedChannel, *members: PlacedChannel) -> ConsistencyState:
    assert summary.space_path is not None
    return ConsistencyState(
        Summary(summary, summary.space_path, members),
        since=_SINCE,
        summary_on=True,
        members_on=0,
    )


_STATES = {
    "0/2/43": _disagreeing(_BUERO_STATUS, _SPOTS, _STRIPE),
    "0/0/113": _disagreeing(_EG_STATUS, _SPOTS, _STRIPE, _SOFA),
}


def _plan(episodes: list[Episode], open_rows: list[OpenEpisodeRow]) -> Plan[HousePublish]:
    measured: Measured[ConsistencyState] = Measured(
        states=_STATES, observations=(), dataless=frozenset(), record={}
    )
    return plan_run(episodes=episodes, open_rows=open_rows, measured=measured, frontier=_FRONTIER)


class TestHousePlan:
    def test_one_summary_ending_does_not_clear_another_still_open(self) -> None:
        # One address for the whole house: the Büro's end must not write the
        # 0 that would hide the floor still standing at warning.
        plan = _plan(
            [_episode("0/2/43", 1, ended=True), _episode("0/0/113", 2)],
            [OpenEpisodeRow(1, "0/2/43", 1), OpenEpisodeRow(2, "0/0/113", 2)],
        )
        (publish,) = plan.publishes
        assert publish.entity is None
        assert publish.severity == 2
        assert [r.ga for r in publish.summaries] == ["0/0/113"]

    def test_a_summary_joining_at_a_tier_already_held_still_publishes(self) -> None:
        # The severity stays, but a payload naming only the floor would now
        # be naming the wrong set.
        plan = _plan(
            [_episode("0/0/113", 2), _episode("0/2/43", 2)],
            [OpenEpisodeRow(2, "0/0/113", 2)],
        )
        (publish,) = plan.publishes
        assert publish.severity == 2
        assert [r.ga for r in publish.summaries] == ["0/0/113", "0/2/43"]
        report = publish.summaries[1]
        assert (report.name, report.space, report.since) == (
            "Allgemein.EG.Büro.Beleuchtung-Status",
            "Büro (E3)",
            _SINCE,
        )
        assert (report.summary_on, report.members_on, report.member_count) == (True, 0, 2)

    def test_nothing_moved_publishes_nothing(self) -> None:
        plan = _plan([_episode("0/2/43", 1)], [OpenEpisodeRow(1, "0/2/43", 1)])
        assert plan.publishes == ()

    def test_the_last_summary_ending_clears_the_address(self) -> None:
        plan = _plan([_episode("0/2/43", 1, ended=True)], [OpenEpisodeRow(1, "0/2/43", 1)])
        (publish,) = plan.publishes
        assert (publish.severity, publish.summaries) == (0, ())
