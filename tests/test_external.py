"""External-severity tests — the Basalte-written faults become episodes.

Basalte detects, writes a severity 0-3 to the fault's group address and
delivers push and e-mail itself; the engine reads those writes back from
the bus archive and only records. Every test feeds invented severity
writes into the pure fold and asserts only what comes out: episodes with
their notification events, or a reconciliation plan against stored rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from lares_diagnostics_engine.episode_store import OpenEpisodeRow
from lares_diagnostics_engine.episodes import Episode, EventKind, NotificationEvent
from lares_diagnostics_engine.external import (
    SETTLE,
    SeverityWrite,
    drop_processed,
    fold_severity_writes,
    plan,
    read_window,
)
from lares_diagnostics_engine.reconcile import Measured, Plan

_T0 = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
_HOUR = timedelta(hours=1)
_GA = "15/2/8"


def _at(hours: float) -> datetime:
    return _T0 + hours * _HOUR


def _writes(*severities: tuple[float, int], subject: str = _GA) -> list[SeverityWrite]:
    return [SeverityWrite(subject=subject, time=_at(h), severity=s) for h, s in severities]


def _events(events: tuple[NotificationEvent, ...]) -> list[tuple[EventKind, datetime, int]]:
    return [(e.kind, e.time, e.severity) for e in events]


def _plan(
    *,
    episodes: Sequence[Episode],
    open_rows: Sequence[OpenEpisodeRow],
    in_scope: frozenset[str],
    now: datetime,
    read_through: datetime | None = None,
) -> Plan[Any]:
    """The kind's plan hook, given the scope the measurement resolved —
    `labels` names every address the catalog still holds, `read_through`
    how far the measurement read the archive."""
    measured: Measured[list[SeverityWrite]] = Measured(
        states={},
        observations=(),
        dataless=frozenset(),
        record={},
        labels=dict.fromkeys(in_scope, "declared"),
        read_through=read_through,
    )
    return plan(episodes=episodes, open_rows=open_rows, measured=measured, frontier=now)


# ------------------------------------------------------- the read cursor


def test_the_first_run_reads_nothing_and_starts_the_record_at_its_frontier() -> None:
    window = read_window(cursor=None, frontier=_at(10))

    assert window.through == _at(10) - SETTLE
    assert window.start == window.through


def test_a_later_run_continues_where_the_cursor_stands() -> None:
    window = read_window(cursor=_at(5), frontier=_at(10))

    assert (window.start, window.through) == (_at(5), _at(10) - SETTLE)


def test_the_plan_carries_the_read_through_for_the_apply_to_advance_the_cursor() -> None:
    result = _plan(
        episodes=[],
        open_rows=[],
        in_scope=frozenset({_GA}),
        now=_at(10),
        read_through=_at(10) - SETTLE,
    )

    assert result.read_through == _at(10) - SETTLE


def test_a_plan_without_a_measured_read_through_moves_no_cursor() -> None:
    result = _plan(episodes=[], open_rows=[], in_scope=frozenset({_GA}), now=_at(10))

    assert result.read_through is None


# ---------------------------------------------------------- fold: lifecycle

def test_severity_write_opens_an_episode() -> None:
    # The acceptance fixture: a severity 2 appearing on a declared Basalte
    # fault address opens an episode.
    [episode] = fold_severity_writes("system_pressure_low", _writes((0, 2)), {})
    assert episode.fault == "system_pressure_low"
    assert episode.subject == _GA
    assert episode.started_at == _at(0)
    assert episode.ended_at is None
    assert episode.severity == 2
    assert _events(episode.events) == [(EventKind.APPEARED, _at(0), 2)]


def test_severity_zero_ends_the_episode() -> None:
    [episode] = fold_severity_writes("f", _writes((0, 2), (3, 0)), {})
    assert episode.ended_at == _at(3)
    assert episode.last_seen_at == _at(0)
    assert _events(episode.events) == [
        (EventKind.APPEARED, _at(0), 2),
        (EventKind.ENDED, _at(3), 0),
    ]


def test_escalation_emits_exactly_one_escalate_event() -> None:
    [episode] = fold_severity_writes("f", _writes((0, 2), (2, 3)), {})
    assert episode.severity == 3
    assert _events(episode.events) == [
        (EventKind.APPEARED, _at(0), 2),
        (EventKind.ESCALATED, _at(2), 3),
    ]


def test_second_rise_rides_the_bus_without_a_second_event() -> None:
    # 2 -> 3 -> 2 -> 3: the escalation budget spends at the first rise.
    [episode] = fold_severity_writes("f", _writes((0, 2), (1, 3), (2, 2), (3, 3)), {})
    escalations = [e for e in episode.events if e.kind is EventKind.ESCALATED]
    assert len(escalations) == 1
    assert escalations[0].time == _at(1)


def test_repeated_severity_stays_one_episode_without_new_events() -> None:
    # Basalte's change detector writes once per change; a re-delivered or
    # cyclically repeated value must not open anything new.
    [episode] = fold_severity_writes("f", _writes((0, 2), (1, 2), (2, 2)), {})
    assert episode.last_seen_at == _at(2)
    assert _events(episode.events) == [(EventKind.APPEARED, _at(0), 2)]


def test_lowering_keeps_the_episode_open_and_its_peak() -> None:
    [episode] = fold_severity_writes("f", _writes((0, 3), (1, 2)), {})
    assert episode.ended_at is None
    assert episode.severity == 3
    assert [e.severity for e in episode.evidence] == [3, 2]
    assert _events(episode.events) == [(EventKind.APPEARED, _at(0), 3)]


def test_zero_with_nothing_open_is_ignored() -> None:
    assert fold_severity_writes("f", _writes((0, 0)), {}) == ()


def test_two_incidents_in_one_window_are_two_episodes() -> None:
    writes = _writes((0, 2), (1, 0), (5, 3), (6, 0))
    first, second = fold_severity_writes("f", writes, {})
    assert (first.started_at, first.ended_at) == (_at(0), _at(1))
    assert (second.started_at, second.ended_at) == (_at(5), _at(6))
    assert _events(second.events) == [
        (EventKind.APPEARED, _at(5), 3),
        (EventKind.ENDED, _at(6), 0),
    ]


def test_subjects_fold_independently() -> None:
    writes = _writes((0, 2)) + _writes((1, 3), subject="15/4/23")
    episodes = fold_severity_writes("f", writes, {})
    assert {(e.subject, e.severity) for e in episodes} == {(_GA, 2), ("15/4/23", 3)}


def test_evidence_carries_the_written_trajectory() -> None:
    [episode] = fold_severity_writes("f", _writes((0, 1), (1, 2), (2, 3)), {})
    assert [(row.time, row.severity) for row in episode.evidence] == [
        (_at(0), 1),
        (_at(1), 2),
        (_at(2), 3),
    ]


# ------------------------------------------------------------- fold: seeding

def test_prior_open_state_continues_without_a_second_appear() -> None:
    # The stored open row seeds the state at window start: new writes
    # continue that episode, they do not re-open it.
    [episode] = fold_severity_writes("f", _writes((0, 3)), {_GA: 2})
    assert _events(episode.events) == [(EventKind.ESCALATED, _at(0), 3)]
    assert episode.severity == 3


def test_prior_open_state_ends_on_a_bare_zero() -> None:
    [episode] = fold_severity_writes("f", _writes((0, 0)), {_GA: 2})
    assert episode.ended_at == _at(0)
    assert episode.severity == 2
    assert episode.evidence == ()
    assert _events(episode.events) == [(EventKind.ENDED, _at(0), 0)]


def test_prior_open_state_with_no_writes_yields_nothing() -> None:
    # No write means no change: an external fault stays open until the
    # explicit 0 arrives — absence of data is not a recovery.
    assert fold_severity_writes("f", [], {_GA: 2}) == ()


def test_prior_state_repeated_at_the_same_severity_never_escalates() -> None:
    [episode] = fold_severity_writes("f", _writes((0, 2), (1, 2)), {_GA: 2})
    assert _events(episode.events) == []


def test_after_a_seeded_end_a_new_write_opens_a_fresh_episode() -> None:
    ended, fresh = fold_severity_writes("f", _writes((0, 0), (4, 2)), {_GA: 3})
    assert ended.ended_at == _at(0)
    assert fresh.started_at == _at(4)
    assert _events(fresh.events) == [(EventKind.APPEARED, _at(4), 2)]


# --------------------------------------------------------------------- plan

def test_new_open_episode_is_inserted() -> None:
    episodes = fold_severity_writes("f", _writes((0, 2)), {})
    plan = _plan(episodes=episodes, open_rows=[], in_scope=frozenset({_GA}), now=_at(6))
    assert [e.subject for e in plan.inserts] == [_GA]
    assert plan.updates == ()
    assert plan.orphan_closes == ()


def test_continuation_updates_the_stored_row() -> None:
    row = OpenEpisodeRow(id=7, subject=_GA, severity=2)
    episodes = fold_severity_writes("f", _writes((0, 0)), {_GA: 2})
    plan = _plan(episodes=episodes, open_rows=[row], in_scope=frozenset({_GA}), now=_at(6))
    assert plan.inserts == ()
    [(row_id, episode)] = plan.updates
    assert row_id == 7
    assert episode.ended_at == _at(0)


def test_ended_continuation_plus_new_incident_splits_update_and_insert() -> None:
    row = OpenEpisodeRow(id=7, subject=_GA, severity=2)
    episodes = fold_severity_writes("f", _writes((0, 0), (4, 3)), {_GA: 2})
    plan = _plan(episodes=episodes, open_rows=[row], in_scope=frozenset({_GA}), now=_at(6))
    [(row_id, ended)] = plan.updates
    assert (row_id, ended.ended_at) == (7, _at(0))
    [fresh] = plan.inserts
    assert (fresh.started_at, fresh.ended_at) == (_at(4), None)


def test_open_row_without_writes_stays_untouched() -> None:
    row = OpenEpisodeRow(id=7, subject=_GA, severity=2)
    plan = _plan(episodes=(), open_rows=[row], in_scope=frozenset({_GA}), now=_at(6))
    assert plan.updates == ()
    assert plan.orphan_closes == ()
    assert plan.stale_opens == (_GA,)


def test_open_row_whose_address_left_the_catalog_is_closed() -> None:
    row = OpenEpisodeRow(id=7, subject=_GA, severity=2)
    plan = _plan(episodes=(), open_rows=[row], in_scope=frozenset(), now=_at(6))
    assert plan.orphan_closes == ((7, _at(6)),)
    assert plan.stale_opens == ()


# ------------------------------------------------- replay safety across runs

def test_processed_writes_are_dropped() -> None:
    writes = _writes((0, 2), (3, 0), (5, 3))
    assert drop_processed(writes, {_GA: _at(3)}) == _writes((5, 3))


def test_unknown_subject_keeps_all_writes() -> None:
    writes = _writes((0, 2), (3, 0))
    assert drop_processed(writes, {}) == writes


def test_second_run_does_not_replay_a_closed_incident() -> None:
    # Run 1 records the whole incident; the row is closed. Run 2 sees the
    # same sliding window, but everything up to the recorded end is already
    # processed — nothing may be inserted again.
    writes = _writes((0, 2), (3, 0))
    [episode] = fold_severity_writes("f", drop_processed(writes, {}), {})
    assert episode.ended_at == _at(3)

    second_run = drop_processed(writes, {_GA: _at(3)})
    episodes = fold_severity_writes("f", second_run, {})
    plan = _plan(episodes=episodes, open_rows=[], in_scope=frozenset({_GA}), now=_at(4))
    assert plan.inserts == ()
    assert plan.updates == ()


def test_reopened_incident_is_not_closed_by_the_stale_zero() -> None:
    # After end-then-reopen the open row's story reaches _at(4); the old 0
    # at _at(0) sits before that and must not touch the fresh episode.
    writes = _writes((0, 0), (4, 2), (5, 3))
    row = OpenEpisodeRow(id=9, subject=_GA, severity=2)
    third_run = drop_processed(writes, {_GA: _at(4)})
    [episode] = fold_severity_writes("f", third_run, {_GA: 2})
    assert episode.ended_at is None
    assert _events(episode.events) == [(EventKind.ESCALATED, _at(5), 3)]
    plan = _plan(episodes=[episode], open_rows=[row], in_scope=frozenset({_GA}), now=_at(6))
    assert plan.inserts == ()
    assert [row_id for row_id, _ in plan.updates] == [9]
