"""Runner-lifecycle tests, through the injected store and publisher ends.

The guarantees the hand-written runners never had tests for live here:
the publish goes out before the write, a dry run touches neither write
side, a missing frontier stops the run, and the measurement sees the same
open rows the reconciliation uses. Fakes sit at both seams — no cluster,
no live database.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from lares_diagnostics_engine.episode_store import OpenEpisodeRow
from lares_diagnostics_engine.episodes import (
    Episode,
    EpisodeEvent,
    EpisodePolicy,
    EventKind,
    Observation,
)
from lares_diagnostics_engine.faults import Fault, MeasurementKind, Target
from lares_diagnostics_engine.reconcile import Measured, Plan, Window
from lares_diagnostics_engine.runner import LOOKBACK, Kind, run_subjects

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

_T0 = datetime(2026, 8, 30, 10, 0, tzinfo=UTC)
_FRONTIER = _T0 + 8 * timedelta(hours=1)

_Applied = tuple[
    tuple[Episode, ...], tuple[tuple[int, Episode], ...], tuple[tuple[int, datetime], ...]
]


class _Store:
    """In-memory store end; every interaction lands in the shared event log."""

    def __init__(
        self,
        events: list[str],
        open_rows: list[OpenEpisodeRow] | None = None,
        recorded: tuple[EpisodeEvent, ...] = (),
    ) -> None:
        self.events = events
        self._open_rows = open_rows or []
        self._recorded = recorded
        self.applied: _Applied | None = None
        self.fingerprint: str | None = None
        self.externally_delivered = False
        self.read_through: datetime | None = None

    @contextmanager
    def read(self) -> Iterator[Any]:
        self.events.append("read")
        yield None

    def open_rows(self, _conn: Any, _fault_name: str) -> list[OpenEpisodeRow]:
        return list(self._open_rows)

    def history_scores(self, _conn: Any, _fault_name: str) -> list[float]:
        return []

    def apply(
        self,
        _fault_name: str,
        inserts: Sequence[Episode],
        updates: Sequence[tuple[int, Episode]],
        orphan_closes: Sequence[tuple[int, datetime]],
        *,
        fingerprint: str,
        externally_delivered: bool,
        read_through: datetime | None = None,
    ) -> tuple[EpisodeEvent, ...]:
        self.events.append("apply")
        self.applied = (tuple(inserts), tuple(updates), tuple(orphan_closes))
        self.fingerprint = fingerprint
        self.externally_delivered = externally_delivered
        self.read_through = read_through
        return self._recorded


class _Publisher:
    """In-memory publisher end sharing the store's event log."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.published: list[tuple[str, str | None, dict[str, Any], str | None, bool]] = []
        self.episode_events: list[EpisodeEvent] = []

    def publish_anomaly(
        self,
        fault_name: str,
        severity: str | None,
        payload: dict[str, Any],
        *,
        entity: str | None,
        firing: bool,
    ) -> None:
        self.events.append("publish")
        self.published.append((fault_name, severity, payload, entity, firing))

    def publish_episode_event(self, event: EpisodeEvent) -> None:
        self.events.append("publish_episode")
        self.episode_events.append(event)


@dataclass(frozen=True, slots=True)
class _Publish:
    subject: str
    severity: int

    @property
    def entity(self) -> str:
        return self.subject


# Frozen, so it is safe as a default; `_fault(target=None)` is the fault a
# self-delivering kind gets.
_PER_DEVICE = Target(per_device=True)
# Likewise: the hourly cadence every kind but the daily-yield shape folds in.
_HOURLY = EpisodePolicy()


def _fault(target: Target | None = _PER_DEVICE, *, explain: bool = True) -> Fault:
    return Fault(
        name="test_fault",
        sentence="a device runs longer than declared",
        unit="h",
        kind=MeasurementKind.DURATION,
        parameters={},
        target=target,
        explain=explain,
    )


def _kind(
    *,
    frontier: datetime | None = _FRONTIER,
    observations: tuple[Observation, ...] = (),
    seen: dict[str, Any] | None = None,
    policy: EpisodePolicy = _HOURLY,
    **overrides: Any,
) -> Kind[Any, Any]:
    def measure(
        _conn: Any, _fault: Fault, window: Window, open_rows: Sequence[OpenEpisodeRow]
    ) -> Measured[Any]:
        if seen is not None:
            seen["window"] = window
            seen["open_rows"] = tuple(open_rows)
        return Measured(
            states={}, observations=observations, dataless=frozenset(), record={"subjects": 1}
        )

    return Kind(
        event="test_kind_run",
        delivery="per_device",
        frontier=lambda _conn: frontier,
        measure=measure,
        publish_for=lambda subject, severity, _state: _Publish(subject, severity),
        payload=lambda p: {"subject": p.subject},
        policy=policy,
        **overrides,
    )


_FIRING = (Observation(subject="2/1/197", time=_FRONTIER, score=1.0),)


def test_the_publish_goes_out_before_the_write() -> None:
    events: list[str] = []
    store, publisher = _Store(events), _Publisher(events)

    run_subjects(store, publisher, _fault(), _kind(observations=_FIRING), dry_run=False)

    assert events == ["read", "publish", "apply"]
    ((fault_name, severity, payload, entity, firing),) = publisher.published
    assert fault_name == "test_fault"
    assert severity is not None
    assert payload == {"subject": "2/1/197"}
    assert entity == "2/1/197"
    assert firing is True
    assert store.applied is not None
    inserts, _, _ = store.applied
    assert [e.subject for e in inserts] == ["2/1/197"]


def test_the_rows_are_stamped_with_the_fingerprint_the_run_measured_by() -> None:
    # What the store writes on every row it makes or re-makes: the fault's
    # fingerprint, so a later rule can tell its own rows from this one's.
    events: list[str] = []
    store, publisher = _Store(events), _Publisher(events)

    run_subjects(store, publisher, _fault(), _kind(observations=_FIRING), dry_run=False)

    assert store.fingerprint == _fault().fingerprint == "duration()"


def test_a_kind_declaring_its_own_fingerprint_stamps_with_that() -> None:
    # A kind whose code decides what it accuses folds its revision in; the
    # runner stamps whatever the kind declares, never the bare declaration.
    events: list[str] = []
    store, publisher = _Store(events), _Publisher(events)
    kind = _kind(observations=_FIRING, fingerprint=lambda fault: f"{fault.fingerprint}@7")

    run_subjects(store, publisher, _fault(), kind, dry_run=False)

    assert store.fingerprint == "duration()@7"


def test_a_clear_is_published_before_its_row_closes() -> None:
    events: list[str] = []
    row = OpenEpisodeRow(id=7, subject="2/1/197", severity=2)
    store, publisher = _Store(events, open_rows=[row]), _Publisher(events)

    run_subjects(store, publisher, _fault(), _kind(), dry_run=False)

    assert events == ["read", "publish", "apply"]
    ((_, severity, _, _, firing),) = publisher.published
    assert severity is None
    assert firing is False
    assert store.applied is not None
    _, _, orphan_closes = store.applied
    assert orphan_closes == ((7, _FRONTIER),)


def test_a_dry_run_touches_neither_write_side() -> None:
    events: list[str] = []
    store, publisher = _Store(events), _Publisher(events)

    run_subjects(store, publisher, _fault(), _kind(observations=_FIRING), dry_run=True)

    assert events == ["read"]
    assert publisher.published == []
    assert publisher.episode_events == []
    assert store.applied is None


def test_a_missing_frontier_stops_the_run_before_the_measurement() -> None:
    events: list[str] = []
    store, publisher = _Store(events), _Publisher(events)
    seen: dict[str, Any] = {}

    run_subjects(store, publisher, _fault(), _kind(frontier=None, seen=seen), dry_run=False)

    assert events == ["read"]
    assert seen == {}
    assert publisher.published == []
    assert store.applied is None


def test_a_fault_declaring_the_wrong_target_form_is_rejected() -> None:
    events: list[str] = []
    store, publisher = _Store(events), _Publisher(events)

    with pytest.raises(ValueError, match="per_device"):
        run_subjects(
            store, publisher, _fault(target=Target(ga="0/0/230")), _kind(), dry_run=False
        )
    assert events == []


def test_a_kind_declaring_its_own_plan_is_planned_by_it() -> None:
    events: list[str] = []
    store, publisher = _Store(events), _Publisher(events)

    def plan(
        *,
        episodes: Sequence[Episode],
        open_rows: Sequence[OpenEpisodeRow],
        measured: Measured[Any],
        frontier: datetime,
    ) -> Plan[_Publish]:
        return Plan(
            inserts=(),
            updates=(),
            orphan_closes=(),
            stale_opens=(),
            publishes=(_Publish("2", 2),),
        )

    kind = Kind(
        event="test_group_run",
        delivery="per_device",
        frontier=lambda _conn: _FRONTIER,
        measure=lambda _conn, _fault, _window, _open_rows: Measured(
            states={}, observations=(), dataless=frozenset(), record={}
        ),
        payload=lambda p: {"subject": p.subject},
        plan=plan,
    )
    run_subjects(store, publisher, _fault(), kind, dry_run=False)

    assert events == ["read", "publish", "apply"]
    ((_, _, payload, entity, firing),) = publisher.published
    assert payload == {"subject": "2"}
    assert entity == "2"
    assert firing is True


def _empty_fold(
    *,
    fault_name: str,
    measured: Measured[Any],
    open_rows: Sequence[OpenEpisodeRow],
) -> tuple[Episode, ...]:
    return ()


def _empty_plan(
    *,
    episodes: Sequence[Episode],
    open_rows: Sequence[OpenEpisodeRow],
    measured: Measured[Any],
    frontier: datetime,
) -> Plan[Any]:
    return Plan((), (), (), (), ())


def _self_delivering_kind(**overrides: Any) -> Kind[Any, Any]:
    """A kind with no target and nothing to publish — external's shape."""
    return Kind(
        event="test_external_run",
        delivery=None,
        frontier=lambda _conn: _FRONTIER,
        measure=lambda _conn, _fault, _window, _open_rows: Measured(
            states={}, observations=(), dataless=frozenset(), record={}
        ),
        plan=_empty_plan,
        **overrides,
    )


def test_a_self_delivering_kind_publishes_nothing_and_marks_its_rows() -> None:
    events: list[str] = []
    store, publisher = _Store(events), _Publisher(events)
    kind = _self_delivering_kind(fold=_empty_fold, externally_delivered=True)

    run_subjects(store, publisher, _fault(target=None), kind, dry_run=False)

    assert events == ["read", "apply"]
    assert publisher.published == []
    assert store.externally_delivered is True


def test_a_plan_that_moves_the_read_cursor_hands_it_to_the_apply() -> None:
    events: list[str] = []
    store, publisher = _Store(events), _Publisher(events)
    read_through = _FRONTIER - timedelta(minutes=5)

    def cursor_plan(**_: Any) -> Plan[Any]:
        return Plan(
            inserts=(),
            updates=(),
            orphan_closes=(),
            stale_opens=(),
            publishes=(),
            read_through=read_through,
        )

    kind = replace(
        _self_delivering_kind(fold=_empty_fold, externally_delivered=True), plan=cursor_plan
    )

    run_subjects(store, publisher, _fault(target=None), kind, dry_run=False)

    assert events == ["read", "apply"]
    assert store.read_through == read_through


def test_a_plan_without_a_cursor_leaves_the_store_cursor_alone() -> None:
    events: list[str] = []
    store, publisher = _Store(events), _Publisher(events)
    kind = _self_delivering_kind(fold=_empty_fold, externally_delivered=True)

    run_subjects(store, publisher, _fault(target=None), kind, dry_run=False)

    assert store.read_through is None


def test_a_self_delivering_kind_rejects_a_fault_declaring_a_target() -> None:
    events: list[str] = []
    store, publisher = _Store(events), _Publisher(events)

    with pytest.raises(ValueError, match="delivers itself, no target"):
        run_subjects(store, publisher, _fault(), _self_delivering_kind(), dry_run=False)
    assert events == []


def test_a_kind_declaring_neither_publish_for_nor_plan_fails_loudly() -> None:
    events: list[str] = []
    store, publisher = _Store(events), _Publisher(events)

    kind: Kind[Any, Any] = Kind(
        event="test_undeclared_run",
        delivery="per_device",
        frontier=lambda _conn: _FRONTIER,
        measure=lambda _conn, _fault, _window, _open_rows: Measured(
            states={}, observations=(), dataless=frozenset(), record={}
        ),
        payload=lambda p: {"subject": p.subject},
    )
    with pytest.raises(ValueError, match="neither publish_for nor plan"):
        run_subjects(store, publisher, _fault(), kind, dry_run=False)
    assert publisher.published == []
    assert store.applied is None


def test_the_measurement_sees_the_window_and_the_open_rows() -> None:
    events: list[str] = []
    row = OpenEpisodeRow(id=7, subject="2/1/197", severity=2)
    store, publisher = _Store(events, open_rows=[row]), _Publisher(events)
    seen: dict[str, Any] = {}

    run_subjects(
        store, publisher, _fault(), _kind(observations=_FIRING, seen=seen), dry_run=False
    )

    assert seen["open_rows"] == (row,)
    assert seen["window"].frontier == _FRONTIER
    assert seen["window"].start == _FRONTIER - LOOKBACK


def test_a_kind_folds_in_its_own_cadence() -> None:
    # A kind that measures days must fold days. Yesterday's observation is
    # the newest this kind can have; under the hourly default it would look
    # long stale and its episode would close on the run that first saw it.
    daily = EpisodePolicy(bucket=timedelta(days=1), quiet_runs=1)
    events: list[str] = []
    store, publisher = _Store(events), _Publisher(events)
    seen: dict[str, Any] = {}

    run_subjects(
        store,
        publisher,
        _fault(),
        _kind(
            observations=(
                Observation(subject="pv", time=_FRONTIER - timedelta(days=1), score=2.0),
            ),
            seen=seen,
            policy=daily,
        ),
        dry_run=False,
    )

    assert seen["window"].policy is daily
    assert store.applied is not None
    inserts, _, _ = store.applied
    (episode,) = inserts
    assert episode.ended_at is None


_APPEARED = EpisodeEvent(
    episode_id=15510,
    fault="test_fault",
    subject="2/1/197",
    kind=EventKind.APPEARED,
    time=_FRONTIER,
    severity=2,
)
_ENDED = EpisodeEvent(
    episode_id=15509,
    fault="test_fault",
    subject="2/1/198",
    kind=EventKind.ENDED,
    time=_FRONTIER,
    severity=0,
)


def test_each_episode_event_the_write_recorded_is_published_once_after_it() -> None:
    # The pointer goes out after the rows, because the row is what gives the
    # event its episode id — and only for what the write actually recorded,
    # so a rerun over the same window repeats no event.
    events: list[str] = []
    store = _Store(events, recorded=(_APPEARED, _ENDED))
    publisher = _Publisher(events)

    run_subjects(store, publisher, _fault(), _kind(observations=_FIRING), dry_run=False)

    assert events == ["read", "publish", "apply", "publish_episode", "publish_episode"]
    assert publisher.episode_events == [_APPEARED, _ENDED]


def test_a_fault_that_is_not_explained_keeps_its_episode_events_off_the_bus() -> None:
    # `explain: false` is about the episode events alone — the fault
    # severities Basalte routes go out either way.
    events: list[str] = []
    store = _Store(events, recorded=(_APPEARED,))
    publisher = _Publisher(events)

    run_subjects(
        store, publisher, _fault(explain=False), _kind(observations=_FIRING), dry_run=False
    )

    assert events == ["read", "publish", "apply"]
    assert publisher.episode_events == []
    assert len(publisher.published) == 1


def test_a_pointer_that_cannot_be_published_fails_the_run() -> None:
    # The rows are written by then, so a later run's conflict clause filters
    # the event away rather than offering it again: the event is lost, and
    # the run says so instead of reporting a clean pass.
    events: list[str] = []
    store = _Store(events, recorded=(_APPEARED,))

    class _DeadBus(_Publisher):
        def publish_episode_event(self, _event: EpisodeEvent) -> None:
            raise ConnectionError("no route to nats")

    publisher = _DeadBus(events)
    with pytest.raises(ConnectionError):
        run_subjects(store, publisher, _fault(), _kind(observations=_FIRING), dry_run=False)

    assert store.applied is not None  # the write went through first
    assert publisher.episode_events == []
