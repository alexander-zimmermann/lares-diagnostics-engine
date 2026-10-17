"""External severities: Basalte-written fault addresses become episodes.

Basalte detects the fault, writes a severity 0-3 to its group address and
delivers push and e-mail itself. The knx-nats-bridge archives those writes
like any other telegram, so the engine reads them back from the raw `knx`
table and treats them as observations: a severity >= 1 opens or continues
an episode, the explicit 0 ends it. No quiet-run window and no score
distribution — the wire value is authoritative, and unlike the measured
kinds an absence of writes means "unchanged", never "recovered".

The engine only records: it publishes nothing for these faults, and the
episodes are marked externally delivered so nothing downstream notifies a
second time. Folding and reconciliation are pure; the SQL edge reads the
declared addresses over the bus archive's own index.

The engine keeps a read cursor per fault (`fault_cursors`): a run reads
the archive from where the last one stopped up to the frontier less a
settling margin, and the apply that writes the episodes moves the cursor
there in the same transaction. A fault switched on has no cursor — its
first run reads nothing and the record starts there, so the thirty days
behind it (group-monitor tests, the writes of a block since rebuilt) never
fold into phantom episodes. The stored episodes still guard the fold
(`drop_processed`): a cursor set back by hand re-reads writes, and the
ones already recorded stay recorded once.

Two things the cursor does not do. A severity already standing on the bus
when a fault is switched on is recorded from its next write, and its
eventual 0 is dropped — nothing is open for it to end. And a row that
reaches the archive later than the settling margin after its bus time —
an ingest pipeline down for longer than that, replayed afterwards — sits
behind a cursor that has moved past it; the 30-day window caught those on
the next run, the cursor does not.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import groupby
from operator import attrgetter
from typing import TYPE_CHECKING, Any

from . import episode_store
from .episodes import Episode, EventKind, EvidenceRow, NotificationEvent
from .logging_setup import get_logger
from .reconcile import Measured, Plan
from .severity import CLEAR, CRITICAL
from .silence import resolve_scope

if TYPE_CHECKING:
    import psycopg
    from psycopg.rows import DictRow

    from .episode_store import OpenEpisodeRow
    from .faults import Fault
    from .reconcile import Window

log = get_logger(__name__)

# How long a telegram's row may lag behind its bus time before the engine
# reads: the bridge, NATS and the ingest pipeline sit between, seconds in
# normal operation. A run reads only up to the frontier less this margin,
# so a row that lands within it is read by the next run; one that lands
# later than this has slipped behind the cursor (see the module docstring).
SETTLE = timedelta(minutes=5)


@dataclass(frozen=True, slots=True)
class ReadWindow:
    """What one run reads off the archive: writes after `start` up to and
    including `through`, the point the cursor moves to afterwards."""

    start: datetime
    through: datetime


def read_window(cursor: datetime | None, frontier: datetime) -> ReadWindow:
    """Where this run reads: from the cursor to the settled frontier — or,
    on a fault's first run, nothing at all. The record starts where the
    engine starts watching; Basalte already delivered what came before.
    """
    through = frontier - SETTLE
    return ReadWindow(start=cursor if cursor is not None else through, through=through)


@dataclass(frozen=True, slots=True)
class SeverityWrite:
    """One archived severity telegram on a declared fault address."""

    subject: str
    time: datetime
    severity: int


def frontier(_conn: psycopg.Connection[DictRow]) -> datetime | None:
    """Wall-clock now: the bus archive has no materialization lag, so
    external runs take wall-clock time where an orphaned row needs closing.
    """
    return datetime.now(tz=UTC)


def measure(
    conn: psycopg.Connection[DictRow],
    fault: Fault,
    window: Window,
    _open_rows: Sequence[OpenEpisodeRow],
) -> Measured[list[SeverityWrite]]:
    """The kind's whole measurement: the declared addresses' severity writes
    off the bus archive between the fault's read cursor and the settled
    frontier, minus everything the stored episodes have already folded.
    What leaves per subject is its fresh writes — the fold's input, not
    observations. A fault without a cursor is on its first run: it reads
    nothing, and the plan starts its record here. The runner's window is
    the measured kinds' 30-day lookback; only its frontier is used here.
    """
    channels = resolve_scope(conn, fault.channel_scope())
    if not channels:
        # Address not in the catalog yet (ETS work pending) — or a typo.
        log.warning("external_no_subjects", fault=fault.name)
    cursor = episode_store.read_cursor(conn, fault.name)
    reading = read_window(cursor, window.frontier)
    processed = episode_store.processed_through(conn, fault.name)
    writes = read_writes(conn, [c.ga for c in channels], reading)
    fresh = drop_processed(writes, processed)
    by_subject: dict[str, list[SeverityWrite]] = {}
    for write in fresh:
        by_subject.setdefault(write.subject, []).append(write)
    return Measured(
        states=by_subject,
        observations=(),
        # Absence of writes means "unchanged", never "no data" — the plan
        # below owns keeping those rows open.
        dataless=frozenset(),
        record={
            "subjects": len(channels),
            "first_run": cursor is None,
            "read_from": reading.start.isoformat(),
            "read_through": reading.through.isoformat(),
            "writes": len(writes),
            "fresh_writes": len(fresh),
        },
        labels={c.ga: c.name for c in channels},
        read_through=reading.through,
    )


def fold(
    *,
    fault_name: str,
    measured: Measured[list[SeverityWrite]],
    open_rows: Sequence[OpenEpisodeRow],
) -> tuple[Episode, ...]:
    """The runner's fold hook: severity writes in, episodes out — seeded
    with the severities the stored open rows carry, so a write continues
    the recorded episode instead of re-opening it.
    """
    prior = {row.subject: row.severity for row in open_rows}
    fresh = [write for per_subject in measured.states.values() for write in per_subject]
    return fold_severity_writes(fault_name, fresh, prior)


@dataclass(slots=True)
class _OpenState:
    """One episode being folded: the trajectory so far and the severity its
    escalation budget measures against.
    """

    baseline: int
    evidence: list[EvidenceRow] = field(default_factory=list)
    events: list[NotificationEvent] = field(default_factory=list)
    escalated: bool = False


def drop_processed(
    writes: Iterable[SeverityWrite], processed_through: Mapping[str, datetime]
) -> list[SeverityWrite]:
    """Writes the stored episodes have not seen yet. Everything at or before
    a subject's processed-through time is already part of a recorded episode;
    the cursor keeps a run from reading it twice, and this keeps a cursor
    set back by hand from re-inserting the closed incidents it re-reads.
    """
    return [
        w
        for w in writes
        if (through := processed_through.get(w.subject)) is None or w.time > through
    ]


def fold_severity_writes(
    fault: str,
    writes: Iterable[SeverityWrite],
    prior: Mapping[str, int],
) -> tuple[Episode, ...]:
    """The pure seam: a fault's severity writes in, its episodes out.

    `prior` seeds each subject with the severity its stored open episode
    holds, so a write continues that episode instead of re-opening it; a
    seeded episode emits no appear event (it already notified). Within an
    episode the escalation budget spends at the first rise above the
    appearing (or seeded) severity — later worsening rides the bus value.
    """
    episodes: list[Episode] = []
    for subject, per_subject in groupby(
        sorted(writes, key=attrgetter("subject", "time")), key=attrgetter("subject")
    ):
        episodes.extend(_fold_subject(fault, subject, list(per_subject), prior.get(subject, 0)))
    return tuple(episodes)


def _fold_subject(
    fault: str, subject: str, ordered: list[SeverityWrite], prior: int
) -> list[Episode]:
    episodes: list[Episode] = []
    # A seeded state opens without an appear event — it already notified.
    state = _OpenState(baseline=prior) if prior > 0 else None

    for write in ordered:
        if write.severity == CLEAR:
            if state is not None:
                state.events.append(NotificationEvent(EventKind.ENDED, write.time, CLEAR))
                episodes.append(_build_episode(fault, subject, state, ended_at=write.time))
                state = None
            continue
        if state is None:
            state = _OpenState(baseline=write.severity)
            state.events.append(NotificationEvent(EventKind.APPEARED, write.time, write.severity))
        elif write.severity > state.baseline and not state.escalated:
            state.events.append(NotificationEvent(EventKind.ESCALATED, write.time, write.severity))
            state.escalated = True
        state.evidence.append(
            EvidenceRow(time=write.time, score=float(write.severity), severity=write.severity)
        )

    if state is not None:
        episodes.append(_build_episode(fault, subject, state, ended_at=None))
    return episodes


def _build_episode(
    fault: str, subject: str, state: _OpenState, ended_at: datetime | None
) -> Episode:
    if state.evidence:
        started_at, last_seen_at = state.evidence[0].time, state.evidence[-1].time
    elif ended_at is not None:
        # Only a seeded episode ending on a bare 0 has no in-window evidence;
        # its times sit at the end write, and the row update — the only thing
        # such an episode exists for — never touches started_at anyway.
        started_at = last_seen_at = ended_at
    else:
        raise RuntimeError(f"open episode for {subject} without evidence")
    severity = max((row.severity for row in state.evidence), default=state.baseline)
    return Episode(
        fault=fault,
        subject=subject,
        started_at=started_at,
        last_seen_at=last_seen_at,
        ended_at=ended_at,
        severity=severity,
        peak_score=float(severity),
        evidence=tuple(state.evidence),
        events=tuple(state.events),
    )


def plan(
    *,
    episodes: Sequence[Episode],
    open_rows: Sequence[OpenEpisodeRow],
    measured: Measured[list[SeverityWrite]],
    frontier: datetime,
) -> Plan[Any]:
    """Pure reconciliation: computed episodes against the stored open rows.

    A subject's first computed episode continues its stored open row; any
    later ones are fresh incidents — a full open-and-closed cycle inside
    the window is recorded whole, which is why this kind cannot ride the
    shared reconcile (that one materializes only open episodes). A row
    without writes stays open (`stale_opens`) — only the explicit 0 ends an
    external episode — unless its address no longer resolves in the
    catalog, which orphans the row at the frontier. Publishes nothing:
    Basalte already delivered. The plan also carries how far the read went,
    so the apply moves the fault's cursor there in the transaction that
    writes what that read produced.
    """
    in_scope = frozenset(measured.labels)
    open_by_subject = {row.subject: row for row in open_rows}

    inserts: list[Episode] = []
    updates: list[tuple[int, Episode]] = []
    seen: set[str] = set()
    for subject, per_subject in groupby(
        sorted(episodes, key=attrgetter("subject", "started_at")), key=attrgetter("subject")
    ):
        seen.add(subject)
        ordered = list(per_subject)
        row = open_by_subject.get(subject)
        if row is not None:
            updates.append((row.id, ordered[0]))
            ordered = ordered[1:]
        inserts.extend(ordered)

    orphan_closes: list[tuple[int, datetime]] = []
    still_open: list[str] = []
    for row in open_rows:
        if row.subject in seen:
            continue
        if row.subject in in_scope:
            still_open.append(row.subject)
        else:
            orphan_closes.append((row.id, frontier))

    return Plan(
        inserts=tuple(inserts),
        updates=tuple(updates),
        orphan_closes=tuple(orphan_closes),
        stale_opens=tuple(still_open),
        publishes=(),
        read_through=measured.read_through,
    )


def read_writes(
    conn: psycopg.Connection[DictRow], gas: Iterable[str], reading: ReadWindow
) -> list[SeverityWrite]:
    """The declared addresses' severity telegrams from the bus archive
    inside the read window, one indexed query per address — external
    faults have one or two. A value off the 0-3 contract is logged and
    dropped, never folded.
    """
    writes: list[SeverityWrite] = []
    for ga in gas:
        main, middle, sub = (int(part) for part in ga.split("/"))
        rows = conn.execute(
            """
            SELECT time, value FROM knx
            WHERE knx_main = %(main)s AND knx_middle = %(middle)s AND knx_sub = %(sub)s
              AND time > %(start)s AND time <= %(through)s
            ORDER BY time
            """,
            {
                "main": main,
                "middle": middle,
                "sub": sub,
                "start": reading.start,
                "through": reading.through,
            },
        ).fetchall()
        for row in rows:
            severity = int(row["value"])
            if not CLEAR <= severity <= CRITICAL or severity != row["value"]:
                log.warning(
                    "severity_off_contract", ga=ga, time=row["time"].isoformat(),
                    value=row["value"],
                )
                continue
            writes.append(SeverityWrite(subject=ga, time=row["time"], severity=severity))
    return writes
