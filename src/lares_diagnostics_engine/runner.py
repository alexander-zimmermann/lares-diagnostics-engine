"""The run lifecycle every fault kind shares.

A `Kind` declares what genuinely differs between kinds — how its series is
measured and how its payload is shaped, both living in the kind's own
module — and `run_subjects` owns the rest: the window off the aggregate's
frontier, the fold behind the pure episode seam, the reconciliation against
the stored open rows, the run record, the dry run, and the publish-then-
write tail. A new kind declares those things and inherits all of it.

The runner touches the world through two ends, injected so the lifecycle's
guarantees are testable through fakes:

* the **store end** — one read connection for the measurement, the open
  rows and score history behind it, and the transactional apply of a
  plan's row changes;
* the **publisher end** — one anomaly publish per moved subject, and one
  episode-event pointer per event the write recorded.

Two adapters sit at each seam: the database and NATS in production, an
in-memory recorder in the runner's tests.

The ordering is load-bearing. Time is the aggregate's frontier throughout —
episodes also *end* in frontier time, so a stalled refresh (or a dead
bridge) freezes the picture instead of clearing every open episode with a
severity 0 nobody earned. Publishes go out before the database writes: a
failed run then repeats the same publish (same value, Basalte's change
detector ignores it) instead of losing it behind an already-updated
database. `--dry-run` computes and logs everything and touches neither
write side.

The episode events go the other way round, *after* the write, and for one
reason: the row is what gives an event its id and what says it is new at
all. Only the events the write recorded reach the bus, and only for a fault
declared `explain` — the severities Basalte routes are untouched by that
flag.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

from . import episode_store, nats_publisher
from .db_write import read_connection, write_connection
from .episodes import Episode, EpisodeEvent, EpisodePolicy, fold_observations
from .logging_setup import get_logger
from .reconcile import Measured, Plan, SubjectPublish, Window, subject_plan
from .severity import severity_name

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    import psycopg
    from psycopg.rows import DictRow

    from .config import Settings
    from .episode_store import OpenEpisodeRow
    from .faults import Fault

log = get_logger(__name__)

# Measurement window: pause estimation, observation reconstruction and the
# score history all live inside it. Matches the 30 days the episode fold-in
# started the comparison basis with.
LOOKBACK = timedelta(days=30)

# How far back the aggregates most kinds read can be trusted: the raw tables
# keep a year (the retention policies in the lares bootstrap schema) and the
# aggregates over them were materialized from that. They hold no retention
# policy of their own and may reach further; nobody can promise it.
HISTORY = timedelta(weeks=52)


class Store(Protocol):
    """The store end: what the runner needs of episode persistence."""

    def read(self) -> AbstractContextManager[psycopg.Connection[DictRow]]:
        """One connection for the whole read phase — frontier, measurement,
        open rows and score history see the same snapshot."""
        ...

    def open_rows(
        self, conn: psycopg.Connection[DictRow], fault_name: str
    ) -> list[OpenEpisodeRow]: ...

    def history_scores(
        self, conn: psycopg.Connection[DictRow], fault_name: str
    ) -> list[float]: ...

    def apply(
        self,
        fault_name: str,
        inserts: Sequence[Episode],
        updates: Sequence[tuple[int, Episode]],
        orphan_closes: Sequence[tuple[int, datetime]],
        *,
        fingerprint: str,
        externally_delivered: bool,
    ) -> Sequence[EpisodeEvent]:
        """One plan's row changes, in one transaction; the rows it makes or
        re-makes stamped with the fingerprint of the rule the run measured
        by. What comes back are the episode events the write recorded — the
        new ones, with the ids their rows just got."""
        ...


class Publisher(Protocol):
    """The publisher end: one anomaly publish per moved subject, and one
    pointer per recorded episode event."""

    def publish_anomaly(
        self,
        fault_name: str,
        severity: str | None,
        payload: dict[str, Any],
        *,
        entity: str | None,
        firing: bool,
    ) -> None: ...

    def publish_episode_event(self, event: EpisodeEvent) -> None: ...


@dataclass(frozen=True, slots=True)
class DbStore:
    """The store end's database adapter — thin by policy, the cluster smoke
    test covers the SQL."""

    settings: Settings

    def read(self) -> AbstractContextManager[psycopg.Connection[DictRow]]:
        return read_connection(self.settings)

    def open_rows(
        self, conn: psycopg.Connection[DictRow], fault_name: str
    ) -> list[OpenEpisodeRow]:
        return episode_store.open_rows(conn, fault_name)

    def history_scores(
        self, conn: psycopg.Connection[DictRow], fault_name: str
    ) -> list[float]:
        return episode_store.history_scores(conn, fault_name)

    def apply(
        self,
        fault_name: str,
        inserts: Sequence[Episode],
        updates: Sequence[tuple[int, Episode]],
        orphan_closes: Sequence[tuple[int, datetime]],
        *,
        fingerprint: str,
        externally_delivered: bool,
    ) -> Sequence[EpisodeEvent]:
        with write_connection(self.settings) as conn, conn.transaction():
            return episode_store.apply(
                conn,
                fault_name,
                inserts,
                updates,
                orphan_closes,
                fingerprint=fingerprint,
                externally_delivered=externally_delivered,
            )


@dataclass(frozen=True, slots=True)
class NatsPublisher:
    """The publisher end's NATS adapter."""

    settings: Settings

    def publish_anomaly(
        self,
        fault_name: str,
        severity: str | None,
        payload: dict[str, Any],
        *,
        entity: str | None = None,
        firing: bool = True,
    ) -> None:
        nats_publisher.publish_anomaly(
            self.settings, fault_name, severity, payload, entity=entity, firing=firing
        )

    def publish_episode_event(self, event: EpisodeEvent) -> None:
        nats_publisher.publish_episode_event(self.settings, event)


def declared_fingerprint(fault: Fault) -> str:
    """The default stamp: the rule as the fault file declares it. A kind
    whose code decides what it accuses declares its own, with a revision
    folded in."""
    return fault.fingerprint


class FoldHook[S](Protocol):
    """A kind's own fold: the measurement to episodes, where the default
    observation pipeline does not fit (external folds severity writes,
    seeded by the stored open severities)."""

    def __call__(
        self,
        *,
        fault_name: str,
        measured: Measured[S],
        open_rows: Sequence[OpenEpisodeRow],
    ) -> tuple[Episode, ...]: ...


class PlanHook[S, P](Protocol):
    """A kind's own planning step: the computed episodes against the stored
    open rows, delivered the kind's way — declared only where the default
    per-subject delivery does not fit (silence reports per main group).
    """

    def __call__(
        self,
        *,
        episodes: Sequence[Episode],
        open_rows: Sequence[OpenEpisodeRow],
        measured: Measured[S],
        frontier: datetime,
    ) -> Plan[P]: ...


@dataclass(frozen=True, slots=True)
class Kind[S, P: SubjectPublish]:
    """One fault kind, reduced to what actually differs between them: how
    its series is measured (`frontier`, `measure`) and how its payload is
    shaped (`publish_for`, `payload`) — both live in the kind's own module,
    so the whole wire story of a kind is read in one place. `event` names
    its log record, `delivery` the target form the fault must declare for
    it.

    `measure` receives the fault's open rows beside the window: most kinds
    ignore them, a kind that prunes its scope by what is already open (or
    seeds its fold with stored severities) reads them instead of re-asking
    the store.

    Folding defaults to the pure observation pipeline; planning defaults to
    the shared per-subject delivery through `publish_for` — a kind that
    works another way declares `fold` or `plan` instead (one of
    `publish_for`/`plan` is required). `policy` is how its observations
    fold — hourly, like the aggregates every kind so far reads, unless the
    kind measures in another cadence and declares its own, so that "a few
    quiet runs" is counted in the unit it actually measures in.
    `delivery` is None for the one kind whose faults declare no target at
    all; `payload` is None for the one kind that publishes nothing;
    `warn_dataless` is off for the one kind whose dataless set is routinely
    huge and already accounted for; and `externally_delivered` marks
    episodes someone else already notified about, so nothing downstream
    notifies a second time. `fingerprint` is what the rows get stamped
    with — the declared rule, unless the kind's code is part of the rule
    and says so.

    `history` is how far back the data this kind reads reaches: the
    retention where the store prunes it, and the window a scheduled run
    already reads where the kind reads the raw archive itself. A scheduled
    run's `LOOKBACK` sits inside every one of them; a back-test asking for
    more would read a shortened past and report fewer episodes than the
    rule produced.
    """

    event: str
    delivery: str | None
    frontier: Callable[[psycopg.Connection[DictRow]], datetime | None]
    measure: Callable[
        [psycopg.Connection[DictRow], Fault, Window, Sequence[OpenEpisodeRow]],
        Measured[S],
    ]
    payload: Callable[[P], dict[str, Any]] | None = None
    publish_for: Callable[[str, int, S | None], P] | None = None
    fold: FoldHook[S] | None = None
    plan: PlanHook[S, P] | None = None
    warn_dataless: bool = True
    externally_delivered: bool = False
    policy: EpisodePolicy = EpisodePolicy()
    fingerprint: Callable[[Fault], str] = declared_fingerprint
    history: timedelta = HISTORY


def publish_subjects[P: SubjectPublish](
    publisher: Publisher,
    fault_name: str,
    publishes: Iterable[P],
    payload: Callable[[P], dict[str, Any]],
) -> None:
    """One publish per moved subject, on the subject's own address: the
    severity decides firing, the kind decides the rest of the payload.
    """
    for publish in publishes:
        firing = publish.severity > 0
        publisher.publish_anomaly(
            fault_name,
            severity_name(publish.severity) if firing else None,
            payload(publish),
            entity=publish.entity,
            firing=firing,
        )


def publish_episode_events(
    publisher: Publisher, fault: Fault, events: Iterable[EpisodeEvent]
) -> None:
    """One pointer per recorded episode event, on `episode.<kind>` — for a
    fault declared `explain`, and for nothing else.

    A failure here is a loss, not a delay: the rows are already written, so
    every later run's conflict clause filters the event away rather than
    offering it again. It is named before the run fails on it.
    """
    if not fault.explain:
        return
    for event in events:
        try:
            publisher.publish_episode_event(event)
        except Exception:
            log.exception(
                "episode_event_lost",
                fault=fault.name,
                episode=event.episode_id,
                kind=str(event.kind),
            )
            raise


def check_delivery(fault: Fault, kind: Kind[Any, Any]) -> None:
    """That the fault declares the target form its kind delivers on. The
    loader already forbids a target on a self-delivering fault; this is the
    other side of the same contract, and it holds for a candidate nobody has
    written down yet as much as for a loaded file's entry.
    """
    if kind.delivery is None:
        if fault.target is not None:
            raise ValueError(f"fault {fault.name}: {fault.kind} delivers itself, no target")
    elif fault.target is None or fault.target.form != kind.delivery:
        raise ValueError(
            f"fault {fault.name}: {fault.kind} delivery needs a {kind.delivery} target"
        )


def fold_measured[S, P: SubjectPublish](
    kind: Kind[S, P],
    fault_name: str,
    measured: Measured[S],
    open_rows: Sequence[OpenEpisodeRow],
    history_scores: Sequence[float],
    frontier: datetime,
) -> tuple[Episode, ...]:
    """A measurement folded into episodes the way the kind declares: its own
    fold where it has one (external folds severity writes), the pure
    observation pipeline otherwise, in the kind's own cadence.

    `frontier` is what decides which episodes are still open — never wall
    time racing ahead of a stalled materialization.
    """
    if kind.fold is not None:
        return kind.fold(fault_name=fault_name, measured=measured, open_rows=open_rows)
    return fold_observations(
        fault_name, measured.observations, history_scores, kind.policy, frontier
    )


def run_subjects[S, P: SubjectPublish](
    store: Store, publisher: Publisher, fault: Fault, kind: Kind[S, P], *, dry_run: bool
) -> None:
    """The one shape a per-subject kind runs in: guard the declaration, take
    the window off the aggregate, measure, fold, reconcile, log — then
    publish before writing.
    """
    check_delivery(fault, kind)
    policy = kind.policy

    with store.read() as conn:
        frontier = kind.frontier(conn)
        if frontier is None:
            log.warning("no_aggregate_data", fault=fault.name)
            return
        window = Window(start=frontier - LOOKBACK, frontier=frontier, policy=policy)
        open_rows = store.open_rows(conn, fault.name)
        measured = kind.measure(conn, fault, window, open_rows)
        history_scores = store.history_scores(conn, fault.name)

    if measured.dataless and kind.warn_dataless:
        # Never at info level: a subject nobody could measure is the one
        # thing that keeps an open episode from ever clearing itself.
        log.warning(
            "subjects_dataless", fault=fault.name, subjects=sorted(measured.dataless)
        )

    episodes = fold_measured(
        kind, fault.name, measured, open_rows, history_scores, frontier
    )

    plan = _plan_for(kind, episodes, open_rows, measured, frontier)
    fingerprint = kind.fingerprint(fault)

    log.info(
        kind.event,
        fault=fault.name,
        fingerprint=fingerprint,
        frontier=frontier.isoformat(),
        **measured.record,
        episodes=len(episodes),
        open_episodes=sum(1 for e in episodes if e.ended_at is None),
        inserts=len(plan.inserts),
        updates=len(plan.updates),
        orphan_closes=len(plan.orphan_closes),
        stale_opens=list(plan.stale_opens),
        publishes=len(plan.publishes),
        explain=fault.explain,
        dry_run=dry_run,
    )

    if dry_run:
        log_dry_run(fault, episodes, plan, measured.labels)
        return

    if plan.publishes:
        if kind.payload is None:
            raise ValueError(f"kind {kind.event}: publishes without a payload declaration")
        publish_subjects(publisher, fault.name, plan.publishes, kind.payload)
    recorded = store.apply(
        fault.name,
        plan.inserts,
        plan.updates,
        plan.orphan_closes,
        fingerprint=fingerprint,
        externally_delivered=kind.externally_delivered,
    )
    publish_episode_events(publisher, fault, recorded)


def _plan_for[S, P: SubjectPublish](
    kind: Kind[S, P],
    episodes: Sequence[Episode],
    open_rows: Sequence[OpenEpisodeRow],
    measured: Measured[S],
    frontier: datetime,
) -> Plan[P]:
    """The kind's own plan where it declares one, the shared per-subject
    delivery otherwise."""
    if kind.plan is not None:
        return kind.plan(
            episodes=episodes, open_rows=open_rows, measured=measured, frontier=frontier
        )
    publish_for = kind.publish_for
    if publish_for is None:
        raise ValueError(f"kind {kind.event}: declares neither publish_for nor plan")

    def payload(subject: str, severity: int) -> P:
        return publish_for(subject, severity, measured.states.get(subject))

    return subject_plan(
        episodes=episodes,
        open_rows=open_rows,
        dataless=measured.dataless,
        stranded=measured.stranded,
        frontier=frontier,
        publish_for=payload,
    )


def log_dry_run[P: SubjectPublish](
    fault: Fault,
    episodes: Sequence[Episode],
    plan: Plan[P],
    labels: Mapping[str, str],
) -> None:
    """What the run would have done, by subject — the dry run's whole point,
    so it names each subject the way a human does where the kind knows it.
    """
    per_subject = Counter(e.subject for e in episodes)
    log.info(
        "dry_run_episodes",
        fault=fault.name,
        per_subject={
            labels.get(subject, subject): count
            for subject, count in sorted(per_subject.items())
        },
        open_subjects=sorted(e.subject for e in episodes if e.ended_at is None),
        would_publish=[
            {"subject": p.subject, "severity": p.severity} for p in plan.publishes
        ],
    )
