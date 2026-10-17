"""The shape every per-subject fault kind has: measure, fold, reconcile,
deliver.

A kind that reports per subject — a device, a room, an appliance's standby
— differs from its siblings in exactly two things: how its series is
measured (what leaves that step is a `Measured`), and how its payload is
shaped (a `SubjectPublish`). Everything between those two ends is the same
for all of them and lives here, so a new kind declares the two ends and
inherits the rest.

Reconciliation is the computed episodes against the stored open rows, and
the guarantees are the interesting part rather than the bookkeeping:

* only open computed episodes materialize as new rows; a computed episode
  that already ended matters only to close the open row it reconciles;
* a stored severity is never lowered — the recompute window may have slid
  past the peak;
* an open row whose subject produced no episode closes at the frontier,
  unless the subject is `dataless` — and closes even then if the row is
  `stranded`.

`dataless` means one thing in every kind, and `measurement_reaches` is
where it is said: a subject is dataless when this run cannot tell a
recovery from a blind spot, because its measurement does not reach the
frontier. With no data to decide a recovery, its episode stays open
instead of self-clearing.

`stranded` is the one thing that overrides it, and it is about the row,
not the subject: a rule other than this run's left the row on a subject
this run declines to judge. Nothing this run does would open it, so
nothing is being recovered from, and the row closes rather than being
held for data that could never vindicate it. Which rows those are is the
kind's to say — silence reads it off the fingerprint every row carries.

What leaves a reconciliation is `after` — the severity every subject with
an open episode carries now, the ones held open for want of data included
— and `moved`, those of them whose severity changed, including the 0 that
ends an episode. A per-subject kind turns `moved` straight into its
publishes; channel silence needs `after`, because it reports the maximum
per main group rather than per subject.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

from .episodes import Episode

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from .episode_store import OpenEpisodeRow
    from .episodes import EpisodePolicy, Observation


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """What one run changes: new episodes, reconciled open rows, orphaned
    rows to close, rows kept open for want of data, the severity every open
    subject carries now, and those of them whose severity moved.
    """

    inserts: tuple[Episode, ...]
    updates: tuple[tuple[int, Episode], ...]
    orphan_closes: tuple[tuple[int, datetime], ...]
    stale_opens: tuple[str, ...]
    after: Mapping[str, int]
    moved: tuple[tuple[str, int], ...]


def measurement_reaches(
    last: datetime | None, *, frontier: datetime, max_gap: timedelta
) -> bool:
    """Whether a subject's measurement reaches the present — the difference
    between "recovered" and "unmeasured", and the whole of what `dataless`
    means, in every kind.

    A subject nobody could measure produces no observations for exactly the
    same reason a repaired one does. Without this test the episode would end
    on missing data and clear an address whose fault still stands, so
    anything short of the frontier by more than the episode pipeline's own
    gap counts as unmeasurable rather than well.
    """
    return last is not None and frontier - last <= max_gap


@dataclass(frozen=True, slots=True)
class Window:
    """The run's measurement window: where it starts, the aggregate's
    frontier it is measured against, and the fold policy — which also
    decides how far behind that frontier a measurement may lag before the
    subject counts as unmeasured.
    """

    start: datetime
    frontier: datetime
    policy: EpisodePolicy

    @property
    def lookback(self) -> timedelta:
        return self.frontier - self.start

    def reaches(self, last: datetime | None) -> bool:
        """`measurement_reaches`, against this window."""
        return measurement_reaches(
            last, frontier=self.frontier, max_gap=self.policy.max_gap
        )


_NO_LABELS: Mapping[str, str] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class Measured[S]:
    """What a kind's measurement saw over its whole scope: the state per
    subject its payload is shaped from, the observations they produced, the
    subjects whose measurement did not reach the frontier, the fields its
    run record names, what a human calls each subject, the open rows it
    disowns as another rule's, by subject, where the kind can tell — and,
    for a kind that reads from a cursor, how far this measurement read.
    """

    states: Mapping[str, S]
    observations: tuple[Observation, ...]
    dataless: frozenset[str]
    record: Mapping[str, Any]
    labels: Mapping[str, str] = _NO_LABELS
    stranded: frozenset[str] = frozenset()
    read_through: datetime | None = None


class SubjectPublish(Protocol):
    """What the runner needs of any per-subject payload: the subject it
    speaks for, the address token it is delivered on (None for the one
    house-wide 1:1 subject), and the severity that decides firing.
    Everything else in the payload is the kind's own.
    """

    @property
    def subject(self) -> str: ...

    @property
    def entity(self) -> str | None: ...

    @property
    def severity(self) -> int: ...


@dataclass(frozen=True, slots=True)
class Plan[P]:
    """What one run changes: the reconciled rows, the publishes the kind
    shaped out of the subjects whose severity moved, and — for a kind that
    reads from a cursor — how far this run read, the point the cursor moves
    to once the rows are written.
    """

    inserts: tuple[Episode, ...]
    updates: tuple[tuple[int, Episode], ...]
    orphan_closes: tuple[tuple[int, datetime], ...]
    stale_opens: tuple[str, ...]
    publishes: tuple[P, ...]
    read_through: datetime | None = None


def plan_from[P](result: Reconciliation, publishes: Iterable[P]) -> Plan[P]:
    """The reconciled rows carried into a plan beside the kind's publishes —
    the one place those four fields are copied across.
    """
    return Plan(
        inserts=result.inserts,
        updates=result.updates,
        orphan_closes=result.orphan_closes,
        stale_opens=result.stale_opens,
        publishes=tuple(publishes),
    )


def subject_plan[P](
    *,
    episodes: Sequence[Episode],
    open_rows: Sequence[OpenEpisodeRow],
    dataless: frozenset[str],
    frontier: datetime,
    publish_for: Callable[[str, int], P],
    stranded: frozenset[str] = frozenset(),
) -> Plan[P]:
    """Reconcile, then deliver per subject: a publish goes out when a
    subject's severity moved, including the 0 when its episode ends.
    """
    result = reconcile(
        episodes=episodes,
        open_rows=open_rows,
        dataless=dataless,
        frontier=frontier,
        stranded=stranded,
    )
    return plan_from(
        result, (publish_for(subject, severity) for subject, severity in result.moved)
    )


def reconcile(
    *,
    episodes: Sequence[Episode],
    open_rows: Sequence[OpenEpisodeRow],
    dataless: frozenset[str],
    frontier: datetime,
    stranded: frozenset[str] = frozenset(),
) -> Reconciliation:
    """Pure reconciliation — the guarantees are in the module docstring."""
    open_by_subject = {row.subject: row for row in open_rows}

    # Several episodes per subject can fall in the window; the stored open
    # row can only correspond to the latest one.
    latest_by_subject: dict[str, Episode] = {}
    for episode in episodes:
        current = latest_by_subject.get(episode.subject)
        if current is None or episode.started_at > current.started_at:
            latest_by_subject[episode.subject] = episode

    inserts: list[Episode] = []
    updates: list[tuple[int, Episode]] = []
    after: dict[str, int] = {}

    for subject, episode in sorted(latest_by_subject.items()):
        row = open_by_subject.get(subject)
        if episode.ended_at is None:
            after[subject] = max(episode.severity, row.severity if row else 0)
            if row is not None:
                updates.append((row.id, episode))
            else:
                inserts.append(episode)
        elif row is not None:
            updates.append((row.id, episode))

    orphan_closes: list[tuple[int, datetime]] = []
    stale_opens: list[str] = []
    for row in open_rows:
        if row.subject in latest_by_subject:
            continue
        if row.subject in dataless and row.subject not in stranded:
            stale_opens.append(row.subject)
            # Still firing as far as anyone can tell — it keeps counting.
            after[row.subject] = row.severity
        else:
            orphan_closes.append((row.id, frontier))

    before = {row.subject: row.severity for row in open_rows}
    moved = tuple(
        (subject, after.get(subject, 0))
        for subject in sorted(set(before) | set(after))
        if before.get(subject) != after.get(subject)
    )

    return Reconciliation(
        inserts=tuple(inserts),
        updates=tuple(updates),
        orphan_closes=tuple(orphan_closes),
        stale_opens=tuple(stale_opens),
        after=MappingProxyType(after),
        moved=moved,
    )
