"""Back-test: what one candidate fault would have found in the last N weeks.

A proposed fault is only reviewable with a number beside it — "this rule
would have opened two episodes in eight weeks, here are the channels and the
dates" — and zero and fifty are both grounds to say no. This module answers
that question for an entry nobody has written down yet: the candidate goes
through the same schema, the same kind registry, the same measurement and
the same fold the detect-faults job runs, over a window of whole weeks off
the aggregate's frontier.

It is a library entry point, not a job: no CronJob, no subcommand, no
endpoint. `lares-mcp-bridge` depends on the engine at its deployed tag and
exposes this as a tool, the way the lares generator imports the loader and
the entity slug.

Read-only by construction. It takes one read connection and no store or
publisher end, so there is nothing here that could write a row or put a
message on the bus; the stored episodes are never even read for the
candidate's own name. Two consequences of that are worth naming:

* the severities are the ones a rule without history produces — info, one
  tier up where an episode stood long enough — because a candidate has no
  stored distribution to be rare against. That is the engine's own rule
  (`episodes.severity_cutoffs`) and not a fallback, and the alternative is
  worse: a window judged against its own scores calls its own worst bucket
  critical every time. What a candidate is judged by is the peak score, in
  the fault's own declared unit;
* nothing is reconciled. What comes back is the episodes the rule produces
  from the data, never a plan against rows somebody else's rule left behind.

The window is the one thing the caller chooses, and it is bounded by what
the measurement can read: a year of aggregates for most shapes, 90 days for
the plant's daily yield (the forecast it is short of is kept that long), 30
days for the duty-cycle signal (the one that reads the bus archive itself).
Each kind declares its own. A window past it would read a shortened past and
report fewer episodes than the rule really produced — the one wrong answer a
back-test must not give — so it is refused rather than trimmed.

One honest limitation: the window is measured in a single pass, where the
job slides a 30-day window hourly. For a rule whose measurement carries its
own trailing basis — silence's normal pause, drift's healthy window — the
basis here is the whole window, so the counts are the right order of
magnitude and not the identical run-by-run history.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from .faults import MeasurementKind, parse_entry
from .kinds import kind_for
from .reconcile import Window
from .runner import Kind, check_delivery, fold_measured

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import psycopg
    from psycopg.rows import DictRow

    from .episodes import Episode
    from .site import Site


@dataclass(frozen=True, slots=True)
class BacktestEpisode:
    """One episode the candidate would have produced: the subject it accuses
    — a channel, a device, a room, the house — as a human and as the engine
    name it, when it ran, how bad it got in tiers and in the fault's own
    unit, and how many per-bucket observations carried it.
    """

    subject: str
    label: str
    started_at: datetime
    ended_at: datetime | None
    severity: int
    peak_score: float
    observations: int


@dataclass(frozen=True, slots=True)
class Backtest:
    """What the candidate would have done: the episodes, the window they were
    found in, and the measurement's own record — how many channels the scope
    resolved to, how many of them were candidates at all. Without that
    record no episodes is unreadable: a rule that found nothing and a scope
    that matched nothing look the same.
    """

    fault: str
    kind: MeasurementKind
    window_start: datetime
    frontier: datetime
    measured: Mapping[str, Any]
    episodes: tuple[BacktestEpisode, ...]


def backtest_fault(
    conn: psycopg.Connection[DictRow],
    entry: Mapping[str, Any],
    *,
    weeks: int,
    site: Site | None = None,
) -> Backtest:
    """Evaluate one candidate fault entry against history.

    `entry` is one entry of the fault file, in that schema — a candidate, or
    a copy of a declared fault with a threshold moved. `weeks` is how far
    back to measure, in whole weeks off the aggregate's frontier. `site` is
    needed only by the one shape that measures the plant's daily yield.

    Everything that cannot be measured is an error naming what to fix, never
    an empty result: a candidate the schema rejects, an `external` fault
    (Basalte detects those, there is nothing here to re-run), a target form
    its kind does not deliver on, a window past what this measurement can
    read, and an aggregate that holds no data at all.
    """
    fault = parse_entry(entry)
    if fault.kind is MeasurementKind.EXTERNAL:
        raise ValueError(
            f"fault {fault.name}: Basalte detects and delivers an external fault — "
            f"the engine only records its writes, so there is no rule to back-test"
        )

    kind = kind_for(fault, site)
    if kind is None:
        raise ValueError(f"fault {fault.name}: no kind in this engine measures {fault.kind}")
    check_delivery(fault, kind)
    lookback = _lookback(weeks, kind, fault.name)

    frontier = kind.frontier(conn)
    if frontier is None:
        raise ValueError(
            f"fault {fault.name}: the aggregate this kind measures holds no data — "
            f"nothing to back-test against"
        )
    window = Window(start=frontier - lookback, frontier=frontier, policy=kind.policy)
    # A candidate has produced no episodes: no open rows to reconcile
    # against, and no score history to be rare against.
    measured = kind.measure(conn, fault, window, ())
    episodes = fold_measured(kind, fault.name, measured, (), (), frontier)

    return Backtest(
        fault=fault.name,
        kind=fault.kind,
        window_start=window.start,
        frontier=frontier,
        measured=dict(measured.record),
        episodes=_reported(episodes, measured.labels),
    )


def _lookback(weeks: int, kind: Kind[Any, Any], fault_name: str) -> timedelta:
    """The window as an interval, or an error saying why this one cannot be
    measured: the kind declares how far back the data it reads reaches."""
    if weeks < 1:
        raise ValueError(f"a back-test window is at least one week, not {weeks}")
    lookback = timedelta(weeks=weeks)
    if lookback > kind.history:
        raise ValueError(
            f"fault {fault_name}: a window of {weeks} weeks reaches past the "
            f"{kind.history.days // 7} weeks this measurement can read — it would "
            f"report fewer episodes than the rule produced"
        )
    return lookback


def _reported(
    episodes: Sequence[Episode], labels: Mapping[str, str]
) -> tuple[BacktestEpisode, ...]:
    """The episodes as a reviewer reads them: oldest first, each subject
    named the way its kind knows it."""
    return tuple(
        BacktestEpisode(
            subject=episode.subject,
            label=labels.get(episode.subject, episode.subject),
            started_at=episode.started_at,
            ended_at=episode.ended_at,
            severity=episode.severity,
            peak_score=episode.peak_score,
            observations=len(episode.evidence),
        )
        for episode in sorted(episodes, key=lambda e: (e.started_at, e.subject))
    )
