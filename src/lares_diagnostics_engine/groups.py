"""Delivery per KNX main group: the shape a channel-scoped fault reports in.

Two kinds measure per channel and report per main group — silence and
constancy. Per-channel addresses are impossible (roughly 500 channels carry
data), so the address plan gives every main group a Diagnose address per
fault instead: the address says which fault and roughly where, the payload
names the channels exactly.

Beyond the shared reconciliation both need one rule, and it lives here
once: **a group publishes when its severity moved or its set of open
channels changed**. Without the second half, a channel going wrong at a
tier another channel already holds would ride a stale payload — the group
would still say "one channel", naming the wrong one.

The consistency kind needs the same rule on one house-wide address, with
every summary of the house in one group. So a plan names how its subjects
group and how one group's publish is built, and per main group is the
first of those groupings.

What differs between the kinds is only what a channel report says — how
long silent, how long stuck, at which value — so each hands in the factory
that builds one of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .reconcile import plan_from, reconcile

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from datetime import datetime

    from .episode_store import OpenEpisodeRow
    from .episodes import Episode
    from .reconcile import Measured, Plan


def main_group(ga: str) -> int:
    """The KNX main group of a group address — the granularity these kinds
    report at (one Zentral diagnosis address per main group).
    """
    return int(ga.split("/", 1)[0])


class GroupChannel(Protocol):
    """What a group publish needs of one channel's report: which address it
    speaks for, the severity it carries, and how bad it is in the kind's own
    unit — the last of these orders the group's list, and is None where this
    run could not measure it.
    """

    @property
    def ga(self) -> str: ...

    @property
    def severity(self) -> int: ...

    @property
    def magnitude(self) -> float | None: ...


@dataclass(frozen=True, slots=True)
class GroupPublish[R: GroupChannel]:
    """One main group's severity, with the reports of the channels that
    earned it — worst first."""

    main_group: int
    severity: int
    channels: tuple[R, ...]

    @property
    def subject(self) -> str:
        return str(self.main_group)

    @property
    def entity(self) -> str:
        return str(self.main_group)


class ReportFor[S, R](Protocol):
    """A kind's channel-report factory. The state is None where the subject
    was not measured this run, the episode where it is held open for want of
    data — a report then names the address and the tier it still carries,
    and nothing it cannot know. Positional, so a kind that reads only one of
    the two is free to say so in the parameter's name.
    """

    def __call__(
        self, subject: str, severity: int, state: S | None, episode: Episode | None, /
    ) -> R: ...


class GroupPublishFor[G, R, P](Protocol):
    """A kind's publish factory: one group, the severity it carries now and
    its reports, worst first — none where nothing is open any more."""

    def __call__(self, group: G, severity: int, reports: tuple[R, ...], /) -> P: ...


def group_plan[S, R: GroupChannel, G: (int, str), P](
    *,
    episodes: Sequence[Episode],
    open_rows: Sequence[OpenEpisodeRow],
    measured: Measured[S],
    frontier: datetime,
    report_for: ReportFor[S, R],
    group_of: Callable[[str], G],
    publish_for: GroupPublishFor[G, R, P],
) -> Plan[P]:
    """The shared reconciliation, delivered per group: one publish for every
    group whose severity moved or whose set of open subjects changed."""
    result = reconcile(
        episodes=episodes,
        open_rows=open_rows,
        dataless=measured.dataless,
        frontier=frontier,
        stranded=measured.stranded,
    )
    # The fold leaves at most one open episode per subject, and it is the
    # one the reconciliation carried into `after`.
    open_episodes = {e.subject: e for e in episodes if e.ended_at is None}
    reports = {
        subject: report_for(
            subject, severity, measured.states.get(subject), open_episodes.get(subject)
        )
        for subject, severity in result.after.items()
    }

    before = _group_state(((row.subject, row.severity) for row in open_rows), group_of)
    after = _group_state(
        ((subject, report.severity) for subject, report in reports.items()), group_of
    )

    publishes: list[P] = []
    for group in sorted(set(before) | set(after)):
        if after.get(group) == before.get(group):
            continue
        severity, _ = after.get(group, (0, frozenset()))
        channels = sorted(
            (r for subject, r in reports.items() if group_of(subject) == group),
            key=lambda r: (-r.severity, -(r.magnitude or 0.0), r.ga),
        )
        publishes.append(publish_for(group, severity, tuple(channels)))

    return plan_from(result, publishes)


def _group_state[G](
    subject_severities: Iterable[tuple[str, int]], group_of: Callable[[str], G]
) -> dict[G, tuple[int, frozenset[str]]]:
    """Per group: the maximum severity and the set of open subjects — the
    two things whose change warrants a publish."""
    severities: dict[G, int] = {}
    subjects: dict[G, set[str]] = {}
    for subject, severity in subject_severities:
        group = group_of(subject)
        severities[group] = max(severities.get(group, 0), severity)
        subjects.setdefault(group, set()).add(subject)
    return {g: (severities[g], frozenset(subjects[g])) for g in severities}
