"""Episode-pipeline tests — seam 3 of the detection rebuild, the load-bearing one.

Every test feeds invented observations into the one seam and asserts only
what comes out: episodes with a severity trajectory and at most three
notification events. Prior art: the severity tests.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from lares_diagnostics_engine.episodes import (
    Entity,
    Episode,
    EpisodePolicy,
    EventKind,
    Observation,
    fold_observations,
)

_T0 = datetime(2026, 8, 30, 10, 0, tzinfo=UTC)
_HOUR = timedelta(hours=1)

# A quarter of score history with a tail: p99 lands at 1.2 and p99.9 just
# above 5, so 1.0 is info, 1.5 is warning and 9.0 is critical.
_HISTORY = [i / 1000 for i in range(989)] + [1.2] * 9 + [5.0, 8.0]


def _hourly(
    scores: list[float], subject: str = "2/2/227", start: datetime = _T0
) -> list[Observation]:
    return [
        Observation(subject=subject, time=start + i * _HOUR, score=score)
        for i, score in enumerate(scores)
    ]


def _fold(
    observations: list[Observation],
    history: list[float] | None = None,
    policy: EpisodePolicy | None = None,
    now: datetime | None = None,
) -> tuple[Episode, ...]:
    return fold_observations(
        fault="freezer_runtime",
        observations=observations,
        history_scores=_HISTORY if history is None else history,
        policy=policy or EpisodePolicy(),
        now=now or (_T0 + 24 * _HOUR),
    )


# --- one incident, one episode -------------------------------------------------


def test_six_hourly_observations_fold_into_one_episode() -> None:
    episodes = _fold(_hourly([1.0] * 6))

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.fault == "freezer_runtime"
    assert episode.subject == "2/2/227"
    assert episode.started_at == _T0
    assert episode.last_seen_at == _T0 + 5 * _HOUR
    # The evidence rows carry one severity per bucket — the course of the night.
    assert [e.time for e in episode.evidence] == [_T0 + i * _HOUR for i in range(6)]
    assert all(1 <= e.severity <= 3 for e in episode.evidence)


def test_flickering_fault_stays_one_episode() -> None:
    # Quiet for one run, back the next — one incident, not three.
    times = [0, 1, 3, 5, 7]
    observations = [Observation("2/2/227", _T0 + i * _HOUR, 1.0) for i in times]

    episodes = _fold(observations, policy=EpisodePolicy(quiet_runs=1))

    assert len(episodes) == 1
    assert episodes[0].started_at == _T0
    assert episodes[0].last_seen_at == _T0 + 7 * _HOUR


def test_quiet_beyond_quiet_runs_splits_episodes() -> None:
    observations = [
        Observation("2/2/227", _T0, 1.0),
        # Five silent runs with quiet_runs=3: the incident is over, this is a new one.
        Observation("2/2/227", _T0 + 6 * _HOUR, 1.0),
    ]

    episodes = _fold(observations, policy=EpisodePolicy(quiet_runs=3))

    assert len(episodes) == 2


def test_subjects_never_share_an_episode() -> None:
    observations = _hourly([1.0] * 3, subject="2/2/227") + _hourly([1.0] * 3, subject="2/2/228")

    episodes = _fold(observations)

    assert len(episodes) == 2
    assert {e.subject for e in episodes} == {"2/2/227", "2/2/228"}


# --- severity from the fault's own distribution --------------------------------


def test_severity_comes_from_the_faults_own_distribution() -> None:
    # 1.05 sits inside the calm quarter, 1.5 beats its p99, 9.0 beats its p99.9.
    episodes = _fold(_hourly([1.05, 1.5, 9.0]))

    assert [e.severity for e in episodes[0].evidence] == [1, 2, 3]
    assert episodes[0].severity == 3
    assert episodes[0].peak_score == 9.0


def test_without_history_every_score_is_info() -> None:
    # No distribution to be rare against — the old global ladder is gone.
    episodes = _fold(_hourly([1000.0, 2000.0]), history=[])

    assert [e.severity for e in episodes[0].evidence] == [1, 1]


def test_duration_promotes_one_tier() -> None:
    policy = EpisodePolicy(promote_after_runs=4)
    episodes = _fold(_hourly([1.0] * 6), policy=policy)

    # Hour 0-3: an info-grade score. Hour 4: still standing — promoted.
    assert [e.severity for e in episodes[0].evidence] == [1, 1, 1, 1, 2, 2]
    assert episodes[0].severity == 2


def test_promotion_never_exceeds_critical() -> None:
    episodes = _fold(_hourly([9.0] * 8), policy=EpisodePolicy(promote_after_runs=2))

    assert all(e.severity == 3 for e in episodes[0].evidence[2:])


# --- notification events -------------------------------------------------------


def test_episode_emits_at_most_three_events() -> None:
    # A long, worsening, recovered night: appear, escalate, end — and no more.
    episodes = _fold(
        _hourly([1.0, 1.5, 9.0, 1.5, 9.0, 1.0]),
        policy=EpisodePolicy(quiet_runs=2),
        now=_T0 + 24 * _HOUR,
    )

    events = episodes[0].events
    assert len(events) <= 3
    assert [e.kind for e in events] == [EventKind.APPEARED, EventKind.ESCALATED, EventKind.ENDED]


def test_appeared_event_carries_first_severity() -> None:
    episodes = _fold(_hourly([1.0, 9.0]))

    appeared = episodes[0].events[0]
    assert appeared.kind is EventKind.APPEARED
    assert appeared.time == _T0
    assert appeared.severity == 1


def test_escalated_fires_once_at_the_first_rise() -> None:
    # Severity rises twice (1 -> 2 -> 3); still exactly one escalation event.
    episodes = _fold(_hourly([1.0, 1.5, 9.0]))

    escalations = [e for e in episodes[0].events if e.kind is EventKind.ESCALATED]
    assert len(escalations) == 1
    assert escalations[0].time == _T0 + 1 * _HOUR
    assert escalations[0].severity == 2


def test_flat_episode_never_escalates() -> None:
    episodes = _fold(_hourly([1.0] * 4))

    assert EventKind.ESCALATED not in {e.kind for e in episodes[0].events}


def test_ended_event_clears_after_quiet_runs() -> None:
    policy = EpisodePolicy(quiet_runs=3)
    episodes = _fold(_hourly([1.0] * 2), policy=policy, now=_T0 + 24 * _HOUR)

    ended = episodes[0].events[-1]
    assert ended.kind is EventKind.ENDED
    assert ended.severity == 0
    # Decidable at the first run that no observation can continue the episode.
    assert ended.time == _T0 + 1 * _HOUR + 4 * _HOUR
    assert episodes[0].ended_at == ended.time


def test_open_episode_has_no_ended_event() -> None:
    policy = EpisodePolicy(quiet_runs=3)
    # Last seen one run ago — still inside the quiet window.
    episodes = _fold(_hourly([1.0] * 2), policy=policy, now=_T0 + 2 * _HOUR)

    assert episodes[0].ended_at is None
    assert EventKind.ENDED not in {e.kind for e in episodes[0].events}


def test_still_open_at_the_deciding_run() -> None:
    # At close_time itself, that run's observation may not have arrived yet —
    # the end is decidable only strictly past it.
    policy = EpisodePolicy(quiet_runs=3)
    close_time = _T0 + 1 * _HOUR + 4 * _HOUR
    episodes = _fold(_hourly([1.0] * 2), policy=policy, now=close_time)

    assert episodes[0].ended_at is None


def test_observation_at_close_time_continues_the_episode() -> None:
    policy = EpisodePolicy(quiet_runs=3)
    observations = [
        Observation("2/2/227", _T0, 1.0),
        # Exactly quiet_runs missed runs, back on the next — still one incident.
        Observation("2/2/227", _T0 + 4 * _HOUR, 1.0),
    ]

    episodes = _fold(observations, policy=policy)

    assert len(episodes) == 1


# --- policy validation ---------------------------------------------------------


def test_policy_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        EpisodePolicy(quiet_runs=0)
    with pytest.raises(ValueError):
        EpisodePolicy(promote_after_runs=0)
    with pytest.raises(ValueError):
        EpisodePolicy(bucket=timedelta(0))
    with pytest.raises(ValueError):
        EpisodePolicy(warning_quantile=0.999, critical_quantile=0.99)


def test_no_observations_no_episodes() -> None:
    assert _fold([]) == ()


def test_episode_carries_what_its_subject_names() -> None:
    """The detector's word travels from the observation to the row, so a
    reader never has to guess what a subject is."""
    room = Entity(kind="room", ref="8/2/80")
    episodes = fold_observations(
        "fbh_cold",
        [
            Observation(subject="eg-buero", time=_T0, score=9.0, entity=room),
            Observation(subject="eg-buero", time=_T0 + _HOUR, score=9.0, entity=room),
        ],
        _HISTORY,
        EpisodePolicy(),
        now=_T0 + 10 * _HOUR,
    )
    assert [e.entity for e in episodes] == [room]


def test_an_episode_without_a_named_entity_keeps_none() -> None:
    """A channel fault says nothing; the store reads its subject instead."""
    episodes = fold_observations(
        "channel_silence",
        [Observation(subject="15/2/0", time=_T0, score=9.0)],
        _HISTORY,
        EpisodePolicy(),
        now=_T0 + 10 * _HOUR,
    )
    assert episodes[0].entity is None
