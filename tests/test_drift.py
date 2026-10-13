"""Drift-measurement tests — the `drift` kind against invented series.

Every test feeds an invented series and asserts only what comes out: the
level the signal reads (a standby valley, a duty cycle), the accumulation,
observations with their scores, the current state, the published payload,
or a resolution error naming the device. The shared reconciliation is
`test_reconcile`'s; no cluster, no live database.

The load-bearing cases are the two the z-score provably could not tell
apart: a slow ramp must fire, and a small permanent step must never fire,
however long it stands. For the duty-cycle signal a third joins them: a
door left open must not read as an iced evaporator.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from lares_diagnostics_engine.drift import (
    Ambient,
    Device,
    DevicePublish,
    DeviceState,
    Exchanger,
    ExchangerPublish,
    ExchangerState,
    OnTime,
    accumulate,
    accumulate_fall,
    ambient_levels,
    capability_levels,
    classify,
    classify_exchanger,
    door_hours,
    drift_observations,
    duty_cycles,
    efficiency_series,
    min_window_samples,
    publish_for,
    publish_for_exchanger,
    reaches_frontier,
    resolve_devices,
    resolve_exchanger,
    standby_floors,
)
from lares_diagnostics_engine.episodes import (
    EpisodePolicy,
    Observation,
    fold_observations,
)
from lares_diagnostics_engine.faults import AmbientReference, DeviceReference, ExchangerRoles
from lares_diagnostics_engine.silence import Channel

_T0 = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)
_HOUR = timedelta(hours=1)
_MINUTE = timedelta(minutes=1)
_DAY = timedelta(hours=24)

# The declared numbers of the appliance_standby fault.
_RISE = 40.0
_BUDGET = 480.0

# The declared numbers of the freezer_icing fault: a healthy freezer runs
# its compressor half the day, 15 points more is the line, and a stretch
# the compressor never leaves for three hours is a door, not ice.
_HEALTHY_DUTY = 50.0
_DUTY_RISE = 15.0
_DUTY_BUDGET = 600.0
_DOOR_RUN = timedelta(hours=3)
# The icing fault's declared coverage, through the engine's own reading of
# it: the hours a door event covers are cut out of the day, so it asks for
# less of the day than the standby fault does.
_DUTY_COVERAGE = min_window_samples(_DAY, 0.6)

_FREEZER = Channel(
    ga="2/2/227", name="Schalten.EG.Küche.K15-L1.Gefrierschrank.Stromwert", dpt="7.012"
)
_KITCHEN = Channel(ga="8/2/151", name="Sensorik.EG.Küche.Sensor.Temperatur", dpt="9.001")
# The freezer's reference as it follows the kitchen: 78 % of the day at
# 21 °C, two points more per kelvin — read off two winter fortnights.
_KITCHEN_REFERENCE = DeviceReference(
    match="Küche.K15-L1.Gefrierschrank",
    healthy=78,
    ambient=AmbientReference(channel="%.Küche.Sensor.Temperatur", at_c=21, per_kelvin=2.0),
)
_KITCHEN_AMBIENT = Ambient(ga=_KITCHEN.ga, name=_KITCHEN.name, at_c=21, per_kelvin=2.0)
_ROOM_RISE = 5.0
_ROOM_BUDGET = 240.0
_WASHER = Channel(
    ga="2/1/197", name="Schalten.KG.Hauswirtschaftsraum.K4-L1.Waschmaschine.Stromwert", dpt="7.012"
)

_REFERENCES = (
    DeviceReference(match="Hauswirtschaftsraum.K4-L1.Waschmaschine", healthy=0),
    DeviceReference(match="Küche.K15-L1.Gefrierschrank", healthy=48),
)


def _device(channel: Channel, reference: DeviceReference) -> Device:
    return Device(
        ga=channel.ga,
        name=channel.name,
        label=reference.match,
        healthy=reference.healthy,
    )


def _series(values: Sequence[float], start: datetime = _T0) -> list[tuple[datetime, float]]:
    return [(start + n * _HOUR, v) for n, v in enumerate(values)]


def _floors(values: Sequence[float], start: datetime = _T0) -> list[tuple[datetime, float]]:
    """A standby-valley series straight from values, skipping the rolling
    window — the accumulation tests are about the CUSUM, not the valley."""
    return _series(values, start)


class TestStandbyFloors:
    def test_valley_is_the_lowest_hour_of_the_trailing_day(self) -> None:
        # A freezer idling at 48 mA whose compressor lifts whole hours.
        hourly = _series([48, 270, 48, 390, 55, 48] * 4)
        floors = standby_floors(hourly, window=_DAY, min_samples=20)
        # The first 19 hours carry no day behind them; the rest read the
        # valley through the compressor's hours.
        assert [v for _, v in floors] == [48.0] * 5

    def test_no_valley_before_the_window_is_covered(self) -> None:
        assert standby_floors(_series([48] * 19), window=_DAY, min_samples=20) == []

    def test_a_thin_window_yields_no_sample(self) -> None:
        # Every third hour missing: 16 buckets span the day, under the gate.
        thin = [(_T0 + n * 3 * _HOUR, 48.0) for n in range(16)]
        assert standby_floors(thin, window=_DAY, min_samples=20) == []

    def test_the_valley_follows_the_window(self) -> None:
        # A day at 48, then a day at 100: the valley crosses over once the
        # last 48 has left the trailing day.
        hourly = _series([48.0] * 24 + [100.0] * 24)
        floors = dict(standby_floors(hourly, window=_DAY, min_samples=20))
        assert floors[_T0 + 23 * _HOUR] == 48.0
        assert floors[_T0 + 40 * _HOUR] == 48.0
        assert floors[_T0 + 47 * _HOUR] == 100.0


def _samples(duty: Sequence[float], start: datetime = _T0) -> list[OnTime]:
    """An hourly on-time series from duty cycles in percent — 100 means the
    compressor never stopped in that hour."""
    return [
        OnTime(time=start + n * _HOUR, on=d / 100 * _HOUR, total=_HOUR)
        for n, d in enumerate(duty)
    ]


def _duty(buckets: Sequence[OnTime], min_samples: int = 0) -> list[tuple[datetime, float]]:
    """The levels the icing fault would read off this series."""
    return duty_cycles(
        buckets,
        window=_DAY,
        min_samples=min_samples or _DUTY_COVERAGE,
        excluded=door_hours(buckets, door_run=_DOOR_RUN),
    )


class TestDoorHours:
    """A door left open is the one thing that looks like icing and is not:
    the compressor runs through instead of cycling. It is excluded by its
    signature — whole hours without a single idle sample, several in a row."""

    def test_a_stretch_the_compressor_never_leaves_is_a_door(self) -> None:
        # Four hours flat out in the middle of ordinary cycling.
        hours = door_hours(_samples([60] * 4 + [100] * 4 + [60] * 4), door_run=_DOOR_RUN)
        assert sorted(hours) == [_T0 + n * _HOUR for n in range(4, 8)]

    def test_a_short_saturated_stretch_is_ordinary_running(self) -> None:
        # Two hours flat out is a long cycle after a warm load, not a door.
        short = _samples([60] * 4 + [100] * 2 + [60] * 4)
        assert door_hours(short, door_run=_DOOR_RUN) == frozenset()

    def test_a_gap_ends_the_stretch(self) -> None:
        # Two saturated hours, a normal hour, two more: no door anywhere.
        broken = _samples([100] * 2 + [70] + [100] * 2)
        assert door_hours(broken, door_run=_DOOR_RUN) == frozenset()


class TestDutyCycles:
    def test_the_level_is_the_share_of_the_trailing_window(self) -> None:
        levels = _duty(_samples([40] * 24))
        # Every bucket with a covered day behind it, from the 15th on.
        assert [t for t, _ in levels] == [_T0 + n * _HOUR for n in range(14, 24)]
        assert [v for _, v in levels] == pytest.approx([40.0] * 10)

    def test_no_level_before_the_window_is_covered(self) -> None:
        assert _duty(_samples([40] * 14)) == []

    def test_hours_weigh_by_their_time_not_by_their_ratio(self) -> None:
        # A quarter-hour of delivery says less about the day than a full
        # one — a mean of hourly ratios would call this 62.5 %.
        uneven = [
            OnTime(time=_T0, on=_HOUR / 4, total=_HOUR / 4),
            OnTime(time=_T0 + _HOUR, on=_HOUR / 4, total=_HOUR),
            OnTime(time=_T0 + 2 * _HOUR, on=_HOUR / 4, total=_HOUR),
        ]
        [(_, level)] = _duty(uneven, min_samples=2)
        # The thin hour is a delivery gap and is left out altogether.
        assert level == pytest.approx(25.0)

    def test_a_door_event_is_not_counted_as_duty(self) -> None:
        # A day of ordinary cycling around a five-hour door event: the level
        # reads the cycling, not the door, so the day stays healthy.
        levels = _duty(_samples([50] * 10 + [100] * 5 + [50] * 9))
        assert levels
        assert [v for _, v in levels] == pytest.approx([50.0] * len(levels))

    def test_the_series_stays_hourly_across_a_door_event(self) -> None:
        # The load-bearing one: a hole here would be an observation gap, and
        # the episode pipeline would split one iced freezer into one episode
        # a day. The level is read *at* the door hours, only never *from*
        # them.
        levels = _duty(_samples([80] * 24 + [100] * 5 + [80] * 24))
        times = [t for t, _ in levels]
        assert max(b - a for a, b in zip(times, times[1:], strict=False)) == _HOUR
        assert times[-1] == _T0 + 52 * _HOUR

    def test_a_day_mostly_spent_with_the_door_open_is_unmeasurable(self) -> None:
        # Ten hours of cycling left of 24 is under the declared coverage:
        # better no reading than one taken around an open door.
        assert _duty(_samples([50] * 6 + [100] * 14 + [50] * 4)) == []

    def test_a_delivery_gap_is_not_a_reading(self) -> None:
        # Two minutes of telegrams in an hour says nothing about that hour.
        thin = [OnTime(time=_T0 + n * _HOUR, on=timedelta(), total=2 * _MINUTE) for n in range(24)]
        assert _duty(thin) == []


class TestIcing:
    """The freezer_icing fault end to end over its own signal: duty cycles
    in, observations out, with the declared numbers of the fault file."""

    def _levels(self, duty: Sequence[float]) -> list[tuple[datetime, float]]:
        return _duty(_samples(duty))

    def _observations(self, duty: Sequence[float]) -> list[Observation]:
        trace = accumulate(self._levels(duty), healthy=_HEALTHY_DUTY, rise=_DUTY_RISE)
        return drift_observations(
            _FREEZER.ga, trace, rise=_DUTY_RISE, budget=_DUTY_BUDGET
        )

    def test_an_iced_evaporator_fires(self) -> None:
        # The freezer as measured in June: the same cold bought with 80 % of
        # the day instead of half of it.
        duty = [80] * 24 * 5
        observations = self._observations(duty)
        assert observations
        # 30 points high, 15 of them past the declared rise: the 600
        # point-hour budget fills 40 h after the first reading, not before.
        first_level, _ = self._levels(duty)[0]
        assert observations[0].time == first_level + 40 * _HOUR
        assert observations[0].value == pytest.approx(30.0)
        assert observations[0].score == pytest.approx(30.0 / _DUTY_RISE)

    def test_a_healthy_freezer_with_a_daily_door_event_never_fires(self) -> None:
        # Five hours of door a day for a month, ordinary cycling around it:
        # the acceptance criterion that door events must not fire icing.
        month = ([50] * 10 + [100] * 5 + [50] * 9) * 30
        assert self._observations(month) == []

    def test_a_warm_load_for_a_day_does_not_fire(self) -> None:
        # A full shopping load run in at 90 % duty for a day, then back to
        # normal: high, but not standing — the budget is what tells them
        # apart, and it must not fill here.
        assert self._observations([50] * 24 + [90] * 24 + [50] * 48) == []

    def test_an_iced_freezer_whose_door_is_used_is_one_episode(self) -> None:
        # Icing and a five-hour door event every afternoon, ten days on end.
        # The door hours are cut out of the reading but not out of the
        # series, so this stays the one situation it is instead of clearing
        # and re-firing the address daily.
        duty = ([80] * 10 + [100] * 5 + [80] * 9) * 10
        observations = self._observations(duty)
        assert observations
        episodes = fold_observations(
            "freezer_icing", observations, [], EpisodePolicy(), self._levels(duty)[-1][0]
        )
        assert len(episodes) == 1
        assert episodes[0].ended_at is None

    def test_a_compressor_that_never_stops_is_unmeasured_not_healthy(self) -> None:
        # Days on end without a single idle minute: at this resolution that
        # is a door standing open, and it is Basalte's fault to report. The
        # engine says nothing rather than guessing — no level at all, which
        # makes the device dataless and holds any open episode open.
        never_idle = _samples([100] * 24 * 3)
        assert _duty(never_idle) == []
        assert self._observations([100] * 24 * 3) == []

    def _room_observations(self, duty: Sequence[float], kitchen_c: float) -> list[Observation]:
        levels = self._levels(duty)
        healthy = {t: _KITCHEN_AMBIENT.healthy_at(78, kitchen_c) for t, _ in levels}
        trace = accumulate(levels, healthy=healthy, rise=_ROOM_RISE)
        return drift_observations(_FREEZER.ga, trace, rise=_ROOM_RISE, budget=_ROOM_BUDGET)

    def test_a_hot_kitchen_is_not_ice(self) -> None:
        # August as measured: 89 % of the day at 26.6 °C is what a healthy
        # freezer does there — 78 at 21 °C plus two points per kelvin is
        # 89.2 — so a reference that follows the room never accumulates.
        assert self._room_observations([89] * 24 * 10, kitchen_c=26.6) == []

    def test_the_same_running_in_a_cool_kitchen_is_ice(self) -> None:
        # The same 89 % at 21 °C is 11 points past healthy, 6 of them past
        # the rise: the 240 point-hour budget fills in 40 hours.
        duty = [89] * 24 * 10
        observations = self._room_observations(duty, kitchen_c=21.0)
        assert observations
        first_level, _ = self._levels(duty)[0]
        assert observations[0].time == first_level + 40 * _HOUR
        assert observations[0].value == pytest.approx(11.0)

    def test_the_episode_ends_after_a_defrost(self) -> None:
        # Iced for a week, defrosted, back to half the day: the episode must
        # close so the address clears.
        duty = [80] * 24 * 7 + [50] * 24 * 4
        observations = self._observations(duty)
        levels = self._levels(duty)
        episodes = fold_observations(
            "freezer_icing", observations, [], EpisodePolicy(), levels[-1][0]
        )
        assert len(episodes) == 1
        assert episodes[0].ended_at is not None


class TestAmbientLevels:
    def _hourly(self, values: Sequence[float | None]) -> dict[datetime, float]:
        return {_T0 + n * _HOUR: v for n, v in enumerate(values) if v is not None}

    def test_the_level_is_the_trailing_mean_of_the_room(self) -> None:
        means = ambient_levels(
            self._hourly([20, 22, 24]),
            window_start=_T0,
            frontier=_T0 + 2 * _HOUR,
            window=2 * _HOUR,
        )
        assert means == {_T0: 20.0, _T0 + _HOUR: 21.0, _T0 + 2 * _HOUR: 23.0}

    def test_a_silent_hour_carries_the_last_reading(self) -> None:
        means = ambient_levels(
            self._hourly([20, None, None]),
            window_start=_T0,
            frontier=_T0 + 2 * _HOUR,
            window=_HOUR,
        )
        assert means == {_T0: 20.0, _T0 + _HOUR: 20.0, _T0 + 2 * _HOUR: 20.0}

    def test_no_reference_before_the_room_has_spoken(self) -> None:
        means = ambient_levels(
            self._hourly([None, None, 20]),
            window_start=_T0,
            frontier=_T0 + 2 * _HOUR,
            window=_HOUR,
        )
        assert means == {_T0 + 2 * _HOUR: 20.0}


class TestMinWindowSamples:
    def test_the_declared_fraction_becomes_a_bucket_count(self) -> None:
        # The number that decides measurability at all: 0.8 of a day is
        # 19.2 hourly buckets, and a fifth of a day may not go missing.
        assert min_window_samples(_DAY, 0.8) == 20

    def test_a_full_window_demands_every_bucket(self) -> None:
        assert min_window_samples(_DAY, 1.0) == 24

    def test_it_rounds_up_so_a_thin_window_cannot_pass(self) -> None:
        assert min_window_samples(timedelta(hours=6), 0.5) == 3
        assert min_window_samples(timedelta(hours=7), 0.5) == 4


class TestReachesFrontier:
    """The difference between "recovered" and "unmeasured" — without it an
    episode ends on missing data and clears a still-stuck relay."""

    _MAX_GAP = EpisodePolicy().max_gap

    def test_valleys_up_to_the_frontier_are_measurable(self) -> None:
        floors = _floors([48] * 6)
        assert reaches_frontier(floors, frontier=_T0 + 5 * _HOUR, max_gap=self._MAX_GAP)

    def test_a_short_lag_still_counts(self) -> None:
        # One late bucket is materialization lag, not a blind device.
        floors = _floors([48] * 6)
        assert reaches_frontier(floors, frontier=_T0 + 7 * _HOUR, max_gap=self._MAX_GAP)

    def test_a_gap_past_the_episode_gap_is_unmeasurable(self) -> None:
        # Five missing hours stop the valley for ~29 h; the episode would
        # otherwise end and publish a clear the device never earned.
        floors = _floors([48] * 6)
        assert not reaches_frontier(floors, frontier=_T0 + 30 * _HOUR, max_gap=self._MAX_GAP)

    def test_no_valley_at_all_is_unmeasurable(self) -> None:
        assert not reaches_frontier([], frontier=_T0, max_gap=self._MAX_GAP)


class TestAccumulate:
    def test_a_reference_per_bucket_is_walked_bucket_by_bucket(self) -> None:
        # The same 60 reads 10 high against a 50 and 10 low against a 70:
        # each hour is compared with what healthy is there.
        levels = _series([60, 60])
        healthy = {levels[0][0]: 50.0, levels[1][0]: 70.0}
        trace = accumulate(levels, healthy=healthy, rise=5.0)
        assert [(s.healthy, s.excess, s.budget_used) for s in trace] == [
            (50.0, 10.0, 5.0),
            (70.0, -10.0, 0.0),
        ]

    def test_a_healthy_device_never_accumulates(self) -> None:
        trace = accumulate(_floors([48, 49, 47, 48]), healthy=48, rise=_RISE)
        assert [a.budget_used for a in trace] == [0.0, 0.0, 0.0, 0.0]
        assert all(a.since is None for a in trace)

    def test_only_the_excess_past_the_declared_rise_accumulates(self) -> None:
        # 148 mA on a 48 mA device is 100 mA high; 60 of them count.
        trace = accumulate(_floors([148, 148, 148]), healthy=48, rise=_RISE)
        assert [a.budget_used for a in trace] == [60.0, 120.0, 180.0]
        assert [a.excess for a in trace] == [100.0, 100.0, 100.0]
        assert all(a.since == _T0 for a in trace)

    def test_a_rise_within_the_band_never_accumulates(self) -> None:
        # The Vorratsraum case: a real, permanent 10 mA step that is not
        # this fault. A budget without this floor would fire eventually.
        trace = accumulate(_floors([58] * 500), healthy=48, rise=_RISE)
        assert max(a.budget_used for a in trace) == 0.0

    def test_returning_into_the_band_starts_the_count_over(self) -> None:
        trace = accumulate(_floors([148, 148, 48, 148]), healthy=48, rise=_RISE)
        assert [a.budget_used for a in trace] == [60.0, 120.0, 0.0, 60.0]
        assert trace[-1].since == _T0 + 3 * _HOUR


class TestDriftObservations:
    def test_a_slow_ramp_fires(self) -> None:
        # 43 mA climbing 6 mA/day for six weeks — the case the 30-day
        # z-score scored at 1.83 whatever the slope (#1593).
        ramp = [43.0 + 6.0 * (n / 24) for n in range(42 * 24)]
        trace = accumulate(_floors(ramp), healthy=43, rise=_RISE)
        observations = drift_observations(_FREEZER.ga, trace, rise=_RISE, budget=_BUDGET)
        assert observations
        first = observations[0]
        # Nothing fires before the ramp has climbed past the declared rise.
        assert first.value is not None
        assert first.value > _RISE
        assert first.score == pytest.approx(first.value / _RISE)
        # The excess passes the declared 40 mA on day 6.7; the budget then
        # fills against a 0.25 mA/h growth in another 2.6 days.
        assert (first.time - _T0) / _DAY == pytest.approx(9.25, abs=0.05)

    def test_a_stuck_relay_fires_within_hours(self) -> None:
        # 500 mA where 48 belong: 460 mA of excess fills a 480 mA·h budget
        # in the second hour.
        trace = accumulate(_floors([500] * 6), healthy=48, rise=_RISE)
        observations = drift_observations(_FREEZER.ga, trace, rise=_RISE, budget=_BUDGET)
        assert [o.time for o in observations] == [_T0 + n * _HOUR for n in range(1, 6)]
        assert observations[0].value == 452.0
        assert observations[0].score == pytest.approx(452.0 / _RISE)

    def test_a_plateau_below_the_rise_never_fires(self) -> None:
        trace = accumulate(_floors([78] * 24 * 90), healthy=48, rise=_RISE)
        assert drift_observations(_FREEZER.ga, trace, rise=_RISE, budget=_BUDGET) == []

    def test_recovery_ends_the_observations(self) -> None:
        trace = accumulate(_floors([500] * 6 + [48] * 6), healthy=48, rise=_RISE)
        observations = drift_observations(_FREEZER.ga, trace, rise=_RISE, budget=_BUDGET)
        assert max(o.time for o in observations) == _T0 + 5 * _HOUR

    def test_the_same_history_yields_the_same_observations(self) -> None:
        # Recomputed from history on every run, never persisted: a rerun
        # over the same window is the same run.
        floors = _floors([500] * 6 + [48] * 3 + [300] * 8)
        again = accumulate(floors, healthy=48, rise=_RISE)
        once = accumulate(floors, healthy=48, rise=_RISE)
        assert drift_observations(
            "2/2/227", once, rise=_RISE, budget=_BUDGET
        ) == drift_observations("2/2/227", again, rise=_RISE, budget=_BUDGET)


class TestEpisodes:
    def test_one_stuck_relay_is_one_episode(self) -> None:
        trace = accumulate(_floors([500] * 12), healthy=48, rise=_RISE)
        observations = drift_observations(_FREEZER.ga, trace, rise=_RISE, budget=_BUDGET)
        episodes = fold_observations(
            "appliance_standby", observations, [], EpisodePolicy(), _T0 + 11 * _HOUR
        )
        assert len(episodes) == 1
        assert episodes[0].subject == _FREEZER.ga
        assert episodes[0].ended_at is None

    def test_the_episode_ends_after_the_device_recovers(self) -> None:
        trace = accumulate(_floors([500] * 6 + [48] * 12), healthy=48, rise=_RISE)
        observations = drift_observations(_FREEZER.ga, trace, rise=_RISE, budget=_BUDGET)
        episodes = fold_observations(
            "appliance_standby", observations, [], EpisodePolicy(), _T0 + 17 * _HOUR
        )
        assert len(episodes) == 1
        assert episodes[0].ended_at is not None


class TestClassify:
    def test_a_drifting_device_names_its_excess_and_since(self) -> None:
        device = _device(_FREEZER, _REFERENCES[1])
        trace = accumulate(_floors([500] * 6), healthy=device.healthy, rise=_RISE)
        state = classify(device, trace, frontier=_T0 + 5 * _HOUR)
        assert state == DeviceState(
            device=device, level=500.0, healthy=48.0, excess=452.0, rising_since=_T0
        )

    def test_a_healthy_device_names_only_its_standby(self) -> None:
        device = _device(_FREEZER, _REFERENCES[1])
        trace = accumulate(_floors([48] * 6), healthy=device.healthy, rise=_RISE)
        state = classify(device, trace, frontier=_T0 + 5 * _HOUR)
        assert state == DeviceState(device=device, level=48.0, healthy=48.0, excess=0.0)

    def test_a_room_following_device_names_the_room_and_healthy_there(self) -> None:
        device = Device(
            ga=_FREEZER.ga,
            name=_FREEZER.name,
            label="Gefrierschrank",
            healthy=78,
            ambient=_KITCHEN_AMBIENT,
        )
        room = {t: 26.0 for t, _ in _series([0] * 6)}
        healthy = {t: _KITCHEN_AMBIENT.healthy_at(78, c) for t, c in room.items()}
        trace = accumulate(_series([95] * 6), healthy=healthy, rise=_ROOM_RISE)
        state = classify(device, trace, frontier=_T0 + 5 * _HOUR, room=room)
        assert state == DeviceState(
            device=device,
            level=95.0,
            healthy=88.0,
            excess=7.0,
            rising_since=_T0,
            ambient_c=26.0,
        )

    def test_a_series_short_of_the_frontier_leaves_the_state_empty(self) -> None:
        device = _device(_FREEZER, _REFERENCES[1])
        trace = accumulate(_floors([500] * 6), healthy=device.healthy, rise=_RISE)
        assert classify(device, trace, frontier=_T0 + 20 * _HOUR) == DeviceState(device)


class TestResolveDevices:
    def test_maps_each_reference_to_its_channel(self) -> None:
        devices = resolve_devices([_FREEZER, _WASHER], _REFERENCES)
        assert devices == [
            _device(_WASHER, _REFERENCES[0]),
            _device(_FREEZER, _REFERENCES[1]),
        ]

    def test_reference_matching_no_channel_is_an_error(self) -> None:
        with pytest.raises(ValueError, match=r"Gefrierschrank.*no channel"):
            resolve_devices([_WASHER], _REFERENCES)

    def test_channel_without_reference_is_an_error(self) -> None:
        with pytest.raises(ValueError, match=r"2/2/227.*no declared reference"):
            resolve_devices([_FREEZER, _WASHER], _REFERENCES[:1])

    def test_a_reference_following_its_room_resolves_the_room_channel(self) -> None:
        rooms = {"%.Küche.Sensor.Temperatur": [_KITCHEN]}
        [device] = resolve_devices([_FREEZER], [_KITCHEN_REFERENCE], rooms)
        assert device.ambient == _KITCHEN_AMBIENT
        assert device.healthy == 78

    def test_a_room_pattern_must_match_exactly_one_channel(self) -> None:
        with pytest.raises(ValueError, match=r"Küche.Sensor.Temperatur.*no channel"):
            resolve_devices([_FREEZER], [_KITCHEN_REFERENCE])
        twice = {"%.Küche.Sensor.Temperatur": [_KITCHEN, _WASHER]}
        with pytest.raises(ValueError, match=r"2 channels: 8/2/151, 2/1/197"):
            resolve_devices([_FREEZER], [_KITCHEN_REFERENCE], twice)


class TestPublishFor:
    def test_the_payload_names_the_device_and_its_numbers(self) -> None:
        device = _device(_FREEZER, _REFERENCES[1])
        state = DeviceState(device, level=500.0, healthy=48.0, excess=452.0, rising_since=_T0)
        publish = publish_for(_FREEZER.ga, 1, state)
        assert publish == DevicePublish(
            ga=_FREEZER.ga,
            severity=1,
            device=device.label,
            name=device.name,
            level=500.0,
            healthy=48.0,
            excess=452.0,
            rising_since=_T0,
        )
        assert publish.subject == _FREEZER.ga
        # Raw GA — the NATS adapter slugs it into the subject token.
        assert publish.entity == _FREEZER.ga

    def test_a_device_that_left_the_scope_still_gets_its_clear(self) -> None:
        publish = publish_for(_FREEZER.ga, 0, None)
        assert publish == DevicePublish(
            ga=_FREEZER.ga,
            severity=0,
            device=_FREEZER.ga,
            name=_FREEZER.ga,
            level=None,
            healthy=None,
            excess=None,
            rising_since=None,
        )


# The recovery seam's invented numbers — the deployed fault declares its own
# (calibrated against history): a healthy exchanger recovers 88 % of the
# gradient, 10 points less is the line, and hours where inside and outside
# are within 10 K of each other measure nothing.
_KWL_HEALTHY = 88.0
_FALL = 10.0
_FALL_BUDGET = 480.0
_RECOVERY_WINDOW = timedelta(hours=72)
_RECOVERY_COVERAGE = min_window_samples(_RECOVERY_WINDOW, 0.1)
_MIN_DELTA = 10.0

_OUTDOOR = Channel(
    ga="15/3/28", name="Versorgungstechnik.KWL.Temperatur-Außenluft", dpt="9.001"
)
_EXTRACT = Channel(
    ga="15/3/26", name="Versorgungstechnik.KWL.Temperatur-Abluft", dpt="9.001"
)
_SUPPLY = Channel(
    ga="15/3/29", name="Versorgungstechnik.KWL.Temperatur-Zuluft", dpt="9.001"
)
_AIRS = ExchangerRoles(
    outdoor="%.KWL.Temperatur-Außenluft",
    extract="%.KWL.Temperatur-Abluft",
    supply="%.KWL.Temperatur-Zuluft",
)
_KWL_REFERENCE = DeviceReference(match="KWL", healthy=_KWL_HEALTHY)
_KWL = Exchanger(
    label="KWL",
    outdoor_ga=_OUTDOOR.ga,
    extract_ga=_EXTRACT.ga,
    supply_ga=_SUPPLY.ga,
    healthy=_KWL_HEALTHY,
)


def _temps(
    outdoor: Sequence[float],
    extract: Sequence[float],
    supply: Sequence[float],
    start: datetime = _T0,
) -> dict[str, dict[datetime, float]]:
    return {
        _KWL.outdoor_ga: {start + n * _HOUR: v for n, v in enumerate(outdoor)},
        _KWL.extract_ga: {start + n * _HOUR: v for n, v in enumerate(extract)},
        _KWL.supply_ga: {start + n * _HOUR: v for n, v in enumerate(supply)},
    }


class TestEfficiencySeries:
    def test_the_efficiency_is_the_supply_share_of_the_gradient(self) -> None:
        by_ga = _temps([0.0, 0.0], [20.0, 20.0], [18.0, 17.0])
        etas = efficiency_series(_KWL, by_ga, _T0, _T0 + _HOUR, min_delta=_MIN_DELTA)
        assert etas == [(_T0, 90.0), (_T0 + _HOUR, 85.0)]

    def test_hours_inside_the_delta_gate_measure_nothing(self) -> None:
        # Inside and outside agree to 5 K: the quotient would be noise over
        # noise, and in summer that is every hour of the day.
        by_ga = _temps([15.0], [20.0], [19.0])
        assert efficiency_series(_KWL, by_ga, _T0, _T0, min_delta=_MIN_DELTA) == []

    def test_the_declared_delta_itself_still_measures(self) -> None:
        by_ga = _temps([10.0], [20.0], [19.0])
        etas = efficiency_series(_KWL, by_ga, _T0, _T0, min_delta=_MIN_DELTA)
        assert etas == [(_T0, 90.0)]

    def test_a_silent_role_carries_its_last_value(self) -> None:
        # KNX channels are state: an outdoor sensor that sent once holds
        # until it sends again; the silence fault owns the dead channel.
        by_ga = _temps([0.0], [20.0, 20.0, 20.0], [18.0, 18.0, 16.0])
        etas = efficiency_series(_KWL, by_ga, _T0, _T0 + 2 * _HOUR, min_delta=_MIN_DELTA)
        assert etas == [(_T0, 90.0), (_T0 + _HOUR, 90.0), (_T0 + 2 * _HOUR, 80.0)]

    def test_no_efficiency_before_every_role_has_appeared(self) -> None:
        by_ga = _temps([0.0, 0.0, 0.0], [20.0, 20.0, 20.0], [])
        by_ga[_KWL.supply_ga] = {_T0 + 2 * _HOUR: 18.0}
        etas = efficiency_series(_KWL, by_ga, _T0, _T0 + 2 * _HOUR, min_delta=_MIN_DELTA)
        assert etas == [(_T0 + 2 * _HOUR, 90.0)]


class TestCapabilityLevels:
    def _levels(
        self, etas: Sequence[tuple[datetime, float]], frontier: datetime
    ) -> list[tuple[datetime, float]]:
        return capability_levels(
            etas,
            window_start=_T0,
            frontier=frontier,
            window=_RECOVERY_WINDOW,
            min_samples=_RECOVERY_COVERAGE,
        )

    def test_the_level_is_the_best_hour_of_the_trailing_window(self) -> None:
        etas = _series([80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0])
        assert self._levels(etas, _T0 + 7 * _HOUR) == [(_T0 + 7 * _HOUR, 87.0)]

    def test_the_level_is_read_at_every_hour_once_covered(self) -> None:
        # Twelve valid hours, then a gated day and a half: the trailing
        # window keeps reading, so a stretch of small gradients does not
        # split one decay into an episode per cold night.
        etas = _series([88.0] * 12)
        levels = self._levels(etas, _T0 + 35 * _HOUR)
        assert levels[0] == (_T0 + 7 * _HOUR, 88.0)
        assert levels[-1] == (_T0 + 35 * _HOUR, 88.0)
        assert len(levels) == 29

    def test_a_window_short_of_valid_hours_yields_nothing(self) -> None:
        etas = _series([88.0] * 5)
        assert self._levels(etas, _T0 + 9 * _HOUR) == []

    def test_good_hours_older_than_the_window_fall_out(self) -> None:
        etas = _series([88.0] * 24 + [70.0] * 72)
        levels = self._levels(etas, _T0 + 95 * _HOUR)
        by_time = dict(levels)
        assert by_time[_T0 + 30 * _HOUR] == 88.0
        assert by_time[_T0 + 95 * _HOUR] == 70.0


class TestAccumulateFall:
    def test_a_healthy_exchanger_never_accumulates(self) -> None:
        trace = accumulate_fall(
            _series([88.0, 87.0, 89.0, 88.5]), healthy=_KWL_HEALTHY, fall=_FALL
        )
        assert [a.budget_used for a in trace] == [0.0, 0.0, 0.0, 0.0]
        assert all(a.since is None for a in trace)

    def test_only_the_deficit_past_the_declared_fall_accumulates(self) -> None:
        # 68 % on an 88 % exchanger is 20 points down; 10 of them count.
        trace = accumulate_fall(_series([68.0, 68.0, 68.0]), healthy=_KWL_HEALTHY, fall=_FALL)
        assert [a.budget_used for a in trace] == [10.0, 20.0, 30.0]
        assert [a.excess for a in trace] == [20.0, 20.0, 20.0]
        assert [a.level for a in trace] == [68.0, 68.0, 68.0]
        assert all(a.since == _T0 for a in trace)

    def test_a_sag_within_the_band_never_accumulates(self) -> None:
        # A real, permanent 8-point sag that is not this fault: the fall is
        # a floor, not a noise band, or this would fire eventually.
        trace = accumulate_fall(_series([80.0] * 500), healthy=_KWL_HEALTHY, fall=_FALL)
        assert all(a.budget_used == 0.0 for a in trace)

    def test_returning_into_the_band_starts_the_count_over(self) -> None:
        trace = accumulate_fall(
            _series([68.0, 85.0, 68.0]), healthy=_KWL_HEALTHY, fall=_FALL
        )
        assert [a.budget_used for a in trace] == [10.0, 0.0, 10.0]


class TestDecay:
    """The heat_recovery_decay fault end to end over its own signal:
    temperatures in, observations out, with the declared numbers of the
    fault file."""

    def _measured(
        self, targets: Sequence[float], by_ga: dict[str, dict[datetime, float]] | None = None
    ) -> tuple[list[Observation], list[tuple[datetime, float]], datetime]:
        frontier = _T0 + (len(targets) - 1) * _HOUR
        if by_ga is None:
            by_ga = _temps(
                [0.0] * len(targets),
                [20.0] * len(targets),
                [e / 100.0 * 20.0 for e in targets],
            )
        etas = efficiency_series(_KWL, by_ga, _T0, frontier, min_delta=_MIN_DELTA)
        levels = capability_levels(
            etas,
            window_start=_T0,
            frontier=frontier,
            window=_RECOVERY_WINDOW,
            min_samples=_RECOVERY_COVERAGE,
        )
        trace = accumulate_fall(levels, healthy=_KWL.healthy, fall=_FALL)
        observations = drift_observations(
            _KWL.slug, trace, rise=_FALL, budget=_FALL_BUDGET
        )
        return observations, levels, frontier

    def test_a_slowly_decaying_exchanger_fires(self) -> None:
        # The fault sentence's case: a week healthy, then four weeks sliding
        # from 88 to 68 % — the exchanger fouling, nothing jumping.
        decline = [88.0 - 20.0 * n / 671 for n in range(672)]
        observations, _, frontier = self._measured([88.0] * 168 + decline)
        assert observations
        episodes = fold_observations(
            "heat_recovery_decay", observations, [], EpisodePolicy(), frontier
        )
        assert len(episodes) == 1
        assert episodes[0].subject == "kwl"
        assert episodes[0].ended_at is None

    def test_a_stable_exchanger_stays_quiet(self) -> None:
        # Five weeks of ordinary jitter around healthy: the acceptance
        # criterion's other half.
        jitter = [88.0, 85.0, 90.0, 83.0, 89.0]
        observations, _, _ = self._measured([jitter[n % 5] for n in range(840)])
        assert observations == []

    def test_bypass_hours_do_not_read_as_fouling(self) -> None:
        # Nights that pass the gate with the exchanger deliberately bypassed
        # look exactly like fouling — but only hour by hour. The best hour
        # of the trailing window is still the exchanger's own.
        days = ([88.0] * 12 + [25.0] * 12) * 10
        observations, _, _ = self._measured(days)
        assert observations == []

    def test_a_summer_stretch_is_unmeasured_not_recovered(self) -> None:
        # Ten days inside the gate after five valid ones: no level reaches
        # the frontier, so an open episode must be held open, not cleared.
        n_valid, n_gated = 120, 240
        by_ga = _temps(
            [0.0] * n_valid + [18.0] * n_gated,
            [20.0] * (n_valid + n_gated),
            [17.6] * n_valid + [19.76] * n_gated,
        )
        observations, levels, frontier = self._measured(
            [88.0] * (n_valid + n_gated), by_ga
        )
        assert observations == []
        assert not reaches_frontier(
            levels, frontier=frontier, max_gap=EpisodePolicy().max_gap
        )

    def test_the_episode_ends_after_a_cleaning(self) -> None:
        # Three days healthy, twelve fouled at 68 %, three healthy again:
        # the drop reaches the level once the last good hour leaves the
        # trailing window (hour 143), the 480 point-hour budget fills 48 h
        # later, and the cleaning closes the episode so the address clears.
        targets = [88.0] * 72 + [68.0] * 288 + [88.0] * 72
        observations, _, frontier = self._measured(targets)
        assert observations
        assert observations[0].time == _T0 + 191 * _HOUR
        assert observations[0].value == pytest.approx(20.0)
        assert observations[0].score == pytest.approx(2.0)
        episodes = fold_observations(
            "heat_recovery_decay", observations, [], EpisodePolicy(), frontier
        )
        assert len(episodes) == 1
        assert episodes[0].ended_at is not None

    def test_the_same_history_yields_the_same_observations(self) -> None:
        decline = [88.0] * 100 + [88.0 - 20.0 * n / 671 for n in range(672)]
        first, _, _ = self._measured(decline)
        second, _, _ = self._measured(decline)
        assert first == second


class TestResolveExchanger:
    def test_maps_each_role_to_its_channel(self) -> None:
        exchanger = resolve_exchanger(
            [_EXTRACT, _OUTDOOR, _SUPPLY], (_KWL_REFERENCE,), _AIRS
        )
        assert exchanger == _KWL

    def test_a_role_matching_no_channel_is_an_error(self) -> None:
        with pytest.raises(ValueError, match=r"supply.*no channel"):
            resolve_exchanger([_EXTRACT, _OUTDOOR], (_KWL_REFERENCE,), _AIRS)

    def test_a_channel_matching_no_role_is_an_error(self) -> None:
        exhaust = Channel(
            ga="15/3/27", name="Versorgungstechnik.KWL.Temperatur-Fortluft", dpt="9.001"
        )
        with pytest.raises(ValueError, match=r"15/3/27.*no role"):
            resolve_exchanger(
                [_EXTRACT, _OUTDOOR, _SUPPLY, exhaust], (_KWL_REFERENCE,), _AIRS
            )

    def test_two_channels_for_one_role_is_an_error(self) -> None:
        twin = Channel(
            ga="15/3/99", name="Keller.KWL.Temperatur-Außenluft", dpt="9.001"
        )
        with pytest.raises(ValueError, match=r"outdoor.*2 channels"):
            resolve_exchanger(
                [_EXTRACT, _OUTDOOR, _SUPPLY, twin], (_KWL_REFERENCE,), _AIRS
            )

    def test_anything_but_one_reference_is_an_error(self) -> None:
        with pytest.raises(ValueError, match=r"exactly one exchanger"):
            resolve_exchanger([_EXTRACT, _OUTDOOR, _SUPPLY], (), _AIRS)


class TestClassifyExchanger:
    def test_a_decaying_exchanger_names_its_deficit_and_since(self) -> None:
        trace = accumulate_fall(_series([68.0] * 6), healthy=_KWL.healthy, fall=_FALL)
        state = classify_exchanger(_KWL, trace, frontier=_T0 + 5 * _HOUR)
        assert state == ExchangerState(
            exchanger=_KWL, efficiency=68.0, deficit=20.0, falling_since=_T0
        )

    def test_a_healthy_exchanger_names_only_its_efficiency(self) -> None:
        trace = accumulate_fall(_series([88.0] * 6), healthy=_KWL.healthy, fall=_FALL)
        state = classify_exchanger(_KWL, trace, frontier=_T0 + 5 * _HOUR)
        assert state == ExchangerState(exchanger=_KWL, efficiency=88.0, deficit=0.0)

    def test_a_trace_short_of_the_frontier_leaves_the_state_empty(self) -> None:
        trace = accumulate_fall(_series([68.0] * 6), healthy=_KWL.healthy, fall=_FALL)
        assert classify_exchanger(_KWL, trace, frontier=_T0 + 20 * _HOUR) == ExchangerState(
            exchanger=_KWL
        )


class TestPublishForExchanger:
    def test_the_payload_names_the_exchanger_and_its_numbers(self) -> None:
        state = ExchangerState(
            exchanger=_KWL, efficiency=68.0, deficit=20.0, falling_since=_T0
        )
        publish = publish_for_exchanger("kwl", 2, state)
        assert publish == ExchangerPublish(
            slug="kwl",
            severity=2,
            exchanger="KWL",
            efficiency=68.0,
            healthy=_KWL_HEALTHY,
            deficit=20.0,
            falling_since=_T0,
        )
        assert publish.subject == "kwl"
        assert publish.entity == "kwl"

    def test_a_subject_that_left_the_scope_still_gets_its_clear(self) -> None:
        publish = publish_for_exchanger("kwl", 0, None)
        assert publish.severity == 0
        assert publish.exchanger == "kwl"
        assert publish.efficiency is None
