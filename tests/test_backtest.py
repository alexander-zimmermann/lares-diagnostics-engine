"""Back-test tests: a candidate fault entry against history.

The entry point's whole boundary is one connection and one entry, so the
tests drive it through exactly that: a fake connection hands each
measurement its rows, and what is asserted is the episodes that come back.
The fake refuses anything but a read, so every one of these tests also
proves the guarantee the entry point exists for — a back-test writes
nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from lares_diagnostics_engine import backtest_fault
from lares_diagnostics_engine.site import Location, Plane, Site

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_HOUR = timedelta(hours=1)
_WEEK = timedelta(weeks=1)
_FRONTIER = datetime(2026, 9, 25, 10, 0, tzinfo=UTC)

# The statements the kinds' measurements run, each named by a fragment that
# only that statement carries. A fake that guessed would answer the wrong
# query with plausible rows, which is the one failure a fixture must not
# have.
_CATALOG = "FROM ga_catalog"
_KNX_FRONTIER = "SELECT max(bucket) AS frontier FROM knx_1h"
_APPLIANCE_FRONTIER = "SELECT max(bucket) AS frontier FROM knx_appliance_1h"
_CHANNEL_STATS = "count(*) AS buckets"
_BUCKET_SERIES = "SELECT ga, bucket FROM knx_1h"


@dataclass(frozen=True, slots=True)
class _Result:
    rows: Sequence[Mapping[str, Any]]

    def fetchall(self) -> list[Mapping[str, Any]]:
        return list(self.rows)

    def fetchone(self) -> Mapping[str, Any] | None:
        return self.rows[0] if self.rows else None


class _Conn:
    """Enough of a psycopg connection for a measurement: `execute` answers
    with the rows registered for the statement, matched on the fragment that
    names it.

    An unregistered statement fails rather than returning nothing — an empty
    answer is a measurement result, and a fixture that forgot a query would
    otherwise read as "the fault found nothing". Anything but a read fails
    too: that is the guarantee under test.
    """

    def __init__(self, rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        self._rows = {_flat(fragment): value for fragment, value in rows.items()}
        self.statements: list[str] = []

    def execute(self, sql: str, _params: Mapping[str, Any] | None = None) -> _Result:
        statement = _flat(sql)
        self.statements.append(statement)
        head = statement.split(None, 1)[0].upper()
        if head not in {"SELECT", "WITH"}:
            raise AssertionError(f"a back-test must not write: {statement}")
        matches = [rows for fragment, rows in self._rows.items() if fragment in statement]
        if len(matches) != 1:
            raise AssertionError(
                f"{len(matches)} fixtures match this statement: {statement}"
            )
        return _Result(matches[0])


def _flat(sql: str) -> str:
    """One line, single-spaced — the fragments are written the way the
    statement reads, not the way it is indented."""
    return " ".join(sql.split())


def _hourly(start: datetime, end: datetime) -> list[datetime]:
    buckets: list[datetime] = []
    t = start
    while t <= end:
        buckets.append(t)
        t += _HOUR
    return buckets


def _channel(ga: str, name: str, dpt: str) -> dict[str, Any]:
    return {"ga": ga, "name": name, "dpt": dpt}


# --- silence ---------------------------------------------------------------

_FREEZER = _channel("2/2/227", "Schalten.EG.Küche.K15-L1.Gefrierschrank.Stromwert", "7.012")

_SILENCE_ENTRY = {
    "name": "candidate_silence",
    "sentence": "Ein Kanal schweigt länger als das Fünffache seiner Sendepause.",
    "unit": "× der üblichen Sendepause",
    "kind": "silence",
    "parameters": {"gap_factor": 5, "gap_quantile": 0.95},
    "scope": {"include": ["Schalten.%.Stromwert"]},
    "target": {"per_main_group": True},
}

# The channel sent hourly until six hours before the frontier: a pause of
# one hour, so the current gap stands at six times it and the fault fires on
# the bucket that crosses five.
_QUIET_SINCE = _FRONTIER - 6 * _HOUR
_SENT = _hourly(_FRONTIER - _WEEK, _QUIET_SINCE)

_SILENCE_ROWS: Mapping[str, Sequence[Mapping[str, Any]]] = {
    _CATALOG: [_FREEZER],
    _KNX_FRONTIER: [{"frontier": _FRONTIER}],
    _CHANNEL_STATS: [
        {
            "ga": _FREEZER["ga"],
            "buckets": len(_SENT),
            "last_bucket": _QUIET_SINCE,
            "floor_value": 40.0,
            "ceil_value": 60.0,
        }
    ],
    _BUCKET_SERIES: [{"ga": _FREEZER["ga"], "bucket": bucket} for bucket in _SENT],
}


def test_a_silence_candidate_returns_the_episode_it_would_have_produced() -> None:
    conn = _Conn(_SILENCE_ROWS)

    result = backtest_fault(conn, _SILENCE_ENTRY, weeks=1)

    assert result.fault == "candidate_silence"
    (episode,) = result.episodes
    assert episode.subject == _FREEZER["ga"]
    assert episode.started_at == _FRONTIER
    assert episode.ended_at is None
    assert episode.observations == 1
    # The gap in units of the channel's own pause — the fault's declared unit.
    assert episode.peak_score == 6.0
    assert episode.severity == 1


def test_the_window_is_the_weeks_asked_for_off_the_frontier() -> None:
    conn = _Conn(_SILENCE_ROWS)

    result = backtest_fault(conn, _SILENCE_ENTRY, weeks=1)

    assert result.frontier == _FRONTIER
    assert result.window_start == _FRONTIER - _WEEK


def test_the_measurement_record_comes_back_with_the_episodes() -> None:
    # Zero episodes is a verdict, not silence: the record says how many
    # channels the scope even resolved to, so an empty result is readable.
    conn = _Conn({**_SILENCE_ROWS, _CATALOG: []})

    result = backtest_fault(conn, _SILENCE_ENTRY, weeks=1)

    assert result.episodes == ()
    assert result.measured["channels"] == 0


def test_a_backtest_neither_writes_nor_publishes(monkeypatch: pytest.MonkeyPatch) -> None:
    from lares_diagnostics_engine import nats_publisher

    def _refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("a back-test publishes nothing")

    monkeypatch.setattr(nats_publisher, "publish_anomaly", _refuse)
    monkeypatch.setattr(nats_publisher, "publish_episode_event", _refuse)
    conn = _Conn(_SILENCE_ROWS)

    result = backtest_fault(conn, _SILENCE_ENTRY, weeks=1)

    assert result.episodes  # it did measure something
    # The fake refuses a write itself; this names what the statements were.
    assert all(s.startswith(("SELECT", "WITH")) for s in conn.statements)


def test_an_external_candidate_is_refused() -> None:
    entry = {
        "name": "candidate_external",
        "sentence": "Der Systemdruck der Gastherme liegt unter 1,0 bar.",
        "unit": "bar",
        "kind": "external",
        "scope": {"include": "%.Gastherme.System-Druck-Anomalie"},
    }
    with pytest.raises(ValueError, match="Basalte"):
        backtest_fault(_Conn({}), entry, weeks=1)


def test_a_window_past_the_year_the_aggregates_reach_is_refused() -> None:
    with pytest.raises(ValueError, match="53 weeks reaches past the 52 weeks"):
        backtest_fault(_Conn({}), _SILENCE_ENTRY, weeks=53)


def test_a_window_shorter_than_a_week_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one week"):
        backtest_fault(_Conn({}), _SILENCE_ENTRY, weeks=0)


def test_an_aggregate_without_data_is_an_error_not_an_empty_result() -> None:
    conn = _Conn({**_SILENCE_ROWS, _KNX_FRONTIER: [{"frontier": None}]})

    with pytest.raises(ValueError, match="no data"):
        backtest_fault(conn, _SILENCE_ENTRY, weeks=1)


def test_a_candidate_the_schema_rejects_names_the_fault_and_the_field() -> None:
    entry = {**_SILENCE_ENTRY, "parameters": {"gap_factor": 5}}

    with pytest.raises(ValueError, match="candidate_silence.*gap_quantile"):
        backtest_fault(_Conn({}), entry, weeks=1)


def test_a_dormant_candidate_is_back_tested_all_the_same() -> None:
    # Activating a dormant fault is a proposal like any other, and the
    # back-test is what says whether its signal delivers yet.
    entry = {
        **_SILENCE_ENTRY,
        "dormant": {
            "reason": "Die Studio-Logik ist noch nicht gebaut.",
            "active_when": "Die Studio-Logik schreibt die Adresse.",
        },
    }
    result = backtest_fault(_Conn(_SILENCE_ROWS), entry, weeks=1)

    assert len(result.episodes) == 1


def test_a_target_form_the_kind_does_not_deliver_on_is_refused() -> None:
    # A candidate that could never run is not a proposal anyone can merge.
    entry = {**_SILENCE_ENTRY, "target": {"ga": "0/0/230"}}

    with pytest.raises(ValueError, match="per_main_group"):
        backtest_fault(_Conn({}), entry, weeks=1)


# --- constancy -------------------------------------------------------------

_WINDOW_STATS = "tail_buckets"
_READINGS = "SELECT ga, bucket, min_value, max_value FROM knx_1h"

_SENSOR = _channel("1/1/12", "Sensorik.OG.Flur.Sensor.Temperatur", "9.001")

_CONSTANCY_ENTRY = {
    "name": "candidate_constancy",
    "sentence": "Ein Kanal sendet weiter, liefert dabei aber seit mehr als 6 h denselben Wert.",
    "unit": "× der erlaubten Konstanz-Dauer",
    "kind": "constancy",
    "parameters": {"constant_hours": 6, "same_within": 0},
    "scope": {"dpt": ["9.001"]},
    "target": {"per_main_group": True},
}

# Bit-identical for the last ten hours: the sender lives, the register does
# not move.
_FROZEN_FROM = _FRONTIER - 9 * _HOUR


def test_a_constancy_candidate_returns_the_episode_it_would_have_produced() -> None:
    conn = _Conn(
        {
            _CATALOG: [_SENSOR],
            _KNX_FRONTIER: [{"frontier": _FRONTIER}],
            _WINDOW_STATS: [
                {
                    "ga": _SENSOR["ga"],
                    "last_bucket": _FRONTIER,
                    "tail_buckets": 7,
                    "tail_low": 21.0,
                    "tail_high": 21.0,
                }
            ],
            _READINGS: [
                {"ga": _SENSOR["ga"], "bucket": bucket, "min_value": 21.0, "max_value": 21.0}
                for bucket in _hourly(_FROZEN_FROM, _FRONTIER)
            ],
        }
    )

    result = backtest_fault(conn, _CONSTANCY_ENTRY, weeks=1)

    (episode,) = result.episodes
    assert episode.subject == _SENSOR["ga"]
    # The run stands ten hours; the first four are inside the declared six.
    assert episode.started_at == _FRONTIER - 3 * _HOUR
    assert episode.observations == 4
    assert episode.peak_score == pytest.approx(10 / 6)
    assert episode.ended_at is None


# --- duration --------------------------------------------------------------

_APPLIANCE_ACTIVITY = "SELECT ga, bucket, on_samples, total_samples FROM knx_appliance_1h"

_DRYER = _channel(
    "2/2/210", "Schalten.KG.Hauswirtschaftsraum.K3-L1.Trockner.Stromwert", "7.012"
)
_DRYER_DEVICE = "KG.Hauswirtschaftsraum.K3-L1.Trockner"

_DURATION_ENTRY = {
    "name": "candidate_runtime",
    "sentence": "Ein Gerät zieht länger Strom, als seine erlaubte Laufzeit zulässt.",
    "unit": "× der erlaubten Laufzeit",
    "kind": "duration",
    "parameters": {"active_hour_fraction": 0.5},
    "scope": {"dpt": "7.012", "include": "%.Trockner.Stromwert"},
    "devices": {_DRYER_DEVICE: {"max_run_hours": 4}},
    "target": {"per_device": True, "name": "Schalten.{entity}.Stromwert-Dauerbetrieb-Anomalie"},
}


def test_a_duration_candidate_returns_the_episode_it_would_have_produced() -> None:
    conn = _Conn(
        {
            _CATALOG: [_DRYER],
            _APPLIANCE_FRONTIER: [{"frontier": _FRONTIER}],
            _APPLIANCE_ACTIVITY: [
                {
                    "ga": _DRYER["ga"],
                    "bucket": bucket,
                    "on_samples": 10,
                    "total_samples": 10,
                }
                for bucket in _hourly(_FRONTIER - 7 * _HOUR, _FRONTIER)
            ],
        }
    )

    result = backtest_fault(conn, _DURATION_ENTRY, weeks=1)

    (episode,) = result.episodes
    assert episode.subject == _DRYER["ga"]
    # The device is named the way the fault's own map names it.
    assert episode.label == _DRYER_DEVICE
    # Eight hours of running, the first four inside the declared limit.
    assert episode.started_at == _FRONTIER - 3 * _HOUR
    assert episode.observations == 4
    assert episode.peak_score == pytest.approx(8 / 4)


# --- drift -----------------------------------------------------------------

_IDLE_FLOORS = "min(idle_floor) AS idle_floor FROM knx_appliance_1h"

_MICROWAVE = _channel("2/2/205", "Schalten.EG.Küche.K12-L1.Mikrowelle.Stromwert", "7.012")
_MICROWAVE_DEVICE = "EG.Küche.K12-L1.Mikrowelle"

_DRIFT_ENTRY = {
    "name": "candidate_standby",
    "sentence": "Die Standby-Aufnahme eines Geräts liegt dauerhaft über ihrem gesunden Wert.",
    "unit": "× der erlaubten Erhöhung",
    "kind": "drift",
    "signal": "standby",
    "parameters": {
        "rise_ma": 40,
        "budget_ma_h": 80,
        "window_hours": 3,
        "min_window_fraction": 1.0,
    },
    "scope": {"dpt": "7.012", "include": "%.Mikrowelle.Stromwert"},
    "references": {_MICROWAVE_DEVICE: {"healthy_ma": 41}},
    "target": {"per_device": True, "name": "Schalten.{entity}.Stromwert-Standby-Anomalie"},
}


def test_a_drift_candidate_returns_the_episode_it_would_have_produced() -> None:
    conn = _Conn(
        {
            _CATALOG: [_MICROWAVE],
            _APPLIANCE_FRONTIER: [{"frontier": _FRONTIER}],
            # A relay that no longer opens: 200 mA of idle draw against the
            # 41 mA declared healthy.
            _IDLE_FLOORS: [
                {"ga": _MICROWAVE["ga"], "bucket": bucket, "idle_floor": 200.0}
                for bucket in _hourly(_FRONTIER - 7 * _HOUR, _FRONTIER)
            ],
        }
    )

    result = backtest_fault(conn, _DRIFT_ENTRY, weeks=1)

    (episode,) = result.episodes
    assert episode.subject == _MICROWAVE["ga"]
    assert episode.label == _MICROWAVE_DEVICE
    # The trailing window swallows the first two hours; the budget is spent
    # on the first level past the rise.
    assert episode.started_at == _FRONTIER - 5 * _HOUR
    assert episode.observations == 6
    assert episode.peak_score == pytest.approx((200 - 41) / 40)
    # Four buckets in, duration promotes the tier — the one thing that moves
    # a severity without a history to be rare against.
    assert episode.severity == 2


# --- deviation -------------------------------------------------------------

_HOURLY_AVERAGES = "SELECT ga, bucket, avg_value FROM knx_1h"

_VALVE = _channel("3/1/40", "Raumklima.OG.Flur.FBH.Stellwert-Status", "5.001")
_SETPOINT = _channel("3/1/41", "Raumklima.OG.Flur.FBH.Soll-Temperatur-Status", "9.001")
_ROOM_TEMP = _channel("1/1/30", "Sensorik.OG.Flur.BWM.Treppe.Temperatur", "9.001")

_DEVIATION_ENTRY = {
    "name": "candidate_fbh_cold",
    "sentence": "Ein Raum liegt bei offenem Ventil weiter unter seiner Soll-Temperatur, "
    "als erlaubt ist.",
    "unit": "× der erlaubten Abweichung",
    "kind": "deviation",
    "parameters": {"min_hours": 2, "gate_min_pct": 50},
    "roles": {
        "reference": "%.FBH.Soll-Temperatur-Status",
        "gate": "%.FBH.Stellwert-Status",
    },
    "scope": {
        "include": [
            "Raumklima.%.FBH.Stellwert-Status",
            "Raumklima.%.FBH.Soll-Temperatur-Status",
            "Sensorik.OG.Flur.BWM.Treppe.Temperatur",
        ]
    },
    "rooms": {"OG.Flur": {"min_gap_k": 1.0, "value": _ROOM_TEMP["name"]}},
    "target": {"per_room": True, "name": "Raumklima.{entity}.FBH.Aktiv-Anomalie"},
}


def test_a_deviation_candidate_returns_the_episode_it_would_have_produced() -> None:
    cold = _hourly(_FRONTIER - 5 * _HOUR, _FRONTIER)
    conn = _Conn(
        {
            _CATALOG: [_ROOM_TEMP, _VALVE, _SETPOINT],
            _KNX_FRONTIER: [{"frontier": _FRONTIER}],
            _HOURLY_AVERAGES: [
                {"ga": ga, "bucket": bucket, "avg_value": value}
                for ga, value in (
                    (_ROOM_TEMP["ga"], 19.0),
                    (_SETPOINT["ga"], 21.5),
                    (_VALVE["ga"], 80.0),
                )
                for bucket in cold
            ],
        }
    )

    result = backtest_fault(conn, _DEVIATION_ENTRY, weeks=1)

    (episode,) = result.episodes
    # The room's slug is the subject the writer rules pin the address to.
    assert episode.subject == "og-flur"
    # Six cold hours, the first of which is inside the declared two.
    assert episode.started_at == _FRONTIER - 4 * _HOUR
    assert episode.observations == 5
    assert episode.peak_score == pytest.approx(2.5)
    assert result.measured["rooms"] == 1


def test_a_daily_yield_candidate_without_a_site_says_so() -> None:
    # The plant's shape measures what the site declares — which inverters,
    # and what each can make in an hour.
    entry = {
        "name": "candidate_pv",
        "sentence": "Die Anlage hat an einem Tag weniger erzeugt als die Prognose erwartet.",
        "unit": "× des erlaubten Fehlbetrags",
        "kind": "deviation",
        "expectation": "forecast_solar",
        "parameters": {"min_shortfall_pct": 35, "min_expected_kwh": 3},
        "target": {"ga": "15/4/11"},
    }
    with pytest.raises(ValueError, match="site"):
        backtest_fault(_Conn({}), entry, weeks=1)


# --- volume ----------------------------------------------------------------

_EPISODE_STARTS = "SELECT started_at, fault FROM episodes"

_VOLUME_ENTRY = {
    "name": "candidate_volume",
    "sentence": "Die Engine meldet mehr als zwei Vorfälle in sieben Tagen.",
    "unit": "× der erlaubten Wochenmenge",
    "kind": "volume",
    "parameters": {"max_episodes_per_week": 2},
    "target": {"ga": "0/0/230"},
}


def test_a_volume_candidate_counts_the_engines_own_episode_stream() -> None:
    conn = _Conn(
        {
            _KNX_FRONTIER: [{"frontier": _FRONTIER}],
            _EPISODE_STARTS: [
                {"started_at": _FRONTIER - hours * _HOUR, "fault": fault}
                for hours, fault in ((4, "channel_silence"), (3, "fbh_cold"), (2, "fbh_cold"))
            ],
        }
    )

    result = backtest_fault(conn, _VOLUME_ENTRY, weeks=1)

    (episode,) = result.episodes
    assert episode.subject == "house"
    # The third incident is what takes the week past two.
    assert episode.started_at == _FRONTIER - 2 * _HOUR
    assert episode.observations == 3
    assert episode.peak_score == pytest.approx(3 / 2)
    assert result.measured["incidents"] == 3


# --- how far back each shape can read --------------------------------------

_SITE = Site(
    location=Location(latitude=50.62598, longitude=6.02435),
    timezone="Europe/Berlin",
    planes=(Plane(key="West", inverter_id=1, tilt=17.0, azimuth=129.0, kwp=6.435),),
)

_YIELD_ENTRY = {
    "name": "candidate_pv",
    "sentence": "Die Anlage hat an einem Tag weniger erzeugt als die Prognose erwartet.",
    "unit": "× des erlaubten Fehlbetrags",
    "kind": "deviation",
    "expectation": "forecast_solar",
    "parameters": {"min_shortfall_pct": 35, "min_expected_kwh": 3},
    "target": {"ga": "15/4/11"},
}

_DUTY_CYCLE_ENTRY = {
    "name": "candidate_icing",
    "sentence": "Der Verdichter läuft über den Tag länger als im gesunden Zustand.",
    "unit": "× der erlaubten Erhöhung",
    "kind": "drift",
    "signal": "duty_cycle",
    "parameters": {
        "rise_pct": 15,
        "budget_pct_h": 600,
        "window_hours": 24,
        "min_window_fraction": 0.6,
        "door_run_hours": 3,
        "on_ma": 120,
    },
    "scope": {"dpt": "7.012", "include": "%.Gefrierschrank.Stromwert"},
    "references": {"Gefrierschrank": {"healthy_duty_pct": 50}},
    "target": {"per_device": True, "name": "Haushaltstechnik.{entity}.Vereisung-Anomalie"},
}


def test_the_daily_yield_shape_stops_where_its_forecast_does() -> None:
    # The counters reach a year, the stored forecast they are short of does
    # not: a longer window would read days with no expectation at all.
    with pytest.raises(ValueError, match="13 weeks reaches past the 12 weeks"):
        backtest_fault(_Conn({}), _YIELD_ENTRY, weeks=13, site=_SITE)


def test_the_duty_cycle_shape_stops_where_a_scheduled_run_does() -> None:
    # The one signal that reads the bus archive rather than an aggregate.
    with pytest.raises(ValueError, match="5 weeks reaches past the 4 weeks"):
        backtest_fault(_Conn({}), _DUTY_CYCLE_ENTRY, weeks=5)
