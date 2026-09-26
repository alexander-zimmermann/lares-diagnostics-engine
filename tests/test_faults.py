"""Fault-file loader tests — seam 1 of the detection rebuild.

Prior art: the writer-rules loader tests in knx-nats-bridge. Every test
feeds an invented YAML file and asserts only what comes out: typed fault
objects, or a load error naming the fault and the field.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from lares_diagnostics_engine.faults import (
    _SCHEMA_PATH,
    DeviationExpectation,
    DeviceLimit,
    DeviceReference,
    Dormant,
    DriftSignal,
    ExchangerRoles,
    FaultList,
    MeasurementKind,
    Roles,
    RoomRule,
    Target,
    parse_entry,
)

# A minimal valid file: the tracer fault plus one drift fault, matching the
# shape the lares fault list uses.
_VALID = """
faults:
  - name: channel_silence
    sentence: "Ein Kanal, der sonst regelmäßig sendet, schweigt länger als das
      Fünffache der Sendepause, die er in neun von zehn Fällen einhält."
    unit: "× der üblichen Sendepause"
    kind: silence
    parameters:
      gap_factor: 5
      gap_quantile: 0.9
    scope:
      include: "%"
    target:
      per_main_group: true
  - name: appliance_standby
    sentence: "Der Ruhestrom eines Geräts steigt dauerhaft um mehr als 40 mA."
    unit: "× der erlaubten Erhöhung"
    kind: drift
    signal: standby
    parameters:
      rise_ma: 40
      budget_ma_h: 480
      window_hours: 24
      min_window_fraction: 0.8
    scope:
      dpt: "7.012"
      include: "%.Stromwert"
    references:
      Küche.K15-L1.Gefrierschrank:
        healthy_ma: 48
    target:
      per_device: true
"""


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "faults.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_loads_valid_file(tmp_path: Path) -> None:
    faults = FaultList.load(_write(tmp_path, _VALID))
    assert len(faults) == 2
    assert [f.name for f in faults] == ["channel_silence", "appliance_standby"]

    silence = faults.get("channel_silence")
    assert silence.kind is MeasurementKind.SILENCE
    assert silence.parameters == {"gap_factor": 5, "gap_quantile": 0.9}
    assert silence.unit == "× der üblichen Sendepause"
    assert silence.target is not None
    assert silence.target.per_main_group is True
    assert silence.target.ga is None
    assert silence.dormant is None

    standby = faults.get("appliance_standby")
    assert standby.kind is MeasurementKind.DRIFT
    assert standby.parameters == {
        "rise_ma": 40,
        "budget_ma_h": 480,
        "window_hours": 24,
        "min_window_fraction": 0.8,
    }
    assert standby.signal is DriftSignal.STANDBY
    assert standby.references == (
        DeviceReference(match="Küche.K15-L1.Gefrierschrank", healthy=48),
    )
    assert standby.target is not None
    assert standby.target.per_device is True
    assert standby.target.per_main_group is False
    assert standby.target.name is None


def test_scope_is_carried_not_resolved(tmp_path: Path) -> None:
    # The loader carries the catalog query verbatim; resolution happens where
    # the catalog lives, so the query text must survive the round trip.
    faults = FaultList.load(_write(tmp_path, _VALID))
    standby = faults.get("appliance_standby")
    assert standby.scope.dpt == ("7.012",)
    assert standby.scope.include == ("%.Stromwert",)
    assert standby.scope.exclude == ()


def test_scope_accepts_lists(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in K."
            unit: "K"
            kind: constancy
            parameters:
              constant_hours: 72
              same_within: 0
            scope:
              dpt: ["9.001", "9.007"]
              exclude: "%.Alarm%"
            target:
              ga: "8/0/1"
        """,
    )
    [f] = FaultList.load(path)
    assert f.scope.dpt == ("9.001", "9.007")
    assert f.scope.exclude == ("%.Alarm%",)


def test_missing_sentence_names_fault_and_field(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: channel_silence
            unit: "h"
            kind: silence
            parameters:
              gap_factor: 5
              gap_quantile: 0.9
            scope:
              include: "%"
            target:
              per_main_group: true
        """,
    )
    with pytest.raises(ValueError, match=r"channel_silence.*sentence"):
        FaultList.load(path)


def test_missing_unit_names_fault_and_field(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: channel_silence
            sentence: "Ein Kanal schweigt."
            kind: silence
            parameters:
              gap_factor: 5
              gap_quantile: 0.9
            scope:
              include: "%"
            target:
              per_main_group: true
        """,
    )
    with pytest.raises(ValueError, match=r"channel_silence.*unit"):
        FaultList.load(path)


def test_missing_parameter_names_fault_and_field(tmp_path: Path) -> None:
    # silence requires gap_factor specifically, not just any parameter
    path = _write(
        tmp_path,
        """
        faults:
          - name: channel_silence
            sentence: "Ein Kanal schweigt."
            unit: "h"
            kind: silence
            parameters:
              other: 1
            scope:
              include: "%"
            target:
              per_main_group: true
        """,
    )
    with pytest.raises(ValueError, match=r"channel_silence.*gap_factor"):
        FaultList.load(path)


_CONSTANCY = """
faults:
  - name: channel_constancy
    sentence: "Ein Kanal liefert seit über 48 h denselben Wert (V)."
    unit: "× der erlaubten Konstanz"
    kind: constancy
    parameters:
      constant_hours: 48
      same_within: 0
    scope:
      dpt: "14.027"
    target:
      per_main_group: true
"""


def test_constancy_fault_loads_with_its_two_parameters(tmp_path: Path) -> None:
    [fault] = FaultList.load(_write(tmp_path, _CONSTANCY))
    assert fault.kind is MeasurementKind.CONSTANCY
    assert fault.parameters == {"constant_hours": 48, "same_within": 0}
    assert fault.target == Target(per_main_group=True)


def test_constancy_without_its_duration_rejected(tmp_path: Path) -> None:
    # Without it nothing says how long one value may stand.
    body = _CONSTANCY.replace("      constant_hours: 48\n", "")
    with pytest.raises(ValueError, match=r"channel_constancy.*constant_hours"):
        FaultList.load(_write(tmp_path, body))


def test_constancy_without_its_band_rejected(tmp_path: Path) -> None:
    # Without it nothing says what counts as the same value.
    body = _CONSTANCY.replace("      same_within: 0\n", "")
    with pytest.raises(ValueError, match=r"channel_constancy.*same_within"):
        FaultList.load(_write(tmp_path, body))


def test_constancy_band_cannot_be_negative(tmp_path: Path) -> None:
    body = _CONSTANCY.replace("same_within: 0", "same_within: -1")
    with pytest.raises(ValueError, match=r"channel_constancy.*same_within"):
        FaultList.load(_write(tmp_path, body))


def test_quantile_outside_its_range_names_fault_and_field(tmp_path: Path) -> None:
    # A quantile is a fraction of the channel's own gaps; 90 is the typo for
    # 0.9 that would otherwise silently read the longest gap ever seen.
    path = _write(
        tmp_path,
        """
        faults:
          - name: channel_silence
            sentence: "Ein Kanal schweigt."
            unit: "h"
            kind: silence
            parameters:
              gap_factor: 5
              gap_quantile: 90
            scope:
              include: "%"
            target:
              per_main_group: true
        """,
    )
    with pytest.raises(ValueError, match=r"channel_silence.*gap_quantile"):
        FaultList.load(path)


def test_empty_parameters_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: constancy
            parameters: {}
            scope:
              include: "%.Stromwert"
            target:
              ga: "2/2/229"
        """,
    )
    with pytest.raises(ValueError, match=r"'x'.*parameters"):
        FaultList.load(path)


def test_non_numeric_parameter_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: constancy
            parameters:
              constant_hours: 48
              same_within: 0
              healthy: "high"
            scope:
              include: "%.Stromwert"
            target:
              ga: "2/2/229"
        """,
    )
    with pytest.raises(ValueError, match=r"'x'.*healthy"):
        FaultList.load(path)


def test_unknown_kind_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in ppm."
            unit: "ppm"
            kind: threshold
            parameters:
              limit: 1400
            scope:
              include: "%.CO2"
            target:
              ga: "8/0/1"
        """,
    )
    with pytest.raises(ValueError, match=r"'x'.*kind"):
        FaultList.load(path)


def test_unknown_field_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: constancy
            parameters:
              constant_hours: 48
              same_within: 0
            scope:
              include: "%.Stromwert"
            target:
              ga: "2/2/229"
            severity: critical
        """,
    )
    with pytest.raises(ValueError, match=r"'x'.*severity"):
        FaultList.load(path)


def test_target_requires_exactly_one_form(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: constancy
            parameters:
              constant_hours: 48
              same_within: 0
            scope:
              include: "%.Stromwert"
            target:
              ga: "2/2/229"
              per_main_group: true
        """,
    )
    with pytest.raises(ValueError, match=r"'x'.*target"):
        FaultList.load(path)


def test_invalid_ga_format_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: constancy
            parameters:
              constant_hours: 48
              same_within: 0
            scope:
              include: "%.Stromwert"
            target:
              ga: "2.2.229"
        """,
    )
    with pytest.raises(ValueError, match=r"'x'.*target"):
        FaultList.load(path)


def test_empty_scope_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: constancy
            parameters:
              constant_hours: 48
              same_within: 0
            scope: {}
            target:
              ga: "2/2/229"
        """,
    )
    with pytest.raises(ValueError, match=r"'x'.*scope"):
        FaultList.load(path)


def test_duplicate_name_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: constancy
            parameters:
              constant_hours: 48
              same_within: 0
            scope:
              include: "%.Stromwert"
            target:
              ga: "2/2/229"
          - name: x
            sentence: "Ein anderer Satz mit Einheit in mA."
            unit: "mA"
            kind: duration
            parameters:
              limit: 240
            scope:
              include: "%.Stromwert"
            target:
              ga: "2/2/230"
        """,
    )
    with pytest.raises(ValueError, match=r"duplicate.*'x'"):
        FaultList.load(path)


def test_dormant_loads_but_is_not_schedulable(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _VALID
        + """
  - name: battery_soc_stuck
    sentence: "Der Batterie-Ladestand ändert sich über 24 h nicht (%)."
    unit: "%"
    kind: constancy
    parameters:
      constant_hours: 24
      same_within: 0
    scope:
      include: "%.Batterie.%"
    target:
      ga: "15/4/40"
    dormant:
      reason: "SolarEdge meter not connected until the 400 V supply is in"
      active_when: "the meter reports non-zero values"
""",
    )
    faults = FaultList.load(path)
    assert len(faults) == 3
    dormant = faults.get("battery_soc_stuck")
    assert dormant.dormant == Dormant(
        reason="SolarEdge meter not connected until the 400 V supply is in",
        active_when="the meter reports non-zero values",
    )
    assert [f.name for f in faults.schedulable()] == ["channel_silence", "appliance_standby"]


def test_dormant_requires_reason_and_condition(tmp_path: Path) -> None:
    # `dormant: true` is exactly the boolean-with-a-comment this field replaces
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: constancy
            parameters:
              constant_hours: 48
              same_within: 0
            scope:
              include: "%.Stromwert"
            target:
              ga: "2/2/229"
            dormant: true
        """,
    )
    with pytest.raises(ValueError, match=r"'x'.*dormant"):
        FaultList.load(path)


def test_dormant_missing_active_when_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: constancy
            parameters:
              constant_hours: 48
              same_within: 0
            scope:
              include: "%.Stromwert"
            target:
              ga: "2/2/229"
            dormant:
              reason: "meter not connected"
        """,
    )
    with pytest.raises(ValueError, match=r"'x'.*active_when"):
        FaultList.load(path)


def test_missing_name_reports_position(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: constancy
            parameters:
              constant_hours: 48
              same_within: 0
            scope:
              include: "%.Stromwert"
            target:
              ga: "2/2/229"
        """,
    )
    with pytest.raises(ValueError, match=r"#1.*name"):
        FaultList.load(path)


def test_empty_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="faults"):
        FaultList.load(_write(tmp_path, ""))


def test_non_mapping_top_level_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mapping"):
        FaultList.load(_write(tmp_path, "- just\n- a\n- list\n"))


def test_fault_objects_are_frozen(tmp_path: Path) -> None:
    faults = FaultList.load(_write(tmp_path, _VALID))
    silence = faults.get("channel_silence")
    with pytest.raises(AttributeError):
        silence.unit = "h"  # type: ignore[misc]


def test_fingerprint_names_the_kind_and_the_parameters_it_measures_by(tmp_path: Path) -> None:
    # The fingerprint is what an episode carries: how the fault measured
    # when it made the row. The signal picks the kind's shape, so it is in.
    faults = FaultList.load(_write(tmp_path, _VALID))
    assert faults.get("channel_silence").fingerprint == (
        "silence(gap_factor=5.0, gap_quantile=0.9)"
    )
    assert faults.get("appliance_standby").fingerprint == (
        "drift/standby(budget_ma_h=480.0, min_window_fraction=0.8, rise_ma=40.0, "
        "window_hours=24.0)"
    )


def test_fingerprint_reads_a_parameter_as_a_number_not_as_its_spelling(tmp_path: Path) -> None:
    # `5` and `5.0` declare the same rule: a cosmetic edit of the fault file
    # must not disown every open row.
    spelled_as_int = FaultList.load(_write(tmp_path, _VALID)).get("channel_silence")
    spelled_as_float = FaultList.load(
        _write(tmp_path, _VALID.replace("gap_factor: 5\n", "gap_factor: 5.0\n"))
    ).get("channel_silence")
    assert spelled_as_int.fingerprint == spelled_as_float.fingerprint


def test_target_union_holds_for_python_construction() -> None:
    # The schema's oneOf, mirrored for Python-side construction
    with pytest.raises(ValueError, match="exactly one"):
        Target()
    with pytest.raises(ValueError, match="exactly one"):
        Target(ga="1/2/3", per_main_group=True)
    with pytest.raises(ValueError, match="exactly one"):
        Target(per_main_group=True, per_device=True)


def test_target_name_rules_hold_for_python_construction() -> None:
    # The loader's name-template rules, mirrored the same way
    with pytest.raises(ValueError, match=r"name template.*\{entity\}"):
        Target(per_device=True, name="Schalten.Stromwert-Standby-Anomalie")
    with pytest.raises(ValueError, match="name template.*per_device or per_room"):
        Target(per_main_group=True, name="Zentral.{entity}.Anomalie")
    with pytest.raises(ValueError, match="name template.*per_device or per_room"):
        Target(ga="1/2/3", name="Zentral.{entity}.Anomalie")


_DURATION = """
faults:
  - name: appliance_runtime
    sentence: "Ein Gerät zieht länger ununterbrochen Strom, als seine erlaubte
      Laufzeit zulässt."
    unit: "× der erlaubten Laufzeit"
    kind: duration
    parameters:
      active_hour_fraction: 0.5
    devices:
      Hauswirtschaftsraum.K4-L1.Waschmaschine:
        max_run_hours: 4
      Küche.K15-L1.Gefrierschrank:
        max_run_hours: 6
    scope:
      include: "%.Stromwert"
    target:
      per_device: true
"""


def test_duration_fault_loads_device_limits(tmp_path: Path) -> None:
    [fault] = FaultList.load(_write(tmp_path, _DURATION))
    assert fault.kind is MeasurementKind.DURATION
    assert fault.parameters == {"active_hour_fraction": 0.5}
    assert fault.target is not None
    assert fault.target.per_device is True
    assert fault.target.ga is None
    assert fault.devices == (
        DeviceLimit(match="Hauswirtschaftsraum.K4-L1.Waschmaschine", max_run_hours=4),
        DeviceLimit(match="Küche.K15-L1.Gefrierschrank", max_run_hours=6),
    )


def test_duration_without_devices_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: appliance_runtime
            sentence: "Ein Gerät zieht zu lange Strom."
            unit: "× der erlaubten Laufzeit"
            kind: duration
            parameters:
              active_hour_fraction: 0.5
            scope:
              include: "%.Stromwert"
            target:
              per_device: true
        """,
    )
    with pytest.raises(ValueError, match=r"'appliance_runtime'.*devices"):
        FaultList.load(path)


def test_duration_requires_active_hour_fraction(tmp_path: Path) -> None:
    body = _DURATION.replace("active_hour_fraction: 0.5", "other: 1")
    with pytest.raises(ValueError, match=r"'appliance_runtime'.*active_hour_fraction"):
        FaultList.load(_write(tmp_path, body))


def test_devices_on_other_kind_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: constancy
            parameters:
              constant_hours: 48
              same_within: 0
            devices:
              Waschmaschine:
                max_run_hours: 4
            scope:
              include: "%.Stromwert"
            target:
              ga: "2/2/229"
        """,
    )
    with pytest.raises(ValueError, match=r"'x'.*devices"):
        FaultList.load(path)


def test_non_positive_device_limit_rejected(tmp_path: Path) -> None:
    body = _DURATION.replace("max_run_hours: 4", "max_run_hours: 0")
    with pytest.raises(ValueError, match=r"'appliance_runtime'.*max_run_hours"):
        FaultList.load(_write(tmp_path, body))


def test_device_entry_without_limit_rejected(tmp_path: Path) -> None:
    body = _DURATION.replace("max_run_hours: 4", "note: 4")
    with pytest.raises(ValueError, match=r"'appliance_runtime'.*Waschmaschine"):
        FaultList.load(_write(tmp_path, body))


def test_per_device_target_excludes_other_forms(tmp_path: Path) -> None:
    body = _DURATION.replace("per_device: true", 'per_device: true\n      ga: "2/2/229"')
    with pytest.raises(ValueError, match=r"'appliance_runtime'.*target"):
        FaultList.load(_write(tmp_path, body))


def _with_target_name(body: str, form: str, name: str) -> str:
    """The fixture's target with a name template beside its form."""
    return body.replace(f"{form}: true", f'{form}: true\n      name: "{name}"')


def test_per_device_target_loads_its_name_template(tmp_path: Path) -> None:
    # The template names the entity as the fault's entity map does, so it
    # renders to a catalog name — the loader carries it, the generator in
    # lares resolves it.
    body = _with_target_name(
        _DURATION, "per_device", "Schalten.{entity}.Stromwert-Dauerbetrieb-Anomalie"
    )
    [fault] = FaultList.load(_write(tmp_path, body))
    assert fault.target == Target(
        per_device=True, name="Schalten.{entity}.Stromwert-Dauerbetrieb-Anomalie"
    )


def test_target_name_without_the_entity_placeholder_rejected(tmp_path: Path) -> None:
    # A template without the placeholder would point every device at one
    # address, silently.
    body = _with_target_name(_DURATION, "per_device", "Schalten.Stromwert-Dauerbetrieb-Anomalie")
    with pytest.raises(ValueError, match=r"'appliance_runtime'.*target\.name.*\{entity\}"):
        FaultList.load(_write(tmp_path, body))


def test_empty_target_name_rejected(tmp_path: Path) -> None:
    body = _with_target_name(_DURATION, "per_device", "")
    with pytest.raises(ValueError, match=r"'appliance_runtime'.*target\.name"):
        FaultList.load(_write(tmp_path, body))


def test_target_name_on_a_main_group_target_rejected(tmp_path: Path) -> None:
    # Per main group the address is named by the group's Zentral block, not
    # by a template; the field means one thing.
    body = _with_target_name(_VALID, "per_main_group", "Zentral.{entity}.Anomalie")
    with pytest.raises(
        ValueError, match=r"'channel_silence'.*target\.name.*per_device or per_room"
    ):
        FaultList.load(_write(tmp_path, body))


def test_target_name_on_a_fixed_address_rejected(tmp_path: Path) -> None:
    body = _DURATION.replace(
        "per_device: true", 'ga: "2/2/229"\n      name: "Schalten.{entity}.Anomalie"'
    )
    with pytest.raises(
        ValueError, match=r"'appliance_runtime'.*target\.name.*per_device or per_room"
    ):
        FaultList.load(_write(tmp_path, body))


_DRIFT = """
faults:
  - name: appliance_standby
    sentence: "Die Standby-Aufnahme eines Geräts liegt dauerhaft mehr als
      40 mA über ihrem gesunden Wert."
    unit: "× der erlaubten Erhöhung"
    kind: drift
    signal: standby
    parameters:
      rise_ma: 40
      budget_ma_h: 480
      window_hours: 24
      min_window_fraction: 0.8
    scope:
      dpt: "7.012"
      include: "%.Stromwert"
    references:
      Küche.K15-L1.Gefrierschrank:
        healthy_ma: 48
      Hauswirtschaftsraum.K4-L1.Waschmaschine:
        healthy_ma: 0
    target:
      per_device: true
"""


def test_drift_fault_loads_its_references(tmp_path: Path) -> None:
    [fault] = FaultList.load(_write(tmp_path, _DRIFT))
    assert fault.kind is MeasurementKind.DRIFT
    assert fault.signal is DriftSignal.STANDBY
    assert fault.references == (
        DeviceReference(match="Küche.K15-L1.Gefrierschrank", healthy=48),
        DeviceReference(match="Hauswirtschaftsraum.K4-L1.Waschmaschine", healthy=0),
    )


def test_drift_without_references_rejected(tmp_path: Path) -> None:
    # A drift fault whose reference is missing would have to invent one from
    # history — the one thing the pinned reference exists to prevent.
    head, _refs = _DRIFT.split("    references:\n", 1)
    body = head + "    target:" + _DRIFT.split("    target:", 1)[1]
    with pytest.raises(ValueError, match=r"'appliance_standby'.*references"):
        FaultList.load(_write(tmp_path, body))


def test_reference_entry_without_healthy_value_rejected(tmp_path: Path) -> None:
    body = _DRIFT.replace("healthy_ma: 48", "note: 48")
    with pytest.raises(ValueError, match=r"'appliance_standby'.*Gefrierschrank"):
        FaultList.load(_write(tmp_path, body))


def test_drift_requires_its_parameters(tmp_path: Path) -> None:
    body = _DRIFT.replace("rise_ma: 40", "other: 40")
    with pytest.raises(ValueError, match=r"'appliance_standby'.*rise_ma"):
        FaultList.load(_write(tmp_path, body))


def test_non_positive_rise_rejected(tmp_path: Path) -> None:
    body = _DRIFT.replace("rise_ma: 40", "rise_ma: 0")
    with pytest.raises(ValueError, match=r"'appliance_standby'.*rise_ma"):
        FaultList.load(_write(tmp_path, body))


def test_references_on_other_kind_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: constancy
            parameters:
              constant_hours: 48
              same_within: 0
            references:
              Waschmaschine:
                healthy_ma: 0
            scope:
              include: "%.Stromwert"
            target:
              ga: "2/2/229"
        """,
    )
    with pytest.raises(ValueError, match=r"'x'.*references"):
        FaultList.load(path)


def test_drift_without_a_signal_rejected(tmp_path: Path) -> None:
    # Which series the CUSUM walks decides what the pinned reference means;
    # a drift fault that does not say it is not loadable.
    body = _DRIFT.replace("    signal: standby\n", "")
    with pytest.raises(ValueError, match=r"'appliance_standby'.*signal"):
        FaultList.load(_write(tmp_path, body))


def test_signal_on_another_kind_rejected(tmp_path: Path) -> None:
    body = _DURATION.replace("    kind: duration\n", "    kind: duration\n    signal: standby\n")
    with pytest.raises(ValueError, match=r"'appliance_runtime'.*signal"):
        FaultList.load(_write(tmp_path, body))


_ICING = """
faults:
  - name: freezer_icing
    sentence: "Der Gefrierschrank verdichtet über den Tag anhaltend mehr als
      15 Prozentpunkte länger als im gesunden Zustand."
    unit: "× der erlaubten Erhöhung"
    kind: drift
    signal: duty_cycle
    parameters:
      rise_pct: 15
      budget_pct_h: 600
      window_hours: 24
      min_window_fraction: 0.6
      door_run_hours: 3
      on_ma: 120
    scope:
      dpt: "7.012"
      include: "%.Gefrierschrank.Stromwert"
    references:
      Küche.K15-L1.Gefrierschrank:
        healthy_duty_pct: 50
    target:
      per_device: true
"""


def test_duty_cycle_drift_loads_its_reference_in_percent(tmp_path: Path) -> None:
    [fault] = FaultList.load(_write(tmp_path, _ICING))
    assert fault.kind is MeasurementKind.DRIFT
    assert fault.signal is DriftSignal.DUTY_CYCLE
    assert fault.references == (
        DeviceReference(match="Küche.K15-L1.Gefrierschrank", healthy=50),
    )
    assert fault.parameters["door_run_hours"] == 3


def test_duty_cycle_drift_requires_its_parameters(tmp_path: Path) -> None:
    body = _ICING.replace("rise_pct: 15", "rise_ma: 15")
    with pytest.raises(ValueError, match=r"'freezer_icing'.*rise_pct"):
        FaultList.load(_write(tmp_path, body))


def test_duty_cycle_drift_needs_its_door_run(tmp_path: Path) -> None:
    # Without the door threshold the fault would count a door left open as
    # icing — the one confusion this fault is built to avoid.
    body = _ICING.replace("      door_run_hours: 3\n", "")
    with pytest.raises(ValueError, match=r"'freezer_icing'.*door_run_hours"):
        FaultList.load(_write(tmp_path, body))


def test_a_parameter_in_the_other_signal_unit_rejected(tmp_path: Path) -> None:
    # A fault switched to the other signal leaves its old numbers behind:
    # they would sit there looking authoritative while nothing reads them.
    body = _ICING.replace("      rise_pct: 15\n", "      rise_pct: 15\n      rise_ma: 40\n")
    with pytest.raises(ValueError, match=r"'freezer_icing'.*rise_ma"):
        FaultList.load(_write(tmp_path, body))

    standby = _DRIFT.replace("      rise_ma: 40\n", "      rise_ma: 40\n      rise_pct: 15\n")
    with pytest.raises(ValueError, match=r"'appliance_standby'.*rise_pct"):
        FaultList.load(_write(tmp_path, standby))


def test_a_reference_in_the_other_signal_unit_rejected(tmp_path: Path) -> None:
    # mA on a duty-cycle fault reads as 50 % where 50 mA was meant: the
    # signal decides the unit, and the file has to say the same thing.
    body = _ICING.replace("healthy_duty_pct: 50", "healthy_ma: 50")
    with pytest.raises(ValueError, match=r"'freezer_icing'.*Gefrierschrank"):
        FaultList.load(_write(tmp_path, body))

    standby = _DRIFT.replace("healthy_ma: 48", "healthy_duty_pct: 48")
    with pytest.raises(ValueError, match=r"'appliance_standby'.*Gefrierschrank"):
        FaultList.load(_write(tmp_path, standby))


_RECOVERY = """
faults:
  - name: heat_recovery_decay
    sentence: "Der Wärmetauscher der KWL holt dauerhaft mehr als 10
      Prozentpunkte weniger Wärme zurück als im gesunden Zustand."
    unit: "× des erlaubten Abfalls"
    kind: drift
    signal: recovery
    parameters:
      fall_pct: 10
      budget_pct_h: 480
      window_hours: 72
      min_window_fraction: 0.1
      min_delta_k: 10
    roles:
      outdoor: "%.KWL.Temperatur-Außenluft"
      extract: "%.KWL.Temperatur-Abluft"
      supply: "%.KWL.Temperatur-Zuluft"
    scope:
      include:
        - "%.KWL.Temperatur-Außenluft"
        - "%.KWL.Temperatur-Abluft"
        - "%.KWL.Temperatur-Zuluft"
    references:
      KWL:
        healthy_pct: 88
    target:
      ga: "15/3/34"
"""


def test_recovery_drift_loads_roles_and_reference(tmp_path: Path) -> None:
    [fault] = FaultList.load(_write(tmp_path, _RECOVERY))
    assert fault.kind is MeasurementKind.DRIFT
    assert fault.signal is DriftSignal.RECOVERY
    assert fault.roles == ExchangerRoles(
        outdoor="%.KWL.Temperatur-Außenluft",
        extract="%.KWL.Temperatur-Abluft",
        supply="%.KWL.Temperatur-Zuluft",
    )
    assert fault.references == (DeviceReference(match="KWL", healthy=88),)
    assert fault.target is not None
    assert fault.target.ga == "15/3/34"


def test_recovery_drift_requires_its_roles(tmp_path: Path) -> None:
    # Which channel is which air decides the sign of every efficiency the
    # fault computes; a recovery fault that does not say it is not loadable.
    head, _roles = _RECOVERY.split("    roles:\n", 1)
    body = head + "    scope:" + _RECOVERY.split("    scope:", 1)[1]
    with pytest.raises(ValueError, match=r"'heat_recovery_decay'.*roles"):
        FaultList.load(_write(tmp_path, body))


def test_recovery_drift_requires_its_gate(tmp_path: Path) -> None:
    # Without the delta gate the efficiency is a quotient of noise whenever
    # inside and outside agree.
    body = _RECOVERY.replace("      min_delta_k: 10\n", "")
    with pytest.raises(ValueError, match=r"'heat_recovery_decay'.*min_delta_k"):
        FaultList.load(_write(tmp_path, body))


def test_recovery_drift_declares_exactly_one_exchanger(tmp_path: Path) -> None:
    body = _RECOVERY.replace(
        "      KWL:\n        healthy_pct: 88\n",
        "      KWL:\n        healthy_pct: 88\n      KWL2:\n        healthy_pct: 88\n",
    )
    with pytest.raises(ValueError, match=r"'heat_recovery_decay'.*references"):
        FaultList.load(_write(tmp_path, body))


def test_recovery_reference_in_another_signal_unit_rejected(tmp_path: Path) -> None:
    body = _RECOVERY.replace("healthy_pct: 88", "healthy_duty_pct: 88")
    with pytest.raises(ValueError, match=r"'heat_recovery_decay'.*KWL"):
        FaultList.load(_write(tmp_path, body))


def test_recovery_parameter_in_another_signal_unit_rejected(tmp_path: Path) -> None:
    # rise_pct belongs to the duty-cycle walk; the recovery walk falls.
    body = _RECOVERY.replace(
        "      fall_pct: 10\n", "      fall_pct: 10\n      rise_pct: 15\n"
    )
    with pytest.raises(ValueError, match=r"'heat_recovery_decay'.*rise_pct"):
        FaultList.load(_write(tmp_path, body))


def test_fall_parameter_on_another_signal_rejected(tmp_path: Path) -> None:
    body = _DRIFT.replace("      rise_ma: 40\n", "      rise_ma: 40\n      fall_pct: 10\n")
    with pytest.raises(ValueError, match=r"'appliance_standby'.*fall_pct"):
        FaultList.load(_write(tmp_path, body))


def test_roles_on_a_standby_drift_rejected(tmp_path: Path) -> None:
    # Standby walks one declared channel per device; air roles on it would
    # be dead configuration nothing reads.
    body = _DRIFT.replace(
        "    scope:\n",
        '    roles:\n      outdoor: "%.Temperatur-Außenluft"\n    scope:\n',
    )
    with pytest.raises(ValueError, match=r"'appliance_standby'.*roles"):
        FaultList.load(_write(tmp_path, body))


def test_rooms_on_a_recovery_drift_rejected(tmp_path: Path) -> None:
    body = _RECOVERY.replace(
        "    scope:\n",
        "    rooms:\n      EG.Büro:\n        min_gap_k: 1.0\n"
        '        value: "Sensorik.EG.Büro.Sensor.Temperatur"\n    scope:\n',
    )
    with pytest.raises(ValueError, match=r"'heat_recovery_decay'.*rooms"):
        FaultList.load(_write(tmp_path, body))


_DEVIATION = """
faults:
  - name: fbh_cold
    sentence: "Ein Raum liegt bei offenem Stellventil zu lange unter seiner
      Soll-Temperatur."
    unit: "× der erlaubten Abweichung"
    kind: deviation
    parameters:
      min_hours: 2
      gate_min_pct: 50
    roles:
      reference: "%.FBH.Soll-Temperatur-Status"
      gate: "%.FBH.Stellwert-Status"
    rooms:
      EG.Büro:
        min_gap_k: 1.0
        value: "Sensorik.EG.Büro.Sensor.Temperatur"
      EG.Flur:
        min_gap_k: 1.0
        value: "Sensorik.EG.Flur.BWM.%.Temperatur"
    scope:
      include:
        - "Sensorik.%.Sensor.Temperatur"
        - "Sensorik.EG.Flur.BWM.%.Temperatur"
        - "Raumklima.%.FBH.Soll-Temperatur-Status"
        - "Raumklima.%.FBH.Stellwert-Status"
    target:
      per_room: true
"""


def test_deviation_fault_loads_roles_and_rooms(tmp_path: Path) -> None:
    [fault] = FaultList.load(_write(tmp_path, _DEVIATION))
    assert fault.kind is MeasurementKind.DEVIATION
    assert fault.parameters == {"min_hours": 2, "gate_min_pct": 50}
    assert fault.target is not None
    assert fault.target.per_room is True
    assert fault.target.ga is None
    assert fault.roles == Roles(
        reference="%.FBH.Soll-Temperatur-Status",
        gate="%.FBH.Stellwert-Status",
    )
    assert fault.rooms == (
        RoomRule(
            match="EG.Büro", min_gap_k=1.0, value="Sensorik.EG.Büro.Sensor.Temperatur"
        ),
        RoomRule(match="EG.Flur", min_gap_k=1.0, value="Sensorik.EG.Flur.BWM.%.Temperatur"),
    )


def test_per_room_target_loads_its_name_template(tmp_path: Path) -> None:
    body = _with_target_name(_DEVIATION, "per_room", "Raumklima.{entity}.FBH.Aktiv-Anomalie")
    [fault] = FaultList.load(_write(tmp_path, body))
    assert fault.target == Target(per_room=True, name="Raumklima.{entity}.FBH.Aktiv-Anomalie")


def test_room_without_a_value_channel_rejected(tmp_path: Path) -> None:
    # There is no shared fallback: a room that names no channel cannot be
    # measured, and the file must say so at load.
    body = _DEVIATION.replace('        value: "Sensorik.EG.Büro.Sensor.Temperatur"\n', "")
    with pytest.raises(ValueError, match=r"'fbh_cold'.*Büro"):
        FaultList.load(_write(tmp_path, body))


def test_value_on_the_roles_block_rejected(tmp_path: Path) -> None:
    # The value belongs to the room, never to the shared roles.
    body = _DEVIATION.replace(
        '      reference: "%.FBH.Soll-Temperatur-Status"',
        '      value: "%.Sensor.Temperatur"\n      reference: "%.FBH.Soll-Temperatur-Status"',
    )
    with pytest.raises(ValueError, match=r"'fbh_cold'.*value"):
        FaultList.load(_write(tmp_path, body))


def test_deviation_without_rooms_rejected(tmp_path: Path) -> None:
    head, _rooms = _DEVIATION.split("    rooms:\n", 1)
    body = head + "    scope:" + _DEVIATION.split("    scope:", 1)[1]
    with pytest.raises(ValueError, match=r"'fbh_cold'.*rooms"):
        FaultList.load(_write(tmp_path, body))


def test_deviation_requires_min_hours(tmp_path: Path) -> None:
    body = _DEVIATION.replace("min_hours: 2", "other: 1").replace(
        "gate_min_pct: 50", "gate_min_pct: 50.0"
    )
    with pytest.raises(ValueError, match=r"'fbh_cold'.*min_hours"):
        FaultList.load(_write(tmp_path, body))


def test_gate_role_without_gate_min_rejected(tmp_path: Path) -> None:
    body = _DEVIATION.replace("      gate_min_pct: 50\n", "")
    with pytest.raises(ValueError, match=r"'fbh_cold'.*gate"):
        FaultList.load(_write(tmp_path, body))


def test_gate_min_without_gate_role_rejected(tmp_path: Path) -> None:
    body = _DEVIATION.replace('      gate: "%.FBH.Stellwert-Status"\n', "")
    with pytest.raises(ValueError, match=r"'fbh_cold'.*gate"):
        FaultList.load(_write(tmp_path, body))


def test_rooms_on_other_kind_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: constancy
            parameters:
              constant_hours: 48
              same_within: 0
            rooms:
              EG.Büro:
                min_gap_k: 1.0
            scope:
              include: "%.Stromwert"
            target:
              ga: "2/2/229"
        """,
    )
    with pytest.raises(ValueError, match=r"'x'.*rooms"):
        FaultList.load(path)


def test_non_positive_room_gap_rejected(tmp_path: Path) -> None:
    body = _DEVIATION.replace("min_gap_k: 1.0\n        value:", "min_gap_k: 0\n        value:")
    with pytest.raises(ValueError, match=r"'fbh_cold'.*min_gap_k"):
        FaultList.load(_write(tmp_path, body))


def test_room_entry_without_gap_rejected(tmp_path: Path) -> None:
    body = _DEVIATION.replace(
        '        min_gap_k: 1.0\n        value: "Sensorik.EG.Büro.Sensor.Temperatur"',
        '        value: "Sensorik.EG.Büro.Sensor.Temperatur"',
    )
    with pytest.raises(ValueError, match=r"'fbh_cold'.*Büro"):
        FaultList.load(_write(tmp_path, body))


_EXPECTATION = """
faults:
  - name: pv_underperformance
    sentence: "Die Anlage hat gestern mehr als 35 % weniger erzeugt, als die
      wetterbereinigte Prognose erwartet hat."
    unit: "× des erlaubten Fehlbetrags"
    kind: deviation
    expectation: forecast_solar
    parameters:
      min_shortfall_pct: 35
      min_expected_kwh: 3
    target:
      ga: "15/4/11"
"""


def test_deviation_fault_loads_a_named_expectation(tmp_path: Path) -> None:
    # The other shape of the kind: the reference is a model named in the
    # entry, so there are no rooms to marry and no catalog query to carry.
    [fault] = FaultList.load(_write(tmp_path, _EXPECTATION))
    assert fault.kind is MeasurementKind.DEVIATION
    assert fault.expectation is DeviationExpectation.FORECAST_SOLAR
    assert fault.scope is None
    assert fault.rooms == ()
    assert fault.roles is None
    assert fault.parameters == {"min_shortfall_pct": 35, "min_expected_kwh": 3}
    assert fault.target == Target(ga="15/4/11")


def test_expectation_fault_needs_its_shortfall(tmp_path: Path) -> None:
    body = _EXPECTATION.replace("min_shortfall_pct: 35", "shortfall: 35")
    with pytest.raises(ValueError, match=r"'pv_underperformance'.*min_shortfall_pct"):
        FaultList.load(_write(tmp_path, body))


def test_expectation_fault_needs_its_floor(tmp_path: Path) -> None:
    # Without it a day that expected nothing would score a shortfall.
    body = _EXPECTATION.replace("      min_expected_kwh: 3\n", "")
    with pytest.raises(ValueError, match=r"'pv_underperformance'.*min_expected_kwh"):
        FaultList.load(_write(tmp_path, body))


def test_expectation_fault_rejects_a_channel_scope(tmp_path: Path) -> None:
    # It measures against a model, so a catalog query here reads nothing.
    body = _EXPECTATION + """    scope:
      include: "%.Photovoltaik.%"
"""
    with pytest.raises(ValueError, match=r"'pv_underperformance'.*scope"):
        FaultList.load(_write(tmp_path, body))


def test_expectation_fault_rejects_rooms(tmp_path: Path) -> None:
    body = _EXPECTATION + """    rooms:
      EG.Büro:
        min_gap_k: 1.0
        value: "Sensorik.EG.Büro.Sensor.Temperatur"
"""
    with pytest.raises(ValueError, match=r"'pv_underperformance'.*rooms"):
        FaultList.load(_write(tmp_path, body))


def test_expectation_fault_rejects_channel_roles(tmp_path: Path) -> None:
    body = _EXPECTATION + """    roles:
      reference: "%.FBH.Soll-Temperatur-Status"
"""
    with pytest.raises(ValueError, match=r"'pv_underperformance'.*roles"):
        FaultList.load(_write(tmp_path, body))


def test_expectation_on_another_kind_rejected(tmp_path: Path) -> None:
    # Only the deviation kind measures against an expectation. The kind it
    # is flipped to brings its own required parameters, so the entry stays
    # schema-valid and the loader's check is what has to catch this.
    body = _EXPECTATION.replace("kind: deviation", "kind: constancy").replace(
        "    parameters:", "    parameters:\n      constant_hours: 48\n      same_within: 0"
    )
    with pytest.raises(ValueError, match=r"'pv_underperformance'.*expectation"):
        FaultList.load(_write(tmp_path, body))


def test_unknown_expectation_rejected(tmp_path: Path) -> None:
    # A model nobody wired up must fail here, not at 03:00 in the cluster.
    body = _EXPECTATION.replace("expectation: forecast_solar", "expectation: clear_sky")
    with pytest.raises(ValueError, match=r"'pv_underperformance'.*expectation"):
        FaultList.load(_write(tmp_path, body))


def test_a_shortfall_over_a_hundred_percent_rejected(tmp_path: Path) -> None:
    # A percentage of the expectation cannot exceed it.
    body = _EXPECTATION.replace("min_shortfall_pct: 35", "min_shortfall_pct: 135")
    with pytest.raises(ValueError, match=r"'pv_underperformance'.*min_shortfall_pct"):
        FaultList.load(_write(tmp_path, body))


_EXTERNAL = """
faults:
  - name: system_pressure_low
    sentence: "Der Systemdruck der Gastherme liegt unter 1,0 bar."
    unit: "bar"
    kind: external
    scope:
      include: "%.Gastherme.System-Druck-Anomalie"
"""


def test_external_fault_loads_without_target_and_parameters(tmp_path: Path) -> None:
    # Basalte owns threshold and delivery; the entry only declares the
    # sentence and where the severity writes appear.
    [fault] = FaultList.load(_write(tmp_path, _EXTERNAL))
    assert fault.kind is MeasurementKind.EXTERNAL
    assert fault.sentence == "Der Systemdruck der Gastherme liegt unter 1,0 bar."
    assert fault.parameters == {}
    assert fault.target is None
    assert fault.scope.include == ("%.Gastherme.System-Druck-Anomalie",)


def test_fingerprint_of_an_external_fault_is_its_kind_alone(tmp_path: Path) -> None:
    [fault] = FaultList.load(_write(tmp_path, _EXTERNAL))
    assert fault.fingerprint == "external()"


def test_external_fault_rejects_a_target(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _EXTERNAL
        + """    target:
      ga: "15/2/2"
""",
    )
    with pytest.raises(ValueError, match=r"'system_pressure_low'.*target"):
        FaultList.load(path)


def test_external_fault_rejects_parameters(tmp_path: Path) -> None:
    # A threshold on an external fault would be a second source of truth
    # beside the Basalte logic.
    path = _write(
        tmp_path,
        _EXTERNAL
        + """    parameters:
      threshold: 1
""",
    )
    with pytest.raises(ValueError, match=r"'system_pressure_low'.*parameters"):
        FaultList.load(path)


def test_engine_kind_still_requires_target(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: constancy
            parameters:
              constant_hours: 48
              same_within: 0
            scope:
              include: "%.Stromwert"
        """,
    )
    with pytest.raises(ValueError, match=r"'x'.*target"):
        FaultList.load(path)


_VOLUME = """
faults:
  - name: notification_volume
    sentence: "Mehr als 5 Vorfälle in sieben Tagen."
    unit: "× der erlaubten Wochenmenge"
    kind: volume
    parameters:
      max_episodes_per_week: 5
    target:
      ga: "0/0/251"
"""


def test_volume_fault_loads_without_a_channel_scope(tmp_path: Path) -> None:
    # The watchdog measures the episode stream, not channels — so it has no
    # catalog query to carry, and the limit is the whole configuration.
    [fault] = FaultList.load(_write(tmp_path, _VOLUME))
    assert fault.kind is MeasurementKind.VOLUME
    assert fault.scope is None
    assert fault.parameters == {"max_episodes_per_week": 5}
    assert fault.target == Target(ga="0/0/251")


def test_volume_fault_needs_its_limit(tmp_path: Path) -> None:
    # N lives in the fault file: without it there is nothing to be over.
    path = _write(tmp_path, _VOLUME.replace("max_episodes_per_week: 5", "window_days: 7"))
    with pytest.raises(ValueError, match=r"'notification_volume'.*max_episodes_per_week"):
        FaultList.load(path)


def test_volume_fault_rejects_a_channel_scope(tmp_path: Path) -> None:
    # A scope here would be configuration nothing reads.
    path = _write(
        tmp_path,
        _VOLUME
        + """    scope:
      include: "%"
""",
    )
    with pytest.raises(ValueError, match=r"'notification_volume'.*scope"):
        FaultList.load(path)


def test_measuring_kinds_still_require_a_scope(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: constancy
            parameters:
              constant_hours: 48
              same_within: 0
            target:
              ga: "2/2/229"
        """,
    )
    with pytest.raises(ValueError, match=r"'x'.*scope"):
        FaultList.load(path)


def test_bundled_schema_is_valid() -> None:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)


def test_a_measured_fault_is_explained_unless_it_says_otherwise(tmp_path: Path) -> None:
    # `explain` decides whether an episode event of this fault reaches the
    # bus at all. On by default for everything the engine measures itself:
    # nobody has said what is wrong yet, which is the whole reason to ask.
    faults = FaultList.load(_write(tmp_path, _VALID))
    assert faults.get("channel_silence").explain is True
    assert faults.get("appliance_standby").explain is True


def test_every_measured_kind_defaults_to_explained(tmp_path: Path) -> None:
    for body in (_CONSTANCY, _DURATION, _DRIFT, _RECOVERY, _DEVIATION, _EXPECTATION, _VOLUME):
        for fault in FaultList.load(_write(tmp_path, body)):
            assert fault.explain is True, f"{fault.name} ({fault.kind})"


def test_an_external_fault_is_not_explained_by_default(tmp_path: Path) -> None:
    # Basalte detected it and its own sentence is the explanation; a second
    # one would say the same thing twice.
    [fault] = FaultList.load(_write(tmp_path, _EXTERNAL))
    assert fault.explain is False


def test_a_fault_overrides_the_default_with_one_line(tmp_path: Path) -> None:
    quiet = FaultList.load(_write(tmp_path, _VOLUME.rstrip() + "\n    explain: false\n"))
    assert quiet.get("notification_volume").explain is False

    loud = FaultList.load(_write(tmp_path, _EXTERNAL.rstrip() + "\n    explain: true\n"))
    assert loud.get("system_pressure_low").explain is True


def test_a_non_boolean_explain_names_fault_and_field(tmp_path: Path) -> None:
    path = _write(tmp_path, _VOLUME.rstrip() + "\n    explain: sometimes\n")
    with pytest.raises(ValueError, match=r"'notification_volume'.*explain"):
        FaultList.load(path)


# --- one entry on its own --------------------------------------------------

_ENTRY = {
    "name": "candidate_runtime",
    "sentence": "Ein Gerät zieht länger Strom, als seine erlaubte Laufzeit zulässt.",
    "unit": "× der erlaubten Laufzeit",
    "kind": "duration",
    "parameters": {"active_hour_fraction": 0.5},
    "scope": {"dpt": "7.012", "include": "%.Trockner.Stromwert"},
    "devices": {"KG.Hauswirtschaftsraum.K3-L1.Trockner": {"max_run_hours": 4}},
    "target": {"per_device": True},
}


def test_one_entry_loads_on_its_own() -> None:
    fault = parse_entry(_ENTRY)
    assert fault.name == "candidate_runtime"
    assert fault.kind is MeasurementKind.DURATION
    assert fault.devices == (
        DeviceLimit(match="KG.Hauswirtschaftsraum.K3-L1.Trockner", max_run_hours=4),
    )
    assert fault.explain is True


def test_one_entry_fails_the_schema_rules_a_file_edit_fails() -> None:
    with pytest.raises(ValueError, match=r"'candidate_runtime'.*active_hour_fraction"):
        parse_entry({**_ENTRY, "parameters": {"max_run_hours": 4}})


def test_one_entry_fails_the_cross_field_rules_a_file_edit_fails() -> None:
    constancy = {
        **_ENTRY,
        "kind": "constancy",
        "parameters": {"constant_hours": 48, "same_within": 0},
    }
    with pytest.raises(ValueError, match="only a duration fault declares per-device limits"):
        parse_entry(constancy)


def test_one_entry_error_says_it_came_from_a_candidate() -> None:
    # Where a loaded entry's error names the file it was read from.
    with pytest.raises(ValueError, match="candidate: fault 'candidate_runtime'"):
        parse_entry({**_ENTRY, "devices": {}})


def test_parsing_an_entry_leaves_the_caller_s_mapping_alone() -> None:
    entry = {**_ENTRY}
    parse_entry(entry)
    assert entry == _ENTRY
