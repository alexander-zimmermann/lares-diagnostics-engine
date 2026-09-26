"""Fault-list loader: YAML file -> validated, typed fault definitions.

The fault list lives in lares next to the GA catalog and the writer rules;
this loader validates it against the bundled JSON Schema and freezes it into
dataclasses — the same pattern the KNX bridge uses for its writer rules. A
bad edit fails here, at load, with an error naming the fault and the field,
never at runtime in the cluster.

The channel scope is a catalog query carried verbatim: the loader never
resolves it, so the file stays loadable without a catalog at hand. Dormant
faults load fully but are excluded from `schedulable()`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .declared import SCHEMAS, describe_field, load_declared, validate_declared

if TYPE_CHECKING:
    import jsonschema

_SCHEMA_PATH = SCHEMAS / "faults.schema.json"


class MeasurementKind(StrEnum):
    DRIFT = "drift"
    DURATION = "duration"
    DEVIATION = "deviation"
    SILENCE = "silence"
    CONSTANCY = "constancy"
    # Not measured here: Basalte detects and delivers, the engine reads the
    # severity writes back off the fault address and records episodes.
    EXTERNAL = "external"
    # Not measured on channels at all: the count of incidents per week, over
    # the episode stream every other fault writes.
    VOLUME = "volume"


class DriftSignal(StrEnum):
    """Which series a drift fault walks its CUSUM over. The method is the
    same either way — accumulate what sits above a pinned healthy level —
    but the series decides what the reference and the rise mean, which is
    why the file declares it instead of the code guessing from the scope.
    """

    # The device's idle draw, in mA: a relay that no longer opens.
    STANDBY = "standby"
    # The share of the day a compressor runs, in percent: an evaporator
    # icing up makes the same cold cost more running.
    DUTY_CYCLE = "duty_cycle"
    # The heat-recovery efficiency of an air exchanger, in percent: the one
    # signal that walks downward — fouling lowers what the exchanger can do.
    RECOVERY = "recovery"


class DeviationExpectation(StrEnum):
    """Where a deviation fault takes its expectation from, where that is not
    a channel of the house. The fault entry names one; the computation
    compares a measured yield against expected kWh and never learns which
    model produced them, which is what makes the swap a line in the file
    and a function here rather than a rewrite.
    """

    # The weather-adjusted forecast.solar curve, integrated per day.
    FORECAST_SOLAR = "forecast_solar"


# What each signal calls its declared healthy level. The unit is in the
# name because that is the whole point of the reference; the loader keeps
# it out of the dataclass, where the signal already says what it is.
_HEALTHY_FIELD = {
    DriftSignal.STANDBY: "healthy_ma",
    DriftSignal.DUTY_CYCLE: "healthy_duty_pct",
    DriftSignal.RECOVERY: "healthy_pct",
}

# And what each signal calls the numbers it walks by. The schema requires
# its own; this is what makes another signal's an error rather than a
# line that sits there looking authoritative while nothing reads it. A
# name two signals share (the budget in percent-hours) is no one's error.
_SIGNAL_PARAMETERS = {
    DriftSignal.STANDBY: ("rise_ma", "budget_ma_h"),
    DriftSignal.DUTY_CYCLE: ("rise_pct", "budget_pct_h", "door_run_hours", "on_ma"),
    DriftSignal.RECOVERY: ("fall_pct", "budget_pct_h", "min_delta_k"),
}


@dataclass(frozen=True, slots=True)
class Scope:
    """Catalog query text. Each field is a conjunctive criterion; list-valued
    criteria are disjunctive within themselves (any DPT of, any pattern of).
    Resolution — including dropping dead and dormant channels — happens where
    the catalog lives, never in the loader.
    """

    dpt: tuple[str, ...] = ()
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Target:
    """One group address, one address per matched main group in that
    group's Zentral block (channel silence), one address per declared
    device (appliance runtime), or one per declared room (room deviation)
    — resolved via the catalog and the writer rules, never listed here.

    The per-entity forms may carry the address's catalog-name template,
    `{entity}` standing for the entity as the fault's device or room map
    names it (`Schalten.{entity}.Stromwert-Standby-Anomalie` renders to
    `Schalten.Küche.K12-L1.Mikrowelle.Stromwert-Standby-Anomalie`). The
    engine only validates it: the lares generator resolves it against the
    catalog into the writer rules, delivery stays the bridge's job.
    """

    ga: str | None = None
    per_main_group: bool = False
    per_device: bool = False
    per_room: bool = False
    name: str | None = None

    def __post_init__(self) -> None:
        # Mirrors the schema's oneOf so the union holds for Python-side
        # construction too, not only for loaded files.
        forms = (self.ga is not None, self.per_main_group, self.per_device, self.per_room)
        if sum(forms) != 1:
            raise ValueError(
                "target is exactly one of ga, per_main_group, per_device or per_room"
            )
        # And the loader's name-template rules, for the same reason.
        if self.name is not None and not (self.per_device or self.per_room):
            raise ValueError("a name template belongs to a per_device or per_room target")
        if self.name is not None and "{entity}" not in self.name:
            raise ValueError("a name template needs an {entity} placeholder")

    @property
    def form(self) -> str:
        """Which of the four this target declares — exactly one, by
        construction, so a runner can compare it against the form its kind
        delivers on."""
        if self.ga is not None:
            return "ga"
        if self.per_main_group:
            return "per_main_group"
        if self.per_device:
            return "per_device"
        return "per_room"


@dataclass(frozen=True, slots=True)
class DeviceLimit:
    """One device's declared limit: a unique fragment of its catalog name,
    and the runtime it may not exceed — written here after someone looked
    at the data once, never derived from history.
    """

    match: str
    max_run_hours: float


@dataclass(frozen=True, slots=True)
class DeviceReference:
    """One device's declared healthy level: a unique fragment of its catalog
    name, and the level it sits at when nothing is wrong — written here
    after someone read it off the data once, never derived from history. A
    reference the detector computed for itself would adopt a months-old
    fault as healthy, which is the whole reason it is declared.

    The unit is the signal's: milliamps of idle draw for `standby`, percent
    of the day for `duty_cycle`. The file names it either way; here the
    fault's signal already says which one this is.
    """

    match: str
    healthy: float


@dataclass(frozen=True, slots=True)
class Roles:
    """The deviation kind's shared channel roles, each a catalog-name LIKE
    pattern matched inside every room's channels: the reference the value
    deviates from, and an optional gate that must stand for the fault to
    count at all. Both follow a uniform naming rule across rooms, which is
    what makes a pattern the honest way to write them; the measured value
    does not, so it is declared per room.
    """

    reference: str
    gate: str | None = None


@dataclass(frozen=True, slots=True)
class ExchangerRoles:
    """The recovery signal's channel roles, each a catalog-name LIKE pattern
    matched against the scope: which air is which decides the sign of every
    efficiency the fault computes, so the file says it instead of the code
    guessing from the names.
    """

    outdoor: str
    extract: str
    supply: str


@dataclass(frozen=True, slots=True)
class RoomRule:
    """One room's declared rule: a unique fragment of its catalog names, the
    channel it is measured on, and the gap under the reference it may not
    exceed — written here after someone looked at the data once. The value
    channel is named per room rather than matched by a shared pattern: rooms
    disagree about which channel is theirs, and naming it makes the file
    readable without resolving anything.
    """

    match: str
    min_gap_k: float
    value: str


@dataclass(frozen=True, slots=True)
class Dormant:
    """Declared dormancy: why the fault cannot run yet, and the observable
    condition under which it starts to — never a bare boolean with prose
    beside it.
    """

    reason: str
    active_when: str


@dataclass(frozen=True, slots=True)
class Fault:
    """One declared fault: a sentence with a unit, measured one way, with
    parameters expressed in the channel's own unit. External faults carry
    neither parameters nor a target — threshold and delivery live in
    Basalte, the scope names the address whose writes come back. A volume
    fault carries no scope: it measures the episode stream, not channels,
    and neither does a deviation fault that names an `expectation` — it
    measures a yield against that model. `explain` is whether an episode
    event of this fault goes on the bus for an agent to explain.
    """

    name: str
    sentence: str
    unit: str
    kind: MeasurementKind
    parameters: Mapping[str, float]
    signal: DriftSignal | None = None
    scope: Scope | None = None
    target: Target | None = None
    dormant: Dormant | None = None
    devices: tuple[DeviceLimit, ...] = ()
    references: tuple[DeviceReference, ...] = ()
    roles: Roles | ExchangerRoles | None = None
    rooms: tuple[RoomRule, ...] = ()
    expectation: DeviationExpectation | None = None
    explain: bool = True

    @property
    def fingerprint(self) -> str:
        """The rule this fault declares, fingerprinted: its kind — with the
        signal or expectation that picks the kind's shape — and its
        `parameters` block, read as numbers (`5` and `5.0` declare the same
        rule). Per-subject declarations — device limits, references, rooms
        — are not part of it: they name who is measured, not how. Every
        episode carries the fingerprint of the rule that last made it, so a
        run can tell its own rows from an earlier rule's leftovers.
        """
        shape = "/".join(
            str(part) for part in (self.kind, self.signal, self.expectation) if part is not None
        )
        parameters = ", ".join(
            f"{name}={float(value)!r}" for name, value in sorted(self.parameters.items())
        )
        return f"{shape}({parameters})"

    def channel_scope(self) -> Scope:
        """The catalog query this fault measures over. Every fault that
        measures channels declares one and the loader enforces it, so a
        missing scope here is a kind that measures channels and arrived
        without saying which — never an empty query, which would resolve to
        the whole catalog. The two that measure none (the volume watchdog,
        and a deviation fault with a named expectation) never ask.
        """
        if self.scope is None:
            raise ValueError(f"fault {self.name}: this kind measures channels and needs a scope")
        return self.scope


class FaultList:
    def __init__(self, faults: list[Fault]) -> None:
        self._faults = tuple(faults)
        self._by_name = {f.name: f for f in faults}

    def __len__(self) -> int:
        return len(self._faults)

    def __iter__(self) -> Iterator[Fault]:
        return iter(self._faults)

    def get(self, name: str) -> Fault:
        return self._by_name[name]

    def schedulable(self) -> tuple[Fault, ...]:
        """The faults a scheduler may run — everything not declared dormant."""
        return tuple(f for f in self._faults if f.dormant is None)

    @classmethod
    def load(cls, path: Path) -> FaultList:
        data = load_declared(path, _SCHEMA_PATH, _describe)
        faults: list[Fault] = []
        seen: set[str] = set()
        for raw in data["faults"]:
            name = raw["name"]
            if name in seen:
                raise ValueError(f"{path}: duplicate fault name {name!r}")
            seen.add(name)
            faults.append(_checked_fault(raw, source=str(path)))
        return cls(faults)


def parse_entry(entry: Mapping[str, Any]) -> Fault:
    """One declared entry on its own — a line of the fault file, or a
    candidate nobody has written down yet — validated against the same
    bundled schema and frozen the same way a loaded file's entries are.

    This is the door for a fault without a file: a back-tested candidate
    fails here on exactly what a real edit would fail on, worded the same
    way, with `candidate` where a loaded entry names its path.
    """
    data = validate_declared(
        {"faults": [dict(entry)]}, _SCHEMA_PATH, _describe, source="candidate"
    )
    return _checked_fault(data["faults"][0], source="candidate")


def _checked_fault(raw: dict[str, Any], *, source: str) -> Fault:
    """One schema-valid entry through the cross-field rules the schema
    cannot word well enough, then frozen."""
    problem = (
        _check_external(raw)
        or _check_volume(raw)
        or _check_target(raw)
        or _check_devices(raw)
        or _check_references(raw)
        or _check_signal(raw)
        or _check_expectation(raw)
        or _check_rooms(raw)
    )
    if problem is not None:
        raise ValueError(f"{source}: {problem}")
    return _parse_fault(raw)


def _describe(error: jsonschema.ValidationError, data: Any) -> str:
    """Prefix the schema error with the fault it belongs to, by name where
    the entry has one, by position where the name itself is what is missing.
    """
    path = list(error.absolute_path)
    if len(path) >= 2 and path[0] == "faults" and isinstance(path[1], int):
        idx = path[1]
        raw = data["faults"][idx]
        name = raw.get("name") if isinstance(raw, dict) else None
        label = repr(name) if isinstance(name, str) else f"#{idx + 1}"
        field = ".".join(str(p) for p in path[2:])
        prefix = f"fault {label}: " + (f"{field}: " if field else "")
        return prefix + error.message
    return describe_field(error, data)


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _check_external(raw: dict[str, Any]) -> str | None:
    """External faults declare neither parameters nor a target: threshold
    and delivery live in Basalte, and a value here would be a second source
    of truth beside the Studio logic. Checked in the loader because the
    schema's false-subschema error loses the field name.
    """
    if raw["kind"] != MeasurementKind.EXTERNAL:
        return None
    for forbidden in ("parameters", "target"):
        if forbidden in raw:
            return (
                f"fault {raw['name']!r}: {forbidden}: "
                f"an external fault declares none — Basalte owns it"
            )
    return None


def _check_volume(raw: dict[str, Any]) -> str | None:
    """A volume fault measures the episode stream, so a channel scope on it
    would be configuration nothing reads. Checked in the loader because the
    schema's if/else error names the branch rather than the field.
    """
    if raw["kind"] == MeasurementKind.VOLUME and "scope" in raw:
        return (
            f"fault {raw['name']!r}: scope: a volume fault counts episodes, "
            f"not channels — it declares none"
        )
    return None


def _check_target(raw: dict[str, Any]) -> str | None:
    """A name template belongs to the per-entity forms — per main group the
    Zentral block names the address, a fixed address needs no name — and
    must keep its placeholder, or every entity would share one address.
    Checked in the loader because the schema's oneOf error loses the field
    name; the forms' exclusivity the schema reports by fault and field.
    """
    target = raw.get("target", {})
    if "name" not in target:
        return None
    if not (target.get("per_device") or target.get("per_room")):
        return (
            f"fault {raw['name']!r}: target.name: "
            f"a name template belongs to a per_device or per_room target"
        )
    if "{entity}" not in target["name"]:
        return (
            f"fault {raw['name']!r}: target.name: the template needs an {{entity}} "
            f"placeholder — without one every entity would share one address"
        )
    return None


def _check_devices(raw: dict[str, Any]) -> str | None:
    """Per-device limits belong to the duration kind alone; anywhere else
    they would be dead configuration nothing reads. Checked in the loader
    because the schema's false-subschema error loses the field name.
    """
    if raw["kind"] != MeasurementKind.DURATION and "devices" in raw:
        return f"fault {raw['name']!r}: devices: only a duration fault declares per-device limits"
    return None


def _check_references(raw: dict[str, Any]) -> str | None:
    """Healthy references belong to the drift kind alone; anywhere else they
    would be dead configuration nothing reads. The recovery signal declares
    exactly one — its subject is the one exchanger, not a fleet of devices.
    Checked in the loader because the schema's false-subschema error loses
    the field name.
    """
    if raw["kind"] != MeasurementKind.DRIFT and "references" in raw:
        return (
            f"fault {raw['name']!r}: references: "
            f"only a drift fault declares healthy references"
        )
    if (
        raw["kind"] == MeasurementKind.DRIFT
        and raw.get("signal") == DriftSignal.RECOVERY
        and len(raw.get("references", {})) != 1
    ):
        return (
            f"fault {raw['name']!r}: references: "
            f"a recovery fault declares exactly one exchanger"
        )
    return None


def _check_signal(raw: dict[str, Any]) -> str | None:
    """The signal belongs to the drift kind alone, and it owns the unit of
    every number the fault walks by: a fault switched from one signal to
    another must lose the parameters of the one it left. Checked in the
    loader because the schema's false-subschema error loses the field name.
    """
    if raw["kind"] != MeasurementKind.DRIFT:
        if "signal" in raw:
            return (
                f"fault {raw['name']!r}: signal: "
                f"only a drift fault declares which series it walks"
            )
        return None
    signal = DriftSignal(raw["signal"])
    mine = set(_SIGNAL_PARAMETERS[signal])
    foreign = {
        name: other
        for other, names in _SIGNAL_PARAMETERS.items()
        if other is not signal
        for name in names
        if name not in mine and name in raw["parameters"]
    }
    if foreign:
        named = ", ".join(f"{name} belongs to {other}" for name, other in sorted(foreign.items()))
        return f"fault {raw['name']!r}: parameters: {named}, not to {signal}"
    return None


def _check_expectation(raw: dict[str, Any]) -> str | None:
    """A named expectation belongs to the deviation kind alone, and it is
    that kind's other shape: a yield measured against a model, not a room
    against its setpoint. So the rooms, their shared roles and the catalog
    query they are resolved through would be dead configuration nothing
    reads. Checked in the loader because the schema's if/else error names
    the branch rather than the field.
    """
    if "expectation" not in raw:
        return None
    if raw["kind"] != MeasurementKind.DEVIATION:
        return (
            f"fault {raw['name']!r}: expectation: only a deviation fault "
            f"names where its expectation comes from"
        )
    for forbidden in ("roles", "rooms", "scope"):
        if forbidden in raw:
            return (
                f"fault {raw['name']!r}: {forbidden}: a deviation fault that names an "
                f"expectation measures against that model, not against channels"
            )
    return None


def _roles_owner(raw: dict[str, Any]) -> bool:
    """Whether this fault's kind (and signal) declares channel roles at all:
    the deviation kind's reference and gate, or the recovery signal's airs.
    """
    return raw["kind"] == MeasurementKind.DEVIATION or (
        raw["kind"] == MeasurementKind.DRIFT and raw.get("signal") == DriftSignal.RECOVERY
    )


def _check_rooms(raw: dict[str, Any]) -> str | None:
    """Roles belong to the deviation kind and the recovery signal, per-room
    rules to the deviation kind alone; anywhere else they would be dead
    configuration nothing reads. A deviation gate role needs its threshold
    (and the other way round) — half a gate would silently measure ungated.
    Checked in the loader because the schema's false-subschema error loses
    the field name.
    """
    if not _roles_owner(raw) and "roles" in raw:
        return (
            f"fault {raw['name']!r}: roles: only a deviation fault or a "
            f"recovery drift declares channel roles"
        )
    if raw["kind"] != MeasurementKind.DEVIATION:
        if "rooms" in raw:
            return f"fault {raw['name']!r}: rooms: only a deviation fault declares rooms"
        return None
    if "expectation" in raw:
        # The other shape of the kind: no rooms, so no gate to pair up.
        return None
    has_gate = "gate" in raw["roles"]
    has_gate_min = "gate_min_pct" in raw.get("parameters", {})
    if has_gate != has_gate_min:
        return (
            f"fault {raw['name']!r}: a gate role and the gate_min_pct parameter "
            f"come together — one without the other measures ungated"
        )
    return None


def _parse_references(
    raw: dict[str, Any], signal: DriftSignal | None
) -> tuple[DeviceReference, ...]:
    """The declared healthy levels, read under the field name the signal
    gives them. Only a drift fault carries references and a drift fault
    always declares its signal — both enforced above — so a reference
    without a signal cannot reach here.
    """
    if signal is None:
        return ()
    field = _HEALTHY_FIELD[signal]
    return tuple(
        DeviceReference(match=match, healthy=reference[field])
        for match, reference in raw.get("references", {}).items()
    )


def _parse_roles(
    roles: dict[str, Any] | None, signal: DriftSignal | None
) -> Roles | ExchangerRoles | None:
    """The declared channel roles in the shape their owner gives them — the
    checks above already pinned roles to the deviation kind or the recovery
    signal, so the signal alone decides which shape this is.
    """
    if roles is None:
        return None
    if signal is DriftSignal.RECOVERY:
        return ExchangerRoles(
            outdoor=roles["outdoor"], extract=roles["extract"], supply=roles["supply"]
        )
    return Roles(reference=roles["reference"], gate=roles.get("gate"))


def _parse_target(target: dict[str, Any]) -> Target:
    return Target(
        ga=target.get("ga"),
        per_main_group=target.get("per_main_group", False),
        per_device=target.get("per_device", False),
        per_room=target.get("per_room", False),
        name=target.get("name"),
    )


def _explained_by_default(kind: MeasurementKind) -> bool:
    """Whether an episode of this kind reaches the bus for an agent to
    explain, where the fault itself says nothing. On for every kind the
    engine measures: nobody has said yet what is wrong, which is the whole
    reason to ask. Off for external, where Basalte detected the fault and
    its own sentence already is the explanation.
    """
    return kind is not MeasurementKind.EXTERNAL


def _parse_fault(raw: dict[str, Any]) -> Fault:
    kind = MeasurementKind(raw["kind"])
    scope = raw.get("scope")
    target = raw.get("target")
    dormant = raw.get("dormant")
    roles = raw.get("roles")
    signal = DriftSignal(raw["signal"]) if "signal" in raw else None
    return Fault(
        name=raw["name"],
        sentence=raw["sentence"],
        unit=raw["unit"],
        kind=kind,
        parameters=MappingProxyType(dict(raw.get("parameters", {}))),
        signal=signal,
        scope=(
            Scope(
                dpt=_as_tuple(scope.get("dpt")),
                include=_as_tuple(scope.get("include")),
                exclude=_as_tuple(scope.get("exclude")),
            )
            if scope is not None
            else None
        ),
        target=_parse_target(target) if target is not None else None,
        devices=tuple(
            DeviceLimit(match=match, max_run_hours=limit["max_run_hours"])
            for match, limit in raw.get("devices", {}).items()
        ),
        references=_parse_references(raw, signal),
        roles=_parse_roles(roles, signal),
        rooms=tuple(
            RoomRule(match=match, min_gap_k=rule["min_gap_k"], value=rule["value"])
            for match, rule in raw.get("rooms", {}).items()
        ),
        expectation=(
            DeviationExpectation(raw["expectation"]) if "expectation" in raw else None
        ),
        explain=raw.get("explain", _explained_by_default(kind)),
        dormant=(
            Dormant(reason=dormant["reason"], active_when=dormant["active_when"])
            if dormant is not None
            else None
        ),
    )
