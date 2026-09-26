"""The registry: which shape a declared fault runs in.

One `runner.Kind` per measurement kind — how its series is measured and how
its payload is shaped — and `kind_for`, which picks the one a fault
declares. The runner owns the lifecycle around a kind, the kind's own module
owns the measurement; this module is only the mapping between the file's
`kind` line and that declaration.

It sits below both callers on purpose: the detect-faults job runs the whole
schedulable list through it, and the back-test runs one candidate through
it, so both measure a fault the same way by construction rather than by
agreement.

Channel silence and constancy measure per channel but report per main group
— each on its own address there — so their declarations carry their own
plan. A deviation fault that names an expectation runs in days rather than
hours: the plant's whole-day yield against the kWh its named model expected
of it. The volume watchdog runs over the engine's own episode stream.
External faults run the loop the other way round: Basalte detects, writes
the severity itself and delivers itself; the engine reads those writes back
off the bus archive, records episodes marked externally delivered, and
publishes nothing.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from . import (
    constancy,
    deviation,
    drift,
    duration,
    external,
    silence,
    volume,
)
from .faults import DriftSignal, MeasurementKind
from .runner import Kind

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .faults import Fault
    from .site import Site


_KINDS: Mapping[MeasurementKind, Kind[Any, Any]] = {
    MeasurementKind.DURATION: Kind(
        event="appliance_runtime_run",
        delivery="per_device",
        frontier=duration.frontier,
        measure=duration.measure,
        publish_for=duration.publish_for,
        payload=duration.payload,
    ),
    MeasurementKind.DEVIATION: Kind(
        event="room_deviation_run",
        delivery="per_room",
        frontier=silence.frontier,
        measure=deviation.measure,
        publish_for=deviation.publish_for,
        payload=deviation.payload,
    ),
    MeasurementKind.SILENCE: Kind(
        event="channel_silence_run",
        delivery="per_main_group",
        frontier=silence.frontier,
        measure=silence.measure,
        # Measured per channel, delivered per main group — the one kind
        # whose plan is its own.
        plan=silence.plan_run,
        payload=silence.group_payload,
        # The unproven guard is code: its revision is part of the stamp.
        fingerprint=silence.fingerprint,
        # The dataless set is every never-sent symmetry address in the
        # catalog — a thousand of them, normal, and already counted by the
        # measurement's scope_drops record. The ones actually held open are
        # `stale_opens` in the run record.
        warn_dataless=False,
    ),
    MeasurementKind.CONSTANCY: Kind(
        event="channel_constancy_run",
        delivery="per_main_group",
        # Same aggregate as silence, and the same delivery form — but its
        # own address per main group: two writers on one would overwrite
        # each other's clears.
        frontier=silence.frontier,
        measure=constancy.measure,
        plan=constancy.plan_run,
        payload=constancy.group_payload,
        # Every never-sent symmetry address in the scope lands in the
        # dataless set, exactly as it does for silence, and the run record
        # already counts them. The ones actually held open are
        # `stale_opens`.
        warn_dataless=False,
    ),
    MeasurementKind.VOLUME: Kind(
        event="notification_volume_run",
        # One house-wide address; declared last in the fault list, so the
        # count already includes what this run's other faults just wrote.
        delivery="ga",
        frontier=silence.frontier,
        measure=volume.measure,
        publish_for=volume.publish_for,
        payload=volume.payload,
    ),
    MeasurementKind.EXTERNAL: Kind(
        event="external_severities_run",
        # Basalte detects and delivers itself: the fault declares no
        # target, the fold walks severity writes instead of observations,
        # the plan keeps rows open until their explicit 0, and nothing
        # is published — the engine only records.
        delivery=None,
        frontier=external.frontier,
        measure=external.measure,
        fold=external.fold,
        plan=external.plan,
        externally_delivered=True,
    ),
}


def _daily_yield(site: Site) -> Kind[Any, Any]:
    """The deviation kind's other shape: a fault that names an expectation
    measures the plant's whole-day yield against it, in daily buckets, on
    the plant's one declared address. The plant is what the site file
    declares — which inverters, and what each can make in an hour — so the
    shape is built around the loaded site rather than declared once.
    """
    return Kind(
        event="daily_yield_run",
        delivery="ga",
        frontier=deviation.yield_frontier,
        measure=partial(deviation.measure_yield, site=site),
        publish_for=deviation.publish_for_plant,
        payload=deviation.payload_yield,
        policy=deviation.YIELD_POLICY,
        history=deviation.YIELD_HISTORY,
    )


# The drift kind runs one shape per signal: same CUSUM, different series,
# so the run record and the payload's units differ with the signal the
# fault declares.
_DRIFT_SIGNALS: Mapping[DriftSignal, Kind[Any, Any]] = {
    DriftSignal.STANDBY: Kind(
        event="appliance_standby_run",
        delivery="per_device",
        frontier=duration.frontier,
        measure=drift.measure_standby,
        publish_for=drift.publish_for,
        payload=drift.payload_standby,
    ),
    DriftSignal.DUTY_CYCLE: Kind(
        event="duty_cycle_drift_run",
        delivery="per_device",
        frontier=duration.frontier,
        measure=drift.measure_duty_cycle,
        publish_for=drift.publish_for,
        payload=drift.payload_duty_cycle,
        # The one shape that reads the bus archive rather than an aggregate.
        history=drift.ARCHIVE_HISTORY,
    ),
    DriftSignal.RECOVERY: Kind(
        event="heat_recovery_run",
        # One exchanger, one declared address — the volume watchdog's form,
        # not the appliances' per-device fan-out.
        delivery="ga",
        frontier=silence.frontier,
        measure=drift.measure_recovery,
        publish_for=drift.publish_for_exchanger,
        payload=drift.payload_recovery,
    ),
}


def kind_for(fault: Fault, site: Site | None) -> Kind[Any, Any] | None:
    """The shape this fault runs in, if it has one. Drift picks it by the
    signal the file declares — the loader rejects one without, so a fault
    that got here signalless is a new kind of drift nobody wired up, and it
    fails rather than reporting nothing. Deviation picks it by whether the
    fault names an expectation: with one it measures the declared site's
    daily yield against that model, without one a room against its setpoint.

    Only that one shape needs the site, which is why it may be None here: a
    candidate about a room or a channel is back-tested without one.
    """
    if fault.kind is MeasurementKind.DRIFT:
        if fault.signal is None:
            raise ValueError(f"fault {fault.name}: a drift fault declares which series it walks")
        return _DRIFT_SIGNALS[fault.signal]
    if fault.kind is MeasurementKind.DEVIATION and fault.expectation is not None:
        if site is None:
            raise ValueError(
                f"fault {fault.name}: measured against {fault.expectation} it is the plant's "
                f"daily yield — that shape needs the site the plant stands on"
            )
        return _daily_yield(site)
    return _KINDS.get(fault.kind)
