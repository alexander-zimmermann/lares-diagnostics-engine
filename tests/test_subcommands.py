"""The engine's job surface.

With the old detection switched off, the forecast pulls, the energy
balance and the one fault runner are dispatchable — faults are YAML
entries behind `detect-faults`, never subcommands of their own. Each run
of one is a trace of its own.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from lares_diagnostics_engine import detect_faults
from lares_diagnostics_engine.__main__ import SUBCOMMANDS, main
from lares_diagnostics_engine.faults import Fault

# Two faults the scheduler runs, one dormant.
_FAULTS = """
faults:
  - name: channel_silence
    sentence: "Ein Kanal schweigt länger als das Fünffache seiner Sendepause."
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
  - name: register_constancy
    sentence: "Ein Kanal liefert seit Tagen denselben Wert."
    unit: "× der erlaubten Dauer"
    kind: constancy
    parameters:
      constant_hours: 48
      same_within: 0
    scope:
      include: "%"
    dormant:
      reason: "Waits for its threshold."
      active_when: "the threshold is measured"
    target:
      per_main_group: true
"""

_SITE = """
location: {latitude: 50.626, longitude: 6.024}
timezone: Europe/Berlin
pv:
  planes:
    West: {inverter_id: 1, tilt: 17, azimuth: 129, kwp: 6.435}
"""


@pytest.fixture
def job_env(base_env: pytest.MonkeyPatch, tmp_path: Path) -> pytest.MonkeyPatch:
    """The env a detect-faults CronJob runs with, minus a reachable database."""
    faults = tmp_path / "faults.yaml"
    faults.write_text(_FAULTS, encoding="utf-8")
    site = tmp_path / "site.yaml"
    site.write_text(_SITE, encoding="utf-8")
    base_env.setenv("MCP_FAULTS_FILE", str(faults))
    base_env.setenv("MCP_SITE_FILE", str(site))
    return base_env


def _root(spans: InMemorySpanExporter) -> ReadableSpan:
    (root,) = [s for s in spans.get_finished_spans() if s.parent is None]
    return root


def _children(spans: InMemorySpanExporter, parent: ReadableSpan) -> list[ReadableSpan]:
    assert parent.context is not None
    return [
        s
        for s in spans.get_finished_spans()
        if s.parent is not None and s.parent.span_id == parent.context.span_id
    ]


def test_no_per_detector_jobs_remain() -> None:
    assert set(SUBCOMMANDS) == {
        "forecast-solar",
        "forecast-weather",
        "energy-balance",
        "detect-faults",
    }


def test_every_subcommand_resolves_to_a_module() -> None:
    for subcommand in SUBCOMMANDS:
        module = importlib.import_module(f"lares_diagnostics_engine.{subcommand.replace('-', '_')}")
        assert callable(module.run)


def test_a_detect_faults_run_is_one_trace_with_a_span_per_fault(
    job_env: pytest.MonkeyPatch, spans: InMemorySpanExporter
) -> None:
    # The hour's evaluation is the root an explanation hangs under: the
    # trigger continues from the episode event a fault's span published.
    ran: list[str] = []

    def run_subjects(_store: Any, _publisher: Any, fault: Fault, _kind: Any, **_: Any) -> None:
        ran.append(fault.name)

    job_env.setattr(detect_faults, "run_subjects", run_subjects)

    assert main(["detect-faults"]) == 0

    root = _root(spans)
    assert root.name == "detect-faults"
    assert [s.name for s in _children(spans, root)] == [
        "fault channel_silence",
        "fault appliance_standby",
    ]
    assert ran == ["channel_silence", "appliance_standby"]


def test_a_failed_fault_marks_its_span_and_the_run_and_the_rest_still_runs(
    job_env: pytest.MonkeyPatch, spans: InMemorySpanExporter
) -> None:
    # The CronJob exits non-zero for one broken fault; the trace shows which.
    def run_subjects(_store: Any, _publisher: Any, fault: Fault, _kind: Any, **_: Any) -> None:
        if fault.name == "channel_silence":
            raise RuntimeError("ga_catalog unreachable")

    job_env.setattr(detect_faults, "run_subjects", run_subjects)

    assert main(["detect-faults"]) == 1

    root = _root(spans)
    assert root.status.status_code is StatusCode.ERROR
    statuses = {s.name: s.status.status_code for s in _children(spans, root)}
    assert statuses == {
        "fault channel_silence": StatusCode.ERROR,
        "fault appliance_standby": StatusCode.UNSET,
    }
