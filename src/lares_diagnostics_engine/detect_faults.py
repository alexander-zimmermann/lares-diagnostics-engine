"""Detect-faults job: run the declared fault list end to end.

For each schedulable fault the runner resolves the scope where the catalog
lives (`ga_catalog` in TSDB — never a hand-written address list), measures,
folds the observations into episodes behind the pure pipeline seam, and
reconciles the result with the episodes the database already holds. What
leaves the engine is a severity 0–3 per subject on `fault.<fault>[.<entity>]`;
the knx-nats-bridge writer rules carry it to the declared diagnosis address,
where Basalte owns the text.

Every kind runs in one shape: a `runner.Kind` declares how its series is
measured and how its payload is shaped, `kinds.kind_for` says which one a
fault declares, and the runner module owns the rest — the window, the fold,
the reconciliation, the log record, the dry run and the publish-then-write
tail, behind its injected store and publisher ends. This module is the job:
it loads the fault list and the site, and runs the list fault by fault.

Time is the aggregate's frontier throughout — episodes also *end* in
frontier time, so a stalled refresh (or a dead bridge) freezes the picture
instead of clearing every open episode with a severity 0 nobody earned.

Publishes go out before the database writes: a failed run then repeats the
same publish (same value, Basalte's change detector ignores it) instead of
losing it behind an already-updated database.

State is recomputed from history on every run — the only stored artifacts
are the episodes themselves. `--dry-run` computes and logs everything and
touches neither the database nor NATS.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .config import Settings
from .faults import Fault, FaultList
from .kinds import kind_for
from .logging_setup import get_logger
from .runner import DbStore, NatsPublisher, run_subjects
from .site import Site

log = get_logger(__name__)


def run(settings: Settings, argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="lares-diagnostics-engine detect-faults")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    fault_list = FaultList.load(Path(settings.faults_file))
    site = Site.load(Path(settings.site_file))
    for fault in fault_list:
        if fault.dormant is not None:
            log.info("fault_dormant", fault=fault.name, active_when=fault.dormant.active_when)
    failed: list[str] = []
    for fault in fault_list.schedulable():
        # One fault must not take the others down with it: a catalog change
        # that leaves a device undeclared fails its own fault loudly, while
        # the rest of the list — the volume watchdog last of all — still
        # runs. The job still exits non-zero, so the CronJob shows it.
        try:
            _run_fault(settings, fault, site, dry_run=args.dry_run)
        except Exception:
            log.exception("fault_run_failed", fault=fault.name, kind=str(fault.kind))
            failed.append(fault.name)
    if failed:
        log.error("detect_faults_incomplete", failed=failed)
        return 1
    return 0


def _run_fault(settings: Settings, fault: Fault, site: Site, *, dry_run: bool) -> None:
    kind = kind_for(fault, site)
    if kind is None:
        # Arrives with its own ticket; a declared fault must not fail
        # the ones already running.
        log.warning("fault_kind_not_implemented", fault=fault.name, kind=str(fault.kind))
        return
    run_subjects(DbStore(settings), NatsPublisher(settings), fault, kind, dry_run=dry_run)
