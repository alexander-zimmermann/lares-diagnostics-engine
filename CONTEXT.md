# Domain model

The vocabulary `lares-diagnostics-engine` (formerly `iot-insights-engine`)
is written in. Terms here are the names used in module docstrings, log
records and tests — a new module should reach for one of these before
inventing its own.

## Fault

One declared thing that can be wrong with the house, written as a
sentence with a unit in the fault file (`faults.yaml`, kept in the lares
repo beside the GA catalog and the writer rules). A fault names what it
measures (its **scope**), how (its **kind**), the parameters it measures
by, and where the verdict is **delivered**. The loader validates the file
against a bundled JSON Schema and freezes it into dataclasses, so a bad
edit fails at load, never at runtime in the cluster.

A **dormant** fault loads fully but is excluded from the schedule: it
declares why it cannot run yet and the observable condition under which it
starts to.

A fault also declares whether it is **explained**: whether its episode
events go on the bus for an agent to pick up. On by default for every kind
the engine measures itself, off for `external` — Basalte detected that one
and its own sentence already is the explanation — and overridable per
fault with one line.

## Site

What the house is, declared in `site.yaml` beside the fault list: where
it stands, its timezone, and its PV **planes** — each keyed by name
(`West`, `Ost`) with the inverter it feeds, its orientation and its peak
power in kWp. The fault list says what counts as wrong; the site says
what is there, so a fact about the plant is never a fault parameter. The
daily-yield shape reads the plant off it: an inverter's lifetime counter
at each hourly **close** is walked step by step, a **rise** of at most
the plane's kWp times the hours the step spans is credited to the day,
and a **drop** (the counter re-based) or a rise past that **bound** (a
bogus reading) is declined — the run record counts the declined steps
per inverter as `ignored_steps`. The forecast jobs read the location,
the planes' orientation and the timezone off it as well; nothing about
the house lives in a job's environment.

## Fingerprint

The rule a fault measures by — its kind, with the signal or expectation
that picks the kind's shape, and its parameters, read as numbers —
condensed to one string. A kind whose code decides what it accuses folds
a **revision** in, bumped when that code changes without a parameter
moving (silence's unproven guard did). Every episode carries the
fingerprint of the rule that last made it, so a run can tell its own rows
from an earlier rule's leftovers. A row older than the stamp carries none
and is held like any other. Per-subject declarations (device limits,
references, rooms) are not part of it: they name who is measured, not how.

## Kind

How a fault is measured. Every fault has exactly one:

- **silence** — a channel that used to send has gone quiet, longer than a
  multiple of its own normal pause. A channel that first appeared inside
  the window is **unproven** until it has shown as many gaps as the
  declared quantile needs, and is unmeasured rather than alive until then;
- **constancy** — a channel keeps sending and keeps delivering the same
  value: the producer works, the register behind it is dead. What silence
  drops as unmeasurable is what this kind reports;
- **duration** — a device draws current for longer than its declared
  limit;
- **drift** — something a device does sits persistently above (or, for
  recovery, below) the healthy level declared for it, walked as a CUSUM
  with a pinned reference. Which series it walks is the fault's declared
  **signal**: `standby`, `duty_cycle` or `recovery`;
- **deviation** — a value sits too far under its declared reference. Two
  shapes, told apart by what the fault declares that reference as: rooms
  against a channel of the house while a gate condition holds, or the
  plant's daily yield against a model named in the entry — its declared
  **expectation**, swappable there without touching the comparison;
- **volume** — more than N incidents in a week, measured over the engine's
  own episode stream;
- **external** — Basalte detects and delivers the fault itself; the engine
  reads its severity writes back off the bus archive and only records.

`runner.Kind` is the code-level declaration of one: what it measures, how
its payload is shaped, how far back the data it reads reaches, and — where
the defaults do not fit — how it folds and plans.

## Subject

The thing a fault's verdict is about: a channel, a device, a room, an
exchanger, the PV plant, or the house itself. Episodes are stored per fault and subject.

## Scope

A fault's channel query, resolved where the catalog lives (`ga_catalog` in
TSDB) — never a hand-written address list. What it resolves to is a
**channel**: a group address, its catalog name and its DPT.

## Observation

One per-bucket measurement of a firing fault on one subject, carrying a
**score**: the fault's own magnitude in its declared unit, compared only
against that fault's own history, never across faults.

## Episode

One incident: when it started, when it was last seen, how bad it got, with
the per-bucket evidence rows that formed it and at most three
**notification events** (appearing, escalating, ending). Episodes are the
only stored artifact — everything else is recomputed from history on every
run, so a redeploy cannot corrupt or lose state.

## Episode event

One of an episode's three notification events — appearing, escalating,
ending — as it reached a row, which is what gives it an episode id and
says it is new. Each one the write records leaves the engine as a message
on `episode.<kind>`, for every fault declared explained: `episode_id`,
`fault`, `subject` (the episode's own subject column — the channel, device
or room the fault was measured on), `severity`, `kind` and `time`.
`ended` goes out like the others so the stream is complete and a consumer
filters rather than guessing what it missed.

The payload is a **pointer**, never a report: the sentence, the parameters
and the evidence are fetched from the episode the id names, the one place
they cannot go stale. An event an earlier run already recorded is never
announced twice, so an hourly recompute of a month-old episode stays
silent.

## Fold

Turning repeated observations into episodes: the pure seam
`episodes.fold_observations`. External is the one kind that folds
something else (severity writes).

## Severity

A tier 0–3 (clear, info, warning, critical) — the delivery contract with
Basalte. Within an episode the tier is the quantile of the fault's own
score distribution, promoted one step by duration; there is no global
ladder. A stored severity is never lowered.

## Reconciliation

The computed episodes against the open episode rows the database already
holds: what to insert, what to update, which orphaned rows to close, and
which subjects moved to a new severity. A subject is **dataless** when
this run cannot tell a recovery from a blind spot — its measurement does
not reach the frontier — and its episode then stays open instead of
self-clearing. A row is **stranded** when a rule other than this run's
left it on a subject this run declines to judge — for silence, an
unproven channel: nothing the current rule does would open it, so it
closes at the frontier though its subject is dataless, and its group is
published anew — clear, if the row was the group's last. The same row
under the current rule is a silence sliding out of view and stays open.

## Frontier

The "now" a run is measured against: the newest bucket of the aggregate it
reads, not the wall clock — for a kind that measures whole days, the newest
*complete* one. The continuous aggregate materializes with an
end offset, so the newest visible bucket lags real time for every channel
at once; the frontier cancels that lag. Episodes also *end* in frontier
time, so a stalled refresh freezes the picture instead of clearing every
open episode with a severity nobody earned. External faults are the
exception — the bus archive has no materialization lag, so they declare
wall-clock time.

## Delivery

How a verdict leaves the engine: one publish per moved subject on
`fault.<fault>[.<entity>]`, carrying a numeric `severity_level` the
knx-nats-bridge writer rules route to a KNX group address, where Basalte
owns the text. The subject shape is pinned by those writer rules. A fault
declares its **target** — one address, one per main group, one per device,
one per room — and the kind's declaration says which form it expects. A
per-device or per-room target may carry a **name template**, the target
address's catalog name with `{entity}` for the entity as the fault's own
map names it; the engine only validates it, the lares generator renders it
into the writer rules. The **entity slug** is the subject's last token —
`2/1/27` to `2-1-27`, `EG.Flur` to `eg-flur` — exported at the package root
so those rules are generated with the very function that publishes. The
two channel-scoped kinds, silence and constancy, share the per-main-group
form but not the addresses behind it — one per fault per group, because two
writers on one address overwrite each other's clears. The address says
which fault and roughly where, the payload names the channels exactly.

## Runner

The run lifecycle every kind shares: window off the frontier, fold in the
kind's own cadence, reconciliation, run record, dry-run gating, and the
publish-before-write tail. It touches the world through two injected ends, the **store** and
the **publisher**, so those guarantees are testable through fakes.
Publishes go out before the database writes: a failed run then repeats the
same publish instead of losing it behind an already-updated database. The
episode events are the one thing published *after* it, because the row is
what gives an event its id and what says it is new.

## Back-test

What a **candidate** — a fault entry nobody has written down yet, or a
declared one with a threshold moved — would have found in the last N weeks:
its episodes, with the subject, the dates, the tier and the peak score in
the fault's own unit. It is the number that makes a proposed fault
reviewable, and zero and fifty are both grounds to say no.

The candidate runs through the same schema, the same kind, the same
measurement and the same fold a scheduled run does, over a window of whole
weeks off the frontier — but nothing else: no reconciliation, no delivery,
no write, and no stored episode read for its own name, so a candidate
sharing a declared fault's name cannot inherit its rows. A candidate has no
history to be rare against, so its severities are the ones any rule without
history produces; what it is judged by is the peak score. It is a library
entry point the bridge exposes as a tool, never a job.
