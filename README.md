# lares-diagnostics-engine

_Formerly `iot-insights-engine`._

TSDB-backed background jobs for the homelab: external forecast pulls,
the daily energy balance, and the declared fault list — history faults
measured over the hourly aggregates and delivered as a severity 0–3 on a
KNX group address.
Companion to [lares-mcp-bridge](https://github.com/alexander-zimmermann/lares-mcp-bridge)
(the MCP server, read-only — where verdicts are given) and
[knx-nats-bridge](https://github.com/alexander-zimmermann/knx-nats-bridge)
(KNX ↔ NATS, owns the GA catalog and the writer rules).

The vocabulary — fault, kind, subject, episode, severity, frontier,
delivery — is defined in [CONTEXT.md](CONTEXT.md); module docstrings,
log records and tests use those names.

## Architecture

```
TSDB (hourly aggregates, ga_catalog) ─┐
faults.yaml + site.yaml (lares)       ├─► lares-diagnostics-engine
api.forecast.solar / api.open-meteo   ┘     │
                                            ├─► TSDB (mcp_forecasts, episodes)
                                            ▼
                             NATS (forecast.pv.*, energy.pv.*, fault.*)
                                            │
                                            ▼
                             knx-nats-bridge (writer rules) ─► KNX GA ─► Basalte
                                             3 = push · 1–2 = indicator · 0 = clear · e-mail for all
```

Downstream of the episodes: lares-mcp-bridge lists them and takes the
binary verdict ("real" / "nonsense") per episode; the Grafana
`knx-episodes` dashboard replaced the weekly mail.

## Subcommands

Run via the single entrypoint:

```
lares-diagnostics-engine <subcommand>
```

| Subcommand         | Schedule (Kubernetes CronJob) | What it does |
|--------------------|-------------------------------|--------------|
| `forecast-solar`   | `15 * * * *`                  | Pull PV forecast → `mcp_forecasts`, publish `forecast.pv.*` |
| `forecast-weather` | `20 * * * *`                  | Pull Open-Meteo (ICON) forecast → `mcp_forecasts` |
| `energy-balance`   | `*/15 * * * *`                | Today's kWh counters → `energy.pv.*` |
| `detect-faults`    | `20 * * * *`                  | Run the fault list: resolve scope, measure, fold into episodes, reconcile, publish `fault.*` |

`detect-faults --dry-run` computes and logs everything and touches
neither the database nor NATS. One failing fault does not take the
others down; the job still exits non-zero so the CronJob shows it.

## Fault detection

### The fault file

Faults are declared, not coded. The list lives in lares beside the GA
catalog and the writer rules
(`kubernetes/applications/lares-diagnostics-engine/base/config/faults.yaml`)
and is mounted into the CronJob at `MCP_FAULTS_FILE`. Every entry
carries:

- `sentence` and `unit` — the fault as one readable sentence. If the
  sentence cannot be written, it is not a detector.
- `kind` — one of the measurement kinds below; `drift` also names its
  `signal`, `deviation` may name an `expectation`.
- `parameters` — in the channel's own unit (mA, %, h, × the usual pause).
- `scope` — a catalog query (`dpt`, `include`, `exclude`; the name
  patterns in SQL `LIKE` form), resolved against `ga_catalog` on every
  run. Never a hand-written address list. Channels that never sent are
  dropped where the scope resolves, logged, not reported.
- `target` — where the severity goes: one `ga`, or `per_main_group`,
  `per_device`, `per_room`. The addresses behind it are the bridge's
  writer rules. A per-device or per-room target may add `name`, the
  address's catalog-name template with `{entity}` standing for the
  entity as the fault's device or room map names it
  (`Raumklima.{entity}.FBH.Aktiv-Anomalie`); the engine validates it, the
  lares generator renders it into those rules.
- `dormant` — optional `reason` and `active_when`; the fault loads and
  validates but does not schedule.

The loader ([faults.py](src/lares_diagnostics_engine/faults.py)) validates
the file against the bundled JSON Schema and freezes it into
dataclasses; a missing sentence, unit or parameter fails at load, naming
the fault and the field. Check an edit before shipping with
`task diagnostics:validate-faults` in lares. Tuning a threshold is a
one-line PR there, not an engine release.

### The site file

The fault list says what counts as wrong; `site.yaml`, beside it in the
same ConfigMap and mounted at `MCP_SITE_FILE`, says what the house is:
`location`, `timezone`, and `pv.planes` keyed by plane name (`West`,
`Ost`) with the `inverter_id` each plane feeds, its `tilt` and `azimuth`
(Open-Meteo convention: 0 south, negative east, positive west) and its
`kwp`. Every job reads it: `forecast-solar` takes the location and the
planes (and expects the account's timezone to be the site's),
`forecast-weather` the location, `energy-balance` the timezone whose
midnight bounds "today", and the daily-yield shape the plant — which
inverters count, and the most each counter can rise in an hour — so a
counter that re-bases or reports a 0 is told from production without a
fault parameter. Nothing about the house lives in a job's environment.
The loader ([site.py](src/lares_diagnostics_engine/site.py)) validates the
file the same way the fault list is validated.

### Measurement kinds

| Kind         | Measures | Reports per |
|--------------|----------|-------------|
| `silence`    | A channel that used to send has been quiet longer than N× its own usual pause (a quantile of its own gaps). A channel new to the window is unproven — unmeasured, not alive — until it has shown enough gaps. | main group |
| `constancy`  | A channel keeps sending but has delivered the exact same value for longer than allowed: the producer works, the register behind it is dead. | main group |
| `duration`   | A device draws current for longer than its declared limit. | device |
| `drift`      | CUSUM against a healthy reference pinned in the entry, never derived from history. `signal` picks the series: `standby` (mA), `duty_cycle` (%), `recovery` (%, the one that walks downward). | device, or one GA |
| `deviation`  | A value sits too far under its reference: a room against its setpoint while a gate holds, or the plant's daily yield — the plausible rises of each declared inverter's counter between hourly closes, summed — against a named `expectation` (`forecast_solar`). | room, or one GA |
| `volume`     | More than N episodes in seven days, over the engine's own episode stream. Declared last, so the count includes what this run just wrote. | one GA |
| `external`   | Basalte detects and delivers itself; the engine reads the severity writes back off the bus archive and only records. Nothing is published. | — |

There is no threshold kind: raw-value limits are Basalte's job.

### Episodes and severity

Repeated observations fold into one episode per fault and subject
(`episodes.fold_observations`, a pure function). An episode keeps its
per-bucket observations as evidence and notifies at most three times:
appeared, escalated, ended. It ends after a few quiet runs, never the
first, so a flickering fault stays one incident.

Severity is 0–3 (clear, info, warning, critical). Within an episode it
is the quantile of the fault's own score distribution, promoted one step
by duration; there is no global ladder. A stored severity is never
lowered.

Episodes (`episodes`, `episode_observations`, `episode_events` in TSDB,
written through the rw role) are the only stored state. Everything else
is recomputed from the last 30 days on every run, so a redeploy cannot
corrupt or lose it. Time is the aggregate's frontier — its newest
bucket, not the wall clock — so a stalled refresh freezes the picture
instead of clearing every open episode. A subject the run cannot measure
up to the frontier is dataless and its episode stays open. Every episode
carries the fingerprint of the rule that last made it — the fault's kind
and parameters — so a rule change does not strand rows on channels it
declines to judge: an open row an earlier rule left on a channel the
current rule holds unproven closes, and its group address is published
anew.

### Delivery

One publish per moved subject on `fault.<fault>[.<entity>]` with a
numeric `severity_level`. The knx-nats-bridge writer rules carry it to
the group address; Basalte owns the text and the channel: 3 pushes,
1–2 shows an indicator, 0 clears, and every situation gets an e-mail.
Publishes go out before the database write, so a failed run repeats the
same publish instead of losing it.

Not in this repo: verdicts (lares-mcp-bridge `set_episode_verdict` →
`episode_verdicts`; collected and shown, never acted on), the Grafana
dashboard, and the Basalte Studio faults — those appear here only as
`external` entries.

## Library surface

Two entry points are imported from this package at its deployed git tag, the
way the sidecars import `nats-bridge-core`. They and the back-test's two
result types (`Backtest`, `BacktestEpisode`) are what the package root
exports; everything else is internal.

| Import | Who uses it | What it does |
|--------|-------------|--------------|
| `entity_slug` | the lares generator (`task knx:create-ga-mappings`) | The subject's last token as the publish dialect spells it (`2/1/27` → `2-1-27`, `EG.Flur` → `eg-flur`), so the writer rules are generated with the very function that publishes |
| `backtest_fault` | lares-mcp-bridge, as the `backtest_fault` tool | What one candidate fault would have found in the last N weeks |

### Back-testing a candidate

```python
from lares_diagnostics_engine import backtest_fault

result = backtest_fault(read_connection, entry, weeks=8)
```

`entry` is one entry of the fault file in that schema — a candidate nobody
has written down yet, or a copy of a declared fault with a threshold moved.
Out comes the episodes that rule would have produced (subject, dates, tier,
peak score in the fault's own unit, observation count), the window they were
found in, and the measurement's own record — so no episodes reads as a
verdict rather than as silence.

It is a library call, not a job: no CronJob, no subcommand, no endpoint. The
candidate runs through the same schema, the same kind registry, the same
measurement and the same fold `detect-faults` runs, and it is read-only by
construction — one read connection and no store or publisher end, so nothing
in it can write a row or reach the bus.

Anything unmeasurable is an error naming what to fix, never an empty result.
The window is bounded by what the measurement can read, and each kind
declares its own: a year of aggregates for most, 90 days for the plant's
daily yield (as long as its stored forecast lives), 30 days for the
duty-cycle signal (the one that reads the bus archive itself). Severities are
the ones a rule without history produces, so what a candidate is judged by is
the peak score — [backtest.py](src/lares_diagnostics_engine/backtest.py) says
why, and what the single-pass window does and does not promise.

## Configuration

All `MCP_*` env vars (kept for compatibility with the existing
SealedSecret + Kyverno-clone topology shared with lares-mcp-bridge). Every
job reads the site file at `MCP_SITE_FILE`; `detect-faults` additionally
needs the write credentials (`MCP_DB_WRITE_*`, episodes only),
`MCP_FAULTS_FILE` and a NATS identity for `fault.*`. See
[config.py](src/lares_diagnostics_engine/config.py) for the full list.

## Local dev

```
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run mypy src
```

Tests follow the rebuild's seams: the fault loader (`test_faults`), the
site loader (`test_site`), each
measurement kind against invented fixtures (`test_silence`,
`test_constancy`, `test_duration`, `test_drift`, `test_deviation`,
`test_volume`, `test_external`), the episode pipeline (`test_episodes`,
`test_severity`, `test_reconcile`), and the runner lifecycle through fake
store and publisher ends (`test_runner`). The back-test is driven through
its own boundary — one candidate entry and a fake connection that hands each
measurement its rows and refuses anything but a read (`test_backtest`).
Delivery is not tested here — the writer rules are tested in the bridge
repo.
