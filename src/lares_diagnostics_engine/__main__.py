from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

from nats_bridge_core import tracing
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import StatusCode

from .config import Settings, load_settings
from .logging_setup import configure_logging, get_logger

# Each subcommand maps to a sibling module `lares_diagnostics_engine.<module>`
# that exposes `def run(settings, argv) -> int`. The dispatcher fails
# with a clear message if a subcommand listed here has no module yet.
SUBCOMMANDS: tuple[str, ...] = (
    "forecast-solar",
    "forecast-weather",
    "energy-balance",
    "detect-faults",
)

_tracer = trace.get_tracer(__name__)


def _start_tracing(settings: Settings) -> TracerProvider | None:
    """The exporting provider, installed globally; without an endpoint the API stays a no-op."""
    if settings.tracing_endpoint is None:
        return None
    provider = tracing.build_provider(
        settings.tracing_endpoint, settings.tracing_sampling_ratio, "lares-diagnostics-engine"
    )
    trace.set_tracer_provider(provider)
    return provider


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lares-diagnostics-engine")
    parser.add_argument("subcommand", choices=SUBCOMMANDS)
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    settings = load_settings()
    configure_logging(settings.log_level, settings.log_format)
    log = get_logger(__name__)

    module_name = args.subcommand.replace("-", "_")
    try:
        module = importlib.import_module(f"lares_diagnostics_engine.{module_name}")
    except ImportError as exc:
        log.error("subcommand_not_implemented", subcommand=args.subcommand, error=str(exc))
        return 2

    # One trace per run.
    provider = _start_tracing(settings)
    try:
        with _tracer.start_as_current_span(args.subcommand) as span:
            try:
                rc: int = module.run(settings, args.rest)
            except Exception as exc:
                log.exception("subcommand_failed", subcommand=args.subcommand)
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR)
                return 1
            if rc != 0:
                span.set_status(StatusCode.ERROR)
            return rc
    finally:
        # The process ends with the run: export the queued spans now.
        if provider is not None:
            provider.shutdown()


if __name__ == "__main__":
    sys.exit(main())
