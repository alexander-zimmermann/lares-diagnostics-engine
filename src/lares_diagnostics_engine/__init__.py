"""The package's import-stable surface, imported at the deployed tag: the
entity slug the lares generator writes the writer rules with, and the
back-test `lares-mcp-bridge` exposes as a tool. The fault-list loader stays
at `lares_diagnostics_engine.faults`."""

from .backtest import Backtest, BacktestEpisode, backtest_fault
from .slug import entity_slug

__all__ = ["Backtest", "BacktestEpisode", "backtest_fault", "entity_slug"]
