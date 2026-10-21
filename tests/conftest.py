"""Shared fixtures: a recording tracer provider for the whole session, and
the minimum env a job's settings validate with."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# One recording provider for the whole session: the global can only be set once.
_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    """The spans finished in this test, every one sampled."""
    _exporter.clear()
    yield _exporter
    _exporter.clear()


@pytest.fixture
def base_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Minimum required env so Settings() validates; tests then override as
    needed. monkeypatch restores the original environment on teardown."""
    for k in list(os.environ):
        if k.startswith("MCP_"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("MCP_DB_HOST", "localhost")
    monkeypatch.setenv("MCP_DB_NAME", "test")
    monkeypatch.setenv("MCP_DB_USERNAME", "ro")
    monkeypatch.setenv("MCP_DB_PASSWORD", "ro-pw")
    return monkeypatch
