"""The NATS adapter: one publish per fault, on the subject the
knx-nats-bridge writer rules pin (`tests/test_slug` locks the entity token),
and one pointer per episode event on `episode.<kind>`, each carrying the
trace it was published in.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind

from lares_diagnostics_engine import nats_publisher
from lares_diagnostics_engine.config import Settings
from lares_diagnostics_engine.episodes import EpisodeEvent, EventKind
from lares_diagnostics_engine.nats_publisher import (
    PublishRejectedError,
    publish,
    publish_anomaly,
    publish_episode_event,
)

_TRACER = trace.get_tracer(__name__)


def _settings() -> Settings:
    return Settings(
        db_host="localhost",
        db_name="x",
        db_username="x",
        db_password="x",  # noqa: S106 — test stub
        nats_servers="nats://localhost:4222",
    )


def test_publish_anomaly_slugs_the_raw_entity_once() -> None:
    # Kinds hand the entity over raw (a GA, a room label); this adapter is
    # the one place that speaks NATS dialect — subject token and payload
    # `entity` carry the same slug.
    settings = _settings()
    with patch.object(nats_publisher, "publish") as pub:
        publish_anomaly(settings, "appliance_runtime", "warning", {}, entity="2/1/197")
    (call,) = pub.call_args_list
    assert call.args[1] == "fault.appliance_runtime.2-1-197"
    assert call.args[2]["entity"] == "2-1-197"


def test_an_episode_event_goes_out_as_a_pointer_on_its_own_subject() -> None:
    # The kind names the subject, the payload names the episode: whoever
    # explains it fetches the sentence and the evidence by that id.
    event = EpisodeEvent(
        episode_id=15510,
        fault="appliance_runtime",
        subject="2/1/197",
        kind=EventKind.ESCALATED,
        time=datetime(2026, 9, 25, 14, 20, tzinfo=UTC),
        severity=3,
    )
    with patch.object(nats_publisher, "publish") as pub:
        publish_episode_event(_settings(), event)
    (call,) = pub.call_args_list
    assert call.args[1] == "episode.escalated"
    assert call.args[2] == {
        "episode_id": 15510,
        "fault": "appliance_runtime",
        "subject": "2/1/197",
        "severity": 3,
        "kind": "escalated",
        "time": "2026-09-25T14:20:00+00:00",
    }


def test_the_subject_is_the_episode_row_not_a_slugged_address_token() -> None:
    # The fault severities slug their entity because a KNX writer rule is
    # pinned to the token. An episode event addresses no rule: its subject
    # is the episode's own column, so a consumer can look the row up by it.
    event = EpisodeEvent(
        episode_id=42,
        fault="room_deviation",
        subject="EG.Flur",
        kind=EventKind.APPEARED,
        time=datetime(2026, 9, 25, 14, 20, tzinfo=UTC),
        severity=2,
    )
    with patch.object(nats_publisher, "publish") as pub:
        publish_episode_event(_settings(), event)
    (call,) = pub.call_args_list
    assert call.args[2]["subject"] == "EG.Flur"


class _FakeClient:
    """A NATS connection that answers a flush, and hands over an async error
    while doing so when the server refused what was published."""

    def __init__(self, refusal: Exception | None) -> None:
        self._refusal = refusal
        self.error_cb: Any = None
        self.published: list[tuple[str, bytes, dict[str, str] | None]] = []
        self.flushed = False
        self.closed = False

    async def publish(
        self, subject: str, body: bytes, headers: dict[str, str] | None = None
    ) -> None:
        self.published.append((subject, body, headers))

    async def flush(self, **_kwargs: Any) -> None:
        # The server answers the PUB's error before the PING's pong, so a
        # refusal is in hand by the time the flush returns.
        if self._refusal is not None:
            await self.error_cb(self._refusal)
        self.flushed = True

    async def close(self) -> None:
        self.closed = True


def _with_bus(refusal: Exception | None) -> tuple[Any, _FakeClient]:
    client = _FakeClient(refusal)

    async def connect(**opts: Any) -> _FakeClient:
        client.error_cb = opts["error_cb"]
        return client

    return connect, client


def test_a_refused_publish_raises_instead_of_reporting_success() -> None:
    # Core NATS acknowledges nothing, so a publish the server refuses — an
    # nkey without the right to that subject — comes back as an asynchronous
    # error, never as a failed call. Left unchecked the engine logged
    # `nats_publish` as if it had worked (it did, for eight runs on
    # 2026-09-26). The flush's round trip is what makes the refusal visible.
    connect, client = _with_bus(
        PermissionError('nats: Permissions Violation for Publication to "episode.appeared"')
    )
    with (
        patch.object(nats_publisher.nats, "connect", connect),
        pytest.raises(PublishRejectedError, match="episode.appeared"),
    ):
        publish(_settings(), "episode.appeared", {"episode_id": 1})
    assert client.closed is True  # the connection is not leaked by the raise


def test_an_accepted_publish_goes_out_and_the_connection_closes() -> None:
    connect, client = _with_bus(None)
    with patch.object(nats_publisher.nats, "connect", connect):
        publish(_settings(), "episode.appeared", {"episode_id": 15510})
    ((subject, body, _headers),) = client.published
    assert subject == "episode.appeared"
    assert json.loads(body) == {"episode_id": 15510}
    assert client.flushed is True
    assert client.closed is True


def _producer(spans: InMemorySpanExporter) -> ReadableSpan:
    (span,) = [s for s in spans.get_finished_spans() if s.kind is SpanKind.PRODUCER]
    return span


def _headers(client: _FakeClient) -> dict[str, str] | None:
    ((_subject, _body, headers),) = client.published
    return headers


def _traceparent(span: ReadableSpan) -> str:
    """The W3C header naming `span` as the parent of whatever handles the message."""
    context = span.context
    assert context is not None
    return f"00-{context.trace_id:032x}-{context.span_id:016x}-{context.trace_flags:02x}"


def test_an_episode_event_carries_the_trace_of_the_fault_that_opened_it(
    spans: InMemorySpanExporter,
) -> None:
    # The trigger continues the run from these headers, so the explanation
    # hangs under the hour's detect-faults trace and the fault that opened it.
    event = EpisodeEvent(
        episode_id=15510,
        fault="appliance_runtime",
        subject="2/1/197",
        kind=EventKind.ESCALATED,
        time=datetime(2026, 9, 25, 14, 20, tzinfo=UTC),
        severity=3,
    )
    connect, client = _with_bus(None)
    with (
        patch.object(nats_publisher.nats, "connect", connect),
        _TRACER.start_as_current_span("fault appliance_runtime") as fault_span,
    ):
        publish_episode_event(_settings(), event)
    send_span = _producer(spans)
    assert send_span.name == "send episode.escalated"
    assert send_span.parent is not None
    assert send_span.parent.span_id == fault_span.get_span_context().span_id
    assert _headers(client) == {"traceparent": _traceparent(send_span)}


def test_a_fault_severity_carries_the_trace_it_was_published_in(
    spans: InMemorySpanExporter,
) -> None:
    connect, client = _with_bus(None)
    with (
        patch.object(nats_publisher.nats, "connect", connect),
        _TRACER.start_as_current_span("fault appliance_runtime") as fault_span,
    ):
        publish_anomaly(_settings(), "appliance_runtime", "warning", {}, entity="2/1/197")
    send_span = _producer(spans)
    assert send_span.name == "send fault.appliance_runtime.2-1-197"
    assert send_span.context is not None
    assert send_span.context.trace_id == fault_span.get_span_context().trace_id
    assert _headers(client) == {"traceparent": _traceparent(send_span)}
