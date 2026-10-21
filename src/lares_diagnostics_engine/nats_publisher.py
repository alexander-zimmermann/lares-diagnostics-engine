from __future__ import annotations

import asyncio
import json
from typing import Any

import nats
from nats_bridge_core import tracing

from .config import Settings
from .episodes import EpisodeEvent
from .logging_setup import get_logger
from .severity import severity_level
from .slug import entity_slug

log = get_logger(__name__)


class PublishRejectedError(RuntimeError):
    """The server refused a publish — an nkey without the right to that
    subject, in every case seen so far.

    Core NATS acknowledges nothing, so a refusal is not a failed call: it
    arrives as an asynchronous `-ERR` that nats-py hands to the connection's
    error callback and otherwise shrugs off. Without this the engine logs the
    publish as done and the run passes while the message is dropped — the
    state it ran in for eight hours on 2026-09-26, between the release that
    began publishing `episode.*` and the nkey gaining the right to it. No
    episode event fell due in that window, so nothing was lost, which is
    precisely what nobody could have told from the logs.
    """


def _connect_opts(settings: Settings) -> dict[str, Any]:
    """Auth precedence: creds-file → NKey-seed-file → user/password →
    anonymous. Mirrors the knx-nats-bridge publisher so operators swap
    auth by changing the mounted secret, not the code."""
    if not settings.nats_servers:
        raise ValueError("MCP_NATS_SERVERS is required to publish to NATS")

    opts: dict[str, Any] = {
        "servers": [s.strip() for s in settings.nats_servers.split(",") if s.strip()],
        "name": "lares-diagnostics-engine",
        "max_reconnect_attempts": 3,
        "connect_timeout": 5,
    }
    if settings.nats_creds_file:
        opts["user_credentials"] = settings.nats_creds_file
    elif settings.nats_nkey_seed_file:
        opts["nkeys_seed"] = settings.nats_nkey_seed_file
    elif settings.nats_user and settings.nats_password:
        opts["user"] = settings.nats_user
        opts["password"] = settings.nats_password
    return opts


async def _publish_async(
    settings: Settings, subject: str, payload: dict[str, Any], headers: dict[str, str]
) -> None:
    """One message on one connection, and the refusal made observable.

    The flush is what turns an asynchronous `-ERR` into something this
    function can see: the server reads PUB and PING in the order they were
    written, so it answers the publish's error before the ping's pong, and
    nats-py delivers both to its reader in that same order. Anything the
    callback collected by the time the flush returns therefore belongs to
    this publish — which is exactly what the connection-per-publish shape
    buys, and why it is worth keeping.
    """
    refusals: list[Exception] = []

    async def collect(error: Exception) -> None:
        refusals.append(error)

    nc = await nats.connect(**_connect_opts(settings), error_cb=collect)
    try:
        body = json.dumps(payload, default=str).encode("utf-8")
        await nc.publish(subject, body, headers=headers or None)
        await nc.flush(timeout=5)
        if refusals:
            raise PublishRejectedError(f"{subject}: {refusals[0]}") from refusals[0]
        log.info("nats_publish", subject=subject, bytes=len(body))
    finally:
        await nc.close()


def publish(settings: Settings, subject: str, payload: dict[str, Any]) -> None:
    """Synchronous wrapper — a job invocation publishes a handful of
    messages (the moved subjects, and the episode events the run recorded),
    so we open/close per call rather than wiring an event loop.

    Each publish is a producer span under whatever the job has open, with
    its `traceparent` in the NATS headers for the consumer to continue.
    """
    with tracing.producer_span(subject):
        asyncio.run(_publish_async(settings, subject, payload, tracing.outbound_headers()))


def publish_anomaly(
    settings: Settings,
    uc: str,
    severity: str | None,
    payload: dict[str, Any],
    *,
    entity: str | None = None,
    firing: bool = True,
) -> None:
    """Publish one fault severity. Subject is **stable per routing target** —
    `fault.<uc>` for a 1:1 UC, `fault.<uc>.<entity>` for a grouped one —
    so the knx-nats-bridge writer-rules map exactly one rule per KNX-GA.

    The entity arrives raw (a GA, a room slug, a main group); this adapter
    owns the NATS dialect and slugs it once, for the subject token and the
    payload's `entity` alike.

    The severity travels as a numeric `severity_level` in the payload (the
    writer-rule reads `$.severity_level`); `firing=False` forces level 0
    (auto-clear → GA falls back to 0), with `severity=None` as the matching
    name-side value.
    """
    token = entity_slug(entity) if entity else None
    subject = f"fault.{uc}.{token}" if token else f"fault.{uc}"
    body = {
        "firing": firing,
        "uc": uc,
        "entity": token,
        "severity": severity,
        "severity_level": severity_level(severity) if firing else 0,
        **payload,
    }
    publish(settings, subject, body)


def publish_episode_event(settings: Settings, event: EpisodeEvent) -> None:
    """Publish one `EpisodeEvent` as the pointer it is, on `episode.<kind>`.
    `ended` goes out like the others so the stream is complete and a
    consumer filters rather than guessing what it missed.
    """
    publish(
        settings,
        f"episode.{event.kind.value}",
        {
            "episode_id": event.episode_id,
            "fault": event.fault,
            "subject": event.subject,
            "severity": event.severity,
            "kind": event.kind.value,
            "time": event.time.isoformat(),
        },
    )
