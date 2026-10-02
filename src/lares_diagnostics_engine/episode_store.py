"""Episode persistence: the thin edge between the pure pipeline and the
episodes tables.

Everything here is idempotent — evidence and events land with ON CONFLICT
DO NOTHING on their natural keys, and the row updates only ever raise
severity and peak and re-stamp the fingerprint, so a rerun after a
half-applied failure converges instead of duplicating.

That idempotence is also what says which episode events are *new*: `apply`
reports back the event rows the conflict clause let through, and a rerun
over the same window reports none. The runner turns exactly those into the
pointers on the bus, so an hourly recompute of a month-old episode
announces it once, on the run that first saw it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import psycopg
from psycopg.rows import DictRow

from .episodes import Entity, Episode, EpisodeEvent, EventKind, NotificationEvent
from .severity import CLEAR

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class OpenEpisodeRow:
    """The open episode the database holds for one fault and subject, with
    the fingerprint of the rule that last made it — None on a row older
    than the stamp, which nobody can attribute to a rule.
    """

    id: int
    subject: str
    severity: int
    fingerprint: str | None = None


def open_rows(conn: psycopg.Connection[DictRow], fault_name: str) -> list[OpenEpisodeRow]:
    rows = conn.execute(
        "SELECT id, subject, severity, fingerprint FROM episodes"
        " WHERE fault = %(fault)s AND ended_at IS NULL",
        {"fault": fault_name},
    ).fetchall()
    return [
        OpenEpisodeRow(
            id=r["id"], subject=r["subject"], severity=r["severity"], fingerprint=r["fingerprint"]
        )
        for r in rows
    ]


def processed_through(
    conn: psycopg.Connection[DictRow], fault_name: str
) -> dict[str, datetime]:
    """Per subject, how far this fault's recorded story reaches: the end of
    a closed episode, the last seen write of an open one. Bus-archive writes
    up to that point are already folded — the replay guard for the sliding
    read window.
    """
    rows = conn.execute(
        """
        SELECT subject, max(COALESCE(ended_at, last_seen_at)) AS through
        FROM episodes WHERE fault = %(fault)s GROUP BY subject
        """,
        {"fault": fault_name},
    ).fetchall()
    return {r["subject"]: r["through"] for r in rows}


def history_scores(conn: psycopg.Connection[DictRow], fault_name: str) -> list[float]:
    # Folded episodes are the imported detector era — a different score
    # scale, never this fault's own distribution.
    rows = conn.execute(
        """
        SELECT o.score FROM episode_observations o
        JOIN episodes e ON e.id = o.episode_id
        WHERE e.fault = %(fault)s AND NOT e.folded
        """,
        {"fault": fault_name},
    ).fetchall()
    return [r["score"] for r in rows]


def apply(
    conn: psycopg.Connection[DictRow],
    fault_name: str,
    inserts: Sequence[Episode],
    updates: Sequence[tuple[int, Episode]],
    orphan_closes: Sequence[tuple[int, datetime]],
    *,
    fingerprint: str,
    externally_delivered: bool = False,
) -> tuple[EpisodeEvent, ...]:
    """`fingerprint` names the rule this run measured by, stamped on every
    row it makes or re-makes — never on a row it merely closes — so a later
    rule can tell its own rows from this one's. `externally_delivered`
    marks episodes whose fault Basalte already delivered itself — the
    engine only records them, and nothing downstream may notify a second
    time.

    What comes back are the episode events this call actually wrote, each
    with the id of the row it hangs under — the new ones, and nothing a
    previous run already recorded.
    """
    recorded: list[EpisodeEvent] = []
    for episode in inserts:
        entity = entity_of(episode)
        inserted = conn.execute(
            """
            INSERT INTO episodes (fault, subject, entity_kind, entity_ref,
                                  started_at, last_seen_at,
                                  ended_at, severity, peak_score, fingerprint,
                                  externally_delivered)
            VALUES (%(fault)s, %(subject)s, %(entity_kind)s, %(entity_ref)s,
                    %(started_at)s, %(last_seen_at)s,
                    %(ended_at)s, %(severity)s, %(peak_score)s, %(fingerprint)s,
                    %(externally_delivered)s)
            RETURNING id
            """,
            {
                "fault": fault_name,
                "subject": episode.subject,
                "entity_kind": entity.kind if entity else None,
                "entity_ref": entity.ref if entity else None,
                "started_at": episode.started_at,
                "last_seen_at": episode.last_seen_at,
                "ended_at": episode.ended_at,
                "severity": episode.severity,
                "peak_score": episode.peak_score,
                "fingerprint": fingerprint,
                "externally_delivered": externally_delivered,
            },
        ).fetchone()
        if inserted is None:  # INSERT … RETURNING always yields the row
            raise RuntimeError(f"episode insert for {episode.subject} returned no id")
        _write_evidence(conn, inserted["id"], episode)
        recorded += _record_events(
            conn, inserted["id"], fault_name, episode.subject, episode.events
        )
    for episode_id, episode in updates:
        conn.execute(
            """
            UPDATE episodes
            SET last_seen_at = GREATEST(last_seen_at, %(last_seen_at)s),
                severity = GREATEST(severity, %(severity)s),
                peak_score = GREATEST(peak_score, %(peak_score)s),
                ended_at = %(ended_at)s,
                fingerprint = %(fingerprint)s
            WHERE id = %(id)s
            """,
            {
                "id": episode_id,
                "last_seen_at": episode.last_seen_at,
                "severity": episode.severity,
                "peak_score": episode.peak_score,
                "ended_at": episode.ended_at,
                "fingerprint": fingerprint,
            },
        )
        _write_evidence(conn, episode_id, episode)
        recorded += _record_events(
            conn, episode_id, fault_name, episode.subject, episode.events
        )
    for episode_id, ended_at in orphan_closes:
        # The subject comes back off the close: the plan carries only the
        # row id, and the pointer on the bus names the channel.
        closed = conn.execute(
            "UPDATE episodes SET ended_at = %(ended_at)s WHERE id = %(id)s RETURNING subject",
            {"id": episode_id, "ended_at": ended_at},
        ).fetchone()
        if closed is None:  # the row was read open in this same transaction
            raise RuntimeError(f"episode close for id {episode_id} matched no row")
        recorded += _record_events(
            conn,
            episode_id,
            fault_name,
            closed["subject"],
            (NotificationEvent(EventKind.ENDED, ended_at, CLEAR),),
        )
    return tuple(recorded)


_GA = re.compile(r"^\d+/\d+/\d+$")


def entity_of(episode: Episode) -> Entity | None:
    """What the subject names: the detector's word, or the channel a group
    address names. None where neither applies — a subject nothing claimed
    stays unresolved rather than guessed."""
    if episode.entity is not None:
        return episode.entity
    return Entity(kind="channel", ref=episode.subject) if _GA.match(episode.subject) else None


def _write_evidence(
    conn: psycopg.Connection[DictRow], episode_id: int, episode: Episode
) -> None:
    """The episode's per-bucket evidence rows under its own row."""
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO episode_observations (episode_id, time, score, severity, value)
            VALUES (%(id)s, %(time)s, %(score)s, %(severity)s, %(value)s)
            ON CONFLICT (episode_id, time) DO NOTHING
            """,
            [
                {
                    "id": episode_id,
                    "time": row.time,
                    "score": row.score,
                    "severity": row.severity,
                    "value": row.value,
                }
                for row in episode.evidence
            ],
        )


def _record_events(
    conn: psycopg.Connection[DictRow],
    episode_id: int,
    fault_name: str,
    subject: str,
    events: Sequence[NotificationEvent],
) -> list[EpisodeEvent]:
    """One statement per event rather than one for all three: the conflict
    clause is what tells a new event from one an earlier run already wrote,
    and only a per-row RETURNING says which was which.
    """
    recorded: list[EpisodeEvent] = []
    for event in events:
        row = conn.execute(
            """
            INSERT INTO episode_events (episode_id, kind, time, severity)
            VALUES (%(id)s, %(kind)s, %(time)s, %(severity)s)
            ON CONFLICT (episode_id, kind) DO NOTHING
            RETURNING episode_id
            """,
            {
                "id": episode_id,
                "kind": event.kind.value,
                "time": event.time,
                "severity": event.severity,
            },
        ).fetchone()
        if row is None:  # an earlier run already announced this one
            continue
        recorded.append(
            EpisodeEvent(
                episode_id=episode_id,
                fault=fault_name,
                subject=subject,
                kind=event.kind,
                time=event.time,
                severity=event.severity,
            )
        )
    return recorded
