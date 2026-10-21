"""Tests for the jobs-extension settings fields (db_write, NATS)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lares_diagnostics_engine.config import Settings


def test_db_write_dsn_raises_when_unset(base_env: pytest.MonkeyPatch) -> None:
    s = Settings()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="MCP_DB_WRITE_USERNAME"):
        _ = s.db_write_dsn


def test_db_write_dsn_uses_inline_env(base_env: pytest.MonkeyPatch) -> None:
    base_env.setenv("MCP_DB_WRITE_USERNAME", "rw")
    base_env.setenv("MCP_DB_WRITE_PASSWORD", "rw-pw")
    s = Settings()  # type: ignore[call-arg]
    assert s.db_write_dsn == "postgresql://rw:rw-pw@localhost:5432/test"


def test_db_write_dsn_reads_file(base_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    user_file = tmp_path / "user"
    pw_file = tmp_path / "pw"
    user_file.write_text("rw\n")
    pw_file.write_text("rw-pw\n")
    base_env.setenv("MCP_DB_WRITE_USERNAME_FILE", str(user_file))
    base_env.setenv("MCP_DB_WRITE_PASSWORD_FILE", str(pw_file))
    s = Settings()  # type: ignore[call-arg]
    assert s.db_write_dsn == "postgresql://rw:rw-pw@localhost:5432/test"


def test_db_dsns_url_encode_password(base_env: pytest.MonkeyPatch) -> None:
    # openssl rand -base64 routinely produces passwords containing '/', '+',
    # '=', '@' — all of which break psycopg's URI parser when not encoded.
    base_env.setenv("MCP_DB_USERNAME", "ro")
    base_env.setenv("MCP_DB_PASSWORD", "abc/def+gh=ij@kl")
    base_env.setenv("MCP_DB_WRITE_USERNAME", "rw")
    base_env.setenv("MCP_DB_WRITE_PASSWORD", "x/y+z=q@r")
    s = Settings()  # type: ignore[call-arg]
    assert s.db_dsn == "postgresql://ro:abc%2Fdef%2Bgh%3Dij%40kl@localhost:5432/test"
    assert s.db_write_dsn == "postgresql://rw:x%2Fy%2Bz%3Dq%40r@localhost:5432/test"


def test_optional_fields_have_sane_defaults(base_env: pytest.MonkeyPatch) -> None:
    s = Settings()  # type: ignore[call-arg]
    assert s.nats_servers is None
    assert s.site_file == "/etc/lares-diagnostics-engine/site.yaml"


def test_nats_secret_file_resolves(base_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pw = tmp_path / "nats-pw"
    pw.write_text("super-secret\n")
    base_env.setenv("MCP_NATS_PASSWORD_FILE", str(pw))
    s = Settings()  # type: ignore[call-arg]
    assert s.nats_password == "super-secret"
