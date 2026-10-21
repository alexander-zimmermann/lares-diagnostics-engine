"""Settings for the lares-diagnostics-engine jobs (forecast pulls +
energy balance). MCP_-prefixed env vars kept as the de-facto homelab
convention — same SealedSecrets and Kyverno-clone topology the MCP
server uses already inject these into the namespace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from urllib.parse import quote

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MCP_", env_file=".env", extra="ignore")

    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"

    db_host: str
    db_port: int = 5432
    db_name: str
    # Settings() validates db_username/password as required at construct
    # time even though most jobs only ever open the *write* connection —
    # this keeps the SealedSecret topology identical between the MCP
    # server and the jobs image.
    db_username: str = ""
    db_password: str = Field(default="", repr=False)
    db_username_file: str | None = None
    db_password_file: str | None = None

    db_write_username: str = ""
    db_write_password: str = Field(default="", repr=False)
    db_write_username_file: str | None = None
    db_write_password_file: str | None = None

    # Mounted from the lares ConfigMap: the fault list for detect-faults
    # (what counts as wrong) and the site file every job reads (what the
    # house is — location, timezone, PV planes).
    faults_file: str = "/etc/lares-diagnostics-engine/faults.yaml"
    site_file: str = "/etc/lares-diagnostics-engine/site.yaml"

    # NATS — jobs publish their results on `forecast.pv.*` / `energy.pv.*`.
    nats_servers: str | None = None
    nats_user: str | None = None
    nats_password: str = Field(default="", repr=False)
    nats_password_file: str | None = None
    nats_creds_file: str | None = None
    nats_nkey_seed_file: str | None = None

    # Tracing: the OTLP/HTTP collector base URL (http://host:4318), unset keeps
    # tracing off; the share of new traces kept, the platform's rendered ratio.
    tracing_endpoint: str | None = None
    tracing_sampling_ratio: float = Field(default=0.1, ge=0.0, le=1.0)

    # Forecast.Solar — PV-production forecast HTTPS API. Location and
    # planes come from the site file; the account's timezone must be the
    # site's, as the API returns naive local timestamps.
    forecast_solar_api_key: str = Field(default="", repr=False)
    forecast_solar_api_key_file: str | None = None
    forecast_solar_base_url: str = "https://api.forecast.solar"

    # Open-Meteo — keyless weather forecast API. We pin the DWD ICON model
    # (`icon_seamless`) which covers the homelab at ~2 km. Requested with
    # timezone=UTC for the site's location, so no local-tz conversion is
    # needed before insert.
    forecast_weather_base_url: str = "https://api.open-meteo.com/v1/forecast"
    forecast_weather_model: str = "icon_seamless"
    forecast_weather_forecast_hours: int = 48

    @model_validator(mode="after")
    def _resolve_db_secret_files(self) -> Settings:
        if self.db_username_file:
            self.db_username = Path(self.db_username_file).read_text(encoding="utf-8").strip()
        if self.db_password_file:
            self.db_password = Path(self.db_password_file).read_text(encoding="utf-8").strip()
        if not self.db_username:
            raise ValueError("MCP_DB_USERNAME or MCP_DB_USERNAME_FILE is required")
        if not self.db_password:
            raise ValueError("MCP_DB_PASSWORD or MCP_DB_PASSWORD_FILE is required")
        return self

    @model_validator(mode="after")
    def _resolve_optional_secret_files(self) -> Settings:
        if self.db_write_username_file:
            self.db_write_username = (
                Path(self.db_write_username_file).read_text(encoding="utf-8").strip()
            )
        if self.db_write_password_file:
            self.db_write_password = (
                Path(self.db_write_password_file).read_text(encoding="utf-8").strip()
            )
        if self.nats_password_file:
            self.nats_password = Path(self.nats_password_file).read_text(encoding="utf-8").strip()
        if self.forecast_solar_api_key_file:
            self.forecast_solar_api_key = (
                Path(self.forecast_solar_api_key_file).read_text(encoding="utf-8").strip()
            )
        return self

    @property
    def db_dsn(self) -> str:
        user = quote(self.db_username, safe="")
        password = quote(self.db_password, safe="")
        return f"postgresql://{user}:{password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def db_write_dsn(self) -> str:
        if not self.db_write_username or not self.db_write_password:
            raise ValueError(
                "MCP_DB_WRITE_USERNAME / MCP_DB_WRITE_PASSWORD (or *_FILE variants) are required"
            )
        user = quote(self.db_write_username, safe="")
        password = quote(self.db_write_password, safe="")
        return f"postgresql://{user}:{password}@{self.db_host}:{self.db_port}/{self.db_name}"


def load_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
