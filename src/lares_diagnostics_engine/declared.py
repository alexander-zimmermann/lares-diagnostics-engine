"""Declared files: a YAML file validated against a bundled JSON Schema and
handed over as a mapping. The fault list and the site file are both read
this way — the file lives in lares, the schema here — so a bad edit fails
at load with an error naming the field, never at runtime in the cluster.

The validation itself takes a mapping rather than a path, so a declaration
that never was a file — a candidate fault entry handed in by an agent —
goes through the same schema and the same wording as a line of the real
file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jsonschema
import yaml

if TYPE_CHECKING:
    from collections.abc import Callable

SCHEMAS = Path(__file__).resolve().parent / "_schemas"


def describe_field(error: jsonschema.ValidationError, _data: Any) -> str:
    """The schema error prefixed with the path of the field it belongs to."""
    field = ".".join(str(p) for p in error.absolute_path)
    return f"{field}: {error.message}" if field else error.message


def load_declared(
    path: Path,
    schema: Path,
    describe: Callable[[jsonschema.ValidationError, Any], str] = describe_field,
) -> dict[str, Any]:
    """The file at `path` as a mapping, validated against the bundled
    `schema`; the first schema error is raised as a `ValueError` worded by
    `describe`, which a loader overrides where it can name the entry
    better than its path does."""
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level, got {type(data).__name__}")
    return validate_declared(data, schema, describe, source=str(path))


def validate_declared(
    data: dict[str, Any],
    schema: Path,
    describe: Callable[[jsonschema.ValidationError, Any], str] = describe_field,
    *,
    source: str,
) -> dict[str, Any]:
    """`data` against the bundled `schema`; the first error is raised as a
    `ValueError` worded by `describe` and prefixed with `source` — the file
    it was read from, or what handed it in where there was no file."""
    validator = jsonschema.Draft202012Validator(json.loads(schema.read_text(encoding="utf-8")))
    error = jsonschema.exceptions.best_match(validator.iter_errors(data))
    if error is not None:
        raise ValueError(f"{source}: {describe(error, data)}") from error
    return data
