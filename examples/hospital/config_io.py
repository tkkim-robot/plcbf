"""Validated YAML/JSON artifacts for the Hospital PL-CBF configuration.

The packaged artifact is the single configuration source used after an
Optuna winner has been selected.  This module deliberately owns only artifact
serialization and replay: field validation and normalization remain in
``hospital_config_from_mapping`` so direct mappings, canonical artifacts, and
legacy tuning summaries all obey the same configuration contract.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping

import yaml

from .config import (
    DEFAULT_CONFIG,
    HospitalConfig,
    hospital_config_from_mapping,
)


CONFIG_ARTIFACT_SCHEMA_VERSION = 1
HOSPITAL_CONFIG_ARTIFACT_SCHEMA_VERSION = CONFIG_ARTIFACT_SCHEMA_VERSION
DEFAULT_HOSPITAL_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "plcbf_optuna_best.yaml"
)


def _read_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text)
    if not isinstance(payload, Mapping):
        raise TypeError("Hospital configuration artifact must be an object")
    return payload


def _configuration_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract configuration values from supported artifact layouts."""

    raw_config: Any
    if "config" in payload:
        raw_config = payload["config"]
    elif "hospital_config" in payload:
        # Accepted for descriptive hand-authored artifacts.
        raw_config = payload["hospital_config"]
    elif "best_config" in payload:
        # Backward-compatible replay of Hospital Optuna summary JSON files.
        raw_config = payload["best_config"]
    else:
        # A direct exact or partial Hospital configuration mapping.
        raw_config = payload
    if not isinstance(raw_config, Mapping):
        raise TypeError(
            "config/hospital_config/best_config must be an object"
        )
    return raw_config


def _validate_artifact_metadata(payload: Mapping[str, Any]) -> None:
    """Reject a canonical artifact for another schema, case, or method."""

    # Direct and legacy mappings do not carry canonical artifact metadata.
    if not ({"config", "hospital_config"} & payload.keys()):
        return
    if (
        "schema_version" in payload
        and payload["schema_version"] != CONFIG_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError(
            "unsupported Hospital configuration artifact schema_version: "
            f"{payload['schema_version']!r}"
        )
    if "case_study" in payload and payload["case_study"] != "hospital":
        raise ValueError(
            "Hospital configuration artifact case_study must be 'hospital'"
        )
    if "method" in payload and payload["method"] != "plcbf":
        raise ValueError(
            "Hospital configuration artifact method must be 'plcbf'"
        )


def hospital_config_from_artifact_mapping(
    payload: Mapping[str, Any],
    *,
    base: HospitalConfig = DEFAULT_CONFIG,
) -> HospitalConfig:
    """Build a validated Hospital config from any supported artifact mapping."""

    if not isinstance(payload, Mapping):
        raise TypeError("Hospital configuration artifact must be an object")
    _validate_artifact_metadata(payload)
    return hospital_config_from_mapping(
        _configuration_mapping(payload),
        base=base,
    )


def load_hospital_config_artifact(
    path: str | Path,
    *,
    base: HospitalConfig = DEFAULT_CONFIG,
) -> tuple[HospitalConfig, Mapping[str, Any]]:
    """Load YAML/JSON config data or a legacy Hospital tuning summary."""

    payload = _read_mapping(Path(path))
    return hospital_config_from_artifact_mapping(payload, base=base), payload


def load_default_hospital_config(
    *,
    base: HospitalConfig = DEFAULT_CONFIG,
) -> tuple[HospitalConfig, Mapping[str, Any]]:
    """Load the packaged winning Hospital PL-CBF configuration."""

    return load_hospital_config_artifact(
        DEFAULT_HOSPITAL_CONFIG_PATH,
        base=base,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        converted = item()
        if converted is not value:
            return converted
    raise TypeError(
        f"object of type {type(value).__name__} is not JSON compatible"
    )


def _json_compatible(value: Any) -> Any:
    """Normalize scalars and tuples into a deterministic JSON value tree."""

    return json.loads(
        json.dumps(
            value,
            allow_nan=False,
            default=_json_default,
            sort_keys=True,
        )
    )


def _serialize_document(path: Path, document: Mapping[str, Any]) -> str:
    if path.suffix.lower() == ".json":
        return (
            json.dumps(
                document,
                indent=2,
                sort_keys=False,
                allow_nan=False,
            )
            + "\n"
        )
    return yaml.safe_dump(
        document,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )


def write_hospital_config_artifact(
    path: str | Path,
    config: HospitalConfig,
    *,
    provenance: Mapping[str, Any],
) -> Path:
    """Atomically write a deterministic Hospital PL-CBF config artifact."""

    if not isinstance(config, HospitalConfig):
        raise TypeError("config must be a HospitalConfig")
    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be an object")
    destination = Path(path)
    document = {
        "schema_version": CONFIG_ARTIFACT_SCHEMA_VERSION,
        "case_study": "hospital",
        "method": "plcbf",
        "config": _json_compatible(asdict(config)),
        "provenance": _json_compatible(dict(provenance)),
    }
    serialized = _serialize_document(destination, document)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = (
        stat.S_IMODE(destination.stat().st_mode)
        if destination.exists()
        else 0o644
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(existing_mode)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


__all__ = [
    "CONFIG_ARTIFACT_SCHEMA_VERSION",
    "DEFAULT_HOSPITAL_CONFIG_PATH",
    "HOSPITAL_CONFIG_ARTIFACT_SCHEMA_VERSION",
    "hospital_config_from_artifact_mapping",
    "load_default_hospital_config",
    "load_hospital_config_artifact",
    "write_hospital_config_artifact",
]
