"""Validated YAML/JSON artifacts for nonlinear Quad3D controllers.

The packaged YAML is the single default used by both the headless benchmark
and the interactive runner.  Keeping that path here prevents the two entry
points from silently drifting after an Optuna study exports a new winner.
"""

from __future__ import annotations

from dataclasses import asdict, fields, replace
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping

import yaml

from .controller import NLQuad3DControllerConfig


CONTROLLER_ARTIFACT_SCHEMA_VERSION = 1
DEFAULT_CONTROLLER_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "plcbf_optuna_best.yaml"
)


def controller_config_from_mapping(
    values: Mapping[str, Any],
    *,
    base: NLQuad3DControllerConfig | None = None,
) -> NLQuad3DControllerConfig:
    """Build a controller config from an exact or partial mapping."""

    source = base or NLQuad3DControllerConfig()
    allowed = {item.name for item in fields(NLQuad3DControllerConfig)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(
            "unknown nonlinear Quad3D controller fields: "
            + ", ".join(sorted(unknown))
        )
    return replace(source, **dict(values))


def _read_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text)
    if not isinstance(payload, Mapping):
        raise TypeError("nonlinear Quad3D configuration artifact must be an object")
    return payload


def _controller_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    raw_config: Any
    if "controller" in payload:
        raw_config = payload["controller"]
    elif "best_controller" in payload:
        # Backward-compatible replay of the original JSON tuning summary.
        raw_config = payload["best_controller"]
    else:
        raw_config = payload
    if not isinstance(raw_config, Mapping):
        raise TypeError("controller/best_controller must be an object")
    return raw_config


def load_controller_config_artifact(
    path: str | Path,
    *,
    base: NLQuad3DControllerConfig | None = None,
) -> tuple[NLQuad3DControllerConfig, Mapping[str, Any]]:
    """Load YAML/JSON controller data or a tuning summary artifact."""

    source = Path(path)
    payload = _read_mapping(source)
    config = controller_config_from_mapping(
        _controller_mapping(payload), base=base
    )
    return config, payload


def load_default_controller_config(
    *,
    base: NLQuad3DControllerConfig | None = None,
) -> tuple[NLQuad3DControllerConfig, Mapping[str, Any]]:
    """Load the packaged winning controller configuration."""

    return load_controller_config_artifact(
        DEFAULT_CONTROLLER_CONFIG_PATH,
        base=base,
    )


def _json_compatible(value: Any) -> Any:
    """Normalize NumPy scalars/tuples before handing data to PyYAML."""

    return json.loads(json.dumps(value, allow_nan=False))


def write_controller_config_artifact(
    path: str | Path,
    config: NLQuad3DControllerConfig,
    *,
    provenance: Mapping[str, Any],
) -> Path:
    """Write a deterministic, human-readable winning-controller YAML."""

    destination = Path(path)
    document = {
        "schema_version": CONTROLLER_ARTIFACT_SCHEMA_VERSION,
        "case_study": "nl_quad3d",
        "method": "plcbf",
        "controller": _json_compatible(asdict(config)),
        "provenance": _json_compatible(dict(provenance)),
    }
    serialized = yaml.safe_dump(
        document,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
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
    "CONTROLLER_ARTIFACT_SCHEMA_VERSION",
    "DEFAULT_CONTROLLER_CONFIG_PATH",
    "controller_config_from_mapping",
    "load_controller_config_artifact",
    "load_default_controller_config",
    "write_controller_config_artifact",
]
