from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import pytest
import yaml

from examples.hospital import config_io
from examples.hospital.config import DEFAULT_CONFIG, HospitalConfig
from examples.hospital.config_io import (
    CONFIG_ARTIFACT_SCHEMA_VERSION,
    DEFAULT_HOSPITAL_CONFIG_PATH,
    hospital_config_from_artifact_mapping,
    load_hospital_config_artifact,
    write_hospital_config_artifact,
)


def test_default_artifact_is_packaged_and_replays_audited_winner() -> None:
    assert DEFAULT_HOSPITAL_CONFIG_PATH == (
        Path(config_io.__file__).resolve().parent
        / "configs"
        / "plcbf_optuna_best.yaml"
    )
    assert DEFAULT_HOSPITAL_CONFIG_PATH.is_file()

    config, payload = load_hospital_config_artifact(
        DEFAULT_HOSPITAL_CONFIG_PATH
    )

    assert config.policies.room_target_speed == pytest.approx(
        2.849995196985561
    )
    assert config.policies.cbf_alpha == pytest.approx(
        2.1774542113693043
    )
    assert config.safety.hocbf_lambda1 == pytest.approx(
        0.5556737985556784
    )
    assert config.safety.hocbf_lambda2 == pytest.approx(
        0.9710970735487752
    )
    assert payload["provenance"]["study"]["best_trial_number"] == 42
    assert payload["provenance"]["exact_result_archive"]["sha256"] == (
        "8b9375ae80a088e9960d2b585c1c07f3b2356c486926a2a6844a3f2313d088f2"
    )
    assert payload["provenance"]["outcomes"] == {
        "success": 86,
        "collision": 12,
        "timeout": 2,
        "error": 0,
        "infeasible": 0,
    }


def test_direct_json_partial_mapping_uses_base_configuration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "partial.json"
    path.write_text(
        json.dumps(
            {
                "dt": 0.04,
                "policies": {"cbf_alpha": 1.25},
                "safety": {"stretcher_margin": 0.7},
            }
        ),
        encoding="utf-8",
    )
    base = replace(DEFAULT_CONFIG, width=155.0)

    actual, payload = load_hospital_config_artifact(path, base=base)

    assert actual.width == 155.0
    assert actual.dt == 0.04
    assert actual.policies.cbf_alpha == 1.25
    assert actual.policies.rollout_dt == base.policies.rollout_dt
    assert actual.safety.stretcher_margin == 0.7
    assert payload["policies"] == {"cbf_alpha": 1.25}


def test_legacy_tuning_summary_replays_exactly_and_preserves_tuple(
    tmp_path: Path,
) -> None:
    expected = replace(
        DEFAULT_CONFIG,
        policies=replace(
            DEFAULT_CONFIG.policies,
            cbf_alpha=1.4,
            gradient_steps=(0.05, 0.04, 0.03, 0.02),
        ),
    )
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "study": {"name": "hospital-test"},
                "best_params": {"cbf_alpha": 1.4},
                "best_config": asdict(expected),
            }
        ),
        encoding="utf-8",
    )

    actual, payload = load_hospital_config_artifact(path)

    assert actual == expected
    assert isinstance(actual.policies.gradient_steps, tuple)
    assert hash(actual) == hash(expected)
    assert payload["study"]["name"] == "hospital-test"


def test_yaml_config_and_hospital_config_wrappers_are_supported(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical.yaml"
    canonical.write_text(
        yaml.safe_dump(
            {
                "schema_version": CONFIG_ARTIFACT_SCHEMA_VERSION,
                "case_study": "hospital",
                "method": "plcbf",
                "config": {"robot": {"v_max": 2.4}},
                "provenance": {"source": "test"},
            }
        ),
        encoding="utf-8",
    )
    descriptive = {"hospital_config": {"refuge": {"waypoint_radius": 0.8}}}

    canonical_config, _ = load_hospital_config_artifact(canonical)
    descriptive_config = hospital_config_from_artifact_mapping(descriptive)

    assert canonical_config.robot.v_max == 2.4
    assert descriptive_config.refuge.waypoint_radius == 0.8


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"not_a_hospital_field": 1}, "unknown hospital"),
        ({"config": {"policies": {"not_a_policy_field": 1}}}, "unknown policies"),
        ({"best_config": {"safety": {"not_a_safety_field": 1}}}, "unknown safety"),
    ],
)
def test_unknown_configuration_fields_are_rejected(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        hospital_config_from_artifact_mapping(payload)


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"config": []}, "must be an object"),
        ({"best_config": None}, "must be an object"),
        ({"config": {"policies": 1}}, "policies.*must be an object"),
    ],
)
def test_non_mapping_configuration_values_are_rejected(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        hospital_config_from_artifact_mapping(payload)


@pytest.mark.parametrize(
    "metadata, message",
    [
        ({"schema_version": 99}, "schema_version"),
        ({"case_study": "warehouse"}, "case_study"),
        ({"method": "mps"}, "method"),
    ],
)
def test_canonical_artifact_metadata_is_validated(
    metadata: dict[str, object],
    message: str,
) -> None:
    payload = {**metadata, "config": {}}
    with pytest.raises(ValueError, match=message):
        hospital_config_from_artifact_mapping(payload)


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_artifact_round_trip_preserves_config_and_provenance(
    tmp_path: Path,
    suffix: str,
) -> None:
    path = tmp_path / f"winner{suffix}"
    expected = replace(
        DEFAULT_CONFIG,
        robot=replace(DEFAULT_CONFIG.robot, sensing_range=15.5),
        policies=replace(
            DEFAULT_CONFIG.policies,
            cbf_value_buffer=0.35,
            gradient_steps=(0.04, 0.05, 0.02, 0.03),
        ),
    )
    provenance = {
        "study": {
            "best_trial_number": 26,
            "name": "hospital_plcbf_v11_100",
        },
        "trial_archive": Path("results/trial-26.json"),
    }

    returned = write_hospital_config_artifact(
        path,
        expected,
        provenance=provenance,
    )
    actual, payload = load_hospital_config_artifact(path)

    assert returned == path
    assert actual == expected
    assert isinstance(actual.policies.gradient_steps, tuple)
    assert payload["schema_version"] == CONFIG_ARTIFACT_SCHEMA_VERSION
    assert payload["case_study"] == "hospital"
    assert payload["method"] == "plcbf"
    assert payload["provenance"]["study"]["best_trial_number"] == 26
    assert payload["provenance"]["trial_archive"] == (
        "results/trial-26.json"
    )


def test_writer_is_deterministic_for_equivalent_provenance_order(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"

    write_hospital_config_artifact(
        first,
        DEFAULT_CONFIG,
        provenance={"z": {"b": 2, "a": 1}, "a": "first"},
    )
    write_hospital_config_artifact(
        second,
        DEFAULT_CONFIG,
        provenance={"a": "first", "z": {"a": 1, "b": 2}},
    )

    assert first.read_bytes() == second.read_bytes()


def test_atomic_replace_failure_preserves_existing_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "winner.yaml"
    write_hospital_config_artifact(
        path,
        DEFAULT_CONFIG,
        provenance={"revision": "existing"},
    )
    path.chmod(0o640)
    original = path.read_bytes()

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(config_io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic replace"):
        write_hospital_config_artifact(
            path,
            replace(DEFAULT_CONFIG, dt=0.05),
            provenance={"revision": "new"},
        )

    assert path.read_bytes() == original
    assert (os.stat(path).st_mode & 0o777) == 0o640
    assert list(tmp_path.glob(".winner.yaml.*.tmp")) == []


def test_non_object_document_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("- not\n- an\n- object\n", encoding="utf-8")

    with pytest.raises(TypeError, match="must be an object"):
        load_hospital_config_artifact(path)


def test_writer_rejects_non_finite_configuration_without_replacing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "winner.yaml"
    write_hospital_config_artifact(
        path,
        DEFAULT_CONFIG,
        provenance={"revision": "existing"},
    )
    original = path.read_bytes()
    invalid = replace(DEFAULT_CONFIG, width=float("nan"))

    with pytest.raises(ValueError, match="Out of range float values"):
        write_hospital_config_artifact(
            path,
            invalid,
            provenance={"revision": "invalid"},
        )

    assert path.read_bytes() == original
