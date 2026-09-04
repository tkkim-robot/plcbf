"""Content-addressed provenance for Hospital benchmark reports.

The benchmark is often executed as independently scheduled method/seed
shards while the working tree still contains uncommitted calibration work.
Git metadata alone therefore cannot prove that two shards used identical
implementations.  This module declares the source closure used by the
Hospital benchmark and hashes repository-relative paths together with the
exact file bytes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


HOSPITAL_BENCHMARK_SOURCE_MANIFEST_SCHEMA = (
    "plcbf.hospital.benchmark-source-manifest/v1"
)

# The order is part of the manifest contract.  Keep paths repository-relative
# so fingerprints are independent of checkout location.
HOSPITAL_BENCHMARK_SOURCE_FILES: tuple[str, ...] = (
    "examples/hospital/__init__.py",
    "examples/hospital/baselines.py",
    "examples/hospital/benchmark.py",
    "examples/hospital/config.py",
    "examples/hospital/config_io.py",
    "examples/hospital/controller.py",
    "examples/hospital/dynamics.py",
    "examples/hospital/environment.py",
    "examples/hospital/feasibility.py",
    "examples/hospital/jax_rollout.py",
    "examples/hospital/obstacles.py",
    "examples/hospital/planner.py",
    "examples/hospital/policies.py",
    "examples/hospital/provenance.py",
    "examples/hospital/reporting.py",
    "examples/hospital/scenario_generation.py",
    "examples/hospital/scenarios.py",
    "examples/hospital/simulation.py",
    "plcbf/backup_cbf.py",
    "plcbf/baselines.py",
    "plcbf/benchmarking.py",
    "plcbf/big_m_mpc.py",
    "plcbf/dynamics_linearization.py",
    "plcbf/policy_library.py",
    "plcbf/trajectory_shielding.py",
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _frame(value: bytes) -> bytes:
    """Prefix bytes with an unsigned length to prevent concatenation ambiguity."""

    return len(value).to_bytes(8, byteorder="big", signed=False) + value


def hospital_benchmark_source_manifest() -> dict[str, object]:
    """Return the exact implementation fingerprint embedded in every report.

    ``combined_sha256`` hashes the schema followed by each declared
    repository-relative UTF-8 path and its raw content, with every component
    length-prefixed by an unsigned 64-bit big-endian integer.  Individual
    hashes make shard mismatches directly diagnosable.
    """

    root = _repository_root()
    combined = hashlib.sha256()
    combined.update(
        _frame(HOSPITAL_BENCHMARK_SOURCE_MANIFEST_SCHEMA.encode("utf-8"))
    )
    files: list[dict[str, object]] = []
    for relative_path in HOSPITAL_BENCHMARK_SOURCE_FILES:
        path_bytes = relative_path.encode("utf-8")
        content = (root / relative_path).read_bytes()
        combined.update(_frame(path_bytes))
        combined.update(_frame(content))
        files.append(
            {
                "path": relative_path,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return {
        "schema": HOSPITAL_BENCHMARK_SOURCE_MANIFEST_SCHEMA,
        "hash_algorithm": "sha256",
        "combined_sha256": combined.hexdigest(),
        "files": files,
    }


__all__ = [
    "HOSPITAL_BENCHMARK_SOURCE_FILES",
    "HOSPITAL_BENCHMARK_SOURCE_MANIFEST_SCHEMA",
    "hospital_benchmark_source_manifest",
]
