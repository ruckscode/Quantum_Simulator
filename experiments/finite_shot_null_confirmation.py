"""Fail-closed driver for the prospective 1,024-shot null confirmation.

This module does not sample unless execute_confirmation(...,
execute_sampling=True) is explicitly called with a hash-bound driver and a
review release record. Unit tests exercise validation and analytical helpers
only; they never call pair_outcome or execute_confirmation with sampling enabled.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import time
import tracemalloc
from typing import Any
import base64
import uuid

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

import numpy as np
import cryptography

from experiments.finite_shot_synthetic_characterization import (
    PROTOCOL_ID,
    _cp_lower_bound,
    _cp_upper_bound,
    pair_outcome,
    stream_seed,
)

ROOT = Path(__file__).resolve().parents[1]
PREREG_PATH = ROOT / "reports/experiments/finite_shot_null_confirmation_preregistration_20261016.json"
BINDING_PATH = ROOT / "reports/experiments/finite_shot_null_confirmation_driver_binding_20261016.json"
DEFAULT_OUTPUT_PATH = ROOT / "reports/experiments/finite_shot_null_confirmation_results_20261016.json"
REFERENCE_RUNNER = ROOT / "experiments/finite_shot_synthetic_characterization.py"
STATISTICS_MODULE = ROOT / "src/quantum_simulator/statistics.py"
# This location is deliberately fixed and outside the checkout. An administrator
# must provision it with ACLs that deny write access to the execution identity.
TRUST_ROOT = Path(r"C:\ProgramData\QuantumSimulator\finite-shot-null-confirmation")
TRUSTED_REVIEWER_PATH = TRUST_ROOT / "trusted_reviewer.json"
APPROVAL_PATH = TRUST_ROOT / "approval.json"

EXPECTED = {
    "schema_version": "finite-shot-null-confirmation-prereg-v1",
    "shot_count": 1024,
    "replicates": 38228,
    "maximum_pairs": 38228,
    "reference_distribution": [0.5, 0.25, 0.15, 0.1],
    "submitted_distribution": [0.5, 0.25, 0.15, 0.1],
    "condition_id": "null_equal_confirmation_20261016_01",
    "threshold": 0.05,
    "alpha": 0.05,
    "fpr_limit": 0.06,
    "maximum_passing_flags": 2217,
    "cpu_seconds_maximum": 600,
    "traced_memory_bytes_maximum": 268435456,
    "serialized_output_bytes_maximum": 268435456,
    "workers": 1,
    "seed_protocol_id": PROTOCOL_ID,
}


def sha256_file(path: Path) -> str:
    _assert_regular_no_reparse(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_regular_no_reparse(path: Path) -> None:
    """Reject symlinks/junctions at every path component before reading an artifact."""
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        try:
            info = current.lstat()
        except OSError as exc:
            raise ValueError(f"required artifact path is inaccessible: {current}") from exc
        attrs = getattr(info, "st_file_attributes", 0)
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if stat.S_ISLNK(info.st_mode) or (reparse and attrs & reparse):
            raise ValueError(f"symlink or reparse point is not allowed: {current}")
    if not stat.S_ISREG(absolute.lstat().st_mode):
        raise ValueError(f"artifact is not a regular file: {absolute}")


def _assert_directory_no_reparse(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        try:
            info = current.lstat()
        except OSError as exc:
            raise ValueError(f"required trust directory is inaccessible: {current}") from exc
        attrs = getattr(info, "st_file_attributes", 0)
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if stat.S_ISLNK(info.st_mode) or (reparse and attrs & reparse):
            raise ValueError(f"symlink or reparse point is not allowed: {current}")
        if current == absolute and not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"trust root is not a directory: {absolute}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_file(path: Path) -> Any:
    _assert_regular_no_reparse(path)
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_preregistration(prereg: dict[str, Any]) -> dict[str, Any]:
    """Validate the scientific configuration against fixed registered values."""
    _require(prereg.get("schema_version") == EXPECTED["schema_version"], "unexpected preregistration schema")
    _require(prereg.get("status") == "prospective_design_frozen_execution_driver_hash_pending", "unexpected preregistration status")

    sampling = prereg.get("sampling")
    _require(isinstance(sampling, dict), "sampling section is missing or malformed")
    expected_p = EXPECTED["reference_distribution"]
    _require(sampling.get("reference_distribution") == expected_p, "reference distribution differs from preregistration")
    _require(sampling.get("submitted_distribution") == EXPECTED["submitted_distribution"], "submitted null distribution differs from preregistration")
    _require(sampling.get("shot_count") == EXPECTED["shot_count"], "shot count differs from preregistration")
    _require(sampling.get("replicates") == EXPECTED["replicates"], "replicate count differs from preregistration")
    _require(sampling.get("pair_unit", "").startswith("One pair is one reference multinomial count vector"), "pair definition differs from registered unit")

    detector = prereg.get("detector")
    _require(isinstance(detector, dict), "detector section is missing or malformed")
    _require(detector.get("threshold_argument") == EXPECTED["threshold"], "threshold argument differs from preregistration")
    _require(detector.get("effective_cutoff") == EXPECTED["threshold"], "effective detector cutoff differs from preregistration")
    _require(detector.get("finite_shot_guard") == "max(0.05, 1/sqrt(shots))", "finite-shot guard differs from preregistration")
    _require("empirical TVD" in detector.get("decision", "") and ">" in detector.get("decision", ""), "detector decision rule is malformed")
    _require("No threshold selection" in detector.get("no_tuning", ""), "threshold must remain fixed")

    decision = prereg.get("decision_rule")
    _require(isinstance(decision, dict), "decision rule is missing or malformed")
    _require(decision.get("significance_level") == EXPECTED["alpha"] and decision.get("direction") == "one-sided", "significance level/direction mismatch")
    _require("BetaPPF(0.95, x+1, n-x)" in decision.get("pass", "") and "<=0.06" in decision.get("pass", ""), "pass rule differs from preregistration")
    _require("BetaPPF(0.05, x, n-x+1)" in decision.get("fail", "") and ">0.06" in decision.get("fail", ""), "fail rule differs from preregistration")
    _require("INCONCLUSIVE" in decision.get("inconclusive", ""), "inconclusive rule is malformed")

    rationale = prereg.get("sample_size_rationale")
    _require(isinstance(rationale, dict), "sample-size rationale is missing")
    _require(rationale.get("minimum_n_for_target_power") == EXPECTED["replicates"], "sample size differs from preregistration")
    _require(rationale.get("maximum_passing_detections_at_n") == EXPECTED["maximum_passing_flags"], "acceptance cutoff differs from preregistration")
    _require(math.isclose(rationale.get("target_power", -1), 0.8), "target power differs from preregistration")
    _require(math.isclose(rationale.get("calculated_power_at_n", -1), 0.8024299192074023, abs_tol=1e-12), "registered power calculation differs")
    _require(math.isclose(rationale.get("upper_bound_at_maximum_passing_count", -1), 0.05999850882877437, abs_tol=1e-12), "registered acceptance boundary differs")
    _require(math.isclose(rationale.get("upper_bound_at_next_count_2218", -1), 0.06002508221219589, abs_tol=1e-12), "registered adjacent count differs")
    _require(math.isclose(rationale.get("power_at_n_minus_1", -1), 0.7966325306577215, abs_tol=1e-12), "registered minimal sample-size check differs")

    resource = prereg.get("resource_cap")
    _require(isinstance(resource, dict), "resource cap is missing or malformed")
    _require(resource.get("count_pairs_planned_and_maximum") == EXPECTED["maximum_pairs"], "pair cap differs from preregistration")
    _require(resource.get("count_vectors") == 2 * EXPECTED["replicates"], "count-vector workload differs")
    _require(resource.get("multinomial_outcome_draws") == 2 * EXPECTED["replicates"] * EXPECTED["shot_count"], "multinomial draw count differs")
    _require(resource.get("workers") == EXPECTED["workers"], "worker count differs from preregistration")
    _require(resource.get("cpu_seconds_maximum") == EXPECTED["cpu_seconds_maximum"], "CPU cap differs from preregistration")
    _require(resource.get("traced_memory_bytes_maximum") == EXPECTED["traced_memory_bytes_maximum"], "traced-memory cap differs from preregistration")
    _require(resource.get("serialized_output_bytes_maximum") == EXPECTED["serialized_output_bytes_maximum"], "output cap differs from preregistration")
    _require(EXPECTED["replicates"] == EXPECTED["maximum_pairs"], "registered workload does not fit registered pair cap")

    seed_policy = sampling.get("seed_policy")
    _require(isinstance(seed_policy, dict), "seed policy missing")
    _require(seed_policy.get("condition_id") == EXPECTED["condition_id"], "condition ID differs from preregistration")
    _require(seed_policy.get("roles") == ["reference", "submitted"], "seed roles differ from preregistration")
    _require(seed_policy.get("seed_prefix_protocol_id", "").startswith(EXPECTED["seed_protocol_id"]), "seed protocol prefix differs from implementation")
    return prereg


def verify_hash_bindings(
    prereg: dict[str, Any],
    binding: dict[str, Any],
    *,
    prereg_path: Path = PREREG_PATH,
    binding_path: Path = BINDING_PATH,
    driver_path: Path | None = None,
) -> None:
    """Check internal bindings; independent authorization is verified separately."""
    driver = driver_path or Path(__file__).resolve()
    verify_loaded_code_origins()
    _require(sha256_file(prereg_path) == binding.get("preregistration_sha256"), "preregistration hash mismatch")
    _require(sha256_file(driver) == binding.get("driver_sha256"), "driver hash mismatch")
    _require(binding.get("schema_version") == "finite-shot-null-confirmation-driver-binding-v1", "unexpected binding schema")
    _require(binding.get("binding_path") == binding_path.relative_to(ROOT).as_posix(), "unexpected binding record path")
    _require(strict_json_file(binding_path) == binding, "binding content differs from bound file")

    detector = prereg["detector"]
    _require(detector.get("statistics_module_sha256") == sha256_file(STATISTICS_MODULE), "statistics-module hash differs from preregistration")
    _require(detector.get("reference_runner_sha256") == sha256_file(REFERENCE_RUNNER), "reference-runner hash differs from preregistration")
    _require(binding.get("statistics_module_sha256") == detector.get("statistics_module_sha256"), "binding statistics hash mismatch")
    _require(binding.get("reference_runner_sha256") == detector.get("reference_runner_sha256"), "binding reference-runner hash mismatch")
    _require(binding.get("driver_sha256") == sha256_file(driver), "binding driver hash mismatch")
    _require(binding.get("configuration_sha256") == configuration_sha256(prereg), "configuration hash mismatch")
    ledger = seed_ledger_summary(prereg)
    _require(binding.get("seed_ledger_sha256") == ledger["ordered_seed_bytes_sha256"], "seed ledger hash mismatch")
    _require(binding.get("seed_streams") == ledger["streams"], "binding seed-stream count mismatch")
    _require(binding.get("unique_seed_streams") == ledger["unique_seeds"], "binding seed uniqueness mismatch")
    _require(binding.get("seed_overlap_with_prior") == ledger["overlap_with_prior_streams"] == 0, "binding seed overlap mismatch")
    _require(binding.get("status") == "implementation_hash_bound", "unexpected implementation-binding status")


def verify_loaded_code_origins() -> None:
    """Reject sampler/detector imports whose reported origins differ from pinned files."""
    runner_module = sys.modules.get("experiments.finite_shot_synthetic_characterization")
    stats_module = sys.modules.get("src.quantum_simulator.statistics")
    _require(runner_module is not None and runner_module.pair_outcome is pair_outcome,
             "loaded reference runner does not match the verified import")
    _require(stats_module is not None and pair_outcome.__globals__.get("compare_distributions") is stats_module.compare_distributions,
             "loaded detector does not match the verified import")
    _require(Path(pair_outcome.__code__.co_filename).resolve() == REFERENCE_RUNNER.resolve(),
             "loaded sampler code origin differs from the registered source")
    _require(Path(stats_module.compare_distributions.__code__.co_filename).resolve() == STATISTICS_MODULE.resolve(),
             "loaded detector code origin differs from the registered source")


def artifact_manifest(prereg: dict[str, Any], binding_path: Path = BINDING_PATH,
                      driver_path: Path | None = None,
                      prereg_path: Path = PREREG_PATH,
                      trust_root_id: str | None = None) -> dict[str, Any]:
    """Build exact digest manifest signed by the independent reviewer."""
    driver = driver_path or Path(__file__).resolve()
    ledger = seed_ledger_summary(prereg)
    return {
        "schema": "finite-shot-null-confirmation-artifact-manifest-v1",
        "experiment_id": "finite_shot_null_confirmation_20261016",
        "trust_root_id": trust_root_id,
        "artifacts": {
            "preregistration_sha256": sha256_file(prereg_path),
            "driver_sha256": sha256_file(driver),
            "binding_sha256": sha256_file(binding_path),
            "statistics_module_sha256": sha256_file(STATISTICS_MODULE),
            "reference_runner_sha256": sha256_file(REFERENCE_RUNNER),
            "pyproject_sha256": sha256_file(ROOT / "pyproject.toml"),
            "configuration_sha256": configuration_sha256(prereg),
            "seed_ledger_sha256": ledger["ordered_seed_bytes_sha256"],
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "cryptography": cryptography.__version__,
            "bit_generator": "numpy.random.default_rng default bit generator",
            "python_executable": str(Path(sys.executable).resolve()),
            "python_implementation": sys.implementation.name,
            "python_isolated_mode": bool(sys.flags.isolated),
            "platform": sys.platform,
        },
    }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def verify_signed_approval(prereg: dict[str, Any], binding_path: Path = BINDING_PATH,
                           driver_path: Path | None = None,
                           prereg_path: Path = PREREG_PATH,
                           trusted_reviewer_path: Path = TRUSTED_REVIEWER_PATH,
                           approval_path: Path = APPROVAL_PATH) -> str:
    """Verify approval and return its signed, single-use execution identifier."""
    try:
        _assert_directory_no_reparse(trusted_reviewer_path.parent)
        trusted = strict_json_file(trusted_reviewer_path)
        approval = strict_json_file(approval_path)
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("trusted reviewer key or signed approval is missing/malformed") from exc
    _require(isinstance(trusted, dict), "trusted reviewer record malformed")
    _require(isinstance(approval, dict), "approval record malformed")
    _require(trusted.get("schema") == "finite-shot-trusted-reviewer-v1", "trusted reviewer record schema mismatch")
    _require(approval.get("schema") == "finite-shot-independent-approval-v1", "approval schema mismatch")
    reviewer_id = trusted.get("reviewer_id")
    _require(isinstance(reviewer_id, str) and reviewer_id.strip(), "trusted reviewer identity missing")
    _require(trusted.get("reviewer_independence_verified") is True, "trusted reviewer independence is not established")
    trust_root_id = trusted.get("trust_root_id")
    _require(isinstance(trust_root_id, str) and trust_root_id.strip(), "trusted root identity missing")
    _require(approval.get("trust_root_id") == trust_root_id, "approval trust-root identity mismatch")
    _require(approval.get("reviewer_id") == reviewer_id, "approval reviewer does not match trusted key identity")
    _require(approval.get("key_id") == trusted.get("key_id"), "approval key identifier does not match trusted key")
    run_id = approval.get("run_id")
    try:
        parsed_run_id = uuid.UUID(run_id)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("approval run_id is missing or invalid") from exc
    _require(parsed_run_id.version == 4 and str(parsed_run_id) == run_id, "approval run_id must be a canonical UUIDv4")
    consumed_path = trusted_reviewer_path.parent / f"{run_id}.consumed"
    try:
        consumed_path.lstat()
    except FileNotFoundError:
        pass
    else:
        raise ValueError("approval run_id was already consumed")
    _require(approval.get("decision") == "approved", "independent review is pending or rejected")
    for field in ("reviewed_at", "independence_basis", "review_scope"):
        value = approval.get(field)
        _require((isinstance(value, str) and value.strip()) if field != "review_scope" else
                 (isinstance(value, list) and value and all(isinstance(x, str) and x.strip() for x in value)),
                 f"approval {field} missing or malformed")
    required_scope = {
        "preregistration", "driver", "detector and statistical dependencies",
        "configuration", "seed ledger", "resource controls",
    }
    _require(required_scope.issubset(set(approval["review_scope"])), "approval review scope is incomplete")
    expected_manifest = artifact_manifest(validate_preregistration(prereg), binding_path, driver_path, prereg_path, trust_root_id)
    _require(approval.get("artifact_manifest") == expected_manifest, "approval artifact manifest is stale or mismatched")
    unsigned = {key: value for key, value in approval.items() if key != "signature_base64"}
    try:
        public_key_hex = trusted["ed25519_public_key_hex"]
        public_key_bytes = bytes.fromhex(public_key_hex)
        _require(hashlib.sha256(public_key_bytes).hexdigest() == trusted.get("key_id"), "trusted public-key identifier mismatch")
        public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
        signature = base64.b64decode(approval["signature_base64"], validate=True)
        public_key.verify(signature, _canonical_bytes(unsigned))
    except (KeyError, TypeError, ValueError, InvalidSignature) as exc:
        raise ValueError("approval signature is invalid") from exc
    return run_id


def consume_approval_run(run_id: str, trust_root: Path = TRUST_ROOT) -> None:
    """Atomically consume a signed run before sampling; consumed runs cannot resume."""
    try:
        parsed = uuid.UUID(run_id)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("invalid run identifier") from exc
    _require(parsed.version == 4 and str(parsed) == run_id, "run identifier must be a canonical UUIDv4")
    _assert_directory_no_reparse(trust_root)
    marker = trust_root / f"{run_id}.consumed"
    try:
        with marker.open("x", encoding="ascii") as stream:
            stream.write("consumed-before-sampling\n")
    except FileExistsError as exc:
        raise ValueError("approval run_id was already consumed") from exc


def configuration_projection(prereg: dict[str, Any]) -> dict[str, Any]:
    sampling = prereg["sampling"]
    resource = prereg["resource_cap"]
    return {
        "protocol_id": EXPECTED["seed_protocol_id"],
        "condition_id": sampling["seed_policy"]["condition_id"],
        "shot_count": sampling["shot_count"],
        "replicates": sampling["replicates"],
        "reference_distribution": sampling["reference_distribution"],
        "submitted_distribution": sampling["submitted_distribution"],
        "threshold_argument": prereg["detector"]["threshold_argument"],
        "effective_cutoff": prereg["detector"]["effective_cutoff"],
        "alpha": prereg["decision_rule"]["significance_level"],
        "fpr_limit": EXPECTED["fpr_limit"],
        "workers": resource["workers"],
        "cpu_seconds_maximum": resource["cpu_seconds_maximum"],
        "traced_memory_bytes_maximum": resource["traced_memory_bytes_maximum"],
        "serialized_output_bytes_maximum": resource["serialized_output_bytes_maximum"],
    }


def configuration_sha256(prereg: dict[str, Any]) -> str:
    return canonical_sha256(configuration_projection(prereg))


def seed_ledger_summary(prereg: dict[str, Any]) -> dict[str, Any]:
    """Derive the full registered 32-bit seed ledger; generates no samples."""
    validate_preregistration(prereg)
    n = prereg["sampling"]["replicates"]
    shots = prereg["sampling"]["shot_count"]
    condition_id = prereg["sampling"]["seed_policy"]["condition_id"]
    ordered = bytearray()
    seeds: list[int] = []
    rows: list[dict[str, int]] = []
    for replicate in range(n):
        ref_seed = stream_seed(shots, condition_id, replicate, "reference")
        sub_seed = stream_seed(shots, condition_id, replicate, "submitted")
        seeds.extend((ref_seed, sub_seed))
        ordered.extend(ref_seed.to_bytes(4, "big"))
        ordered.extend(sub_seed.to_bytes(4, "big"))
        rows.append({"replicate_index": replicate, "reference_seed": ref_seed, "submitted_seed": sub_seed})
    unique = len(set(seeds))
    ledger_digest = hashlib.sha256(ordered).hexdigest()
    registered = prereg["sampling"]["seed_policy"]["seed_ledger_integrity"]
    _require(len(seeds) == registered.get("streams"), "seed stream count differs from preregistration")
    _require(unique == registered.get("unique_seeds") and unique == len(seeds), "seed collision or uniqueness mismatch")
    _require(ledger_digest == registered.get("ordered_seed_bytes_sha256"), "seed ledger digest differs from preregistration")

    prior = [("null_equal", 2000), ("tvd_025", 500), ("tvd_050", 500), ("tvd_100", 500), ("tvd_200", 500)]
    prior_seeds = {
        stream_seed(shots, condition, replicate, role)
        for condition, repetitions in prior
        for replicate in range(repetitions)
        for role in ("reference", "submitted")
    }
    overlap = len(set(seeds) & prior_seeds)
    _require(overlap == registered.get("overlap_with_prior_1024_shot_seed_streams"), "seed overlap differs from preregistration")
    _require(overlap == 0, "confirmation seed stream overlaps prior 1,024-shot streams")
    return {"rows": rows, "streams": len(seeds), "unique_seeds": unique, "ordered_seed_bytes_sha256": ledger_digest, "overlap_with_prior_streams": overlap}


def decision_for_flags(flags: int, trials: int = EXPECTED["replicates"]) -> dict[str, Any]:
    if isinstance(flags, bool) or not isinstance(flags, int) or flags < 0 or flags > trials:
        raise ValueError("flags must be an integer in [0, trials]")
    if trials != EXPECTED["replicates"]:
        raise ValueError("trial count must equal the registered confirmation workload")
    upper = 1.0 if flags == trials else _cp_upper_bound(flags, trials, EXPECTED["alpha"])
    lower = 0.0 if flags == 0 else _cp_lower_bound(flags, trials, EXPECTED["alpha"])
    if upper <= EXPECTED["fpr_limit"]:
        verdict = "PASS"
    elif lower > EXPECTED["fpr_limit"]:
        verdict = "FAIL"
    else:
        verdict = "INCONCLUSIVE"
    return {"flags": flags, "trials": trials, "fpr": flags / trials, "one_sided_95_lower": lower, "one_sided_95_upper": upper, "verdict": verdict}


def check_pair_slot(processed: int, pair_cap: int) -> None:
    if processed >= pair_cap:
        raise RuntimeError("pair cap would be exceeded; stopped before sampling another pair")


def check_live_resources(cpu_seconds: float, traced_bytes: int, resource: dict[str, Any]) -> None:
    if cpu_seconds >= resource["cpu_seconds_maximum"]:
        raise TimeoutError("CPU-time cap reached")
    if traced_bytes > resource["traced_memory_bytes_maximum"]:
        raise MemoryError("traced-memory cap exceeded")


def check_output_size(serialized_bytes: int, resource: dict[str, Any]) -> None:
    if serialized_bytes > resource["serialized_output_bytes_maximum"]:
        raise RuntimeError("serialized-output cap exceeded")


def check_output_projection(seed_rows: list[dict[str, int]], resource: dict[str, Any]) -> int:
    """Bound output size before sampling; metadata has a fixed 64 KiB reserve."""
    seed_bytes = len(json.dumps(seed_rows, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    projected = seed_bytes + 65536
    check_output_size(projected, resource)
    return projected


def execute_confirmation(
    prereg_path: Path = PREREG_PATH,
    binding_path: Path = BINDING_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    execute_sampling: bool = False,
) -> dict[str, Any]:
    """Execute only with explicit authorization and a complete hash/review gate."""
    if not execute_sampling:
        raise PermissionError("sampling requires explicit execute_sampling=True")
    _require(prereg_path.resolve() == PREREG_PATH.resolve(), "confirmation preregistration path is fixed")
    _require(binding_path.resolve() == BINDING_PATH.resolve(), "confirmation binding path is fixed")
    _require(output_path.resolve() == DEFAULT_OUTPUT_PATH.resolve(), "confirmation output path is fixed and append-only")
    _assert_directory_no_reparse(output_path.parent)
    if output_path.exists():
        raise FileExistsError(f"create-only result path already exists: {output_path}")
    prereg = validate_preregistration(strict_json_file(prereg_path))
    binding = strict_json_file(binding_path)
    verify_hash_bindings(prereg, binding, prereg_path=prereg_path, binding_path=binding_path)
    run_id = verify_signed_approval(
        prereg, binding_path=binding_path, prereg_path=prereg_path,
        trusted_reviewer_path=TRUSTED_REVIEWER_PATH, approval_path=APPROVAL_PATH,
    )
    version = prereg["runtime_reproducibility"]["planning_environment"]
    _require(sys.version.split()[0] == version["python"].split()[0], "Python version differs from registered planning environment")
    _require(np.__version__ == version["numpy"].split()[0], "NumPy version differs from registered planning environment")
    consume_approval_run(run_id, trust_root=TRUST_ROOT)

    # Account for seed-ledger preflight and the workload in the registered budgets.
    n = prereg["sampling"]["replicates"]
    cap = prereg["resource_cap"]
    _require(n == cap["count_pairs_planned_and_maximum"] == EXPECTED["maximum_pairs"], "workload/cap mismatch")
    start_cpu = time.process_time()
    start_wall = time.monotonic()
    tracemalloc.start()
    ledger = seed_ledger_summary(prereg)
    projected_output_bytes = check_output_projection(ledger["rows"], cap)
    _, preflight_peak = tracemalloc.get_traced_memory()
    check_live_resources(time.process_time() - start_cpu, preflight_peak, cap)
    if output_path.exists():
        tracemalloc.stop()
        raise FileExistsError(f"create-only result path already exists: {output_path}")

    p = tuple(prereg["sampling"]["reference_distribution"])
    q = tuple(prereg["sampling"]["submitted_distribution"])
    shots = prereg["sampling"]["shot_count"]
    condition_id = prereg["sampling"]["seed_policy"]["condition_id"]
    flags = 0
    processed = 0
    for row in ledger["rows"]:
        try:
            check_pair_slot(processed, cap["count_pairs_planned_and_maximum"])
            check_live_resources(time.process_time() - start_cpu, tracemalloc.get_traced_memory()[1], cap)
        except (RuntimeError, TimeoutError, MemoryError):
            tracemalloc.stop()
            raise
        flags += bool(pair_outcome(shots, condition_id, row["replicate_index"], p, q))
        processed += 1
        _, peak = tracemalloc.get_traced_memory()
        try:
            check_live_resources(time.process_time() - start_cpu, peak, cap)
        except (TimeoutError, MemoryError):
            tracemalloc.stop()
            raise
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _require(processed == n, "actual pair count differs from registered workload")
    result = {
        "schema_version": "finite-shot-null-confirmation-result-v1",
        "experiment_id": "finite_shot_null_confirmation_20261016",
        "run_id": run_id,
        "preregistration_sha256": sha256_file(prereg_path),
        "binding_sha256": sha256_file(binding_path),
        "driver_sha256": sha256_file(Path(__file__).resolve()),
        "statistics_module_sha256": sha256_file(STATISTICS_MODULE),
        "configuration_sha256": configuration_sha256(prereg),
        "configuration": configuration_projection(prereg),
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "cryptography": cryptography.__version__, "bit_generator": "numpy.random.default_rng default bit generator"},
        "seed_ledger": {key: value for key, value in ledger.items() if key != "rows"},
        "seed_ledger_rows": ledger["rows"],
        "endpoint": decision_for_flags(flags),
        "resource_accounting": {"planned_pairs": n, "actual_pairs": processed, "maximum_pairs": cap["count_pairs_planned_and_maximum"], "workers": 1, "cpu_seconds": time.process_time() - start_cpu, "wall_seconds": time.monotonic() - start_wall, "tracemalloc_peak_bytes": peak, "projected_output_upper_bound_bytes": projected_output_bytes},
    }
    encoded = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    check_output_size(len(encoded), cap)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        stream.write(encoded)
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, default=PREREG_PATH)
    parser.add_argument("--binding", type=Path, default=BINDING_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--execute-sampling", action="store_true", help="explicitly authorize the registered sampling workload")
    args = parser.parse_args()
    result = execute_confirmation(
        args.prereg, args.binding, args.output,
        execute_sampling=args.execute_sampling,
    )
    print(json.dumps({"verdict": result["endpoint"]["verdict"], "actual_pairs": result["resource_accounting"]["actual_pairs"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
