"""Finite-shot calibration plan, deterministic generation, and runner helpers.

The fixed pilot topology is the primary frozen recipe. Varied paired-CNOT
topology is available for exploratory comparison only. Evaluation labels stay
outside ``diagnosis_inputs``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "finite_shot_distribution_calibration.json"
)

STRATA = (
    ("identical_control", False),
    ("measurement_equivalent_control", False),
    ("subtle", True),
    ("moderate", True),
    ("large", True),
)


def load_spec() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _configuration_id(fields: dict[str, Any]) -> str:
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def build_ledger(spec: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return all planned split/stratum/template/shot rows in stable order."""
    spec = spec or load_spec()
    plan = spec["proposed_next_calibration_plan"]
    design = plan["design"]
    workload = plan["workload"]
    per_split = int(plan["trials_per_stratum_per_shot_per_split"])
    shots = tuple(int(value) for value in plan["shot_counts"])
    roots = design["template_generation_seed_roots"]
    sample_roots = design["sampling_seed_roots"]
    output_root = Path(design["calibration_output_directory"])
    rows: list[dict[str, Any]] = []

    split_names = (("calibration", "calibration"), ("untouched_validation", "untouched_validation"))
    for split, seed_key in split_names:
        for stratum_index, (stratum, expected_anomaly) in enumerate(STRATA):
            for template_index in range(per_split):
                template_seed = int(roots[seed_key]) + template_index
                template_id = f"{split}:template:{template_index:03d}"
                for shot_index, shot_count in enumerate(shots):
                    sampling_seed = (
                        int(sample_roots[seed_key])
                        + 10000 * stratum_index
                        + 10 * template_index
                        + shot_index
                    )
                    fields = {
                        "split": split,
                        "template_id": template_id,
                        "scenario": stratum,
                        "fault_strength_stratum": stratum,
                        "shot_count": shot_count,
                        "template_seed": template_seed,
                        "seed": sampling_seed,
                    }
                    config_id = _configuration_id(fields)
                    output_path = output_root / split / stratum / f"{config_id}.json"
                    rows.append({
                        "configuration_id": config_id,
                        **fields,
                        "expected_label": "ANOMALY" if expected_anomaly else "NO_ANOMALY",
                        "output_location": output_path.as_posix(),
                    })

    expected = int(workload["diagnosis_configurations"])
    if len(rows) != expected:
        raise ValueError(f"ledger has {len(rows)} configurations; expected {expected}")
    if len({row["configuration_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate deterministic configuration identifier")
    return rows


def diagnosis_inputs(case: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Build the diagnosis call payload without any expected/evaluation metadata."""
    return {
        "circuit": case["submitted"],
        "reference_circuit": case["reference"],
        "shots": row["shot_count"],
        "seed": row["seed"],
        "threshold": 0.05,
    }


def output_state(row: dict[str, Any], repository_root: Path | None = None) -> str:
    """Classify output as missing, incomplete, or completed; never infer completion."""
    root = repository_root or Path(__file__).resolve().parents[1]
    path = root / Path(row["output_location"])
    if not path.exists():
        return "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "incomplete"
    return "completed" if is_completed_payload(payload, row["configuration_id"]) else "incomplete"


def is_completed_payload(payload: Any, configuration_id: str) -> bool:
    """Require an explicit completion state, matching ID, and nonempty result."""
    return (
        isinstance(payload, dict)
        and payload.get("status") == "completed"
        and payload.get("configuration_id") == configuration_id
        and isinstance(payload.get("result"), dict)
        and bool(payload["result"])
    )


def _validate_calibration_payload(payload: Any, row: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate a stored result against its ledger row before counting it."""
    if not is_completed_payload(payload, row["configuration_id"]):
        return False, "completion envelope or configuration ID mismatch"
    result = payload["result"]
    expected_fields = {
        "split": row["split"], "template_id": row["template_id"],
        "template_seed": row["template_seed"],
        "configuration_id": row["configuration_id"],
        "scenario": row["scenario"],
        "fault_strength_stratum": row["fault_strength_stratum"],
        "shot_count": row["shot_count"], "sampling_seed": row["seed"],
        "expected_label": row["expected_label"], "status": "completed",
    }
    for key, expected in expected_fields.items():
        if result.get(key) != expected:
            return False, f"result field {key} does not match ledger"
    for key in ("current_detected", "permutation_detected", "abstained"):
        if not isinstance(result.get(key), bool):
            return False, f"result field {key} is not boolean"
    if result.get("diagnosis_category") not in {
        "NO_ANOMALY", "PROGRAM_FAULT", "HARDWARE_ANOMALY",
        "AMBIGUOUS / INSUFFICIENT_EVIDENCE",
    }:
        return False, "invalid diagnosis category"
    for key in ("permutation_p", "tvd", "finite_shot_guard", "effective_threshold"):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return False, f"result field {key} is not finite numeric data"
    for key in ("expected_distribution", "observed_distribution"):
        distribution = result.get(key)
        if not isinstance(distribution, dict) or set(distribution) != {format(i, "04b") for i in range(16)}:
            return False, f"result field {key} has invalid outcome support"
        values = list(distribution.values())
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
            return False, f"result field {key} contains invalid probabilities"
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-8):
            return False, f"result field {key} does not sum to one"
        if not math.isclose(sum(round(float(v) * row["shot_count"]) for v in values), row["shot_count"], abs_tol=0.0):
            return False, f"result field {key} does not encode the declared shot count"
    return True, None


def _read_valid_payload(path: Path, row: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "unreadable or malformed JSON"
    valid, reason = _validate_calibration_payload(payload, row)
    return (payload, "valid") if valid else (None, str(reason))


def inspect_resume_state(
    spec: dict[str, Any] | None = None, repository_root: Path | None = None,
) -> dict[str, Any]:
    """Read-only reconciliation of ledger rows and resumable output artifacts."""
    spec = spec or load_spec()
    require_generation_ready(spec)
    rows = build_ledger(spec)
    root = repository_root or Path(__file__).resolve().parents[1]
    valid: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    missing: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for row in rows:
        path = root / Path(row["output_location"])
        if not path.exists():
            missing.append(row)
            continue
        payload, reason = _read_valid_payload(path, row)
        if payload is None:
            recovered = path.with_name(f"{row['configuration_id']}.recovered.json")
            if recovered.exists():
                recovered_payload, recovered_reason = _read_valid_payload(recovered, row)
                if recovered_payload is not None:
                    valid.append((row, recovered_payload["result"], recovered))
                    continue
                reason = f"{reason}; recovery artifact: {recovered_reason}"
            invalid.append({"row": row, "path": path, "reason": reason})
        else:
            valid.append((row, payload["result"], path))
    plan = spec["proposed_next_calibration_plan"]
    planned_paths = {(root / Path(r["output_location"])).resolve() for r in rows}
    recovered_paths = {p.with_name(f"{r['configuration_id']}.recovered.json").resolve()
                       for r in rows for p in [root / Path(r["output_location"])]}
    extras = [p for p in (root / Path(plan["design"]["calibration_output_directory"])).rglob("*.json")
              if p.name not in {"run_manifest.json", "summary.json"}
              and "failures" not in p.relative_to(
                  root / Path(plan["design"]["calibration_output_directory"])
              ).parts
              and p.resolve() not in planned_paths and p.resolve() not in recovered_paths]
    return {
        "rows": rows, "valid": valid, "missing": missing, "invalid": invalid,
        "extra_json_paths": extras,
        "planned_configurations": len(rows),
        "valid_configurations": len(valid),
        "missing_configurations": len(missing),
        "invalid_configurations": len(invalid),
        "preserved_existing_valid": len(valid),
    }


def resume_preflight(spec: dict[str, Any] | None = None, repository_root: Path | None = None) -> dict[str, Any]:
    """Check that all incomplete ledger destinations are safely writable."""
    spec = spec or load_spec()
    state = inspect_resume_state(spec, repository_root)
    root = repository_root or Path(__file__).resolve().parents[1]
    blockers: list[str] = []
    if state["extra_json_paths"]:
        blockers.append(f"unexpected JSON artifacts found: {len(state['extra_json_paths'])}")
    output_root = root / Path(spec["proposed_next_calibration_plan"]["design"]["calibration_output_directory"])
    summary_path = output_root / "summary.json"
    if summary_path.exists():
        blockers.append(f"final summary already exists; refusing to resume into a finalized run: {summary_path}")
    # Existing invalid files are preserved. Their repaired version is written
    # to a deterministic sibling path, never over the original artifact.
    for item in state["invalid"]:
        alternate = item["path"].with_name(f"{item['row']['configuration_id']}.recovered.json")
        if alternate.exists():
            payload, _ = _read_valid_payload(alternate, item["row"])
            if payload is None:
                blockers.append(f"recovery destination exists but is invalid: {alternate}")
    return {
        "ready": not blockers,
        "planned_configurations": state["planned_configurations"],
        "valid_configurations": state["valid_configurations"],
        "missing_configurations": state["missing_configurations"],
        "invalid_configurations": state["invalid_configurations"],
        "remaining_configurations": state["missing_configurations"] + state["invalid_configurations"],
        "blockers": blockers,
        "state": state,
    }


def validate_preflight(spec: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Validate seed namespaces, split separation, workload, and output safety."""
    spec = spec or load_spec()
    plan = spec["proposed_next_calibration_plan"]
    design = plan["design"]
    n = int(plan["trials_per_stratum_per_shot_per_split"])
    strata_count = len(STRATA)
    shot_counts = [int(value) for value in plan["shot_counts"]]
    tr = design["template_generation_seed_roots"]
    sr = design["sampling_seed_roots"]
    max_sample_offset = 10000 * (strata_count - 1) + 10 * (n - 1) + (len(shot_counts) - 1)
    namespaces = {
        "calibration_template_generation": (int(tr["calibration"]), int(tr["calibration"]) + n - 1),
        "calibration_sampling": (int(sr["calibration"]), int(sr["calibration"]) + max_sample_offset),
        "validation_template_generation": (int(tr["untouched_validation"]), int(tr["untouched_validation"]) + n - 1),
        "validation_sampling": (int(sr["untouched_validation"]), int(sr["untouched_validation"]) + max_sample_offset),
    }
    ranges = list(namespaces.items())
    for index, (name_a, range_a) in enumerate(ranges):
        for name_b, range_b in ranges[index + 1:]:
            if max(range_a[0], range_b[0]) <= min(range_a[1], range_b[1]):
                raise ValueError(f"seed namespaces overlap: {name_a} and {name_b}")

    declared_ranges = design.get("seed_namespace_ranges", {})
    for name, bounds in namespaces.items():
        if declared_ranges.get(name) != list(bounds):
            raise ValueError(f"declared seed range mismatch for {name}")

    expected_configs = 2 * strata_count * n * len(shot_counts)
    expected_executions = 2 * strata_count * n * 2 * sum(shot_counts)
    workload = plan["workload"]
    if int(workload["diagnosis_configurations"]) != expected_configs:
        raise ValueError("preregistered diagnosis configuration arithmetic mismatch")
    if int(workload["shot_circuit_executions"]) != expected_executions:
        raise ValueError("preregistered shot-circuit execution arithmetic mismatch")

    # Keep this metadata-only check repeatable after a run has started. The
    # execution entry point enforces create-only output semantics, while the
    # resume preflight reconciles existing artifacts before any simulation.
    return build_ledger(spec)


class CalibrationDesignBlocked(ValueError):
    """Raised when the preregistered circuit-generation recipe is incomplete."""


def generation_decisions(spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the frozen primary recipe and any remaining generation gap."""
    spec = spec or load_spec()
    protocol = spec["proposed_next_calibration_plan"]["design"].get("generation_protocol")
    if not isinstance(protocol, dict):
        return {"status": "missing", "specified": {}, "missing_decisions": [
            "No generation protocol is recorded for the 120 distinct reference templates per split."
        ], "proposed_minimal_option": None}
    return {
        "status": protocol.get("status", "missing"),
        "specified": protocol.get("specified_by_existing_evidence", {}),
        "missing_decisions": list(protocol.get("missing_decisions", [])),
        "primary_recipe": protocol.get("primary_recipe"),
        "exploratory_recipes": protocol.get("candidate_recipes_for_comparison_only", []),
    }


def require_generation_ready(spec: dict[str, Any] | None = None) -> None:
    """Fail closed unless the recorded primary recipe is supported by code."""
    decision = generation_decisions(spec)
    primary = decision["primary_recipe"]
    if (decision["status"] != "approved_and_frozen" or decision["missing_decisions"]
            or not isinstance(primary, dict)
            or primary.get("id") != "fixed_pilot_topology_phase_tags"
            or primary.get("status") != "selected_for_primary_calibration"):
        missing = "; ".join(decision["missing_decisions"])
        if not missing:
            missing = "a supported and explicitly frozen primary recipe is required"
        raise CalibrationDesignBlocked(
            "finite-shot calibration case generation is not preregistered: " + missing
        )


def validate_execution_preflight(spec: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Validate metadata and require a complete circuit-generation protocol."""
    rows = validate_preflight(spec)
    require_generation_ready(spec)
    return rows


PRIMARY_RECIPE = "fixed_pilot_topology_phase_tags"
EXPLORATORY_RECIPES = ("seeded_paired_topology_phase_tags",)
COMPARISON_RECIPES = (PRIMARY_RECIPE, *EXPLORATORY_RECIPES)


def generate_candidate_template(template_seed: int, recipe_id: str) -> dict[str, Any]:
    """Generate a deterministic template for the primary or exploratory recipe."""
    if isinstance(template_seed, bool) or not isinstance(template_seed, int) or template_seed < 0:
        raise ValueError("template_seed must be a nonnegative integer")
    if recipe_id not in COMPARISON_RECIPES:
        raise ValueError(f"unsupported calibration template recipe: {recipe_id}")
    rng = np.random.default_rng(template_seed)

    if recipe_id == "fixed_pilot_topology_phase_tags":
        h_order = [0, 1, 2, 3]
        cnot_pairs = [(0, 2), (1, 3)]
    else:
        h_order = [int(q) for q in rng.permutation(4)]
        pairings = (
            ((0, 1), (2, 3)),
            ((0, 2), (1, 3)),
            ((0, 3), (1, 2)),
        )
        selected = pairings[int(rng.integers(len(pairings)))]
        cnot_pairs = []
        for pair_index in rng.permutation(2):
            control, target = selected[int(pair_index)]
            if int(rng.integers(2)):
                control, target = target, control
            cnot_pairs.append((control, target))

    phase_angles = [float(value) for value in rng.uniform(-math.pi, math.pi, size=4)]
    reference = [("h", q) for q in h_order]
    reference.extend(("cx", control, target) for control, target in cnot_pairs)
    reference.extend(("rz", q, phase_angles[q]) for q in range(4))
    reference.append(("measure", (0, 1, 2, 3)))
    canonical = json.dumps(reference, separators=(",", ":"), allow_nan=False)
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "template_seed": template_seed,
        "recipe_id": recipe_id,
        "reference": reference,
        "reference_fingerprint": fingerprint,
        "generation_metadata": {
            "h_order": h_order,
            "paired_cnot_edges": [list(edge) for edge in cnot_pairs],
            "terminal_rz_angles": phase_angles,
        },
    }


def generate_primary_template(template_seed: int) -> dict[str, Any]:
    """Generate the frozen fixed-pilot-topology primary calibration template."""
    require_generation_ready()
    return generate_candidate_template(template_seed, PRIMARY_RECIPE)


def generate_candidate_template_sets(
    spec: dict[str, Any] | None = None,
    recipe_id: str = "fixed_pilot_topology_phase_tags",
) -> dict[str, list[dict[str, Any]]]:
    """Generate seed-separated candidate sets and reject any fingerprint reuse."""
    spec = spec or load_spec()
    plan = spec["proposed_next_calibration_plan"]
    count = int(plan["trials_per_stratum_per_shot_per_split"])
    roots = plan["design"]["template_generation_seed_roots"]
    result: dict[str, list[dict[str, Any]]] = {}
    for split, root_key in (("calibration", "calibration"), ("untouched_validation", "untouched_validation")):
        result[split] = [
            generate_candidate_template(int(roots[root_key]) + index, recipe_id)
            for index in range(count)
        ]
        fingerprints = [item["reference_fingerprint"] for item in result[split]]
        if len(set(fingerprints)) != count:
            raise ValueError(f"duplicate template fingerprint within {split} candidate set")
    calibration = {item["reference_fingerprint"] for item in result["calibration"]}
    validation = {item["reference_fingerprint"] for item in result["untouched_validation"]}
    if not calibration.isdisjoint(validation):
        raise ValueError("duplicate circuit fingerprints across calibration and validation splits")
    return result


def generate_paired_case(template: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Apply the registered pilot intervention to the template for one stratum."""
    scenario = row["fault_strength_stratum"]
    labels = {
        "identical_control": "NO_ANOMALY",
        "measurement_equivalent_control": "NO_ANOMALY",
        "subtle": "ANOMALY",
        "moderate": "ANOMALY",
        "large": "ANOMALY",
    }
    if scenario not in labels:
        raise ValueError(f"unsupported calibration stratum: {scenario}")
    if row.get("template_seed") != template.get("template_seed"):
        raise ValueError("ledger row and template seed do not match")

    reference = [tuple(instruction) for instruction in template["reference"]]
    submitted = list(reference)
    if scenario == "identical_control":
        intervention = {"kind": "identity"}
    elif scenario == "measurement_equivalent_control":
        measure_indices = [i for i, operation in enumerate(submitted) if operation[0] == "measure"]
        if len(measure_indices) != 1:
            raise ValueError("reference template must have exactly one terminal measurement")
        submitted.insert(measure_indices[0], ("z", 3))
        intervention = {"kind": "measurement_equivalent_phase", "gate": "z", "qubit": 3}
    else:
        angles = {"subtle": 1.45, "moderate": 1.2, "large": 0.4}
        matches = [i for i, operation in enumerate(submitted) if operation[:2] == ("h", 0)]
        if len(matches) != 1:
            raise ValueError("reference template must contain exactly one H gate on qubit 0")
        index = matches[0]
        angle = angles[scenario]
        submitted[index] = ("ry", 0, angle)
        intervention = {"kind": "replace_gate", "index": index, "from": ["h", 0], "to": ["ry", 0, angle]}

    return {
        "configuration_id": row["configuration_id"],
        "template_id": row["template_id"],
        "split": row["split"],
        "scenario": scenario,
        "fault_strength_stratum": scenario,
        "template_seed": template["template_seed"],
        "sampling_seed": row["seed"],
        "shot_count": row["shot_count"],
        "expected_label": labels[scenario],
        "recipe_id": template["recipe_id"],
        "reference_fingerprint": template["reference_fingerprint"],
        "generation_metadata": template["generation_metadata"],
        "intervention": intervention,
        "reference": reference,
        "submitted": submitted,
        "output_location": row["output_location"],
    }


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    terms = [
        math.comb(n, successes)
        * probability ** successes
        * (1.0 - probability) ** (n - successes)
        for successes in range(k + 1)
    ]
    return sum(terms)


def _cp_upper(successes: int, trials: int, alpha: float = 0.05) -> float | None:
    if trials <= 0:
        return None
    if successes >= trials:
        return 1.0
    low, high = successes / trials, 1.0
    for _ in range(60):
        middle = (low + high) / 2
        if _binomial_cdf(successes, trials, middle) > alpha:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _cp_lower(successes: int, trials: int, alpha: float = 0.05) -> float | None:
    """Return the one-sided exact Clopper-Pearson lower confidence bound."""
    if trials <= 0:
        return None
    if successes <= 0:
        return 0.0
    low, high = 0.0, successes / trials
    for _ in range(60):
        middle = (low + high) / 2
        survival = 1.0 - _binomial_cdf(successes - 1, trials, middle)
        # The lower bound solves P_p[X >= successes] = alpha.
        if survival < alpha:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def summarize_results(records: list[dict[str, Any]], alpha: float = 0.05) -> list[dict[str, Any]]:
    """Summarize detector counts and one-sided exact binomial bounds by stratum.

    Each input record represents one independent generated template at one shot
    count. Repeated shot levels are grouped separately and never pooled.
    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        label = record.get("expected_label")
        if label is None:
            if not isinstance(record.get("expected_anomaly"), bool):
                raise ValueError("result record requires an explicit valid expected label")
        elif label not in {"ANOMALY", "NO_ANOMALY"}:
            raise ValueError(f"unsupported expected label: {label}")
        split = record["split"]
        shots = record.get("shot_count", record.get("shots"))
        scenario = record.get("scenario", record.get("fault_strength_stratum"))
        strength = record.get("fault_strength_stratum", scenario)
        groups.setdefault((split, shots, scenario, strength), []).append(record)

    summaries = []
    for (split, shots, scenario, strength), group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        def is_positive(record: dict[str, Any]) -> bool:
            label = record["expected_label"] if "expected_label" in record else (
                "ANOMALY" if record["expected_anomaly"] else "NO_ANOMALY"
            )
            return label == "ANOMALY"

        positives = [r for r in group if is_positive(r)]
        negatives = [r for r in group if not is_positive(r)]
        for detector in ("current_detected", "permutation_detected"):
            tp = sum(bool(r[detector]) for r in positives)
            fn = len(positives) - tp
            fp = sum(bool(r[detector]) for r in negatives)
            tn = len(negatives) - fp
            summaries.append({
                "split": split, "shot_count": shots, "scenario": scenario,
                "fault_strength_stratum": strength, "detector": detector,
                "n_positive": len(positives), "tp": tp, "fn": fn,
                "sensitivity": tp / len(positives) if positives else None,
                "sensitivity_lower_95_one_sided": _cp_lower(tp, len(positives), alpha),
                "n_negative": len(negatives), "fp": fp, "tn": tn,
                "false_positive_rate": fp / len(negatives) if negatives else None,
                "false_positive_upper_95_one_sided": _cp_upper(fp, len(negatives), alpha),
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / len(positives) if positives else None,
                "abstentions": sum(bool(r.get("abstained", False)) for r in group),
                "n_configurations": len(group),
            })
    return summaries


def _permutation_p_value(
    first_counts: list[int], second_counts: list[int], repetitions: int, seed: int,
) -> float:
    """Match the fixture's pooled-label two-sample TVD permutation test."""
    first = np.asarray(first_counts, dtype=np.int64)
    second = np.asarray(second_counts, dtype=np.int64)
    n_first, n_second = int(first.sum()), int(second.sum())
    pooled = np.repeat(np.arange(len(first)), first + second)
    observed = 0.5 * float(np.abs(first / n_first - second / n_second).sum())
    rng = np.random.default_rng(seed)
    at_least_observed = 0
    for _ in range(repetitions):
        permuted = rng.permutation(pooled)
        perm_first = np.bincount(permuted[:n_first], minlength=len(first))
        perm_second = np.bincount(permuted[n_first:], minlength=len(first))
        permuted_tvd = 0.5 * float(np.abs(perm_first / n_first - perm_second / n_second).sum())
        if permuted_tvd >= observed - 1e-12:
            at_least_observed += 1
    return (at_least_observed + 1) / (repetitions + 1)


def _atomic_create_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically create JSON without ever replacing an existing destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
    # Keep the staging filename short. Repeating a long configuration ID here
    # can exceed Windows' effective path limit under deeply nested pytest
    # basetemp paths, which surfaces as FileNotFoundError from mkstemp.
    handle, temporary = tempfile.mkstemp(prefix=".q.", suffix=".partial", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link creation is atomic and fails if ``path`` already exists.
        os.link(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _atomic_replace_owned_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically update this run's manifest, which is created at run start."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(8):
            try:
                os.replace(temp_path, path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                # Cloud sync and antivirus scanners can briefly hold the
                # manifest on Windows. Retry the atomic replace; result files
                # remain independently committed and create-only.
                time.sleep(min(0.1 * (2 ** attempt), 1.0))
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _empirical_counts(distribution: dict[str, float], support: list[str], shots: int) -> list[int]:
    counts = [int(round(float(distribution.get(key, 0.0)) * shots)) for key in support]
    if sum(counts) != shots:
        raise ValueError("sampled distribution does not map to the declared shot count")
    return counts


def evaluate_preregistered_targets(summaries: list[dict[str, Any]], spec: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Evaluate the fixture's per-stratum, per-shot count targets unchanged."""
    spec = spec or load_spec()
    plan = spec["proposed_next_calibration_plan"]
    minimum_counts = plan["minimum_detected_at_n_120"]
    sensitivities = plan["sensitivity_targets_by_effect_and_shots"]
    fp_limit = int(plan["false_positive_targets"]["maximum_false_positives_at_n"])
    outcomes = []
    for summary in summaries:
        detector = summary["detector"]
        stratum = summary["scenario"]
        shots = str(summary["shot_count"])
        if stratum in {"identical_control", "measurement_equivalent_control"}:
            count = int(summary["fp"])
            outcomes.append({
                "split": summary["split"], "detector": detector, "shot_count": int(shots),
                "scenario": stratum, "observed": count, "maximum": fp_limit,
                "passed": count <= fp_limit,
            })
        else:
            target_rate = float(sensitivities[stratum][shots])
            minimum = int(minimum_counts[f"{target_rate:.2f}"])
            count = int(summary["tp"])
            outcomes.append({
                "split": summary["split"], "detector": detector, "shot_count": int(shots),
                "scenario": stratum, "observed": count, "minimum": minimum,
                "target_rate": target_rate, "passed": count >= minimum,
            })
    return outcomes


def run_full_calibration(spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute the frozen 3,600-row plan, atomically preserving each result.

    Expected labels and scenario metadata are added only after the public
    diagnosis call has returned. Any exception leaves prior completed JSONs
    and a manifest identifying the failed configuration.
    """
    from datetime import datetime, timezone
    from time import monotonic

    from quantum_simulator.diagnosis_engine import diagnose_circuit

    spec = spec or load_spec()
    rows = validate_execution_preflight(spec)
    plan = spec["proposed_next_calibration_plan"]
    design = plan["design"]
    repository_root = Path(__file__).resolve().parents[1]
    output_root = repository_root / Path(design["calibration_output_directory"])
    output_root.mkdir(parents=True, exist_ok=False)
    manifest_path = output_root / "run_manifest.json"
    manifest = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "primary_recipe": PRIMARY_RECIPE,
        "planned_configurations": len(rows),
        "planned_shot_circuit_executions": int(plan["workload"]["shot_circuit_executions"]),
        "completed_configurations": 0,
        "completed_shot_circuit_executions": 0,
        "failed_configuration_id": None,
        "failed_error": None,
    }
    _atomic_create_json(manifest_path, manifest)
    started = monotonic()
    templates: dict[tuple[str, int], dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    support = [format(index, "04b") for index in range(16)]
    permutations = int(spec["permutation_repetitions"])
    alpha = float(spec["permutation_alpha"])

    for row in rows:
        try:
            template_key = (row["split"], row["template_seed"])
            template = templates.get(template_key)
            if template is None:
                template = generate_primary_template(row["template_seed"])
                templates[template_key] = template
            case = generate_paired_case(template, row)
            inputs = diagnosis_inputs(case, row)
            diagnosis = diagnose_circuit(**inputs)
            stats = diagnosis.statistics_result
            if stats is None:
                raise RuntimeError("diagnosis adapter returned no distribution comparison")
            expected_counts = _empirical_counts(stats.expected_distribution, support, row["shot_count"])
            observed_counts = _empirical_counts(stats.observed_distribution, support, row["shot_count"])
            permutation_seed = row["seed"] + sum(row["scenario"].encode("utf-8"))
            p_value = _permutation_p_value(expected_counts, observed_counts, permutations, permutation_seed)
            record = {
                "split": row["split"],
                "template_id": row["template_id"],
                "template_seed": row["template_seed"],
                "reference_fingerprint": template["reference_fingerprint"],
                "configuration_id": row["configuration_id"],
                "scenario": row["scenario"],
                "fault_strength_stratum": row["fault_strength_stratum"],
                "shot_count": row["shot_count"],
                "sampling_seed": row["seed"],
                "expected_label": row["expected_label"],
                "intervention": case["intervention"],
                "generation_metadata": case["generation_metadata"],
                "current_detected": bool(stats.anomaly_detected),
                "permutation_p": p_value,
                "permutation_detected": p_value <= alpha,
                "abstained": diagnosis.category.value == "AMBIGUOUS / INSUFFICIENT_EVIDENCE",
                "diagnosis_category": diagnosis.category.value,
                "tvd": stats.total_variation_distance,
                "finite_shot_guard": stats.finite_shot_guard,
                "effective_threshold": max(stats.threshold, stats.finite_shot_guard),
                "expected_distribution": stats.expected_distribution,
                "observed_distribution": stats.observed_distribution,
                "status": "completed",
            }
            destination = repository_root / Path(row["output_location"])
            _atomic_create_json(destination, {"status": "completed", "configuration_id": row["configuration_id"], "result": record})
            records.append(record)
            manifest["completed_configurations"] += 1
            manifest["completed_shot_circuit_executions"] += 2 * int(row["shot_count"])
            manifest["last_completed_configuration_id"] = row["configuration_id"]
            _atomic_replace_owned_json(manifest_path, manifest)
        except Exception as exc:
            failure = {
                "status": "failed",
                "configuration_id": row["configuration_id"],
                "split": row["split"],
                "template_id": row["template_id"],
                "scenario": row["scenario"],
                "shot_count": row["shot_count"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failed_path = output_root / "failures" / f"{row['configuration_id']}.json"
            _atomic_create_json(failed_path, failure)
            manifest.update({
                "status": "failed",
                "failed_configuration_id": row["configuration_id"],
                "failed_error": f"{type(exc).__name__}: {exc}",
            })
            _atomic_replace_owned_json(manifest_path, manifest)
            raise

    summaries = summarize_results(records)
    target_results = evaluate_preregistered_targets(summaries, spec)
    summary = {
        "status": "completed",
        "primary_recipe": PRIMARY_RECIPE,
        "exploratory_recipe_excluded": "seeded_paired_topology_phase_tags",
        "planned_configurations": len(rows),
        "completed_configurations": len(records),
        "planned_shot_circuit_executions": int(plan["workload"]["shot_circuit_executions"]),
        "completed_shot_circuit_executions": sum(2 * int(record["shot_count"]) for record in records),
        "runtime_seconds": monotonic() - started,
        "configuration_fingerprints": sorted(row["configuration_id"] for row in rows),
        "summaries": summaries,
        "acceptance_targets": target_results,
        "all_targets_passed": all(item["passed"] for item in target_results),
    }
    _atomic_create_json(output_root / "summary.json", summary)
    manifest.update({
        "status": "completed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "completed_configurations": len(records),
        "completed_shot_circuit_executions": summary["completed_shot_circuit_executions"],
        "runtime_seconds": summary["runtime_seconds"],
    })
    _atomic_replace_owned_json(manifest_path, manifest)
    return summary


def run_resume_calibration(spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resume only ledger rows without valid output, preserving every existing file."""
    from datetime import datetime, timezone
    from time import monotonic

    from quantum_simulator.diagnosis_engine import diagnose_circuit

    spec = spec or load_spec()
    preflight = resume_preflight(spec)
    if not preflight["ready"]:
        raise RuntimeError("resume preflight blocked: " + "; ".join(preflight["blockers"]))
    state = preflight["state"]
    rows = state["rows"]
    root = Path(__file__).resolve().parents[1]
    plan = spec["proposed_next_calibration_plan"]
    output_root = root / Path(plan["design"]["calibration_output_directory"])
    manifest_path = output_root / "run_manifest.json"
    started = monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    valid_by_id = {row["configuration_id"]: result for row, result, _ in state["valid"]}
    todo = {row["configuration_id"]: row for row in state["missing"]}
    todo.update({item["row"]["configuration_id"]: item["row"] for item in state["invalid"]})
    templates: dict[tuple[str, int], dict[str, Any]] = {}
    support = [format(index, "04b") for index in range(16)]
    permutations = int(spec["permutation_repetitions"])
    alpha = float(spec["permutation_alpha"])

    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {"prior_manifest_invalid": True}
    # Reconcile from artifacts on every invocation. A prior manifest may be
    # stale or may describe an earlier resume attempt that itself stopped.
    original_remaining = len(todo)
    original_completed = len(valid_by_id)
    manifest.update({
        "status": "resuming",
        "resume_started_utc": manifest.get("resume_started_utc", started_utc),
        "primary_recipe": PRIMARY_RECIPE,
        "planned_configurations": len(rows),
        "planned_shot_circuit_executions": int(plan["workload"]["shot_circuit_executions"]),
        "completed_configurations": len(valid_by_id),
        "completed_shot_circuit_executions": sum(2 * int(r["shot_count"]) for r, _, _ in state["valid"]),
        "remaining_configurations_at_resume_start": original_remaining,
        "failed_configuration_id": None,
        "failed_error": None,
    })
    _atomic_replace_owned_json(manifest_path, manifest)
    new_count = 0
    try:
        for row in rows:
            config_id = row["configuration_id"]
            if config_id not in todo:
                continue
            try:
                template_key = (row["split"], row["template_seed"])
                template = templates.get(template_key)
                if template is None:
                    template = generate_primary_template(row["template_seed"])
                    templates[template_key] = template
                case = generate_paired_case(template, row)
                diagnosis = diagnose_circuit(**diagnosis_inputs(case, row))
                stats = diagnosis.statistics_result
                if stats is None:
                    raise RuntimeError("diagnosis adapter returned no distribution comparison")
                expected_counts = _empirical_counts(stats.expected_distribution, support, row["shot_count"])
                observed_counts = _empirical_counts(stats.observed_distribution, support, row["shot_count"])
                permutation_seed = row["seed"] + sum(row["scenario"].encode("utf-8"))
                p_value = _permutation_p_value(expected_counts, observed_counts, permutations, permutation_seed)
                record = {
                    "split": row["split"], "template_id": row["template_id"],
                    "template_seed": row["template_seed"],
                    "reference_fingerprint": template["reference_fingerprint"],
                    "configuration_id": config_id, "scenario": row["scenario"],
                    "fault_strength_stratum": row["fault_strength_stratum"],
                    "shot_count": row["shot_count"], "sampling_seed": row["seed"],
                    "expected_label": row["expected_label"], "intervention": case["intervention"],
                    "generation_metadata": case["generation_metadata"],
                    "current_detected": bool(stats.anomaly_detected),
                    "permutation_p": p_value, "permutation_detected": p_value <= alpha,
                    "abstained": diagnosis.category.value == "AMBIGUOUS / INSUFFICIENT_EVIDENCE",
                    "diagnosis_category": diagnosis.category.value,
                    "tvd": stats.total_variation_distance,
                    "finite_shot_guard": stats.finite_shot_guard,
                    "effective_threshold": max(stats.threshold, stats.finite_shot_guard),
                    "expected_distribution": stats.expected_distribution,
                    "observed_distribution": stats.observed_distribution,
                    "status": "completed",
                }
                payload = {"status": "completed", "configuration_id": config_id, "result": record}
                canonical = root / Path(row["output_location"])
                destination = canonical if not canonical.exists() else canonical.with_name(f"{config_id}.recovered.json")
                _atomic_create_json(destination, payload)
                valid, reason = _validate_calibration_payload(payload, row)
                if not valid:
                    raise RuntimeError(f"new result failed schema validation: {reason}")
                valid_by_id[config_id] = record
                new_count += 1
                manifest["completed_configurations"] = len(valid_by_id)
                manifest["completed_shot_circuit_executions"] += 2 * int(row["shot_count"])
                manifest["last_completed_configuration_id"] = config_id
                manifest["last_completed_output"] = str(destination.relative_to(root))
                _atomic_replace_owned_json(manifest_path, manifest)
            except Exception as exc:
                failure = {
                    "status": "failed", "configuration_id": config_id,
                    "split": row["split"], "template_id": row["template_id"],
                    "scenario": row["scenario"], "shot_count": row["shot_count"],
                    "error_type": type(exc).__name__, "error": str(exc),
                }
                failures_root = output_root / "failures"
                failures_root.mkdir(parents=True, exist_ok=True)
                failure_path = failures_root / f"{config_id}.resume.json"
                attempt = 2
                while failure_path.exists():
                    failure_path = failures_root / f"{config_id}.resume.{attempt}.json"
                    attempt += 1
                _atomic_create_json(failure_path, failure)
                manifest.update({"status": "failed", "failed_configuration_id": config_id,
                                 "failed_error": f"{type(exc).__name__}: {exc}"})
                _atomic_replace_owned_json(manifest_path, manifest)
                raise
    except Exception:
        raise

    if len(valid_by_id) != len(rows):
        raise RuntimeError(f"resume ended with {len(valid_by_id)} valid outputs of {len(rows)}")
    ordered_records = [valid_by_id[row["configuration_id"]] for row in rows]
    summaries = summarize_results(ordered_records)
    target_results = evaluate_preregistered_targets(summaries, spec)
    runtime = monotonic() - started
    summary = {
        "status": "completed", "primary_recipe": PRIMARY_RECIPE,
        "exploratory_recipe_excluded": "seeded_paired_topology_phase_tags",
        "planned_configurations": len(rows), "completed_configurations": len(ordered_records),
        "newly_completed_configurations": max(0, len(ordered_records) - original_completed),
        "newly_completed_this_invocation": new_count,
        "planned_shot_circuit_executions": int(plan["workload"]["shot_circuit_executions"]),
        "completed_shot_circuit_executions": sum(2 * int(r["shot_count"]) for r in rows),
        "runtime_seconds": runtime,
        "configuration_fingerprints": sorted(row["configuration_id"] for row in rows),
        "summaries": summaries, "acceptance_targets": target_results,
        "all_targets_passed": all(item["passed"] for item in target_results),
        "failed_configurations": [],
    }
    summary_path = output_root / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite existing final report: {summary_path}")
    _atomic_create_json(summary_path, summary)
    manifest.update({
        "status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "completed_configurations": len(rows),
        "completed_shot_circuit_executions": summary["completed_shot_circuit_executions"],
        "newly_completed_configurations": summary["newly_completed_configurations"],
        "newly_completed_this_invocation": new_count, "runtime_seconds": runtime,
        "failed_configuration_id": None, "failed_error": None,
    })
    _atomic_replace_owned_json(manifest_path, manifest)
    return summary
