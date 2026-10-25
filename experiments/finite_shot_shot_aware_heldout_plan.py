"""Metadata-only preregistration for a future shot-aware detector evaluation.

This module builds deterministic ledger rows only. It does not simulate circuits,
read quantum hardware, or execute detector comparisons.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.finite_shot_calibration_plan import (
    PRIMARY_RECIPE, STRATA, generate_candidate_template, load_spec,
)

SPLIT = "new_heldout_20261014"
TEMPLATE_COUNT = 600
SHOT_COUNTS = (256, 1024, 4096)
TEMPLATE_SEED_ROOT = 60_271_014
SAMPLING_SEED_ROOT = 70_361_015
OUTPUT_ROOT = "C:/Quantum_Simulator_runs/finite_shot_shot_aware_heldout_run"
CANDIDATE_TVD_CUTOFFS = {256: 0.18359375, 1024: 0.09375, 4096: 0.0478515625}
SENSITIVITY_MINIMUMS = {
    "subtle": {256: 0.50, 1024: 0.80, 4096: 0.90},
    "moderate": {256: 0.80, 1024: 0.90, 4096: 0.95},
    "large": {256: 0.95, 1024: 0.95, 4096: 0.95},
}
MAX_FALSE_POSITIVE_RATE = 0.05
CONFIDENCE_LEVEL = 0.95
MIN_DETECTIONS_AT_N_600 = {0.50: 321, 0.80: 497, 0.90: 553, 0.95: 580}
MAX_FALSE_POSITIVES_AT_N_600 = 20


def build_heldout_ledger() -> list[dict[str, Any]]:
    """Build the frozen 9,000-row metadata ledger; never executes simulations."""
    rows: list[dict[str, Any]] = []
    shot_index = {shots: index for index, shots in enumerate(SHOT_COUNTS)}
    for stratum_index, (scenario, is_positive) in enumerate(STRATA):
        for template_index in range(TEMPLATE_COUNT):
            template_seed = TEMPLATE_SEED_ROOT + template_index
            template_id = f"{SPLIT}:template:{template_index:04d}"
            for shots in SHOT_COUNTS:
                sampling_seed = SAMPLING_SEED_ROOT + 10_000 * stratum_index + 10 * template_index + shot_index[shots]
                fields = {
                    "split": SPLIT, "template_id": template_id,
                    "scenario": scenario, "fault_strength_stratum": scenario,
                    "shot_count": shots, "template_seed": template_seed,
                    "seed": sampling_seed,
                }
                canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
                config_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
                rows.append({
                    "configuration_id": config_id, **fields,
                    "expected_label": "ANOMALY" if is_positive else "NO_ANOMALY",
                    "output_location": f"{OUTPUT_ROOT}/{scenario}/{config_id}.json",
                })
    if len(rows) != 9_000 or len({row["configuration_id"] for row in rows}) != len(rows):
        raise AssertionError("held-out ledger size or configuration ID uniqueness failed")
    return rows


def frozen_criteria() -> dict[str, Any]:
    """Return the preregistered criteria in a fresh object."""
    return {
        "candidate_cutoffs": dict(CANDIDATE_TVD_CUTOFFS),
        "sensitivity_minimums": {k: dict(v) for k, v in SENSITIVITY_MINIMUMS.items()},
        "maximum_false_positive_rate": MAX_FALSE_POSITIVE_RATE,
        "minimum_detections_at_n_600": dict(MIN_DETECTIONS_AT_N_600),
        "maximum_false_positives_at_n_600": MAX_FALSE_POSITIVES_AT_N_600,
        "one_sided_confidence": CONFIDENCE_LEVEL,
        "abstentions": "retain every row in denominator; an abstention is a missed detection for sensitivity, not a false positive for FPR; report abstention rate and exact two-sided 95% CI separately; no exclusion",
        "acceptance": "candidate passes only if every positive stratum x shot sensitivity one-sided exact 95% lower Clopper-Pearson bound meets its registered minimum and every control scenario x shot FPR one-sided exact 95% upper bound is <= 0.05",
    }


def check_disjointness() -> dict[str, int]:
    """Check new template IDs, seeds, and fingerprints against both old splits."""
    spec = load_spec()
    old_design = spec["proposed_next_calibration_plan"]["design"]
    old_template_seeds: set[int] = set()
    old_template_ids = {f"{split}:template:{index:03d}" for split in ("calibration", "untouched_validation") for index in range(120)}
    old_sampling_seeds: set[int] = set()
    old_fingerprints: set[str] = set()
    for key in ("calibration", "untouched_validation"):
        for seed in range(int(old_design["template_generation_seed_roots"][key]), int(old_design["template_generation_seed_roots"][key]) + 120):
            old_template_seeds.add(seed)
            old_fingerprints.add(generate_candidate_template(seed, PRIMARY_RECIPE)["reference_fingerprint"])
        root = int(old_design["sampling_seed_roots"][key])
        old_sampling_seeds.update(root + 10_000 * s + 10 * i + j
            for s in range(5) for i in range(120) for j in range(3))
    new_template_seeds = {TEMPLATE_SEED_ROOT + i for i in range(TEMPLATE_COUNT)}
    rows = build_heldout_ledger()
    new_sampling_seeds = {row["seed"] for row in rows}
    new_fingerprints = {generate_candidate_template(seed, PRIMARY_RECIPE)["reference_fingerprint"] for seed in new_template_seeds}
    new_template_ids = {row["template_id"] for row in rows}
    if old_template_ids & new_template_ids or old_template_seeds & new_template_seeds or old_sampling_seeds & new_sampling_seeds:
        raise ValueError("held-out seed namespaces overlap prior calibration or validation")
    if old_fingerprints & new_fingerprints:
        raise ValueError("held-out reference fingerprint reused a prior calibration or validation template")
    if len(new_template_ids) != TEMPLATE_COUNT or len(new_fingerprints) != TEMPLATE_COUNT:
        raise ValueError("held-out template identities or fingerprints are not unique")
    return {
        "prior_templates_checked": len(old_fingerprints),
        "new_templates": len(new_fingerprints),
        "new_configurations": len(rows),
        "prior_sampling_seeds_checked": len(old_sampling_seeds),
        "new_sampling_seeds": len(new_sampling_seeds),
    }


def workload() -> dict[str, int]:
    """Compute deterministic workload arithmetic without running a simulator."""
    configs = TEMPLATE_COUNT * len(STRATA) * len(SHOT_COUNTS)
    executions = 2 * TEMPLATE_COUNT * len(STRATA) * sum(SHOT_COUNTS)
    return {"configurations": configs, "shot_circuit_executions": executions}
